#!/usr/bin/env python3

import argparse
import time

import cv2
import numpy as np
import rclpy
import torch

from cv_bridge import CvBridge
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from lerobot.common.policies.act.modeling_act import ACTPolicy


# 学習時の observation.state と同じ順番
STATE_JOINT_NAMES = [
    "shoulder_joint",
    "elbow_joint",
    "left_finger_joint",
    "right_finger_joint",
]

# 学習時の action と同じ順番
ACTION_NAMES = [
    "linear_x",
    "angular_z",
    "shoulder_joint",
    "elbow_joint",
    "left_finger_joint",
    "right_finger_joint",
]

ACTION_DIMENSION = len(ACTION_NAMES)


class ImitationPolicyNode(Node):

    def __init__(
        self,
        model_path,
        inference_hz=10.0,
        enable_image=True,
    ):
        super().__init__(
            "color_sorting_act_infer"
        )

        self.device = torch.device("cpu")
        self.bridge = CvBridge()
        self.enable_image = enable_image

        # ---------------------------------
        # 安全設定
        # ---------------------------------

        self.max_linear_x = 0.02
        self.max_angular_z = 0.10

        self.shoulder_limit = (-1.2, 1.2)
        self.elbow_limit = (-1.5, 1.5)
        self.left_finger_limit = (0.0, 0.08)
        self.right_finger_limit = (0.0, 0.08)

        # 学習時の画像サイズに合わせる
        self.image_width = 640
        self.image_height = 480

        # ---------------------------------
        # 最新観測
        # ---------------------------------

        self.latest_image = None
        self.latest_state = None
        self.last_image_time = None
        self.last_state_time = None

        self.last_action = np.zeros(
            ACTION_DIMENSION,
            dtype=np.float32,
        )

        # ---------------------------------
        # ROS 2 publisher
        # ---------------------------------

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        self.arm_trajectory_publisher = (
            self.create_publisher(
                JointTrajectory,
                "/arm_controller/joint_trajectory",
                10,
            )
        )

        # ---------------------------------
        # ROS 2 subscriber
        # ---------------------------------

        self.joint_state_subscriber = (
            self.create_subscription(
                JointState,
                "/joint_states",
                self.joint_state_callback,
                10,
            )
        )

        self.image_subscriber = (
            self.create_subscription(
                Image,
                "/sorting_camera/image_raw",
                self.image_callback,
                10,
            )
        )

        # ---------------------------------
        # 推論タイマー
        # ---------------------------------

        timer_period = 1.0 / inference_hz

        self.timer = self.create_timer(
            timer_period,
            self.inference_callback,
        )

        # ---------------------------------
        # ACTモデル読み込み
        # ---------------------------------

        self.get_logger().info(
            f"Loading ACT policy from: {model_path}"
        )

        self.policy = ACTPolicy.from_pretrained(
            model_path
        )

        self.policy.to(self.device)
        self.policy.eval()

        self.get_logger().info(
            f"Expected action dimension: "
            f"{ACTION_DIMENSION}"
        )

        self.get_logger().info(
            "Action order: "
            + ", ".join(ACTION_NAMES)
        )

        self.get_logger().info(
            "ACT policy loaded"
        )

    # ==================================================
    # 観測コールバック
    # ==================================================

    def image_callback(self, msg):
        """
        ROS ImageをRGBのnumpy配列へ変換する。
        """

        try:
            image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="rgb8",
            )

            image = np.asarray(
                image,
                dtype=np.uint8,
            )

            self.latest_image = image
            self.last_image_time = time.monotonic()

        except Exception as error:
            self.get_logger().error(
                f"Image conversion failed: {error}"
            )

    def joint_state_callback(self, msg):
        """
        /joint_statesから4関節を抽出する。
        """

        name_to_position = dict(
            zip(msg.name, msg.position)
        )

        positions = []

        for joint_name in STATE_JOINT_NAMES:
            if joint_name not in name_to_position:
                self.get_logger().warning(
                    f"Missing joint: {joint_name}"
                )
                return

            positions.append(
                float(name_to_position[joint_name])
            )

        self.latest_state = np.asarray(
            positions,
            dtype=np.float32,
        )

        self.last_state_time = time.monotonic()

    # ==================================================
    # 観測の作成
    # ==================================================

    def make_state_tensor(self):
        """
        shape: (1, 4)
        """

        return torch.from_numpy(
            self.latest_state.copy()
        ).to(
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)

    def make_image_tensor(self):
        """
        学習画像をACT入力用tensorへ変換する。

        numpy:
            (H, W, C), uint8, RGB

        torch:
            (B, C, H, W), float32, [0, 1]
        """

        image = self.latest_image

        if image is None:
            return None

        image = cv2.resize(
            image,
            (
                self.image_width,
                self.image_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        image = image.astype(
            np.float32
        ) / 255.0

        image_tensor = torch.from_numpy(
            image
        ).permute(
            2, 0, 1
        ).unsqueeze(0).to(
            device=self.device,
            dtype=torch.float32,
        )

        return image_tensor

    def make_observation(self):
        """
        学習時のfeaturesと同じキーを使う。

        学習時に画像をfeaturesへ含めていない場合は、
        --without-imageで画像入力を無効化できる。
        """

        state_tensor = self.make_state_tensor()

        observation = {
            "observation.state": state_tensor,
            "observation.environment_state":
                state_tensor.clone(),
        }

        if self.enable_image:
            image_tensor = (
                self.make_image_tensor()
            )

            if image_tensor is None:
                return None

            observation[
                "observation.images.camera"
            ] = image_tensor

        return observation

    # ==================================================
    # 推論
    # ==================================================

    def inference_callback(self):
        """
        最新観測からACT actionを推論する。
        """

        if self.latest_state is None:
            return

        if self.enable_image:
            if self.latest_image is None:
                return

        # 古い観測を使わない
        if self.last_state_time is not None:
            state_age = (
                time.monotonic()
                - self.last_state_time
            )

            if state_age > 1.0:
                self.get_logger().warning(
                    "Joint state is too old"
                )
                self.publish_stop()
                return

        if self.enable_image:
            if self.last_image_time is not None:
                image_age = (
                    time.monotonic()
                    - self.last_image_time
                )

                if image_age > 1.0:
                    self.get_logger().warning(
                        "Camera image is too old"
                    )
                    self.publish_stop()
                    return

        try:
            observation = (
                self.make_observation()
            )

            if observation is None:
                return

            with torch.no_grad():
                action_tensor = (
                    self.policy.select_action(
                        observation
                    )
                )

            action = (
                action_tensor
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

        except Exception as error:
            self.get_logger().error(
                f"Policy inference failed: {error}"
            )
            self.publish_stop()
            return

        if len(action) != ACTION_DIMENSION:
            self.get_logger().error(
                "Invalid action dimension: "
                f"expected {ACTION_DIMENSION}, "
                f"got {len(action)}"
            )
            self.publish_stop()
            return

        if not np.all(
            np.isfinite(action)
        ):
            self.get_logger().error(
                "Action contains NaN or infinity"
            )
            self.publish_stop()
            return

        self.last_action = action.astype(
            np.float32
        )

        self.publish_base_action(
            self.last_action
        )

        self.publish_arm_action(
            self.last_action
        )

        self.get_logger().info(
            "action="
            + np.array2string(
                self.last_action,
                precision=4,
            )
        )

    # ==================================================
    # 移動action
    # ==================================================

    def publish_base_action(self, action):
        """
        action[0] : linear.x
        action[1] : angular.z
        """

        linear_x = float(
            np.clip(
                action[0],
                -self.max_linear_x,
                self.max_linear_x,
            )
        )

        angular_z = float(
            np.clip(
                action[1],
                -self.max_angular_z,
                self.max_angular_z,
            )
        )

        message = Twist()
        message.linear.x = linear_x
        message.angular.z = angular_z

        self.cmd_vel_publisher.publish(
            message
        )

    # ==================================================
    # アームaction
    # ==================================================

    def publish_arm_action(self, action):
        """
        action[2:6]を4関節の
        JointTrajectoryとして公開する。
        """

        positions = [
            float(
                np.clip(
                    action[2],
                    self.shoulder_limit[0],
                    self.shoulder_limit[1],
                )
            ),
            float(
                np.clip(
                    action[3],
                    self.elbow_limit[0],
                    self.elbow_limit[1],
                )
            ),
            float(
                np.clip(
                    action[4],
                    self.left_finger_limit[0],
                    self.left_finger_limit[1],
                )
            ),
            float(
                np.clip(
                    action[5],
                    self.right_finger_limit[0],
                    self.right_finger_limit[1],
                )
            ),
        ]

        trajectory = JointTrajectory()

        trajectory.header.stamp = (
            self.get_clock().now().to_msg()
        )

        trajectory.joint_names = (
            STATE_JOINT_NAMES.copy()
        )

        point = JointTrajectoryPoint()
        point.positions = positions

        # 次の推論指令までに到達させる時間
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(
            0.2 * 1e9
        )

        trajectory.points.append(
            point
        )

        self.arm_trajectory_publisher.publish(
            trajectory
        )

    # ==================================================
    # 停止
    # ==================================================

    def publish_stop(self):
        """
        ベースを停止する。
        """

        message = Twist()
        message.linear.x = 0.0
        message.angular.z = 0.0

        self.cmd_vel_publisher.publish(
            message
        )

    def destroy_node(self):
        self.publish_stop()
        super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(
        description=(
            "ACT inference node for "
            "color sorting robot"
        )
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help=(
            "Path to trained ACTPolicy "
            "model directory"
        ),
    )

    parser.add_argument(
        "--inference-hz",
        type=float,
        default=10.0,
        help="Inference frequency",
    )

    parser.add_argument(
        "--without-image",
        action="store_true",
        help=(
            "Use this only when the model was "
            "trained without image observations"
        ),
    )

    parsed_args, ros_args = (
        parser.parse_known_args(args)
    )

    rclpy.init(args=ros_args)

    node = None

    try:
        node = ImitationPolicyNode(
            model_path=parsed_args.model_path,
            inference_hz=parsed_args.inference_hz,
            enable_image=not parsed_args.without_image,
        )

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        print(
            f"Failed to start inference node: "
            f"{error}"
        )

    finally:
        if node is not None:
            node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
