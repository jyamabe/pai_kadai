#!/usr/bin/env python3

import argparse
import os
import time

import cv2
import numpy as np
import rclpy
import torch

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from lerobot.common.policies.act.modeling_act import ACTPolicy


# ============================================================
# 学習時と同じ関節順序
# ============================================================

STATE_JOINT_NAMES = [
    "shoulder_joint",
    "elbow_joint",
    "left_finger_joint",
    "right_finger_joint",
]


# ============================================================
# 学習時と同じaction順序
# ============================================================

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

        # ----------------------------------------------------
        # 基本設定
        # ----------------------------------------------------

        self.device = torch.device("cpu")
        self.bridge = CvBridge()
        self.enable_image = enable_image

        self.image_topic = (
            "/sorting_camera/image_raw"
        )

        self.joint_state_topic = (
            "/joint_states"
        )

        self.cmd_vel_topic = (
            "/cmd_vel"
        )

        # アーム・グリッパのJointTrajectory入力先
        self.arm_trajectory_topic = (
            "/arm_controller/joint_trajectory"
        )

        # ----------------------------------------------------
        # 安全制限
        # ----------------------------------------------------

        self.max_linear_x = 0.02
        self.max_angular_z = 0.10

        self.shoulder_limit = (-1.2, 1.2)
        self.elbow_limit = (-1.5, 1.5)
        self.left_finger_limit = (0.0, 0.08)
        self.right_finger_limit = (0.0, 0.08)

        # ----------------------------------------------------
        # 画像サイズ
        # 学習時のサイズと一致させる
        # ----------------------------------------------------

        self.image_width = 640
        self.image_height = 480

        # ----------------------------------------------------
        # 最新観測
        # ----------------------------------------------------

        self.latest_image = None
        self.latest_state = None

        self.last_image_time = None
        self.last_state_time = None

        self.last_action = np.zeros(
            ACTION_DIMENSION,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Publisher
        # ----------------------------------------------------

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            10,
        )

        self.arm_trajectory_publisher = (
            self.create_publisher(
                JointTrajectory,
                self.arm_trajectory_topic,
                10,
            )
        )

        # ----------------------------------------------------
        # Subscriber
        # ----------------------------------------------------

        self.joint_state_subscriber = (
            self.create_subscription(
                JointState,
                self.joint_state_topic,
                self.joint_state_callback,
                10,
            )
        )

        if self.enable_image:
            self.image_subscriber = (
                self.create_subscription(
                    Image,
                    self.image_topic,
                    self.image_callback,
                    10,
                )
            )

        # ----------------------------------------------------
        # ACTモデルの読み込み
        # ----------------------------------------------------

        self.get_logger().info(
            f"Loading ACT policy from: {model_path}"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model directory not found: {model_path}"
            )

        self.policy = ACTPolicy.from_pretrained(
            model_path
        )

        self.policy.to(self.device)
        self.policy.eval()

        self.get_logger().info(
            "ACT policy loaded successfully"
        )

        self.get_logger().info(
            f"Expected action dimension: "
            f"{ACTION_DIMENSION}"
        )

        self.get_logger().info(
            "Action order: "
            + ", ".join(ACTION_NAMES)
        )

        # ----------------------------------------------------
        # 推論タイマー
        # ----------------------------------------------------

        timer_period = 1.0 / inference_hz

        self.timer = self.create_timer(
            timer_period,
            self.inference_callback,
        )

        self.get_logger().info(
            "Imitation policy node started"
        )

        self.get_logger().info(
            f"Image topic: {self.image_topic}"
        )

        self.get_logger().info(
            f"Joint state topic: "
            f"{self.joint_state_topic}"
        )

        self.get_logger().info(
            f"Command velocity topic: "
            f"{self.cmd_vel_topic}"
        )

        self.get_logger().info(
            f"Arm trajectory topic: "
            f"{self.arm_trajectory_topic}"
        )

    # ========================================================
    # 観測コールバック
    # ========================================================

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

            if (
                image.shape[1] != self.image_width
                or image.shape[0] != self.image_height
            ):
                image = cv2.resize(
                    image,
                    (
                        self.image_width,
                        self.image_height,
                    ),
                    interpolation=cv2.INTER_AREA,
                )

            self.latest_image = image
            self.last_image_time = time.monotonic()

        except Exception as error:
            self.get_logger().error(
                f"Image conversion failed: {error}"
            )

    def joint_state_callback(self, msg):
        """
        /joint_statesから、学習時と同じ4関節を抽出する。
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
                float(
                    name_to_position[joint_name]
                )
            )

        self.latest_state = np.asarray(
            positions,
            dtype=np.float32,
        )

        self.last_state_time = time.monotonic()

    # ========================================================
    # 観測作成
    # ========================================================

    def make_observation(self):
        """
        学習時と同じ観測辞書を作成する。

        observation.environment_stateは、
        学習時の定義に合わせてstateと同じ値を使用する。
        """

        if self.latest_state is None:
            return None

        if self.enable_image:
            if self.latest_image is None:
                return None

        state = np.asarray(
            self.latest_state,
            dtype=np.float32,
        )

        environment_state = state.copy()

        observation = {
            "observation.state": state,
            "observation.environment_state": (
                environment_state
            ),
        }

        if self.enable_image:
            observation[
                "observation.images.camera"
            ] = np.asarray(
                self.latest_image,
                dtype=np.uint8,
            )

        return observation

    # ========================================================
    # 推論
    # ========================================================

    def inference_callback(self):
        observation = self.make_observation()

        if observation is None:
            return

        try:
            observation_batch = (
                self.make_policy_input(
                    observation
                )
            )

            with torch.no_grad():
                action = self.policy.select_action(
                    observation_batch
                )

            action = self.convert_action(
                action
            )

            self.publish_base_action(
                action
            )

            self.publish_arm_action(
                action
            )

            self.last_action = action.copy()

        except Exception as error:
            self.get_logger().error(
                f"Inference failed: {error}"
            )

            # 推論失敗時は必ず停止する
            self.publish_stop()

    def make_policy_input(self, observation):
        """
        numpy形式の観測をACT入力用Tensorへ変換する。

        state：
        (4,) -> (1, 4)

        image：
        (H, W, C) -> (1, C, H, W)
        """

        policy_input = {}

        state = torch.from_numpy(
            observation[
                "observation.state"
            ]
        ).float()

        environment_state = torch.from_numpy(
            observation[
                "observation.environment_state"
            ]
        ).float()

        policy_input[
            "observation.state"
        ] = state.unsqueeze(0).to(
            self.device
        )

        policy_input[
            "observation.environment_state"
        ] = environment_state.unsqueeze(0).to(
            self.device
        )

        if self.enable_image:
            image = torch.from_numpy(
                observation[
                    "observation.images.camera"
                ]
            ).float()

            # HWC -> CHW
            image = image.permute(
                2,
                0,
                1,
            )

            # uint8画像を0～1へ変換
            image = image / 255.0

            # CHW -> BCHW
            image = image.unsqueeze(0)

            policy_input[
                "observation.images.camera"
            ] = image.to(
                self.device
            )

        return policy_input

    def convert_action(self, action):
        """
        ACTの出力を1本の6次元actionへ変換する。

        ACTの出力例：
        (6,)
        (1, 6)
        (1, 1, 6)
        (1, chunk_size, 6)
        """

        if isinstance(action, dict):
            if "action" in action:
                action = action["action"]
            elif "actions" in action:
                action = action["actions"]
            else:
                raise KeyError(
                    "No action key in ACT output"
                )

        if isinstance(action, torch.Tensor):
            action = action.detach()
            action = action.cpu().numpy()

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        if action.ndim == 3:
            # (batch, chunk, action_dim)
            action = action[0, 0]

        elif action.ndim == 2:
            # (batch, action_dim)
            action = action[0]

        elif action.ndim == 1:
            pass

        else:
            raise ValueError(
                f"Unexpected action shape: "
                f"{action.shape}"
            )

        if action.shape[0] != ACTION_DIMENSION:
            raise ValueError(
                "Unexpected action dimension: "
                f"expected {ACTION_DIMENSION}, "
                f"got {action.shape[0]}"
            )

        if not np.all(
            np.isfinite(action)
        ):
            raise ValueError(
                "Action contains NaN or inf"
            )

        return action

    # ========================================================
    # 移動・旋回
    # ========================================================

    def publish_base_action(self, action):
        """
        action[0]をlinear.x、
        action[1]をangular.zとして公開する。
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
        message.linear.y = 0.0
        message.linear.z = 0.0

        message.angular.x = 0.0
        message.angular.y = 0.0
        message.angular.z = angular_z

        self.cmd_vel_publisher.publish(
            message
        )

    # ========================================================
    # アーム・グリッパ
    # ========================================================

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

    # ========================================================
    # 停止
    # ========================================================

    def publish_stop(self):
        """
        移動ベースを停止する。
        """

        message = Twist()

        message.linear.x = 0.0
        message.linear.y = 0.0
        message.linear.z = 0.0

        message.angular.x = 0.0
        message.angular.y = 0.0
        message.angular.z = 0.0

        self.cmd_vel_publisher.publish(
            message
        )

    def publish_hold_arm(self):
        """
        最後のアーム姿勢を保持する。

        ACT推論で異常が起きたときに、アームへ
        不正な値を送らないための補助処理。
        """

        if self.last_action.shape[0] != 6:
            return

        try:
            self.publish_arm_action(
                self.last_action
            )
        except Exception:
            pass

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
            "Use this only when the model "
            "was trained without image observations"
        ),
    )

    parsed_args, ros_args = (
        parser.parse_known_args(args)
    )

    rclpy.init(
        args=ros_args
    )

    node = None

    try:
        node = ImitationPolicyNode(
            model_path=parsed_args.model_path,
            inference_hz=parsed_args.inference_hz,
            enable_image=(
                not parsed_args.without_image
            ),
        )

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        if node is not None:
            node.get_logger().error(
                f"Node stopped because of error: "
                f"{error}"
            )
        else:
            print(
                "Failed to start inference node: "
                f"{error}"
            )

    finally:
        if node is not None:
            node.publish_stop()
            node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
