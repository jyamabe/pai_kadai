#!/usr/bin/env python3

import argparse
import os
import time

import cv2
import numpy as np
import rclpy
import torch

from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from lerobot.common.policies.act.modeling_act import ACTPolicy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


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

        # ====================================================
        # 基本設定
        # ====================================================

        self.device = torch.device("cpu")
        self.bridge = CvBridge()
        self.enable_image = enable_image

        self.model_path = model_path

        # 学習時の画像サイズに合わせる
        self.image_width = 640
        self.image_height = 480

        # ====================================================
        # 安全設定
        # ====================================================

        # 移動速度の上限
        self.max_linear_x = 0.02
        self.max_angular_z = 0.10

        # 関節角度の上限
        self.joint_limits = {
            "shoulder_joint": (-1.2, 1.2),
            "elbow_joint": (-1.5, 1.5),
            "left_finger_joint": (0.0, 0.08),
            "right_finger_joint": (0.0, 0.08),
        }

        # 推論結果をアームへ送る周期
        # 推論10 Hz、アーム指令5 Hzの例
        self.arm_command_period = 0.2
        self.last_arm_command_time = 0.0

        # 観測が古い場合の安全停止時間
        self.observation_timeout = 1.0

        # ====================================================
        # 最新観測
        # ====================================================

        self.latest_image = None
        self.latest_state = None

        self.last_image_time = None
        self.last_state_time = None

        # 最後にpublishしたaction
        self.last_action = np.zeros(
            ACTION_DIMENSION,
            dtype=np.float32,
        )

        # ====================================================
        # ROS 2 Publisher
        # ====================================================

        # 移動指令
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        # rosbag記録・確認用のアーム軌道
        self.arm_trajectory_publisher = (
            self.create_publisher(
                JointTrajectory,
                "/arm_controller/joint_trajectory",
                10,
            )
        )

        # ====================================================
        # アームAction Client
        #
        # 学習済みactionを実際のアーム制御へ送る。
        # 環境によってAction serverを使わない場合は、
        # send_arm_goal()を無効化してもよい。
        # ====================================================

        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )

        # ====================================================
        # ROS 2 Subscriber
        # ====================================================

        self.joint_state_subscriber = (
            self.create_subscription(
                JointState,
                "/joint_states",
                self.joint_state_callback,
                10,
            )
        )

        if self.enable_image:
            self.image_subscriber = (
                self.create_subscription(
                    Image,
                    "/sorting_camera/image_raw",
                    self.image_callback,
                    10,
                )
            )
        else:
            self.image_subscriber = None

        # ====================================================
        # 推論タイマー
        # ====================================================

        timer_period = 1.0 / inference_hz

        self.timer = self.create_timer(
            timer_period,
            self.inference_callback,
        )

        # ====================================================
        # ACTモデル読み込み
        # ====================================================

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
            f"Expected action dimension: "
            f"{ACTION_DIMENSION}"
        )

        self.get_logger().info(
            "Action order: "
            + ", ".join(ACTION_NAMES)
        )

        self.get_logger().info(
            "ACT policy loaded successfully"
        )

        self.get_logger().info(
            "Imitation policy node started"
        )

        self.get_logger().info(
            "Image topic: /sorting_camera/image_raw"
        )

        self.get_logger().info(
            "Joint state topic: /joint_states"
        )

        self.get_logger().info(
            "Command velocity topic: /cmd_vel"
        )

        self.get_logger().info(
            "Arm trajectory topic: "
            "/arm_controller/joint_trajectory"
        )

    # ========================================================
    # 画像コールバック
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

    # ========================================================
    # 関節状態コールバック
    # ========================================================

    def joint_state_callback(self, msg):
        """
        /joint_statesから、学習時と同じ4関節を抽出する。
        """

        name_to_position = dict(
            zip(
                msg.name,
                msg.position,
            )
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
    # 観測の有効性確認
    # ========================================================

    def observations_are_ready(self):
        if self.latest_state is None:
            return False

        if self.last_state_time is None:
            return False

        now = time.monotonic()

        if (
            now - self.last_state_time
            > self.observation_timeout
        ):
            self.get_logger().warning(
                "Joint state is too old"
            )
            return False

        if self.enable_image:
            if self.latest_image is None:
                return False

            if self.last_image_time is None:
                return False

            if (
                now - self.last_image_time
                > self.observation_timeout
            ):
                self.get_logger().warning(
                    "Image is too old"
                )
                return False

        return True

    # ========================================================
    # 観測作成
    # ========================================================

    def make_observation(self):
        """
        学習時と同じ観測辞書を作成する。

        今回の重要な点：
        observation.environment_stateを
        observation.stateと同じ4次元で追加する。
        """

        state = np.asarray(
            self.latest_state,
            dtype=np.float32,
        )

        # 学習時に同じ関節値をenvironment_stateへ
        # 入れているため、推論時にも同じ構造にする。
        environment_state = state.copy()

        observation = {
            "observation.state": state,
            "observation.environment_state": (
                environment_state
            ),
        }

        if self.enable_image:
            image = np.asarray(
                self.latest_image,
                dtype=np.uint8,
            )

            observation[
                "observation.images.camera"
            ] = image

        return observation

    # ========================================================
    # ACTへ渡す入力テンソルを作成
    # ========================================================

    def observation_to_tensors(
        self,
        observation,
    ):
        """
        numpy配列の観測を、ACT用のtorch Tensorへ変換する。

        state：
            (4,) -> (1, 4)

        image：
            (H, W, C) -> (1, C, H, W)
        """

        observation_tensors = {}

        state = np.asarray(
            observation["observation.state"],
            dtype=np.float32,
        )

        environment_state = np.asarray(
            observation[
                "observation.environment_state"
            ],
            dtype=np.float32,
        )

        if state.shape != (4,):
            raise ValueError(
                "Invalid observation.state shape: "
                f"{state.shape}, expected (4,)"
            )

        if environment_state.shape != (4,):
            raise ValueError(
                "Invalid observation.environment_state "
                f"shape: {environment_state.shape}, "
                "expected (4,)"
            )

        observation_tensors[
            "observation.state"
        ] = torch.from_numpy(
            state
        ).unsqueeze(0).to(
            self.device
        )

        observation_tensors[
            "observation.environment_state"
        ] = torch.from_numpy(
            environment_state
        ).unsqueeze(0).to(
            self.device
        )

        if self.enable_image:
            image = np.asarray(
                observation[
                    "observation.images.camera"
                ],
                dtype=np.uint8,
            )

            expected_shape = (
                self.image_height,
                self.image_width,
                3,
            )

            if image.shape != expected_shape:
                raise ValueError(
                    "Invalid image shape: "
                    f"{image.shape}, "
                    f"expected {expected_shape}"
                )

            # HWC RGB uint8
            # -> BCHW float32 [0, 1]
            image_tensor = torch.from_numpy(
                image
            ).permute(
                2,
                0,
                1,
            ).float() / 255.0

            observation_tensors[
                "observation.images.camera"
            ] = image_tensor.unsqueeze(
                0
            ).to(
                self.device
            )

        return observation_tensors

    # ========================================================
    # ACT推論
    # ========================================================

    def predict_action(self, observation):
        """
        ACTからactionを取得する。

        LeRobotのバージョンにより、
        select_actionまたはpredict_actionを使う。
        """

        observation_tensors = (
            self.observation_to_tensors(
                observation
            )
        )

        with torch.no_grad():
            if hasattr(
                self.policy,
                "select_action",
            ):
                action = self.policy.select_action(
                    observation_tensors
                )

            elif hasattr(
                self.policy,
                "predict_action",
            ):
                action = self.policy.predict_action(
                    observation_tensors
                )

            else:
                action = self.policy(
                    observation_tensors
                )

        if isinstance(action, dict):
            if "action" in action:
                action = action["action"]

            elif "actions" in action:
                action = action["actions"]

            else:
                raise KeyError(
                    "Model output does not contain "
                    "'action' or 'actions'"
                )

        if isinstance(action, torch.Tensor):
            action = (
                action.detach()
                .cpu()
                .numpy()
            )

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        return action

    # ========================================================
    # actionの形状を統一
    # ========================================================

    def prepare_action(self, action):
        """
        ACTの出力を1本の6次元actionへ変換する。

        想定される出力：
            (6,)
            (1, 6)
            (1, 1, 6)
            (1, chunk_size, 6)

        現在は、最初のactionを使用する。
        """

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        if action.ndim == 1:
            pass

        elif action.ndim == 2:
            # (batch, action_dim)
            action = action[0]

        elif action.ndim == 3:
            # (batch, chunk, action_dim)
            action = action[0, 0]

        else:
            raise ValueError(
                "Unexpected action shape: "
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
    # 推論コールバック
    # ========================================================

    def inference_callback(self):
        """
        一定周期で観測を取得し、
        ACTのactionをROS 2へ出力する。
        """

        if not self.observations_are_ready():
            self.publish_stop()
            return

        try:
            observation = (
                self.make_observation()
            )

            action = self.predict_action(
                observation
            )

            action = self.prepare_action(
                action
            )

            self.last_action = action.copy()

            self.publish_action(
                action
            )

        except Exception as error:
            self.get_logger().error(
                f"Inference failed: {error}"
            )

            # 推論失敗時はベースを停止
            self.publish_stop()

    # ========================================================
    # actionのROS 2出力
    # ========================================================

    def publish_action(self, action):
        """
        actionの順番：

        0: linear_x
        1: angular_z
        2: shoulder_joint
        3: elbow_joint
        4: left_finger_joint
        5: right_finger_joint
        """

        linear_x = float(
            action[0]
        )

        angular_z = float(
            action[1]
        )

        arm_positions = [
            float(action[2]),
            float(action[3]),
            float(action[4]),
            float(action[5]),
        ]

        # -----------------------------
        # 安全制限
        # -----------------------------

        linear_x = float(
            np.clip(
                linear_x,
                -self.max_linear_x,
                self.max_linear_x,
            )
        )

        angular_z = float(
            np.clip(
                angular_z,
                -self.max_angular_z,
                self.max_angular_z,
            )
        )

        arm_positions = (
            self.clip_joint_positions(
                arm_positions
            )
        )

        # -----------------------------
        # 移動指令
        # -----------------------------

        velocity_message = Twist()

        velocity_message.linear.x = linear_x
        velocity_message.linear.y = 0.0
        velocity_message.linear.z = 0.0

        velocity_message.angular.x = 0.0
        velocity_message.angular.y = 0.0
        velocity_message.angular.z = angular_z

        self.cmd_vel_publisher.publish(
            velocity_message
        )

        # -----------------------------
        # アーム指令
        # -----------------------------

        now = time.monotonic()

        if (
            now - self.last_arm_command_time
            >= self.arm_command_period
        ):
            self.publish_arm_trajectory(
                arm_positions
            )

            # 実際のアームをActionで制御する構成
            self.send_arm_goal(
                arm_positions
            )

            self.last_arm_command_time = now

        self.get_logger().info(
            "Action: "
            f"linear_x={linear_x:.5f}, "
            f"angular_z={angular_z:.5f}, "
            f"arm={arm_positions}"
        )

    # ========================================================
    # 関節値の制限
    # ========================================================

    def clip_joint_positions(self, positions):
        clipped = []

        for joint_name, position in zip(
            STATE_JOINT_NAMES,
            positions,
        ):
            lower, upper = (
                self.joint_limits[joint_name]
            )

            clipped.append(
                float(
                    np.clip(
                        position,
                        lower,
                        upper,
                    )
                )
            )

        return clipped

    # ========================================================
    # アーム指令トピック
    # ========================================================

    def publish_arm_trajectory(
        self,
        positions,
    ):
        """
        アーム・グリッパの4関節目標を
        /arm_controller/joint_trajectoryへ公開する。

        rosbag記録やデバッグにも利用する。
        """

        trajectory = JointTrajectory()

        trajectory.header.stamp = (
            self.get_clock().now().to_msg()
        )

        trajectory.joint_names = (
            STATE_JOINT_NAMES.copy()
        )

        point = JointTrajectoryPoint()

        point.positions = [
            float(position)
            for position in positions
        ]

        # 次の指令までに到達させる時間
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(
            self.arm_command_period * 1e9
        )

        trajectory.points.append(
            point
        )

        self.arm_trajectory_publisher.publish(
            trajectory
        )

    # ========================================================
    # アームAction
    # ========================================================

    def send_arm_goal(self, positions):
        """
        arm_controllerのFollowJointTrajectoryへ
        関節目標を送る。

        /arm_controller/joint_trajectoryを
        実際のコントローラが購読している環境では、
        このAction送信を無効化してもよい。
        """

        if not self.arm_client.server_is_ready():
            self.get_logger().warning(
                "Arm action server is not ready"
            )
            return

        goal = FollowJointTrajectory.Goal()

        goal.trajectory.joint_names = (
            STATE_JOINT_NAMES.copy()
        )

        point = JointTrajectoryPoint()

        point.positions = [
            float(position)
            for position in positions
        ]

        # 頻繁な目標更新に対応するため、
        # 1秒後の目標として送る
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0

        goal.trajectory.points.append(
            point
        )

        self.arm_client.send_goal_async(
            goal
        )

    # ========================================================
    # 停止処理
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

    # ========================================================
    # 終了処理
    # ========================================================

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
                f"Node stopped: {error}"
            )
        else:
            print(
                f"Failed to start inference node: "
                f"{error}"
            )

    finally:
        if node is not None:
            node.publish_stop()
            node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
