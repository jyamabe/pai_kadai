#!/usr/bin/env python3

import os
import time

import cv2
import numpy as np
import rclpy
import torch

from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectoryPoint


class ImitationPolicyNode(Node):

    def __init__(self):
        super().__init__('imitation_policy')

        # ==================================================
        # 基本設定
        # ==================================================

        self.model_path = (
            '/home/ubuntu/pai_kadai/checkpoints/'
            'color_sorting_robot_act'
        )

        self.image_topic = (
            '/sorting_camera/image_raw'
        )

        self.joint_state_topic = (
            '/joint_states'
        )

        self.cmd_vel_topic = (
            '/cmd_vel'
        )

        self.arm_action_name = (
            '/arm_controller/'
            'follow_joint_trajectory'
        )

        # task_plannerからFalseを受け取るまで
        # 推論を有効にする場合はTrue
        self.policy_enabled = True

        # 学習時の画像サイズ
        self.image_width = 640
        self.image_height = 480

        # 推論周期
        self.inference_period = 0.1

        # 学習時のaction次元
        self.action_dimension = 6

        # actionの順番
        #
        # [linear_x,
        #  angular_z,
        #  shoulder_joint,
        #  elbow_joint,
        #  left_finger_joint,
        #  right_finger_joint]
        self.action_names = [
            'linear_x',
            'angular_z',
            'shoulder_joint',
            'elbow_joint',
            'left_finger_joint',
            'right_finger_joint',
        ]

        self.joint_names = [
            'shoulder_joint',
            'elbow_joint',
            'left_finger_joint',
            'right_finger_joint',
        ]

        # ==================================================
        # 安全制限
        # ==================================================

        # 学習モデルが出した速度をそのまま使わず、
        # まずは小さい値に制限する
        self.max_linear_x = 0.03
        self.max_angular_z = 0.15

        self.joint_limits = {
            'shoulder_joint': (-1.2, 1.2),
            'elbow_joint': (-1.5, 1.5),
            'left_finger_joint': (0.0, 0.08),
            'right_finger_joint': (0.0, 0.08),
        }

        # アーム指令の送信間隔
        # 毎回actionを送るとアクションが混雑するため、
        # 0.3秒に1回程度に制限する
        self.arm_command_period = 0.3
        self.last_arm_command_time = 0.0

        # ==================================================
        # 学習時の正規化
        # ==================================================
        #
        # ACTPolicy.from_pretrained()が正規化を内部処理する
        # 構成ならFalseのままにする。
        #
        # 自作学習コードが正規化済みactionを返し、
        # 推論側で戻す必要がある場合だけTrueにする。
        #
        self.apply_manual_action_denormalization = False

        self.action_mean = np.zeros(
            self.action_dimension,
            dtype=np.float32,
        )

        self.action_std = np.ones(
            self.action_dimension,
            dtype=np.float32,
        )

        # ==================================================
        # 最新の観測
        # ==================================================

        self.bridge = CvBridge()

        self.latest_image = None
        self.latest_joint_state = None

        self.latest_image_time = 0.0
        self.latest_joint_time = 0.0

        # ==================================================
        # Subscriber
        # ==================================================

        self.image_subscriber = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.joint_subscriber = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.joint_state_callback,
            10,
        )

        self.enable_subscriber = self.create_subscription(
            Bool,
            '/imitation_policy/enable',
            self.enable_callback,
            10,
        )

        # ==================================================
        # Publisher
        # ==================================================

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            10,
        )

        # ==================================================
        # Arm Action Client
        # ==================================================

        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.arm_action_name,
        )

        # ==================================================
        # ACTモデルの読み込み
        # ==================================================

        self.policy = self.load_policy()

        # ==================================================
        # 推論タイマー
        # ==================================================

        self.inference_timer = self.create_timer(
            self.inference_period,
            self.inference_step,
        )

        self.get_logger().info(
            'Imitation policy node started'
        )

        self.get_logger().info(
            f'Image topic: {self.image_topic}'
        )

        self.get_logger().info(
            f'Joint state topic: '
            f'{self.joint_state_topic}'
        )

        self.get_logger().info(
            f'Command velocity topic: '
            f'{self.cmd_vel_topic}'
        )

        self.get_logger().info(
            f'Arm action: {self.arm_action_name}'
        )

        self.get_logger().info(
            'Expected action order: '
            f'{self.action_names}'
        )

    # ==================================================
    # モデル読み込み
    # ==================================================

    def load_policy(self):
        self.get_logger().info(
            f'Loading ACT policy from: '
            f'{self.model_path}'
        )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f'Policy directory not found: '
                f'{self.model_path}'
            )

        try:
            from lerobot.common.policies.act.modeling_act import (
                ACTPolicy,
            )

            policy = ACTPolicy.from_pretrained(
                self.model_path
            )

            if hasattr(policy, 'eval'):
                policy.eval()

            if hasattr(policy, 'to'):
                policy.to(
                    torch.device('cpu')
                )

            self.get_logger().info(
                'ACT policy loaded successfully'
            )

            return policy

        except Exception as error:
            self.get_logger().error(
                f'Failed to load ACT policy: {error}'
            )
            raise

    # ==================================================
    # ROS 2 callback
    # ==================================================

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='rgb8',
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
            self.latest_image_time = time.monotonic()

        except Exception as error:
            self.get_logger().error(
                f'Image conversion failed: {error}'
            )

    def joint_state_callback(self, msg):
        try:
            name_to_position = dict(
                zip(
                    msg.name,
                    msg.position,
                )
            )

            positions = []

            for joint_name in self.joint_names:
                if joint_name not in name_to_position:
                    self.get_logger().warning(
                        f'Missing joint: {joint_name}'
                    )
                    return

                positions.append(
                    float(
                        name_to_position[joint_name]
                    )
                )

            self.latest_joint_state = np.asarray(
                positions,
                dtype=np.float32,
            )

            self.latest_joint_time = time.monotonic()

        except Exception as error:
            self.get_logger().error(
                f'Joint state conversion failed: {error}'
            )

    def enable_callback(self, msg):
        self.policy_enabled = bool(
            msg.data
        )

        self.get_logger().info(
            f'Policy enabled: {self.policy_enabled}'
        )

        if not self.policy_enabled:
            self.publish_stop()

    # ==================================================
    # 観測作成
    # ==================================================

    def make_observation(self):
        if self.latest_image is None:
            return None

        if self.latest_joint_state is None:
            return None

        state = np.asarray(
            self.latest_joint_state,
            dtype=np.float32,
        )

        # 学習時に同じ値をenvironment_stateとして
        # 保存しているため、推論時も同じ値を渡す
        environment_state = state.copy()

        image = np.asarray(
            self.latest_image,
            dtype=np.uint8,
        )

        return {
            'observation.state': state,
            'observation.environment_state': (
                environment_state
            ),
            'observation.images.camera': image,
        }

    # ==================================================
    # 推論
    # ==================================================

    def inference_step(self):
        if not self.policy_enabled:
            self.publish_stop()
            return

        observation = self.make_observation()

        if observation is None:
            self.publish_stop()
            return

        try:
            self.validate_observation(
                observation
            )

            observation_torch = (
                self.observation_to_torch(
                    observation
                )
            )

            with torch.no_grad():
                raw_action = (
                    self.predict_action(
                        observation_torch
                    )
                )

            action = self.prepare_action(
                raw_action
            )

            if action is None:
                self.publish_stop()
                return

            # デバッグ用：
            # まずlinear_x/angular_zが0でないか確認する
            self.get_logger().info(
                'Predicted action: '
                f'linear_x={action[0]:.5f}, '
                f'angular_z={action[1]:.5f}, '
                f'arm={action[2:].tolist()}'
            )

            self.publish_action(action)

        except Exception as error:
            self.get_logger().error(
                f'Inference failed: {error}'
            )

            # 推論エラー時は必ず停止
            self.publish_stop()

    def observation_to_torch(self, observation):
        result = {}

        image = observation[
            'observation.images.camera'
        ]

        # HWC uint8 -> BCHW float32
        image_tensor = torch.from_numpy(
            image
        ).permute(
            2,
            0,
            1,
        ).float() / 255.0

        result[
            'observation.images.camera'
        ] = image_tensor.unsqueeze(0)

        state = torch.from_numpy(
            observation[
                'observation.state'
            ]
        ).float().unsqueeze(0)

        environment_state = torch.from_numpy(
            observation[
                'observation.environment_state'
            ]
        ).float().unsqueeze(0)

        result[
            'observation.state'
        ] = state

        result[
            'observation.environment_state'
        ] = environment_state

        return result

    def predict_action(self, observation):
        if hasattr(
            self.policy,
            'select_action'
        ):
            action = self.policy.select_action(
                observation
            )

        elif hasattr(
            self.policy,
            'predict_action'
        ):
            action = self.policy.predict_action(
                observation
            )

        else:
            action = self.policy(
                observation
            )

        if isinstance(action, dict):
            if 'action' in action:
                action = action['action']

            elif 'actions' in action:
                action = action['actions']

            else:
                raise KeyError(
                    'No action key in policy output'
                )

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        return action

    def prepare_action(self, raw_action):
        action = np.asarray(
            raw_action,
            dtype=np.float32,
        )

        self.get_logger().info(
            f'Raw action shape: {action.shape}'
        )

        # 例：
        # (6,)
        # (1, 6)
        # (1, 1, 6)
        # (1, chunk_size, 6)
        #
        # 現在は最初のactionを使う
        if action.ndim == 3:
            action = action[0, 0, :]

        elif action.ndim == 2:
            action = action[0, :]

        elif action.ndim == 1:
            pass

        else:
            raise ValueError(
                f'Unexpected action shape: '
                f'{action.shape}'
            )

        action = action.reshape(-1)

        if action.shape[0] != self.action_dimension:
            raise ValueError(
                'Unexpected action dimension: '
                f'expected {self.action_dimension}, '
                f'got {action.shape[0]}'
            )

        if not np.all(
            np.isfinite(action)
        ):
            raise ValueError(
                'Action contains NaN or inf'
            )

        # 学習コードが正規化済みactionを返す場合のみ
        # Trueにする
        if self.apply_manual_action_denormalization:
            action = (
                action * self.action_std
                + self.action_mean
            )

        return action

    def validate_observation(self, observation):
        state = np.asarray(
            observation[
                'observation.state'
            ]
        )

        environment_state = np.asarray(
            observation[
                'observation.environment_state'
            ]
        )

        image = np.asarray(
            observation[
                'observation.images.camera'
            ]
        )

        if state.shape != (4,):
            raise ValueError(
                f'Invalid state shape: '
                f'{state.shape}'
            )

        if environment_state.shape != (4,):
            raise ValueError(
                'Invalid environment state shape: '
                f'{environment_state.shape}'
            )

        if image.shape != (
            self.image_height,
            self.image_width,
            3,
        ):
            raise ValueError(
                f'Invalid image shape: '
                f'{image.shape}'
            )

    # ==================================================
    # actionのROS 2出力
    # ==================================================

    def publish_action(self, action):
        # actionの順番：
        #
        # 0 linear_x
        # 1 angular_z
        # 2 shoulder_joint
        # 3 elbow_joint
        # 4 left_finger_joint
        # 5 right_finger_joint

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

        # 学習済みactionが正規化値の場合、
        # ここでは移動指令が小さすぎる可能性がある。
        #
        # まずは実際の値をログで確認する。
        # linear_xが常に0.0付近なら、
        # actionの正規化設定を確認する。
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
        # 移動指令を毎推論周期送る
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
        # アーム指令は頻度を制限して送る
        # -----------------------------

        now = time.monotonic()

        if (
            now - self.last_arm_command_time
            >= self.arm_command_period
        ):
            self.send_arm_goal(
                arm_positions
            )

            self.last_arm_command_time = now

    def clip_joint_positions(self, positions):
        clipped = []

        for joint_name, position in zip(
            self.joint_names,
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

    def send_arm_goal(self, positions):
        if not self.arm_client.server_is_ready():
            self.get_logger().warning(
                'Arm action server is not ready'
            )
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = (
            self.joint_names
        )

        point = JointTrajectoryPoint()
        point.positions = positions

        # 頻繁な目標更新に対応するため、
        # 短すぎない時間を指定する
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0

        goal.trajectory.points.append(
            point
        )

        self.arm_client.send_goal_async(
            goal
        )

    # ==================================================
    # 停止処理
    # ==================================================

    def publish_stop(self):
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

    def destroy_node(self):
        self.publish_stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = ImitationPolicyNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        if node is not None:
            node.get_logger().error(
                f'Node stopped: {error}'
            )

    finally:
        if node is not None:
            node.publish_stop()
            node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()
