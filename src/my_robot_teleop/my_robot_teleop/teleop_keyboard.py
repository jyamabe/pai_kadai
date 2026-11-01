#!/usr/bin/env python3

import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('my_robot_keyboard_teleop')

        # 実際のコントローラー名に変更する
        self.declare_parameter(
            'arm_topic',
            '/arm_controller/joint_trajectory'
        )

        # /joint_statesで確認した関節名に変更する
        self.declare_parameter(
            'joint_names',
            [
                'shoulder_joint',
                'elbow_joint',
                'left_finger_joint',
                'right_finger_joint',
            ]
        )

        self.declare_parameter('step', 0.05)
        self.declare_parameter('duration', 0.5)

        self.arm_topic = self.get_parameter(
            'arm_topic'
        ).value

        self.joint_names = list(
            self.get_parameter('joint_names').value
        )

        self.step = float(
            self.get_parameter('step').value
        )

        self.duration = float(
            self.get_parameter('duration').value
        )

        self.publisher = self.create_publisher(
            JointTrajectory,
            self.arm_topic,
            10
        )

        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.positions = [0.0] * len(self.joint_names)
        self.received_joint_state = False

        self.get_logger().info(
            f'Publishing to: {self.arm_topic}'
        )
        self.get_logger().info(
            f'Joints: {self.joint_names}'
        )

    def joint_state_callback(self, msg):
        """現在の関節角度を/joint_statesから取得する。"""
        position_map = dict(zip(msg.name, msg.position))

        for index, joint_name in enumerate(self.joint_names):
            if joint_name in position_map:
                self.positions[index] = position_map[joint_name]

        self.received_joint_state = True

    def publish_positions(self):
        """現在の目標角度をJointTrajectoryとして送信する。"""
        message = JointTrajectory()
        message.joint_names = list(self.joint_names)

        point = JointTrajectoryPoint()
        point.positions = list(self.positions)
        point.time_from_start.sec = int(self.duration)
        point.time_from_start.nanosec = int(
            (self.duration - int(self.duration)) * 1e9
        )

        message.points.append(point)
        self.publisher.publish(message)

        self.get_logger().info(
            f'positions: {[round(p, 3) for p in self.positions]}'
        )

    def change_joint(self, index, amount):
        if index < 0 or index >= len(self.positions):
            return

        self.positions[index] += amount
        self.publish_positions()

    def reset(self):
        self.positions = [0.0] * len(self.joint_names)
        self.publish_positions()


def get_key(settings, timeout=0.1):
    """端末からキーを1文字読み取る。"""
    tty.setraw(sys.stdin.fileno())

    readable, _, _ = select.select(
        [sys.stdin],
        [],
        [],
        timeout
    )

    if readable:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(
        sys.stdin,
        termios.TCSADRAIN,
        settings
    )

    return key


def print_help(joint_names):
    print()
    print('=== Keyboard Teleop ===')
    print('Q / E : select previous / next joint')
    print('W / S : increase / decrease selected joint')
    print('R / F : increase / decrease all joints')
    print('X     : reset all joints to zero')
    print('ESC   : quit')
    print()
    print(f'Joints: {joint_names}')
    print('Selected joint index: 0')
    print()


def main(args=None):
    rclpy.init(args=args)

    node = KeyboardTeleop()
    settings = termios.tcgetattr(sys.stdin)

    selected_joint = 0

    print_help(node.joint_names)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            key = get_key(settings, timeout=0.05)

            if key == '':
                continue

            if key == '\x1b':
                print('\n終了します。')
                break

            elif key in ('q', 'Q'):
                selected_joint -= 1
                if selected_joint < 0:
                    selected_joint = len(node.joint_names) - 1

                print(
                    f'\nSelected joint: '
                    f'{node.joint_names[selected_joint]}'
                )

            elif key in ('e', 'E'):
                selected_joint += 1
                if selected_joint >= len(node.joint_names):
                    selected_joint = 0

                print(
                    f'\nSelected joint: '
                    f'{node.joint_names[selected_joint]}'
                )

            elif key in ('w', 'W'):
                node.change_joint(
                    selected_joint,
                    node.step
                )

            elif key in ('s', 'S'):
                node.change_joint(
                    selected_joint,
                    -node.step
                )

            elif key in ('r', 'R'):
                node.positions = [
                    position + node.step
                    for position in node.positions
                ]
                node.publish_positions()

            elif key in ('f', 'F'):
                node.positions = [
                    position - node.step
                    for position in node.positions
                ]
                node.publish_positions()

            elif key in ('x', 'X'):
                node.reset()

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
