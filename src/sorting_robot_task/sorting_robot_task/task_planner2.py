#!/usr/bin/env python3

import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist
#from trajectory_msgs.msg import JointTrajectoryPoint
from trajectory_msgs.msg import (
    JointTrajectory,
    JointTrajectoryPoint
)


class TaskPlanner2(Node):

    def __init__(self):
        super().__init__('task_planner2')

        # =====================================
        # 動作設定
        # ここを書き換えて動作を調整する
        # =====================================

        # 前進設定
        self.forward_speed = 0.10
        self.forward_duration = 15.0

        # 後退設定
        self.backward_speed = 0.10
        self.backward_duration = 15.0

        # 旋回設定
        # 正：左旋回、負：右旋回
        self.turn_angular_speed = -0.1
        self.turn_duration = 3.0

        # 待機時間
        self.wait_after_grasp = 2.0
        self.wait_before_start = 1.0

        # アーム動作時間
        self.arm_open_duration = 5
        self.arm_extend_duration = 8
        self.gripper_close_duration = 5
        self.arm_lift_duration = 8

        # =====================================
        # アーム用アクションクライアント
        # =====================================

        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )
        
        # =====================================
        # アーム用パブリッシャ
        # =====================================
        
        self.arm_trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )

        # =====================================
        # 移動用パブリッシャ
        # =====================================

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # URDFのジョイント名と一致させる
        self.joint_names = [
            'shoulder_joint',
            'elbow_joint',
            'left_finger_joint',
            'right_finger_joint'
        ]

        self.get_logger().info(
            'Task planner 2 started'
        )

        self.get_logger().info(
            f'forward: speed={self.forward_speed}, '
            f'duration={self.forward_duration}'
        )

        self.get_logger().info(
            f'backward: speed={self.backward_speed}, '
            f'duration={self.backward_duration}'
        )

        self.get_logger().info(
            f'turn: angular_speed={self.turn_angular_speed}, '
            f'duration={self.turn_duration}'
        )

    # =====================================
    # アーム制御
    # =====================================

    def send_arm_goal(self, positions, duration):
        self.get_logger().info(
            f'Arm goal: positions={positions}, '
            f'duration={duration}'
        )

        if not self.arm_client.wait_for_server(
            timeout_sec=5.0
        ):
            self.get_logger().error(
                'arm_controller action server is not available'
            )
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = 0

        goal.trajectory.points.append(point)

        send_future = self.arm_client.send_goal_async(goal)

        rclpy.spin_until_future_complete(
            self,
            send_future
        )
        
        #アクションを送る前に、同じ軌道をトピックへ公開
        #--------------------------------------------------
        trajectory_message = JointTrajectory()
        trajectory_message.joint_names = self.joint_names

        trajectory_point = JointTrajectoryPoint()
        trajectory_point.positions = positions
        trajectory_point.time_from_start.sec = int(duration)
        trajectory_point.time_from_start.nanosec = 0

        trajectory_message.points.append(
        trajectory_point
        )

        self.arm_trajectory_publisher.publish(
            trajectory_message
        )
        #--------------------------------------------------

        goal_handle = send_future.result()

        if goal_handle is None:
            self.get_logger().error(
                'No goal handle returned'
            )
            return False

        if not goal_handle.accepted:
            self.get_logger().error(
                'Arm goal was rejected'
            )
            return False

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self,
            result_future
        )

        self.get_logger().info(
            'Arm movement completed'
        )

        return True

    # =====================================
    # 移動制御
    # =====================================

    def publish_velocity(
        self,
        linear_x=0.0,
        angular_z=0.0
    ):
        message = Twist()

        message.linear.x = float(linear_x)
        message.linear.y = 0.0
        message.linear.z = 0.0

        message.angular.x = 0.0
        message.angular.y = 0.0
        message.angular.z = float(angular_z)

        self.cmd_vel_publisher.publish(message)

    def publish_stop(self):
        self.publish_velocity(
            linear_x=0.0,
            angular_z=0.0
        )

    def move_forward(self):
        self.get_logger().info(
            f'Moving forward: '
            f'{self.forward_speed} m/s for '
            f'{self.forward_duration} s'
        )

        self.move_linear(
            speed=self.forward_speed,
            duration=self.forward_duration
        )

    def move_backward(self):
        self.get_logger().info(
            f'Moving backward: '
            f'{self.backward_speed} m/s for '
            f'{self.backward_duration} s'
        )

        self.move_linear(
            speed=-self.backward_speed,
            duration=self.backward_duration
        )

    def move_linear(self, speed, duration):
        start_time = time.monotonic()

        try:
            while (
                time.monotonic() - start_time
                < duration
            ):
                self.publish_velocity(
                    linear_x=speed,
                    angular_z=0.0
                )

                # 20 Hzで指令を送る
                time.sleep(0.05)

        finally:
            self.publish_stop()

    def turn(self):
        direction = (
            'left'
            if self.turn_angular_speed > 0.0
            else 'right'
        )

        self.get_logger().info(
            f'Turning {direction}: '
            f'angular_z={self.turn_angular_speed} '
            f'for {self.turn_duration} s'
        )

        start_time = time.monotonic()

        try:
            while (
                time.monotonic() - start_time
                < self.turn_duration
            ):
                self.publish_velocity(
                    linear_x=0.0,
                    angular_z=self.turn_angular_speed
                )

                time.sleep(0.05)

        finally:
            self.publish_stop()

    # =====================================
    # 一連のタスク
    # =====================================

    def execute_task(self):
        self.get_logger().info(
            'Task started'
        )

        # 開始時に停止
        self.publish_stop()
        time.sleep(self.wait_before_start)

        # ---------------------------------
        # 1. グリッパを開いた待機姿勢
        # positions:
        # shoulder, elbow, left finger, right finger
        # ---------------------------------

        self.send_arm_goal(
            positions=[
                0.0,
                0.0,
                0.0,
                0.0
            ],
            duration=self.arm_open_duration
        )

        # ---------------------------------
        # 2. 前進
        # ---------------------------------

        self.move_forward()

        # ---------------------------------
        # 3. アームを箱へ伸ばす
        # ---------------------------------

        self.send_arm_goal(
            positions=[
                -0.05,
                0.10,
                0.0,
                0.0
            ],
            duration=self.arm_extend_duration
        )

        # ---------------------------------
        # 4. グリッパを閉じる
        # ---------------------------------

        self.send_arm_goal(
            positions=[
                -0.05,
                0.10,
                0.07,
                0.07
            ],
            duration=self.gripper_close_duration
        )

        # ---------------------------------
        # 5. 把持後に待機
        # ---------------------------------

        time.sleep(self.wait_after_grasp)

        # ---------------------------------
        # 6. 箱を持ち上げる
        # ---------------------------------

        self.send_arm_goal(
            positions=[
                -0.02,
                -0.20, #0.08
                0.07,
                0.07
            ],
            duration=self.arm_lift_duration
        )

        # ---------------------------------
        # 7. 後退
        # ---------------------------------

        self.move_backward()
        
        # ---------------------------------
        # 8. グリッパを開ける
        # ---------------------------------
        
        self.send_arm_goal(
            positions=[
                0.0,
                0.0,
                0.0,
                0.0
            ],
            duration=self.arm_open_duration
        )

        # ---------------------------------
        # 9. アームを上げる
        # ---------------------------------
        
        self.send_arm_goal(
            positions=[
                -0.3,
                -0.5,
                0.0,
                0.0
            ],
            duration=self.arm_open_duration
        )


        # ---------------------------------
        # 10. 旋回
        # ---------------------------------

        self.turn()

        # ---------------------------------
        # 11. 最後に停止
        # ---------------------------------

        self.publish_stop()

        self.get_logger().info(
            'Task completed'
        )


def main(args=None):
    rclpy.init(args=args)

    node = TaskPlanner2()

    try:
        node.execute_task()

    except KeyboardInterrupt:
        node.get_logger().warning(
            'Task interrupted'
        )

    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
