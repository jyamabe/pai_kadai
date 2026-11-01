#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectoryPoint


class ManipulationSequence(Node):
    def __init__(self):
        super().__init__('manipulation_sequence')

        self.client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )

        self.joint_names = [
            'shoulder_joint',
            'elbow_joint',
            'left_finger_joint',
            'right_finger_joint'
        ]

    def send_goal(self, positions, duration):
        if not self.client.wait_for_server(
            timeout_sec=5.0
        ):
            self.get_logger().error(
                'arm_controller action server not available'
            )
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = duration

        goal.trajectory.points.append(point)

        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(
            self,
            future
        )

        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error(
                'Trajectory goal rejected'
            )
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self,
            result_future
        )

        self.get_logger().info(
            'Trajectory completed'
        )

        return True

    def execute_sequence(self):
    
        # A. 現在の位置
        self.send_goal(
            [0.71, -1.16, 7.25, 0.00],
            2
        )
        
        """
        # TEST1. shoulder_joint を少しずつ移動
        for i in range(9,0,-1):
            self.send_goal(
                [0.71*i*0.1, -1.16, 0.05, 0.05],
                2
            )
        # TEST2. elbow_joint 少しずつ移動
        for i in range(9,0,-1):
            self.send_goal(
                [0.0, -1.16*i*0.1, 0.05, 0.05],
                2
            )
        """    

        # 0. 先ず shoulder_joint を移動
        self.send_goal(
            [0.0, -1.16, 7.25, 0.00],
            3
        )
        
        # 1. 待機姿勢・グリッパを開く
        self.send_goal(
            [0.0, 0.0, 0.05, 0.05],
            3
        )
       
        # 2. 箱へアームを伸ばす
        self.send_goal(
            [-0.3, 0.8, 0.05, 0.05],
            3
        )
        
        # 3. グリッパを閉じる
        self.send_goal(
            [-0.3, 0.8, 0.0, 0.0],
            3
        )

        # 4. 箱を持ち上げる
        self.send_goal(
            [0.2, 0.4, 0.0, 0.0],
            3
        )

        # 5. 待機姿勢へ戻す
        self.send_goal(
            [0.0, 0.0, 0.0, 0.0],
            3
        )

def main(args=None):
    rclpy.init(args=args)

    node = ManipulationSequence()

    try:
        node.execute_sequence()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
