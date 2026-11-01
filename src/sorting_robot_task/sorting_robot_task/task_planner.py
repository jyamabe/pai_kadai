#!/usr/bin/env python3

import json
from enum import Enum

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import String


class TaskState(Enum):
    SEARCH_OBJECT = 0
    SELECT_OBJECT = 1
    APPROACH_OBJECT = 2
    PICK_OBJECT = 3
    MOVE_TO_AREA = 4
    PLACE_OBJECT = 5
    VERIFY = 6
    ERROR = 7


class TaskPlanner(Node):
    def __init__(self):
        super().__init__('task_planner')

        self.state = TaskState.SEARCH_OBJECT
        self.selected_object = None
        self.target_area = None
        
        #-------------------------------------------
        self.image_width = 640
        self.image_height = 480

        self.image_center_x = self.image_width / 2.0

        # この範囲内なら「ほぼ正面」とみなす
        self.center_tolerance = 40.0

        # 画像上の箱の面積がこの値以上なら接近完了
        self.approach_area_threshold = 30000.0

        # 最小信頼度
        self.confidence_threshold = 0.60

        # AI検出がこの秒数以上途切れたら停止
        self.detection_timeout_sec = 10.0

        # 安全のため低速にする
        self.linear_speed = 0.15
        self.angular_speed = 0.10

        self.last_detection_time = None
        
        self.declare_parameter(
        'target_color',
        'green'
        )

        target_color = self.get_parameter(
            'target_color'
        ).value.strip().lower()

        if target_color not in ['red', 'blue', 'green']:
            self.get_logger().warning(
                f'Unknown target_color: {target_color}. '
                'Use green instead.'
            )
            target_color = 'green'

        self.target_color = target_color
        self.target_class = f'{target_color}_box'
        
        #-------------------------------------------

        # AI検出結果を購読する
        self.ai_detection_subscriber = self.create_subscription(
            String,
            '/ai_detection',
            self.ai_detection_callback,
            10
        )

        # 状態を公開する
        self.state_publisher = self.create_publisher(
            String,
            '/sorting_task/state',
            10
        )

        # タスク結果を公開する
        self.result_publisher = self.create_publisher(
            String,
            '/sorting_task/result',
            10
        )

        # 移動指令用
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.timer = self.create_timer(
            0.5,
            self.state_machine_step
        )

        # AIモデルのクラス名と仕分け先の対応
        self.destinations = {
            'red_box': 'red_area',
            'blue_box': 'blue_area',
            'green_box': 'green_area',
        }

        # AI検出の信頼度しきい値
        self.confidence_threshold = 0.60

        # 同じ検出を何度も処理しないための設定
        self.detection_timeout = 1.0
        self.last_detection_time = None

        self.get_logger().info(
            'Task planner started. '
            'Subscribing to /ai_detection'
        )
        
        #-------------------------------------------        
        self.get_logger().info(
            f'Target color: {self.target_color}'
        )

        self.get_logger().info(
            f'Target class: {self.target_class}'
        )
        #-------------------------------------------
        
        self.change_state(TaskState.SEARCH_OBJECT)

    def change_state(self, new_state):
        if self.state != new_state:
            self.get_logger().info(
                f'State: {self.state.name} -> '
                f'{new_state.name}'
            )

        self.state = new_state

        state_message = String()
        state_message.data = self.state.name
        self.state_publisher.publish(state_message)

        #-----------------------------------------------
    def ai_detection_callback(self, msg):
        """
        /ai_detectionからAI検出結果を受信する。

              例：
        {
          "class_name": "red_box",
          "confidence": 0.92,
          "center_x": 315,
          "center_y": 238,
          "width": 105,
          "height": 98
        }
        """
    
        try:
            detection = json.loads(msg.data)
        except json.JSONDecodeError as error:
            self.get_logger().warning(
                f'Invalid AI detection JSON: {error}'
            )
            return
    
        class_name = detection.get('class_name')
        confidence = float(
            detection.get('confidence', 0.0)
        )
    
        # 指定色以外の箱は無視する
        if class_name != self.target_class:
            return
    
        # 信頼度が低い検出は無視する
        if confidence < self.confidence_threshold:
            return
    
        if 'center_x' not in detection:
            self.get_logger().warning(
                'AI detection does not contain center_x'
            )
            return
    
        # 指定色の検出結果を常に最新状態へ更新する
        self.selected_object = detection
        self.target_area = self.destinations[
            self.target_class
        ]
    
        self.last_detection_time = (
            self.get_clock().now()
        )
    
        # SEARCH_OBJECT中に対象色を発見したら接近開始
        if self.state == TaskState.SEARCH_OBJECT:
            self.get_logger().info(
                f'Target found: {class_name}, '
                f'confidence={confidence:.2f}'
            )
    
            self.get_logger().info(
                f'Approach target area: {self.target_area}'
            )
    
            self.change_state(TaskState.SELECT_OBJECT)
    
        # APPROACH_OBJECT中は、最新のcenter_xで追従する
        elif self.state == TaskState.APPROACH_OBJECT:
            self.get_logger().debug(
                f'Updated target position: '
                f'center_x={detection["center_x"]}'
            )
        #-----------------------------------------------

    def state_machine_step(self):
        if self.state == TaskState.SEARCH_OBJECT:
            self.handle_search_object()

        elif self.state == TaskState.SELECT_OBJECT:
            self.handle_select_object()

        elif self.state == TaskState.APPROACH_OBJECT:
            self.handle_approach_object()

        elif self.state == TaskState.PICK_OBJECT:
            self.handle_pick_object()

        elif self.state == TaskState.MOVE_TO_AREA:
            self.handle_move_to_area()

        elif self.state == TaskState.PLACE_OBJECT:
            self.handle_place_object()

        elif self.state == TaskState.VERIFY:
            self.handle_verify()

        elif self.state == TaskState.ERROR:
            self.handle_error()

    def handle_search_object(self):
        # 待機中は停止
        self.publish_stop()
        
    #-----------------------------------------------     
    def handle_select_object(self):
        if self.selected_object is None:
            self.change_state(TaskState.SEARCH_OBJECT)
            return

        class_name = self.selected_object['class_name']
        confidence = float(
            self.selected_object['confidence']
        )

        self.get_logger().info(
            f'Selected target: {class_name}, '
            f'confidence={confidence:.2f}'
        )
    
        self.get_logger().info(
            f'Target area: {self.target_area}'
        )
    
        self.change_state(TaskState.APPROACH_OBJECT)
    #-----------------------------------------------

    def handle_approach_object(self):
        if self.selected_object is None:
            self.publish_stop()
            self.change_state(TaskState.SEARCH_OBJECT)
            return
    
        # AI検出が途切れたら停止して探索へ戻る
        if self.last_detection_time is None:
            self.publish_stop()
            self.change_state(TaskState.SEARCH_OBJECT)
            return
    
        now = self.get_clock().now()
    
        elapsed = (
            now - self.last_detection_time
        ).nanoseconds / 1e9
    
        if elapsed > self.detection_timeout_sec:
            self.get_logger().warning(
                'Target detection timeout. Stopping robot.'
            )
    
            self.publish_stop()
            self.change_state(TaskState.SEARCH_OBJECT)
            return
    
        center_x = float(
            self.selected_object.get(
                'center_x',
                self.image_center_x
            )
        )
    
        width = float(
            self.selected_object.get('width', 0.0)
        )
    
        height = float(
            self.selected_object.get('height', 0.0)
        )
    
        object_area = width * height
        error_x = center_x - self.image_center_x
    
        # 箱が十分近づいたら停止
        if object_area >= self.approach_area_threshold:
            self.publish_stop()
    
            self.get_logger().info(
                f'Target reached: '
                f'{self.target_class}, '
                f'area={object_area:.1f}'
            )
    
            self.change_state(TaskState.PICK_OBJECT)
            return
    
        message = Twist()
    
        # 箱が画像中心より右側
        if error_x > self.center_tolerance:
            message.linear.x = 0.0
            message.angular.z = -self.angular_speed
    
            self.get_logger().info(
                f'Turning right: '
                f'center_x={center_x:.1f}, '
                f'error_x={error_x:.1f}'
            )
    
        # 箱が画像中心より左側
        elif error_x < -self.center_tolerance:
            message.linear.x = 0.0
            message.angular.z = self.angular_speed
    
            self.get_logger().info(
                f'Turning left: '
                f'center_x={center_x:.1f}, '
                f'error_x={error_x:.1f}'
            )
    
        # 箱が中央付近
        else:
            message.linear.x = self.linear_speed
            message.angular.z = 0.0
    
            self.get_logger().info(
                f'Moving forward toward '
                f'{self.target_class}: '
                f'area={object_area:.1f}'
            )
    
        self.cmd_vel_publisher.publish(message)    

    #-----------------------------------------------

    def handle_pick_object(self):
        self.publish_stop()

        self.get_logger().info(
            'Pick object: arm control is not connected yet'
        )

        # 後でアーム・グリッパ制御を追加する
        self.change_state(TaskState.MOVE_TO_AREA)

    def handle_move_to_area(self):
        self.publish_stop()

        self.get_logger().info(
            f'Move to {self.target_area}: '
            'navigation is not connected yet'
        )

        # 後で目的地への移動を追加する
        self.change_state(TaskState.PLACE_OBJECT)

    def handle_place_object(self):
        self.publish_stop()

        self.get_logger().info(
            'Place object: gripper control is not connected yet'
        )

        # 後でグリッパを開く処理を追加する
        self.change_state(TaskState.VERIFY)

    def handle_verify(self):
        class_name = self.selected_object['class_name']

        result_message = String()
        result_message.data = json.dumps({
            'status': 'completed',
            'object': class_name,
            'destination': self.target_area,
        })

        self.result_publisher.publish(result_message)

        self.get_logger().info(
            f'Sorting task completed: {class_name} '
            f'-> {self.target_area}'
        )

        self.selected_object = None
        self.target_area = None

        self.change_state(TaskState.SEARCH_OBJECT)

    def handle_error(self):
        self.publish_stop()

        self.get_logger().error(
            'Task planner entered ERROR state'
        )

        self.selected_object = None
        self.target_area = None

        self.change_state(TaskState.SEARCH_OBJECT)

    def publish_stop(self):
        message = Twist()
        message.linear.x = 0.0
        message.angular.z = 0.0
        self.cmd_vel_publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)

    node = TaskPlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
