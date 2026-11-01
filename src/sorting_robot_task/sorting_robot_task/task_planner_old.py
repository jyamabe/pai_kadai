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
        self.detection_timeout_sec = 1.0

        # 安全のため低速にする
        self.linear_speed = 0.10
        self.angular_speed = 0.30

        self.last_detection_time = None
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

    """
    def ai_detection_callback(self, msg):

     #   /ai_detectionのJSONを受信する。
     #
     #例：
     #   {
     #     "class_name": "red_box",
     #     "confidence": 0.92,
     #     "center_x": 315,
     #     "center_y": 238,
     #     "width": 105,
     #     "height": 98
     #   }
     

        # 作業中は新しい荷物を選ばない
        if self.state != TaskState.SEARCH_OBJECT:
            return

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

        if class_name not in self.destinations:
            self.get_logger().warning(
                f'Unknown class: {class_name}'
            )
            return

        if confidence < self.confidence_threshold:
            self.get_logger().info(
                f'Low confidence: {confidence:.2f}'
            )
            return

        # 検出時刻と検出結果を保存
        self.selected_object = detection
        self.target_area = self.destinations[class_name]
        self.last_detection_time = (
            self.get_clock().now()
        )
        
        self.change_state(TaskState.SELECT_OBJECT)
    """
        #-----------------------------------------------
    def ai_detection_callback(self, msg):
        # SEARCH_OBJECT中だけ新しい対象物を受け付ける
        if self.state != TaskState.SEARCH_OBJECT:
            return

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

        if class_name not in self.destinations:
            self.get_logger().warning(
                f'Unknown class: {class_name}'
            )
            return

        if confidence < self.confidence_threshold:
            self.get_logger().warning(
                f'Low confidence: {confidence:.2f}'
            )
            return

        # 必要な値があるか確認
        if 'center_x' not in detection:
            self.get_logger().warning(
                'center_x is not included in detection'
            )
            return

        # AI検出結果を保存
        self.selected_object = detection
        self.target_area = self.destinations[class_name]

        self.last_detection_time = (
            self.get_clock().now()
        )

        self.get_logger().info(
            f'Object selected: {class_name}, '
            f'confidence={confidence:.2f}'
        )

        self.change_state(TaskState.SELECT_OBJECT)
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

    def handle_select_object(self):
        class_name = self.selected_object['class_name']
        confidence = float(
            self.selected_object['confidence']
        )

        self.get_logger().info(
            f'Selected object: {class_name}, '
            f'confidence={confidence:.2f}'
        )

        self.get_logger().info(
            f'Target area: {self.target_area}'
        )

        self.change_state(TaskState.APPROACH_OBJECT)
        
    """
    def handle_approach_object(self):
        self.publish_stop()

        self.get_logger().info(
            'Approach object: simulation only'
        )   

        self.change_state(TaskState.PICK_OBJECT)
    """
    """
    #====================一時的な確認用 Strat===========================
    def handle_approach_object(self):
        center_x = float(
            self.selected_object.get('center_x', 320)
        )

        error_x = center_x - self.image_center_x

        if abs(error_x) > self.center_tolerance:
            self.get_logger().info(
                f'Would turn: error_x={error_x:.1f}'
            )
        else:
            self.get_logger().info(
                f'Would move forward: error_x={error_x:.1f}'
            )

        self.publish_stop()
    #====================一時的な確認用 End ===========================
    """

    #-----------------------------------------------
    def handle_approach_object(self):
        if self.selected_object is None:
            self.publish_stop()
            self.change_state(TaskState.SEARCH_OBJECT)
            return

        # AI検出が一定時間途切れたら停止
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
                'AI detection timeout. Stop robot.'
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

        error_x = center_x - self.image_center_x

        # widthとheightから画像上の面積を計算
        width = float(
            self.selected_object.get('width', 0.0)
        )

        height = float(
            self.selected_object.get('height', 0.0)
        )

        object_area = width * height

        # 箱が十分大きく映っていれば接近完了
        if object_area >= self.approach_area_threshold:
            self.publish_stop()

            self.get_logger().info(
                f'Approach completed. '
                f'area={object_area:.1f}'
            )

            self.change_state(TaskState.PICK_OBJECT)
            return

        message = Twist()

        # 箱が左・右にある場合は旋回
        if abs(error_x) > self.center_tolerance:
            message.linear.x = 0.0

            if error_x > 0:
                # 箱が画像の右側にある
                message.angular.z = -self.angular_speed
                direction = 'right'
            else:
                # 箱が画像の左側にある
                message.angular.z = self.angular_speed
                direction = 'left'

            self.get_logger().info(
                f'Turn {direction}: '
                f'center_x={center_x:.1f}, '
                f'error={error_x:.1f}'
            )

        # 箱が中央にある場合は前進
        else:
            message.linear.x = self.linear_speed
            message.angular.z = 0.0

            self.get_logger().info(
                f'Forward: '
                f'center_x={center_x:.1f}, '
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
