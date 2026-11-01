import json

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ultralytics import YOLO


class AIDetector(Node):
    def __init__(self):
        super().__init__('ai_detector')

        self.bridge = CvBridge()

        self.model = YOLO(
            '/home/ubuntu/pai_kadai/src/'
            'sorting_robot_perception/models/best.pt'
        )

        self.image_subscriber = self.create_subscription(
            Image,
            '/sorting_camera/image_raw',
            self.image_callback,
            10
        )

        self.detection_publisher = self.create_publisher(
            String,
            '/ai_detection',
            10
        )

        self.debug_image_publisher = self.create_publisher(
            Image,
            '/ai_detection/image',
            10
        )

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        result = self.model(
            image,
            verbose=False
        )[0]

        best_detection = None

        for box in result.boxes:
            confidence = float(box.conf[0])

            if confidence < 0.6:
                continue

            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            x1, y1, x2, y2 = [
                int(value)
                for value in box.xyxy[0]
            ]

            candidate = {
                'class_name': class_name,
                'confidence': confidence,
                'center_x': (x1 + x2) // 2,
                'center_y': (y1 + y2) // 2,
                'width': x2 - x1,
                'height': y2 - y1,
            }

            if (
                best_detection is None or
                confidence > best_detection['confidence']
            ):
                best_detection = candidate

        if best_detection is None:
            return

        message = String()
        message.data = json.dumps(
            best_detection,
            ensure_ascii=False
        )

        self.detection_publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = AIDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
