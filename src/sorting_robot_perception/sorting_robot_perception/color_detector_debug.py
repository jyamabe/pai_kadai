#!/usr/bin/env python3

import json

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')

        self.bridge = CvBridge()

        self.declare_parameter(
            'image_topic',
            '/sorting_camera/image_raw'
        )

        image_topic = self.get_parameter(
            'image_topic'
        ).get_parameter_value().string_value

        self.image_subscriber = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        self.color_publisher = self.create_publisher(
            String,
            '/detected_color',
            10
        )

        self.object_publisher = self.create_publisher(
            String,
            '/colored_object',
            10
        )

        self.debug_image_publisher = self.create_publisher(
            Image,
            '/color_detection/image',
            10
        )

        self.get_logger().info(
            f'Color detector started: {image_topic}'
        )

    def image_callback(self, msg):
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
        except Exception as error:
            self.get_logger().error(
                f'Image conversion failed: {error}'
            )
            return

        hsv_image = cv2.cvtColor(
            bgr_image,
            cv2.COLOR_BGR2HSV
        )

        detections = []

        red_mask_1 = cv2.inRange(
            hsv_image,
            np.array([0, 100, 80]),
            np.array([10, 255, 255])
        )

        red_mask_2 = cv2.inRange(
            hsv_image,
            np.array([170, 100, 80]),
            np.array([179, 255, 255])
        )

        red_mask = cv2.bitwise_or(
            red_mask_1,
            red_mask_2
        )

        blue_mask = cv2.inRange(
            hsv_image,
            np.array([100, 100, 80]),
            np.array([130, 255, 255])
        )

        green_mask = cv2.inRange(
            hsv_image,
            np.array([35, 70, 60]),
            np.array([85, 255, 255])
        )

        color_masks = {
            'red': red_mask,
            'blue': blue_mask,
            'green': green_mask,
        }

        for color_name, mask in color_masks.items():
            mask = self.clean_mask(mask)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)

                if area < 500.0:
                    continue

                x, y, width, height = cv2.boundingRect(contour)

                center_x = x + width // 2
                center_y = y + height // 2

                detections.append({
                    'color': color_name,
                    'center_x': int(center_x),
                    'center_y': int(center_y),
                    'width': int(width),
                    'height': int(height),
                    'area': float(area),
                })
                
        debug_image = bgr_image.copy()
                
        # 検出されたすべての色の確認デバッグ用ログ追加
        detected_colors = [
           item['color']
           for item in detections
        ]
    
        self.get_logger().info(
           f'Detected colors: {detected_colors}'
        )
        # ログ追加終わり

        if len(detections) == 0:
            self.publish_no_detection(debug_image)
            return

        # 面積が最大の対象物を採用
        detection = max(
            detections,
            key=lambda item: item['area']
        )

        color_name = detection['color']

        # 簡易的な信頼度。
        # 本格的なAIモデルの信頼度とは異なります。
        image_area = (
            bgr_image.shape[0] *
            bgr_image.shape[1]
        )

        confidence = min(
            detection['area'] / (image_area * 0.05),
            1.0
        )

        detection['confidence'] = round(
            float(confidence),
            3
        )

        x = (
            detection['center_x'] -
            detection['width'] // 2
        )

        y = (
            detection['center_y'] -
            detection['height'] // 2
        )

        cv2.rectangle(
            debug_image,
            (x, y),
            (
                x + detection['width'],
                y + detection['height']
            ),
            (0, 255, 0),
            2
        )

        label = (
            f"{color_name} "
            f"{detection['confidence']:.2f}"
        )

        cv2.putText(
            debug_image,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        color_message = String()
        color_message.data = color_name
        self.color_publisher.publish(color_message)

        object_message = String()
        object_message.data = json.dumps(
            detection,
            ensure_ascii=False
        )
        self.object_publisher.publish(object_message)

        self.publish_debug_image(debug_image)

    def clean_mask(self, mask):
        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        return mask

    def publish_no_detection(self, image):
        message = String()
        message.data = 'none'
        self.color_publisher.publish(message)

        self.publish_debug_image(image)

    def publish_debug_image(self, image):
        image_message = self.bridge.cv2_to_imgmsg(
            image,
            encoding='bgr8'
        )

        image_message.header.stamp = (
            self.get_clock().now().to_msg()
        )

        image_message.header.frame_id = 'camera_link'

        self.debug_image_publisher.publish(
            image_message
        )


def main(args=None):
    rclpy.init(args=args)

    node = ColorDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
