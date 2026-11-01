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

        # 赤色のマスクを作成
        # 赤色はHSV色相の両端に分かれるため、2つの範囲を使用する
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

        # 青色のマスクを作成
        blue_mask = cv2.inRange(
            hsv_image,
            np.array([100, 100, 80]),
            np.array([130, 255, 255])
        )

        # 緑色のマスクを作成
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

        # 色ごとに対象物を検出
        for color_name, mask in color_masks.items():
            mask = self.clean_mask(mask)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)

                # 小さなノイズを除外
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

        if len(detections) == 0:
            self.publish_no_detection(debug_image)
            return

        # 検出結果を左上から右下の順番に並べる
        detections.sort(
            key=lambda item: (
                item['center_y'],
                item['center_x']
            )
        )

        image_area = (
            bgr_image.shape[0] *
            bgr_image.shape[1]
        )

        # 色ごとの矩形描画色
        # OpenCVでは色の順番がBGRであることに注意
        box_colors = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
        }

        # 検出されたすべての対象物を処理する
        for detection in detections:
            color_name = detection['color']

            # 対象物ごとに簡易的な信頼度を計算
            # 本格的なAIモデルの信頼度とは異なる
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

            box_color = box_colors.get(
                color_name,
                (255, 255, 255)
            )

            # 検出対象を矩形で囲む
            cv2.rectangle(
                debug_image,
                (x, y),
                (
                    x + detection['width'],
                    y + detection['height']
                ),
                box_color,
                2
            )

            # 色名と信頼度を表示
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
                box_color,
                2
            )

        # 検出された色の一覧を作成
        # 同じ色の対象物が複数ある場合は、色名を1回だけにする
        detected_colors = list(dict.fromkeys(
            detection['color']
            for detection in detections
        ))

        # 検出されたすべての色を発行
        #
        # 例:
        # ["red", "blue", "green"]
        color_message = String()
        color_message.data = json.dumps(
            detected_colors,
            ensure_ascii=False
        )
        self.color_publisher.publish(color_message)

        # 検出されたすべての対象物を発行
        #
        # 例:
        # [
        #     {
        #         "color": "red",
        #         "center_x": 120,
        #         "center_y": 200,
        #         ...
        #     },
        #     ...
        # ]
        object_message = String()
        object_message.data = json.dumps(
            detections,
            ensure_ascii=False
        )
        self.object_publisher.publish(object_message)

        # 全対象物を描画したデバッグ画像を発行
        self.publish_debug_image(debug_image)

    def clean_mask(self, mask):
        """
        マスク画像から小さなノイズを除去する。
        """

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
        """
        対象物が検出されなかった場合の処理。
        """

        message = String()
        message.data = 'none'
        self.color_publisher.publish(message)

        self.publish_debug_image(image)

    def publish_debug_image(self, image):
        """
        デバッグ用画像をROS 2トピックに発行する。
        """

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
