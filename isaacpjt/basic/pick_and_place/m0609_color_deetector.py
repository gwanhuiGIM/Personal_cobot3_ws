#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge



from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,  # 재전송 없음
    history=HistoryPolicy.KEEP_LAST,
    depth=10,                                     # 최신 1개만 유지
    durability=DurabilityPolicy.VOLATILE
)

class ColorDetector(Node):

    def __init__(self):
        super().__init__('color_detector')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.image_callback,
            qos
        )

        self.publisher = self.create_publisher(
            Int32,
            '/detected_color',
            10
        )

    def detect_color(self, hsv, lower, upper):

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None, mask

        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) < 300:
            return None, mask

        M = cv2.moments(c)

        if M["m00"] == 0:
            return None, mask

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return (cx, cy), mask

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Blue
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Green
        lower_green = np.array([45, 50, 50])
        upper_green = np.array([75, 255, 255])

        blue_pos, blue_mask = self.detect_color(
            hsv,
            lower_blue,
            upper_blue
        )

        green_pos, green_mask = self.detect_color(
            hsv,
            lower_green,
            upper_green
        )

        detected = 0

        if blue_pos is not None:
            x, y = blue_pos
            cv2.circle(frame, (x, y), 8, (255,0,0), -1)
            detected = 1

        if green_pos is not None:
            x, y = green_pos
            cv2.circle(frame, (x, y), 8, (0,255,0), -1)
            if detected == 0:
                detected = 2

        if detected != 0:
            msg_out = Int32()
            msg_out.data = detected
            self.publisher.publish(msg_out)

        cv2.imshow("camera", frame)
        cv2.imshow("blue", blue_mask)
        cv2.imshow("green", green_mask)

        cv2.waitKey(1)


def main(args=None):

    rclpy.init(args=args)

    node = ColorDetector()

    rclpy.spin(node)

    node.destroy_node()

    cv2.destroyAllWindows()

    rclpy.shutdown()


if __name__ == '__main__':
    main()