#!/usr/bin/env python3
import json
from pathlib import Path
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

IMAGE_TOPIC = "/rgb"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "tray_rois.json"
TRAY_ORDER = [0, 1, 2, 3, 4, 5]  # 0 BR, 1 TR, 2 BC, 3 TC, 4 BL, 5 TL

class TrayRoiCalibrator(Node):
    def __init__(self):
        super().__init__("tray_roi_calibrator")
        self.bridge = CvBridge()
        self.frame = None
        self.current_index = 0
        self.current_points = []
        self.trays = {}
        self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, 10)
        cv2.namedWindow("Tray ROI Calibrator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Tray ROI Calibrator", self.mouse_callback)
        self.get_logger().info("Tray order: 0 bottom-right, 1 top-right, 2 bottom-center, 3 top-center, 4 bottom-left, 5 top-left")
        self.get_logger().info("Keys: r=redo previous tray, s=save, q=quit")

    def image_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.current_index >= 6:
            return
        tray_id = TRAY_ORDER[self.current_index]
        self.current_points.append([int(x), int(y)])
        print(f"Tray {tray_id}: point {len(self.current_points)}/4 = ({x}, {y})")
        if len(self.current_points) == 4:
            self.trays[str(tray_id)] = self.current_points.copy()
            self.current_points.clear()
            self.current_index += 1
            print(f"Tray {tray_id} completed.")

    def reset_previous(self):
        self.current_points.clear()
        if self.current_index > 0:
            self.current_index -= 1
            tray_id = TRAY_ORDER[self.current_index]
            self.trays.pop(str(tray_id), None)
            print(f"Redo Tray {tray_id}")

    def save(self):
        if len(self.trays) != 6:
            print(f"Cannot save: {len(self.trays)}/6 trays completed.")
            return
        payload = {"image_topic": IMAGE_TOPIC, "layout": [[5, 3, 1], [4, 2, 0]], "trays": self.trays}
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved: {OUTPUT_PATH}")

    def draw(self):
        canvas = self.frame.copy()
        for tray_id, points in self.trays.items():
            polygon = np.array(points, dtype=np.int32)
            cv2.polylines(canvas, [polygon], True, (0, 255, 255), 3)
            center = polygon.mean(axis=0).astype(int)
            cv2.putText(canvas, f"Tray {tray_id}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        for point in self.current_points:
            cv2.circle(canvas, tuple(point), 6, (0, 0, 255), -1)
        status = "All trays completed. Press s to save."
        if self.current_index < 6:
            status = f"Click Tray {TRAY_ORDER[self.current_index]} corners clockwise"
        cv2.putText(canvas, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        cv2.putText(canvas, "Layout: 5 3 1 / 4 2 0 | r redo | s save | q quit", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return canvas

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.frame is None:
                continue
            cv2.imshow("Tray ROI Calibrator", self.draw())
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                self.reset_previous()
            elif key == ord("s"):
                self.save()
            elif key == ord("q"):
                break
        cv2.destroyAllWindows()

def main():
    rclpy.init()
    node = TrayRoiCalibrator()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
