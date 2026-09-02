#!/usr/bin/env python3
import json
import time
from collections import Counter, deque
from pathlib import Path

import torch
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "weights" / "best.pt"
ROI_PATH = BASE_DIR / "tray_rois.json"

IMAGE_TOPIC = "/rgb"
ANNOTATED_TOPIC = "/yolo/tracking_image"
DETECTION_TOPIC = "/m0609/tool_detection"

CONFIDENCE_THRESHOLD = 0.50
PUBLISH_RATE_HZ = 5.0
HISTORY_LENGTH = 5
MIN_VOTES = 3

CLASS_TO_TOOL_ID = {
    0: "caliper",
    1: "clamps",
    2: "handsaws",
    3: "knife",
    4: "ligature_needle",
    5: "mallet",
}

# Temporary values for this integration test.
# Replace later or let RobotManager resolve the real position from tray_id.
TEMP_TRAY_POSITIONS = {i: (0.0, 0.0, 0.0) for i in range(6)}

class VisionToolDetectionNode(Node):
    def __init__(self):
        super().__init__("vision_tool_detection_node")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        if not ROI_PATH.exists():
            raise FileNotFoundError(f"ROI file not found: {ROI_PATH}")

        self.model = YOLO(str(MODEL_PATH))
        self.bridge = CvBridge()
        self.tray_polygons = self.load_rois()
        self.history = {i: deque(maxlen=HISTORY_LENGTH) for i in range(6)}
        self.stable_state = {i: None for i in range(6)}
        self.latest_confidence = {i: 0.0 for i in range(6)}

        self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, 10)
        self.annotated_pub = self.create_publisher(Image, ANNOTATED_TOPIC, 10)
        self.detection_pub = self.create_publisher(String, DETECTION_TOPIC, 10)
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self.publish_detections)

        self.get_logger().info(f"Model loaded: {MODEL_PATH}")
        self.get_logger().info(f"ROI loaded: {ROI_PATH}")
        self.get_logger().info(f"Input topic: {IMAGE_TOPIC}")
        self.get_logger().info(f"Detection topic: {DETECTION_TOPIC}")
        self.get_logger().info("Tray layout: 5 3 1 / 4 2 0")

    def load_rois(self):
        data = json.loads(ROI_PATH.read_text(encoding="utf-8"))
        result = {}
        for tray_id in range(6):
            points = data["trays"].get(str(tray_id))
            if not points or len(points) < 3:
                raise ValueError(f"Invalid ROI for tray {tray_id}")
            result[tray_id] = np.array(points, dtype=np.int32)
        return result

    def find_tray(self, cx, cy):
        for tray_id, polygon in self.tray_polygons.items():
            if cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0:
                return tray_id
        return None

    def update_stable_states(self, frame_assignments):
        for tray_id in range(6):
            current = frame_assignments.get(tray_id)
            signature = None if current is None else current["class_id"]
            self.history[tray_id].append(signature)
            candidate, count = Counter(self.history[tray_id]).most_common(1)[0]
            if count >= MIN_VOTES:
                if candidate is None:
                    self.stable_state[tray_id] = None
                    self.latest_confidence[tray_id] = 0.0
                else:
                    self.stable_state[tray_id] = {"class_id": candidate}
                    if current and current["class_id"] == candidate:
                        self.latest_confidence[tray_id] = current["confidence"]

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        annotated = frame.copy()
        frame_assignments = {}

        results = self.model.predict(frame, conf=CONFIDENCE_THRESHOLD, device="cuda:0" if torch.cuda.is_available() else "cpu", verbose=False)
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                tray_id = self.find_tray(cx, cy)
                if tray_id is None:
                    continue
                old = frame_assignments.get(tray_id)
                if old is None or confidence > old["confidence"]:
                    frame_assignments[tray_id] = {"class_id": class_id, "confidence": confidence}
                label = f"{CLASS_TO_TOOL_ID[class_id]} {confidence:.2f} T{tray_id}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(annotated, label, (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        self.update_stable_states(frame_assignments)

        for tray_id, polygon in self.tray_polygons.items():
            state = self.stable_state[tray_id]
            text = f"Tray {tray_id}: EMPTY/NOT DETECTED"
            if state is not None:
                text = f"Tray {tray_id}: {CLASS_TO_TOOL_ID[state['class_id']]}"
            cv2.polylines(annotated, [polygon], True, (0, 255, 255), 2)
            center = polygon.mean(axis=0).astype(int)
            cv2.putText(annotated, text, (int(center[0] - 80), int(center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out.header = msg.header
        self.annotated_pub.publish(out)

    def publish_detections(self):
        now = time.time()

        for tray_id in range(6):
            state = self.stable_state[tray_id]
            x, y, z = TEMP_TRAY_POSITIONS[tray_id]

            if state is None:
                tool_id = "EMPTY"
            else:
                class_id = int(state["class_id"])
                tool_id = CLASS_TO_TOOL_ID[class_id]

            payload = {
                "tool_id": tool_id,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "tray_id": int(tray_id),
                "timestamp": float(now),
            }

            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.detection_pub.publish(msg)

def main():
    rclpy.init()
    node = VisionToolDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
