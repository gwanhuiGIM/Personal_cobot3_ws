"""
yolo_ros 2D 탐지 결과 + depth/camera_info로 3D 좌표를 직접 계산해 이동 명령을 발행하는 노드.

[실행 순서]

1. yolo_ros yolo_node 기동 (2D 탐지만, use_3d:=False):
   ros2 launch yolo_bringup yolo.launch.py \
     use_3d:=False \
     use_tracking:=False \
     model:=/home/kimkh/cobot3_ws/0622_v2_yolov8/best.pt \
     input_image_topic:=/rgb \
     threshold:=0.5

2. 이 노드 실행:
   python3 yolo3d_move_command.py

[토픽 구성]
구독:
  /yolo/detections   (yolo_msgs/DetectionArray) - yolo_node 2D 탐지 결과
  /depth             (sensor_msgs/Image, 32FC1)  - Isaac Sim 뎁스 이미지
  /camera_info       (sensor_msgs/CameraInfo)    - 카메라 내부 파라미터

발행:
  /m0609/move_command  (std_msgs/String) - JSON 이동 명령
"""

import json
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
from yolo_msgs.msg import DetectionArray

# 유효 depth 범위 (미터). Isaac Sim 환경에 맞게 조정
DEPTH_MIN = 0.05   # 5cm 미만은 노이즈로 간주
DEPTH_MAX = 3.0    # 3m 초과는 작업 범위 밖

# 중심 픽셀 주변 샘플링 ROI 반경 (픽셀)
DEPTH_ROI_HALF = 5


class Yolo3DMoveCommandNode(Node):
    def __init__(self):
        super().__init__('yolo3d_move_command')

        self.bridge       = CvBridge()
        self.camera_info  = None
        self.latest_depth = None
        self._request_counter = 0

        # ── 구독 ────────────────────────────────────────────
        self.create_subscription(
            DetectionArray, '/yolo/detections',
            self._detection_callback, 10
        )
        self.create_subscription(
            Image, '/depth',
            self._depth_callback, 10
        )
        self.create_subscription(
            CameraInfo, '/camera_info',
            self._camera_info_callback, 10
        )

        # ── 발행 ────────────────────────────────────────────
        self._cmd_pub = self.create_publisher(String, '/m0609/move_command', 10)

        self.get_logger().info("yolo3d_move_command 노드 시작")
        self.get_logger().info("구독: /yolo/detections | /depth | /camera_info")
        self.get_logger().info("발행: /m0609/move_command")

    # ── 카메라 파라미터 수신 (최초 1회) ─────────────────────
    def _camera_info_callback(self, msg: CameraInfo):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info(
                f"camera_info 수신 | "
                f"fx={msg.k[0]:.1f}  fy={msg.k[4]:.1f}  "
                f"cx={msg.k[2]:.1f}  cy={msg.k[5]:.1f}"
            )

    # ── depth 이미지 수신 ─────────────────────────────────────
    def _depth_callback(self, msg: Image):
        self.latest_depth = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='32FC1'
        )

    # ── ROI 중앙값으로 depth 샘플링 ──────────────────────────
    def _sample_depth(self, u: float, v: float) -> float:
        """
        바운딩박스 중심 (u,v) 주변 ROI에서 유효 depth 값의 중앙값을 반환.
        단일 픽셀 대비 노이즈와 경계 오류에 강함.
        유효 값이 없으면 -1.0 반환.
        """
        h, w = self.latest_depth.shape
        u_i = int(np.clip(u, 0, w - 1))
        v_i = int(np.clip(v, 0, h - 1))

        u0 = max(0, u_i - DEPTH_ROI_HALF)
        u1 = min(w, u_i + DEPTH_ROI_HALF + 1)
        v0 = max(0, v_i - DEPTH_ROI_HALF)
        v1 = min(h, v_i + DEPTH_ROI_HALF + 1)

        roi = self.latest_depth[v0:v1, u0:u1]
        valid = roi[
            (roi > DEPTH_MIN) & (roi < DEPTH_MAX) & ~np.isnan(roi)
        ]

        if valid.size == 0:
            return -1.0
        return float(np.median(valid))

    # ── 핵심: 2D 탐지 → 핀홀 역투영 → 3D 좌표 → 발행 ───────
    def _detection_callback(self, msg: DetectionArray):
        # depth 또는 camera_info 미수신 시 처리 불가
        if self.camera_info is None or self.latest_depth is None:
            return

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        for det in msg.detections:
            cls_id   = det.class_id
            score    = det.score
            cls_name = det.class_name if det.class_name else f'class_{cls_id}'

            # yolo_msgs/BoundingBox2D → center.position (Point2D, 픽셀)
            u = det.bbox.center.position.x
            v = det.bbox.center.position.y

            # ROI 중앙값 depth 샘플링
            z = self._sample_depth(u, v)
            if z < 0:
                self.get_logger().warn(
                    f"[{cls_name}] 유효 depth 없음 | pixel=({u:.0f},{v:.0f})"
                )
                continue

            # ── 핀홀 역투영 (2D + depth → 카메라 좌표계 3D) ──
            # X = (u - cx) * z / fx
            # Y = (v - cy) * z / fy
            # Z = z
            x_cam = (u - cx) * z / fx
            y_cam = (v - cy) * z / fy
            z_cam = z

            # 이동 명령 JSON 생성
            self._request_counter += 1
            request_id = f"{cls_name}_{self._request_counter:04d}"

            payload = {
                "request_id": request_id,
                "x": round(x_cam, 4),
                "y": round(y_cam, 4),
                "z": round(z_cam, 4),
            }

            self.get_logger().info(
                f"[{cls_name}] conf={score:.2f} | "
                f"pixel=({u:.0f},{v:.0f}) depth={z:.3f}m | "
                f"3D=({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})m"
            )

            cmd = String()
            cmd.data = json.dumps(payload)
            self._cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = Yolo3DMoveCommandNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
