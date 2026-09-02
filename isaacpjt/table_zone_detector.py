"""
테이블 구역 탐지 노드 (온디맨드 모드).

[동작 방식 - 리소스 절약]
  /target_request 수신 전  : /yolo/detections 구독 없음 → CPU 유휴
  /target_request 수신 후  : /yolo/detections 구독 시작 → 탐지 시작
  zone_detection 발행 완료 : /yolo/detections 구독 해제 → 다시 유휴

  즉, 요청이 들어올 때만 YOLO 탐지 결과를 처리하고, 1회 성공 후 즉시 정지.
  다음 요청이 오면 다시 활성화.

[구역(Zone) 개념]
  카메라 이미지를 GRID_COLS × GRID_ROWS 격자로 나눔.
  예) GRID_COLS=4, GRID_ROWS=4 → 16개 구역 (zone_id 0~15)

  zone_id 배치 예시 (4×4):
  ┌────┬────┬────┬────┐
  │  0 │  1 │  2 │  3 │  row 0
  ├────┼────┼────┼────┤
  │  4 │  5 │  6 │  7 │  row 1
  ├────┼────┼────┼────┤
  │  8 │  9 │ 10 │ 11 │  row 2
  ├────┼────┼────┼────┤
  │ 12 │ 13 │ 14 │ 15 │  row 3
  └────┴────┴────┴────┘

[실행 순서]
  1) yolo_ros 기동:
     ros2 launch yolo_bringup yolo.launch.py \
       use_3d:=False use_tracking:=False \
       model:/home/kimkh/cobot3_ws/0624_YOLOv1_pt/best.pt \
       input_image_topic:=/camera1/image_raw threshold:=0.3

  2) 이 노드 실행:
     python3 table_zone_detector.py

  3) 탐지 요청 전송 (보낼 때만 탐지 시작):
     ros2 topic pub --once /target_request std_msgs/String "data: 'scissors'"
     또는 class_id 정수 문자열:
     ros2 topic pub --once /target_request std_msgs/String "data: '2'"

[발행 토픽 - /zone_detection, JSON 형식]
  {
    "zone_id" : 5    # 격자 번호 (0-based, row-major)
  }
"""

import json
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from yolo_msgs.msg import DetectionArray


# ── 구역 격자 설정 ────────────────────────────────────────────────
GRID_COLS = 4   # 가로 분할 수
GRID_ROWS = 4   # 세로 분할 수

# 3×3 전용 가독성 레이블 (다른 격자 크기 사용 시 자동 비활성화)
_ZONE_LABELS_3X3 = {
    0: "TOP_LEFT",    1: "TOP_CENTER",    2: "TOP_RIGHT",
    3: "MID_LEFT",    4: "CENTER",        5: "MID_RIGHT",
    6: "BOT_LEFT",    7: "BOT_CENTER",    8: "BOT_RIGHT",
}

# 이미지 해상도 (camera_info 수신 전 fallback용)
DEFAULT_IMG_WIDTH  = 640
DEFAULT_IMG_HEIGHT = 640

# 최소 신뢰도 (yolo_ros threshold와 이중 필터링)
MIN_CONFIDENCE = 0.4


def _zone_label(zone_id: int) -> str:
    """3×3 격자일 때 레이블 반환, 그 외에는 숫자 문자열 반환."""
    if GRID_COLS == 3 and GRID_ROWS == 3:
        return _ZONE_LABELS_3X3.get(zone_id, str(zone_id))
    return str(zone_id)


def _pixel_to_zone(u: float, v: float, img_w: int, img_h: int):
    """
    픽셀 좌표 (u, v) → 격자 구역 정보 반환.

    Returns:
        (zone_id, zone_row, zone_col)
    """
    col = int(np.clip(u / img_w * GRID_COLS, 0, GRID_COLS - 1))
    row = int(np.clip(v / img_h * GRID_ROWS, 0, GRID_ROWS - 1))
    zone_id = row * GRID_COLS + col
    return zone_id, row, col


class TableZoneDetector(Node):
    """
    외부 요청 클래스를 테이블 영상에서 탐지하고
    해당 물체가 속한 픽셀 구역을 publish하는 노드.

    [온디맨드 동작]
    - 평소: /yolo/detections 구독 없음 (리소스 0)
    - 요청 수신 시: 구독 생성 → 탐지 시작
    - 탐지 성공 1회: 구독 해제 → 다시 유휴 상태
    """

    def __init__(self):
        super().__init__('table_zone_detector')

        self._target: str | None = None
        self._img_width  = DEFAULT_IMG_WIDTH
        self._img_height = DEFAULT_IMG_HEIGHT
        self._request_counter = 0

        # /yolo/detections 구독 핸들 (None = 비활성)
        self._detection_sub = None

        # /target_request, /camera_info는 항상 구독 (초경량)
        self.create_subscription(
            String, '/tool_command',
            self._target_callback, 10
        )
        self.create_subscription(
            CameraInfo, '/camera_info',
            self._camera_info_callback, 10
        )

        self._zone_pub = self.create_publisher(String, '/zone_detection', 10)

        self.get_logger().info(
            f"table_zone_detector 대기 중 | 격자={GRID_COLS}×{GRID_ROWS}"
        )
        self.get_logger().info(
            "요청 예시: ros2 topic pub --once /tool_command "
            "std_msgs/String \"data: 'scissors'\""
        )

    # ── 구독 동적 제어 ───────────────────────────────────────────
    def _start_detection(self):
        """/yolo/detections 구독 생성 (없을 때만)."""
        if self._detection_sub is None:
            self._detection_sub = self.create_subscription(
                DetectionArray, '/yolo/detections',
                self._detection_callback, 10
            )
            self.get_logger().info(
                f"탐지 활성화 → target='{self._target}' | "
                "/yolo/detections 구독 시작"
            )

    def _stop_detection(self):
        """/yolo/detections 구독 해제."""
        if self._detection_sub is not None:
            self.destroy_subscription(self._detection_sub)
            self._detection_sub = None
            self.get_logger().info(
                "탐지 완료 → /yolo/detections 구독 해제 (유휴 상태)"
            )

    # ── 외부 요청 수신 ───────────────────────────────────────────
    def _target_callback(self, msg: String):
        """
        /target_request 수신 시 탐지 대상을 설정하고 구독을 활성화.
        빈 문자열이면 현재 진행 중인 탐지를 취소하고 유휴로 전환.
        """
        raw = msg.data.strip()
        if raw == '':
            self._target = None
            self._stop_detection()
            self.get_logger().info("탐지 취소 → 유휴 상태")
        else:
            self._target = raw
            self._start_detection()

    # ── 이미지 해상도 파악 ───────────────────────────────────────
    def _camera_info_callback(self, msg: CameraInfo):
        self._img_width  = msg.width
        self._img_height = msg.height

    # ── 핵심 파이프라인 (활성 상태일 때만 호출됨) ───────────────
    def _detection_callback(self, msg: DetectionArray):
        if not msg.detections:
            return

        for det in msg.detections:
            cls_name = det.class_name if det.class_name else f'class_{det.class_id}'
            cls_id   = det.class_id
            score    = det.score

            # ① 신뢰도 필터
            if score < MIN_CONFIDENCE:
                continue

            # ② 요청 클래스 매칭
            if self._target is not None:
                match_by_name = (cls_name.lower() == self._target.lower())
                match_by_id   = (self._target.isdigit() and
                                 int(self._target) == cls_id)
                if not (match_by_name or match_by_id):
                    continue

            # ③ bbox 중심 픽셀 추출
            bc = det.bbox.center.position
            u  = float(bc.x)
            v  = float(bc.y)

            # ④ 픽셀 → 구역 변환
            zone_id, zone_row, zone_col = _pixel_to_zone(
                u, v, self._img_width, self._img_height
            )
            label = _zone_label(zone_id)

            # ⑤ 발행
            self._request_counter += 1
            payload = {"zone_id": zone_id}

            self.get_logger().info(
                f"[{cls_name}] conf={score:.2f} | "
                f"pixel=({round(u)},{round(v)}) | "
                f"zone={zone_id}({label}) [row={zone_row}, col={zone_col}]"
            )

            msg_out = String()
            msg_out.data = json.dumps(payload)
            self._zone_pub.publish(msg_out)

            # ⑥ 탐지 성공 → 구독 해제 (1회 응답 후 유휴로 전환)
            self._stop_detection()
            return   # 같은 프레임의 나머지 탐지는 무시


def main():
    rclpy.init()
    node = TableZoneDetector()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
