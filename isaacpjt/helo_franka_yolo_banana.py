import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2


# 클래스 ID별 시각화 색상 (BGR)
CLASS_COLORS = [
    (0, 255, 0),    # class 0: 초록
    (255, 0, 0),    # class 1: 파랑
    (0, 0, 255),    # class 2: 빨강
    (0, 255, 255),  # class 3: 노랑
]


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # 커스텀 모델 로드 (클래스 0,1,2,3)
        self.model = YOLO('0622_v2_yolov8/best.pt')
        self.bridge       = CvBridge()
        self.camera_info  = None
        self.latest_depth = None
        self.frame_count  = 0

        # 모델 클래스 이름 (추론 후 자동으로 채워짐)
        self.class_names = {}

        # 클래스별 PointStamped 퍼블리셔 (탐지 시 동적 생성)
        self.point_pubs = {}

        # 구독
        self.create_subscription(
            CameraInfo, '/camera_info',
            self.camera_info_callback, 10
        )
        self.create_subscription(
            Image, '/depth',
            self.depth_callback, 10
        )
        self.create_subscription(
            Image, '/rgb',
            self.image_callback, 10
        )

        # 시각화 발행
        self.vis_pub = self.create_publisher(
            Image, '/yolo/visualization', 10
        )

        self.get_logger().info("YOLO 다중 클래스 탐지 노드 시작")
        self.get_logger().info("구독: /camera/rgb, /camera/depth, /camera_info")
        self.get_logger().info("발행: /yolo/visualization, /detected/class_N/point")

    def _get_or_create_publisher(self, cls_id: int):
        """클래스 ID에 대한 퍼블리셔를 반환. 없으면 생성."""
        if cls_id not in self.point_pubs:
            topic = f'/detected/class_{cls_id}/point'
            self.point_pubs[cls_id] = self.create_publisher(
                PointStamped, topic, 10
            )
            self.get_logger().info(f"새 퍼블리셔 생성: {topic}")
        return self.point_pubs[cls_id]

    def _get_color(self, cls_id: int):
        return CLASS_COLORS[cls_id % len(CLASS_COLORS)]

    # ── camera_info 수신 (최초 1회) ──────────────────────────
    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info(
                f"camera_info 수신 | "
                f"fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}, "
                f"cx={msg.k[2]:.1f}, cy={msg.k[5]:.1f}"
            )

    # ── depth 이미지 수신 ─────────────────────────────────────
    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='32FC1'
        )

    # ── RGB → YOLO 추론 → 클래스별 3D 좌표 계산 ─────────────
    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % 20 != 0:  # 약 3Hz
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        # 모든 클래스 탐지 (classes=None)
        results = self.model(cv_image, conf=0.3, verbose=False)

        # 모델 클래스 이름 저장 (최초 1회)
        if not self.class_names and results[0].names:
            self.class_names = results[0].names

        for r in results[0].boxes:
            x1, y1, x2, y2 = r.xyxy[0].tolist()
            conf   = float(r.conf[0])
            cls_id = int(r.cls[0])
            label  = self.class_names.get(cls_id, str(cls_id))
            color  = self._get_color(cls_id)

            # 바운딩박스 중심 픽셀
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0

            # 시각화
            cv2.rectangle(cv_image,
                          (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.circle(cv_image, (int(u), int(v)), 5, (0, 0, 255), -1)
            cv2.putText(cv_image, f"{label} {conf:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            # 3D 좌표 계산
            if self.camera_info is None or self.latest_depth is None:
                continue

            fx = self.camera_info.k[0]
            fy = self.camera_info.k[4]
            cx = self.camera_info.k[2]
            cy = self.camera_info.k[5]

            u_int = int(np.clip(u, 0, self.latest_depth.shape[1] - 1))
            v_int = int(np.clip(v, 0, self.latest_depth.shape[0] - 1))
            z = float(self.latest_depth[v_int, u_int])

            if z <= 0.0 or np.isnan(z) or z > 3.0:
                self.get_logger().warn(
                    f"[class {cls_id}] 유효하지 않은 depth: {z:.3f}m"
                )
                continue

            # 핀홀 역투영
            x_cam = (u - cx) * z / fx
            y_cam = (v - cy) * z / fy
            z_cam = z

            self.get_logger().info(
                f"[class {cls_id} '{label}'] "
                f"픽셀=({u:.1f},{v:.1f}), "
                f"conf={conf:.2f}, "
                f"3D=({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})m"
            )

            # 클래스별 토픽으로 발행
            point = PointStamped()
            point.header.stamp    = self.get_clock().now().to_msg()
            point.header.frame_id = self.camera_info.header.frame_id
            point.point.x = x_cam
            point.point.y = y_cam
            point.point.z = z_cam
            self._get_or_create_publisher(cls_id).publish(point)

        # 시각화 이미지 발행
        vis_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
        vis_msg.header = msg.header
        self.vis_pub.publish(vis_msg)


def main():
    rclpy.init()
    node = YoloDetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
