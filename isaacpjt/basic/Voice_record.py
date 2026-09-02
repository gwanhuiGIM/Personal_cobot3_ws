import rclpy
from rclpy.node import Node

# 변경 후
from std_msgs.msg import UInt8MultiArray as AudioData
from std_msgs.msg import String as AudioInfo   # info는 JSON String으로 대체
import pyaudio
import threading


class MicPublisher(Node):
    def __init__(self):
        super().__init__('mic_publisher')

        # ── 파라미터 선언 (ros2 run 시 --ros-args -p 로 덮어쓰기 가능) ──
        self.declare_parameter('device_index',  -1)    # -1 = 시스템 기본 마이크
        self.declare_parameter('sample_rate',   16000) # Hz
        self.declare_parameter('channels',      1)     # 모노
        self.declare_parameter('chunk_size',    1024)  # 프레임 수 / 콜백
        self.declare_parameter('format',        'S16') # S16 | S32 | F32

        self.device_index = self.get_parameter('device_index').value
        self.rate         = self.get_parameter('sample_rate').value
        self.channels     = self.get_parameter('channels').value
        self.chunk        = self.get_parameter('chunk_size').value
        fmt_str           = self.get_parameter('format').value

        fmt_map = {
            'S16': pyaudio.paInt16,
            'S32': pyaudio.paInt32,
            'F32': pyaudio.paFloat32,
        }
        self.pa_format = fmt_map.get(fmt_str, pyaudio.paInt16)

        # ── Publishers ──────────────────────────────────────────────
        self.pub_audio = self.create_publisher(AudioData, '/audio/raw',  10)
        self.pub_info  = self.create_publisher(AudioInfo, '/audio/info', 10)

        # ── AudioInfo (정적 메타데이터, 1회 퍼블리시 후 타이머로 주기 발행) ──
        self.info_msg              = AudioInfo()
        self.info_msg.channels     = self.channels
        self.info_msg.sample_rate  = self.rate
        self.info_msg.sample_format = fmt_str
        self.info_msg.coding_format = 'wave'

        self.create_timer(2.0, self._publish_info)   # 2초마다 info 퍼블리시
        self._publish_info()

        # ── PyAudio 스트림 시작 (별도 스레드) ───────────────────────
        self._pa      = pyaudio.PyAudio()
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        dev_name = self._pa.get_device_info_by_index(
            self.device_index if self.device_index >= 0
            else self._pa.get_default_input_device_info()['index']
        )['name']

        self.get_logger().info(f'MicPublisher started')
        self.get_logger().info(f'  Device : [{self.device_index}] {dev_name}')
        self.get_logger().info(f'  Rate   : {self.rate} Hz')
        self.get_logger().info(f'  Format : {fmt_str}  Channels: {self.channels}')
        self.get_logger().info(f'  Topic  : /audio/raw  (AudioData)')
        self.get_logger().info(f'  Topic  : /audio/info (AudioInfo)')

    def _publish_info(self):
        self.pub_info.publish(self.info_msg)

    def _capture_loop(self):
        """PyAudio 캡처 루프 — ROS2 spin 과 별도 스레드로 실행"""
        kwargs = dict(
            format=self.pa_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )
        if self.device_index >= 0:
            kwargs['input_device_index'] = self.device_index

        stream = self._pa.open(**kwargs)
        self.get_logger().info('Audio stream opened.')

        while self._running:
            try:
                raw = stream.read(self.chunk, exception_on_overflow=False)
                msg = AudioData()
                msg.data = list(raw)          # bytes → list[uint8]
                self.pub_audio.publish(msg)
            except Exception as e:
                self.get_logger().warn(f'Capture error: {e}')

        stream.stop_stream()
        stream.close()

    def destroy_node(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._pa.terminate()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MicPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()