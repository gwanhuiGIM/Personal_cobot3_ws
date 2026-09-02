
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

from pathlib import Path
import sys
import random
import threading

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image as RosImage

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
try:
    from isaacsim.sensors.camera import Camera  # type: ignore[import-untyped]
except ImportError:
    Camera = None  # IsaacSIM 런타임 외부에서는 미사용

_THIS_DIR = Path(__file__).resolve().parent

RMPFLOW_DIR = str(_THIS_DIR / "rmpflow")
if RMPFLOW_DIR not in sys.path:
    sys.path.insert(0, RMPFLOW_DIR)

from m0609_pick_place_controller import PickPlaceController


# ╔══════════════════════════════════════════════════════════════╗
# ║  A. 파라미터                                                  ║
# ╚══════════════════════════════════════════════════════════════╝

# ── 로봇 ──────────────────────────────────────────────────────
USD_PATH        = str(_THIS_DIR / "Collected_m0609_camera/m0609_camera.usd")
ROBOT_PRIM_PATH = "/World/m0609"
EE_LINK_NAME    = "link_6"
GRIPPER_JOINTS  = ["finger_joint", "right_inner_knuckle_joint"]

DRIVE_STIFFNESS = 1e8
DRIVE_DAMPING   = 1e4
DRIVE_MAX_FORCE = 1e8

GRIPPER_OPEN  = [0.0, 0.0]
GRIPPER_CLOSE = [0.5, 0.5]
GRIPPER_DELTA = [-0.5, -0.5]

FINGER_STATIC  = 1.8
FINGER_DYNAMIC = 1.4
CUBE_STATIC    = 1.2
CUBE_DYNAMIC   = 1.0

# ── RMPFlow ────────────────────────────────────────────────────
M0609_URDF_PATH           = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
M0609_DESCRIPTION_PATH    = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

EE_OFFSET = np.array([0.0, 0.0, 0.2])

EVENTS_DT = [
    0.008,   # 0. 접근 이동
    0.005,   # 1. 하강
    0.02,    # 2. 그리퍼 닫기 대기
    0.1,     # 3. 그리퍼 닫힘 유지
    0.0025,  # 4. 들어올리기
    0.01,    # 5. Place 위치로 이동
    0.0025,  # 6. 하강
    1,       # 7. 그리퍼 열기 대기
    0.008,   # 8. 상승
    0.08,    # 9. 복귀
]

# ── 큐브 / 마커 ────────────────────────────────────────────────
COLOR_BLUE  = 1
COLOR_GREEN = 2

BLUE_CUBE_STANDBY  = np.array([0.0,  0.50, 0.30])  # 공중 대기
GREEN_CUBE_STANDBY = np.array([0.0, -0.50, 0.30])  # 공중 대기

PICK_AREA_X  = (0.25, 0.42)   # Pick 영역 X 범위
PICK_AREA_Y  = (-0.18, 0.18)  # Pick 영역 Y 범위
CUBE_HALF_H  = 0.025          # 큐브 반높이 (5 cm 큐브)

BLUE_MARKER_POS  = np.array([0.55,  0.20, 0.001])
GREEN_MARKER_POS = np.array([0.55, -0.25, 0.001])

SETTLE_FRAMES = 60  # 큐브 낙하 안정화 대기 프레임

# ── ROS2 토픽 ──────────────────────────────────────────────────
CAMERA_TOPIC    = "/wrist_camera/image_raw"
DETECTION_TOPIC = "/color_detection"


# ╔══════════════════════════════════════════════════════════════╗
# ║  B. 유틸리티                                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def find_prim_path_by_name(root_path: str, name: str):
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def find_camera_prim(root_path: str):
    """root_path 하위에서 첫 번째 Camera prim 경로를 반환."""
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetTypeName() == "Camera":
            return str(prim.GetPath())
    return None


def initialize_robot(robot, world):
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )
    robot.set_joint_positions(np.zeros(robot.num_dof))


# ╔══════════════════════════════════════════════════════════════╗
# ║  C. ROS2 노드                                                 ║
# ║     - 카메라 이미지 발행 (wrist camera → PC B)                 ║
# ║     - 색상 감지 결과 구독 (PC B → IsaacSIM)                    ║
# ║     포맷: "color,px,py" 또는 "color1,px1,py1;color2,px2,py2" ║
# ╚══════════════════════════════════════════════════════════════╝

class PickPlaceROS2Node(Node):

    def __init__(self):
        super().__init__("isaac_pick_place")
        self._image_pub = self.create_publisher(RosImage, CAMERA_TOPIC, 10)
        self._sub = self.create_subscription(
            String, DETECTION_TOPIC, self._on_detection, 10
        )
        self._lock = threading.Lock()
        # {color_id: (pixel_x, pixel_y)}  — 마지막 수신 메시지 기준
        self._detections: dict = {}

    def _on_detection(self, msg: String):
        """포맷: "1,857,490" 또는 "1,857,490;2,737,424" """
        new_dets = {}
        for part in msg.data.strip().split(";"):
            vals = part.strip().split(",")
            if len(vals) >= 3:
                try:
                    cid = int(vals[0])
                    new_dets[cid] = (int(vals[1]), int(vals[2]))
                except ValueError:
                    continue
        with self._lock:
            self._detections = new_dets

    def get_detection_for_color(self, color_id: int):
        """지정 color_id 감지 결과 (px, py) 반환. 없으면 None."""
        with self._lock:
            return self._detections.get(color_id)

    def publish_image(self, rgb: np.ndarray):
        if rgb is None or rgb.size == 0:
            return
        msg = RosImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "wrist_camera"
        h, w = rgb.shape[:2]
        msg.height = h
        msg.width  = w
        msg.encoding     = "rgb8"
        msg.is_bigendian = False
        msg.step = w * 3
        msg.data = rgb.astype(np.uint8).flatten().tobytes()
        self._image_pub.publish(msg)

    def reset_detection(self):
        with self._lock:
            self._detections = {}


# ╔══════════════════════════════════════════════════════════════╗
# ║  D. Task                                                      ║
# ╚══════════════════════════════════════════════════════════════╝

class M0609Task(BaseTask):

    def __init__(self, name):
        super().__init__(name=name, offset=None)

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._load_usd()
        self._discover_links()
        self._setup_physics()
        self._register_robot(scene)
        self._create_scene(scene)
        print("\n[완료] 씬 구성 성공!\n")

    # ── 내부 초기화 ──────────────────────────────────────────────

    def _load_usd(self):
        print(f"\n[USD 로드] {USD_PATH}")
        stage = omni.usd.get_context().get_stage()
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim.IsValid():
            world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        world_prim.GetReferences().AddReference(USD_PATH)
        for _ in range(15):
            simulation_app.update()

    def _discover_links(self):
        self._ee_path = find_prim_path_by_name(ROBOT_PRIM_PATH, EE_LINK_NAME)
        if self._ee_path is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' prim을 찾지 못했습니다.")
        print(f"[EE 경로] {self._ee_path}")

    def _setup_physics(self):
        stage = omni.usd.get_context().get_stage()
        for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM_PATH)):
            for dt in ["angular", "linear"]:
                drive = UsdPhysics.DriveAPI.Get(prim, dt)
                if drive:
                    drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                    drive.GetDampingAttr().Set(DRIVE_DAMPING)
                    drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)

    def _register_robot(self, scene):
        gripper = ParallelGripper(
            end_effector_prim_path=self._ee_path,
            joint_prim_names=GRIPPER_JOINTS,
            joint_opened_positions=np.array(GRIPPER_OPEN),
            joint_closed_positions=np.array(GRIPPER_CLOSE),
            action_deltas=np.array(GRIPPER_DELTA),
        )
        self._robot = scene.add(
            SingleManipulator(
                prim_path=ROBOT_PRIM_PATH,
                name="m0609_robot",
                end_effector_prim_path=self._ee_path,
                gripper=gripper,
            )
        )

    def _create_scene(self, scene):
        cube_mat = PhysicsMaterial(
            prim_path="/World/Physics_Materials/cube_material",
            static_friction=CUBE_STATIC,
            dynamic_friction=CUBE_DYNAMIC,
            restitution=0.0,
        )

        # 파란 큐브 (초기: 공중 대기)
        self._blue_cube = scene.add(DynamicCuboid(
            prim_path="/World/blue_cube",
            name="blue_cube",
            position=BLUE_CUBE_STANDBY,
            scale=np.array([0.05, 0.05, 0.05]),
            color=np.array([0.0, 0.0, 1.0]),
            mass=0.05,
            physics_material=cube_mat,
        ))

        # 초록 큐브 (초기: 공중 대기)
        self._green_cube = scene.add(DynamicCuboid(
            prim_path="/World/green_cube",
            name="green_cube",
            position=GREEN_CUBE_STANDBY,
            scale=np.array([0.05, 0.05, 0.05]),
            color=np.array([0.0, 1.0, 0.0]),
            mass=0.05,
            physics_material=cube_mat,
        ))

        # 파란 마커 (Place 목표)
        scene.add(VisualCuboid(
            prim_path="/World/blue_marker",
            name="blue_marker",
            position=BLUE_MARKER_POS,
            scale=np.array([0.07, 0.07, 0.001]),
            color=np.array([0.0, 0.0, 1.0]),
        ))

        # 초록 마커 (Place 목표)
        scene.add(VisualCuboid(
            prim_path="/World/green_marker",
            name="green_marker",
            position=GREEN_MARKER_POS,
            scale=np.array([0.07, 0.07, 0.001]),
            color=np.array([0.0, 1.0, 0.0]),
        ))

        # 손가락 마찰 재질
        finger_mat = PhysicsMaterial(
            prim_path="/World/Physics_Materials/finger_material",
            static_friction=FINGER_STATIC,
            dynamic_friction=FINGER_DYNAMIC,
            restitution=0.0,
        )
        for link_name in ["left_inner_finger", "right_inner_finger"]:
            path = find_prim_path_by_name(ROBOT_PRIM_PATH, link_name)
            if path:
                SingleGeometryPrim(
                    prim_path=path, name=f"{link_name}_geom"
                ).apply_physics_material(finger_mat)

        print("[씬] 큐브 2개 + 마커 2개 + 손가락 마찰 설정 완료")

    # ── 외부 접근 헬퍼 ──────────────────────────────────────────

    def get_cube(self, color_id: int) -> DynamicCuboid:
        return self._blue_cube if color_id == COLOR_BLUE else self._green_cube

    def get_standby_pos(self, color_id: int) -> np.ndarray:
        return BLUE_CUBE_STANDBY.copy() if color_id == COLOR_BLUE else GREEN_CUBE_STANDBY.copy()

    def get_marker_pos(self, color_id: int) -> np.ndarray:
        return BLUE_MARKER_POS.copy() if color_id == COLOR_BLUE else GREEN_MARKER_POS.copy()

    def freeze_cube(self, color_id: int):
        """큐브를 대기 위치에 정지시킨다 (매 프레임 호출)."""
        cube = self.get_cube(color_id)
        cube.set_world_pose(self.get_standby_pos(color_id))
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))

    def get_observations(self, active_color: int) -> dict:
        cube_pos, _ = self.get_cube(active_color).get_world_pose()
        return {
            "joint_positions": self._robot.get_joint_positions(),
            "cube_position":   cube_pos,
        }

    def post_reset(self):
        self._robot.gripper.set_joint_positions(
            self._robot.gripper.joint_opened_positions
        )
        for cid in [COLOR_BLUE, COLOR_GREEN]:
            cube = self.get_cube(cid)
            cube.set_world_pose(self.get_standby_pos(cid))
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))


# ╔══════════════════════════════════════════════════════════════╗
# ║  E. 상태 머신 상수                                            ║
# ╚══════════════════════════════════════════════════════════════╝

class State:
    IDLE              = "IDLE"
    DROPPING          = "DROPPING"           # 큐브 pick 위치로 이동 후 안정화 대기
    WAITING_DETECTION = "WAITING_DETECTION"  # 카메라 스트리밍 + 색상 감지 대기
    EXECUTING         = "EXECUTING"          # Pick & Place 실행 중
    DONE              = "DONE"               # 완료


# ╔══════════════════════════════════════════════════════════════╗
# ║  F. 메인                                                      ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    # ROS2 초기화 (별도 스레드에서 spin)
    rclpy.init()
    ros_node = PickPlaceROS2Node()
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    # World + Task
    my_world = World(stage_units_in_meters=1.0)
    task = M0609Task(name="m0609_task")
    my_world.add_task(task)
    my_world.reset()

    robot = my_world.scene.get_object("m0609_robot")
    initialize_robot(robot, my_world)

    # 손목 카메라 초기화
    camera_path = find_camera_prim(ROBOT_PRIM_PATH)
    wrist_cam = None
    if camera_path:
        print(f"[카메라] {camera_path}")
        wrist_cam = Camera(prim_path=camera_path, resolution=(640, 480))
        wrist_cam.initialize()
    else:
        print("[경고] 손목 카메라 prim 없음 — 이미지 발행 불가")

    # 홈 포지션 안정화
    for _ in range(30):
        my_world.step(render=True)

    # PickPlaceController 생성
    controller = PickPlaceController(
        name="m0609_pick_place_controller",
        gripper=robot.gripper,
        robot_articulation=robot,
        end_effector_initial_height=0.30,
        events_dt=EVENTS_DT,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
    )

    # 상태 머신 변수
    state        = State.IDLE
    was_playing  = False
    active_color = None   # 이번 사이클에서 선택된 큐브 색상
    place_pos    = None   # 감지 색상 기반 Place 목표 위치
    settle_count = 0

    print("\n[Ready] Play 버튼을 눌러 Pick & Place를 시작하세요.\n")

    while simulation_app.is_running():
        my_world.step(render=True)
        is_playing = my_world.is_playing()

        # ── Play 버튼 감지 → 초기화 + 큐브 Drop ────────────────
        if is_playing and not was_playing:
            my_world.reset()
            initialize_robot(robot, my_world)
            controller.reset()
            ros_node.reset_detection()

            # 랜덤으로 파란/초록 큐브 선택
            active_color   = random.choice([COLOR_BLUE, COLOR_GREEN])
            standby_color  = COLOR_GREEN if active_color == COLOR_BLUE else COLOR_BLUE
            active_name    = "파란" if active_color == COLOR_BLUE else "초록"

            # 선택된 큐브 → 랜덤 Pick 위치로 이동
            px = random.uniform(*PICK_AREA_X)
            py = random.uniform(*PICK_AREA_Y)
            pick_pos = np.array([px, py, CUBE_HALF_H])
            task.get_cube(active_color).set_world_pose(pick_pos)

            # 비활성 큐브 → 대기 위치 고정
            task.freeze_cube(standby_color)

            settle_count = 0
            place_pos    = None
            state        = State.DROPPING

            print(f"\n{'='*55}")
            print(f"[선택] {active_name} 큐브  →  pick 위치 ({px:.3f}, {py:.3f})")
            print(f"{'='*55}")

        if is_playing:
            # 매 프레임: 비활성 큐브 대기 위치 유지 (중력 무시)
            if active_color is not None:
                standby_color = COLOR_GREEN if active_color == COLOR_BLUE else COLOR_BLUE
                task.freeze_cube(standby_color)

            # ── 상태 머신 ─────────────────────────────────────
            if state == State.DROPPING:
                # 큐브가 테이블 위에 안착할 때까지 대기
                settle_count += 1
                if settle_count >= SETTLE_FRAMES:
                    state = State.WAITING_DETECTION
                    active_name = "파란" if active_color == COLOR_BLUE else "초록"
                    print(f"[감지 대기] {active_name}({active_color}) 큐브 감지 중... (topic: {DETECTION_TOPIC})")

            elif state == State.WAITING_DETECTION:
                # 카메라 이미지 발행
                if wrist_cam:
                    frame = wrist_cam.get_current_frame()
                    if frame is not None and "rgba" in frame and frame["rgba"] is not None:
                        ros_node.publish_image(frame["rgba"][:, :, :3])

                # PC B 감지 결과 확인 — active_color와 일치하는 항목만 사용
                pixel = ros_node.get_detection_for_color(active_color)
                if pixel is not None:
                    active_name = "파란" if active_color == COLOR_BLUE else "초록"
                    place_pos   = task.get_marker_pos(active_color)
                    print(f"[감지 완료] 색상={active_name}({active_color}), 픽셀={pixel}")
                    print(f"[목표] {active_name} 마커 = {place_pos[:2]}")
                    state = State.EXECUTING

            elif state == State.EXECUTING:
                # 카메라 이미지 계속 발행
                if wrist_cam:
                    frame = wrist_cam.get_current_frame()
                    if frame is not None and "rgba" in frame and frame["rgba"] is not None:
                        ros_node.publish_image(frame["rgba"][:, :, :3])

                obs        = task.get_observations(active_color)
                cube_pos   = obs["cube_position"]
                joints     = obs["joint_positions"]

                actions = controller.forward(
                    picking_position=cube_pos,
                    placing_position=place_pos,
                    current_joint_positions=joints,
                    end_effector_offset=EE_OFFSET,
                )
                robot.apply_action(actions)

                event  = controller.get_current_event()
                ee_pos, _ = robot.end_effector.get_world_pose()
                print(f"  [event={event}] cube_z={cube_pos[2]:.4f}  ee_z={ee_pos[2]:.4f}")

                if controller.is_done():
                    active_name = "파란" if active_color == COLOR_BLUE else "초록"
                    print(f"\n[완료] {active_name} 큐브 Pick & Place 성공!")
                    state = State.DONE
                    my_world.pause()

        was_playing = is_playing

    ros_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
