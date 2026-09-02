from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension

# ROS2 Bridge 활성화
enable_extension("isaacsim.ros2.bridge")

# 일부 Isaac Sim 버전에서는 Generic ROS2 node가 isaacsim.ros2.nodes 쪽에 있음.
try:
    enable_extension("isaacsim.ros2.nodes")
except Exception:
    pass

simulation_app.update()

from pathlib import Path
import sys
import time

import numpy as np
import random

import omni.usd
import omni.graph.core as og

from pxr import Usd, UsdGeom, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator


_THIS_DIR = Path(__file__).resolve().parent

# rmpflow 인프라 폴더 경로 등록
RMPFLOW_DIR = str(_THIS_DIR / "rmpflow")
if RMPFLOW_DIR not in sys.path:
    sys.path.insert(0, RMPFLOW_DIR)

from m0609_pick_place_controller import PickPlaceController


# ╔══════════════════════════════════════════════════════════════╗
# ║  A. Task 파라미터                                             ║
# ╚══════════════════════════════════════════════════════════════╝

USD_PATH        = str(_THIS_DIR / "Collected_m0609_camera/m0609_camera.usd")
ROBOT_PRIM_PATH = "/World/m0609"
EE_LINK_NAME    = "link_6"
GRIPPER_JOINTS  = ["finger_joint", "right_inner_knuckle_joint"]

DRIVE_STIFFNESS = 1e8
DRIVE_DAMPING   = 1e4
DRIVE_MAX_FORCE = 1e8

GRIPPER_OPEN    = [0.0, 0.0]
GRIPPER_CLOSE   = [0.5, 0.5]
GRIPPER_DELTA   = [-0.5, -0.5]

FINGER_STATIC   = 1.8
FINGER_DYNAMIC  = 1.4
CUBE_STATIC     = 1.2
CUBE_DYNAMIC    = 1.0


# ╔══════════════════════════════════════════════════════════════╗
# ║  B. Controller 파라미터                                       ║
# ╚══════════════════════════════════════════════════════════════╝

M0609_URDF_PATH           = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
M0609_DESCRIPTION_PATH    = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

GREEN_CUBE_INIT_POS = np.array([0.3, 0.23, 0.5])
BLUE_CUBE_INIT_POS  = np.array([0.3, -0.23, 0.5])

# 선택되지 않은 큐브를 카메라/작업공간 밖으로 치우는 위치
HIDDEN_CUBE_POS = np.array([-10.0, -10.0, 1.0])

GREEN_GOAL_POS = np.array([0.30, 0.4, 0.0])
BLUE_GOAL_POS  = np.array([0.55, -0.35, 0.0])

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


# ╔══════════════════════════════════════════════════════════════╗
# ║  C. ROS2 Bridge Subscriber Action Graph 설정                 ║
# ╚══════════════════════════════════════════════════════════════╝

COLOR_GRAPH_PATH = "/World/ROS_Color_Subscribe_Graph"
COLOR_SUB_NODE_NAME = "SubscribeDetectedColor"
COLOR_SUB_NODE_PATH = COLOR_GRAPH_PATH + "/" + COLOR_SUB_NODE_NAME

DETECTED_COLOR_TOPIC = "/detected_color"

# PC B에서 발행할 메시지:
# topic: /detected_color
# type : std_msgs/msg/Int32
# data : blue=1, green=2


def _try_create_color_subscriber_graph(subscriber_node_type: str) -> bool:
    try:
        og.Controller.edit(
            {
                "graph_path": COLOR_GRAPH_PATH,
                "evaluator_name": "execution",
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                    (COLOR_SUB_NODE_NAME, subscriber_node_type),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        "OnPlaybackTick.outputs:tick",
                        f"{COLOR_SUB_NODE_NAME}.inputs:execIn",
                    ),
                    (
                        "ROS2Context.outputs:context",
                        f"{COLOR_SUB_NODE_NAME}.inputs:context",
                    ),
                ],
                og.Controller.Keys.SET_VALUES: [
                    (f"{COLOR_SUB_NODE_NAME}.inputs:topicName", DETECTED_COLOR_TOPIC),
                    (f"{COLOR_SUB_NODE_NAME}.inputs:messagePackage", "std_msgs"),
                    (f"{COLOR_SUB_NODE_NAME}.inputs:messageSubfolder", "msg"),
                    (f"{COLOR_SUB_NODE_NAME}.inputs:messageName", "Int32"),
                ],
            },
        )

        print(f"[OK] ROS2 color subscriber graph created: {subscriber_node_type}")
        return True

    except Exception as e:
        print(f"[WARN] failed to create subscriber graph with {subscriber_node_type}: {e}")
        return False


def setup_color_subscriber_graph():
    subscriber_node_types = [
        "isaacsim.ros2.bridge.ROS2Subscriber",
        "isaacsim.ros2.nodes.ROS2Subscriber",
    ]

    for node_type in subscriber_node_types:
        if _try_create_color_subscriber_graph(node_type):
            for _ in range(5):
                simulation_app.update()
            return

    raise RuntimeError(
        "ROS2 Subscriber Action Graph 생성 실패. "
        "Action Graph UI에서 ROS2 Subscriber 노드 타입 이름을 확인해야 함."
    )


def reset_color_subscriber_graph():
    """
    ROS2 Subscriber Action Graph를 삭제 후 재생성해서
    이전 outputs:data 값이 남아있는 문제를 방지한다.
    """
    stage = omni.usd.get_context().get_stage()

    graph_prim = stage.GetPrimAtPath(COLOR_GRAPH_PATH)
    if graph_prim.IsValid():
        stage.RemovePrim(COLOR_GRAPH_PATH)
        print("[RESET] old ROS color subscriber graph removed")

        for _ in range(5):
            simulation_app.update()

    setup_color_subscriber_graph()
    print("[RESET] ROS color subscriber graph recreated")


def _og_get_attr(attr_path: str):
    try:
        return og.Controller.get(attr_path)
    except Exception:
        try:
            return og.Controller.get(og.Controller.attribute(attr_path))
        except Exception:
            return None


def read_detected_color_from_graph():
    """
    ROS2 Subscriber 노드의 outputs:data 값을 읽음.
    1=blue, 2=green만 유효.
    0 또는 None은 색상 미수신으로 처리.
    """
    value = _og_get_attr(COLOR_SUB_NODE_PATH + ".outputs:data")

    if value is None:
        return None

    try:
        color_id = int(value)
    except Exception:
        return None

    if color_id == 0:
        return None

    if color_id not in [1, 2]:
        return None

    return color_id


# ============================================================
# 유틸
# ============================================================

def find_prim_path_by_name(root_path: str, name: str):
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)

    if not root_prim.IsValid():
        return None

    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
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


# ============================================================
# Task
# ============================================================

class M0609Task(BaseTask):

    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._task_achieved = False

        self.target_cube = None
        self.non_target_cube = None
        self.pick_cube_position = None

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._load_usd()
        self._discover_links()
        self._setup_physics()
        self._register_robot(scene)
        self._create_scene(scene)
        print("\n  [완료] 씬 구성 성공!\n")

    def _load_usd(self):
        print("\n" + "=" * 60)
        print("[1.LOAD] USD 로드")
        print("=" * 60)

        stage = omni.usd.get_context().get_stage()

        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim.IsValid():
            world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()

        world_prim.GetReferences().AddReference(USD_PATH)

        for _ in range(15):
            simulation_app.update()

        print(f"  [OK] {USD_PATH}")

    def _discover_links(self):
        print("\n" + "=" * 60)
        print("[2.DISCOVER] 링크 경로 탐색")
        print("=" * 60)

        self._ee_path = find_prim_path_by_name(ROBOT_PRIM_PATH, EE_LINK_NAME)

        if self._ee_path is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found")

        print(f"  EE ({EE_LINK_NAME}) = {self._ee_path}")

        for jn in GRIPPER_JOINTS:
            print(f"  {jn:<35} = {find_prim_path_by_name(ROBOT_PRIM_PATH, jn)}")

    def _setup_physics(self):
        print("\n" + "=" * 60)
        print("[3.PHYSICS] 물리 설정")
        print("=" * 60)

        stage = omni.usd.get_context().get_stage()

        drive_count = 0

        for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM_PATH)):
            for dt in ["angular", "linear"]:
                drive = UsdPhysics.DriveAPI.Get(prim, dt)

                if drive:
                    drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                    drive.GetDampingAttr().Set(DRIVE_DAMPING)
                    drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                    drive_count += 1

        print(f"  [OK] drive updated: {drive_count}")

    def _register_robot(self, scene):
        print("\n" + "=" * 60)
        print("[4.REGISTER] 로봇 등록")
        print("=" * 60)

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

        print(f"  [OK] SingleManipulator: {ROBOT_PRIM_PATH}")

    def _create_scene(self, scene):
        print("\n" + "=" * 60)
        print("[5.SCENE] 작업 환경 구성")
        print("=" * 60)

        cube_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/cube_material",
            static_friction=CUBE_STATIC,
            dynamic_friction=CUBE_DYNAMIC,
            restitution=0.0,
        )

        self.blue_cube = scene.add(
            DynamicCuboid(
                prim_path="/World/blue_cube",
                name="blue_cube",
                position=BLUE_CUBE_INIT_POS,
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([0.0, 0.0, 1.0]),
                mass=0.05,
                physics_material=cube_material,
            )
        )

        self.green_cube = scene.add(
            DynamicCuboid(
                prim_path="/World/green_cube",
                name="green_cube",
                position=GREEN_CUBE_INIT_POS,
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([0.0, 1.0, 0.0]),
                mass=0.05,
                physics_material=cube_material,
            )
        )

        print(f"  [OK] green cube @ {GREEN_CUBE_INIT_POS}")
        print(f"  [OK] blue cube  @ {BLUE_CUBE_INIT_POS}")

        scene.add(
            VisualCuboid(
                prim_path="/World/green_goal_marker",
                name="green_goal_marker",
                position=GREEN_GOAL_POS,
                scale=np.array([0.06, 0.06, 0.001]),
                color=np.array([0.0, 1.0, 0.0]),
            )
        )

        scene.add(
            VisualCuboid(
                prim_path="/World/blue_goal_marker",
                name="blue_goal_marker",
                position=BLUE_GOAL_POS,
                scale=np.array([0.06, 0.06, 0.001]),
                color=np.array([0.0, 0.0, 1.0]),
            )
        )

        print(f"  [OK] green goal @ {GREEN_GOAL_POS}")
        print(f"  [OK] blue goal  @ {BLUE_GOAL_POS}")

        finger_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/finger_material",
            static_friction=FINGER_STATIC,
            dynamic_friction=FINGER_DYNAMIC,
            restitution=0.0,
        )

        for link_name in ["left_inner_finger", "right_inner_finger"]:
            link_path = find_prim_path_by_name(ROBOT_PRIM_PATH, link_name)

            if link_path:
                SingleGeometryPrim(
                    prim_path=link_path,
                    name=f"{link_name}_geom",
                ).apply_physics_material(finger_material)

                print(f"  [OK] friction: {link_path}")

    def get_observations(self):
        green_cube_pos, _ = self.green_cube.get_world_pose()
        blue_cube_pos, _ = self.blue_cube.get_world_pose()

        return {
            self._robot.name: {
                "joint_positions": self._robot.get_joint_positions(),
            },
            self.green_cube.name: {
                "position": green_cube_pos,
                "goal_position": GREEN_GOAL_POS,
            },
            self.blue_cube.name: {
                "position": blue_cube_pos,
                "goal_position": BLUE_GOAL_POS,
            },
        }

    def pre_step(self, control_index, simulation_time):
        pass

    def post_reset(self):
        self._robot.gripper.set_joint_positions(
            self._robot.gripper.joint_opened_positions
        )

        # 1. 랜덤으로 하나 선택
        self.target_cube = random.choice([self.blue_cube, self.green_cube])

        # 2. 선택되지 않은 큐브 결정
        if self.target_cube.name == "blue_cube":
            self.non_target_cube = self.green_cube
        else:
            self.non_target_cube = self.blue_cube

        # 3. 선택된 큐브를 pick 영역 랜덤 위치로 이동
        pick_pos = np.array([
            random.uniform(0.3, 0.5),
            random.uniform(-0.2, 0.2),
            0.05,
        ])

        self.target_cube.set_world_pose(position=pick_pos)

        # 4. 선택되지 않은 큐브는 카메라/작업공간 밖으로 이동
        self.non_target_cube.set_world_pose(position=HIDDEN_CUBE_POS)

        # 5. pick 위치 저장
        # place 위치는 여기서 정하지 않음.
        # place 위치는 로봇이 큐브를 잡은 뒤 ROS2 색상값으로 결정.
        self.pick_cube_position = pick_pos

        print(
            f"[RESET] target={self.target_cube.name}, "
            f"pick_pos={self.pick_cube_position}, "
            f"hidden={self.non_target_cube.name}@{HIDDEN_CUBE_POS}"
        )


# ============================================================
# Main
# ============================================================

def main():
    my_world = World(stage_units_in_meters=1.0)

    task = M0609Task(name="m0609_task")
    my_world.add_task(task)
    my_world.reset()

    robot = my_world.scene.get_object("m0609_robot")
    initialize_robot(robot, my_world)

    # 홈 포지션 안정화 대기
    for _ in range(30):
        my_world.step(render=True)

    print("\n" + "=" * 60)
    print("[C-2] PickPlaceController 생성")
    print("=" * 60)
    print(f"  URDF         = {M0609_URDF_PATH}")
    print(f"  description  = {M0609_DESCRIPTION_PATH}")
    print(f"  rmpflow      = {M0609_RMPFLOW_CONFIG_PATH}")
    print(f"  events_dt    = {EVENTS_DT}")
    print(f"  EE frame     = {EE_LINK_NAME}")

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

    print("  [OK] Controller 생성 완료")

    # ROS2 Bridge Generic Subscriber Action Graph 생성
    setup_color_subscriber_graph()

    ee_pos, _ = robot.end_effector.get_world_pose()
    print(f"\n  EE 초기 위치      = {ee_pos}")
    print(f"  초록 큐브 위치    = {GREEN_CUBE_INIT_POS}")
    print(f"  파랑 큐브 위치    = {BLUE_CUBE_INIT_POS}")
    print(f"  숨김 큐브 위치    = {HIDDEN_CUBE_POS}")
    print(f"  초록 목표 위치    = {GREEN_GOAL_POS}")
    print(f"  파랑 목표 위치    = {BLUE_GOAL_POS}")

    print("\n[Pick & Place 시작]\n")
    print(f"[WAIT TOPIC] {DETECTED_COLOR_TOPIC} / std_msgs/msg/Int32")
    print("[COLOR ID] blue=1, green=2")
    print("[COLOR READ] after grasp only")
    print("[COLOR RESET] subscriber graph recreated on restart\n")

    was_playing = False
    task_done = False

    # 로봇이 큐브를 잡은 뒤 한 번만 색상 결과를 고정하기 위한 변수
    locked_color_id = None
    placing_position = None

    # 색상값을 받기 전까지 controller.forward에 임시로 넘길 place 위치
    # event 0~2에서는 실제 place 위치를 거의 쓰지 않는다는 전제
    temporary_place_position = BLUE_GOAL_POS

    try:
        while simulation_app.is_running():
            my_world.step(render=True)
            time.sleep(0.01)

            is_playing = my_world.is_playing()

            # Play 시작 감지 → 리셋
            if is_playing and not was_playing:
                my_world.reset()
                initialize_robot(robot, my_world)
                controller.reset()

                # 이전 ROS2 Subscriber outputs:data 값 제거
                reset_color_subscriber_graph()

                task_done = False
                locked_color_id = None
                placing_position = None
                temporary_place_position = BLUE_GOAL_POS

                print("[START] simulation reset complete")
                print("[RESET] locked_color_id cleared")

            # 매 스텝 제어
            if is_playing and not task_done:

                obs = task.get_observations()
                current_joints = obs["m0609_robot"]["joint_positions"]

                if task.target_cube is None:
                    print("[ERROR] target_cube is None")
                    was_playing = is_playing
                    continue

                # pick 위치는 랜덤으로 이동한 큐브의 현재 위치를 사용
                cube_position, _ = task.target_cube.get_world_pose()

                # 현재 controller 이벤트 확인
                event = controller.get_current_event()

                # ------------------------------------------------------------
                # event 기준
                # 0: 접근 이동
                # 1: 하강
                # 2: 그리퍼 닫기 대기
                # 3: 그리퍼 닫힘 유지
                # 4: 들어올리기
                # 5: Place 위치로 이동
                #
                # event >= 3 이면 큐브를 잡은 상태로 보고
                # ROS2 Bridge에서 색상값을 한 번만 읽어 place 위치를 고정한다.
                # ------------------------------------------------------------
                if locked_color_id is None and event >= 3:
                    color_id = read_detected_color_from_graph()

                    if color_id is None:
                        print("[WAIT] cube grasped, waiting color result from ROS2 Bridge...")
                        was_playing = is_playing
                        continue

                    if color_id == 1:
                        locked_color_id = 1
                        placing_position = BLUE_GOAL_POS
                    elif color_id == 2:
                        locked_color_id = 2
                        placing_position = GREEN_GOAL_POS
                    else:
                        print(f"[ERROR] invalid color id: {color_id}")
                        was_playing = is_playing
                        continue

                    print(
                        f"[LOCK AFTER GRASP] "
                        f"event={event}, "
                        f"color_id={locked_color_id}, "
                        f"placing_position={placing_position}"
                    )

                # 색상값을 아직 못 받은 상태에서는 임시 place 위치를 넘김.
                # event 0~2에서는 실제 place 위치를 거의 쓰지 않으므로 괜찮음.
                active_place_position = (
                    placing_position
                    if locked_color_id is not None
                    else temporary_place_position
                )

                actions = controller.forward(
                    picking_position=cube_position,
                    placing_position=active_place_position,
                    current_joint_positions=current_joints,
                    end_effector_offset=EE_OFFSET,
                )

                robot.apply_action(actions)

                if controller.is_done():
                    print("[완료] Pick & Place 성공!")
                    task_done = True
                    my_world.pause()

                event = controller.get_current_event()
                ee_pos, _ = robot.end_effector.get_world_pose()

                print(
                    f"  [event={event}] "
                    f"color_id={locked_color_id} "
                    f"target={task.target_cube.name} "
                    f"hidden={task.non_target_cube.name if task.non_target_cube else None} "
                    f"pick={cube_position} "
                    f"place={active_place_position} "
                    f"ee_z={ee_pos[2]:.4f}"
                )

            was_playing = is_playing

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()