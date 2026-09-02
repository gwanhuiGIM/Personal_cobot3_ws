from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")
enable_extension("isaacsim.robot.surface_gripper")
simulation_app.update()

from pathlib import Path
import sys
import time

import numpy as np
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators.manipulators import SingleManipulator

_THIS_DIR = Path(__file__).resolve().parent

RMPFLOW_DIR = str(_THIS_DIR / "rmpflow")
if RMPFLOW_DIR not in sys.path:
    sys.path.insert(0, RMPFLOW_DIR)

from dual_surface_gripper_adapter import DualSurfaceGripperAdapter
from m0609_pick_place_controller_surface import PickPlaceController


# ============================================================
# A. 에셋/Prim 경로
# ============================================================
ROBOT_USD_PATH = str(
    _THIS_DIR / "Collected_test_dual_suction_surface_grippers_fixed_paths/test_dual_suction_surface_grippers_fixed_paths.usda"
    
)
TRAY_USD_PATH = str(
    _THIS_DIR / "Collected_model_redtray_scaled_for_180mm_pads/model_redtray_scaled_for_180mm_pads.usda"
)

ROBOT_PRIM_PATH = "/World/m0609"
EE_LINK_NAME = "link_6"

SURFACE_GRIPPER_PATHS = [
    "/World/m0609/onrobot_rg2ft/gripper_body/dual_suction_tool/"
    "suction_contact_left/SurfaceGripper_left",
    "/World/m0609/onrobot_rg2ft/gripper_body/dual_suction_tool/"
    "suction_contact_right/SurfaceGripper_right",
]

SUCTION_CONTACT_PATHS = [
    "/World/m0609/onrobot_rg2ft/gripper_body/dual_suction_tool/"
    "suction_contact_left",
    "/World/m0609/onrobot_rg2ft/gripper_body/dual_suction_tool/"
    "suction_contact_right",
]

TRAY_REFERENCE_PRIM_PATH = "/World/target_tray"
TRAY_RIGID_PRIM_PATH = "/World/target_tray/E_redtray_28"

DRIVE_STIFFNESS = 1e8
DRIVE_DAMPING = 1e4
DRIVE_MAX_FORCE = 1e8


# ============================================================
# B. 작업 위치/컨트롤러 설정
# ============================================================
M0609_URDF_PATH = str(
    _THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf"
)
M0609_DESCRIPTION_PATH = str(
    _THIS_DIR / "rmpflow/m0609_description.yaml"
)
M0609_RMPFLOW_CONFIG_PATH = str(
    _THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml"
)

# 트레이 원점은 바닥 근처이고, 블록 윗면은 약 18.6 mm 높이다.
TRAY_TOP_Z = 0.0186
TRAY_INIT_POS = np.array([0.30, 0.40, 0.001])
GOAL_FLOOR_POS = np.array([0.55, -0.35, 0.0])
GOAL_CONTACT_POS = GOAL_FLOOR_POS + np.array([0.0, 0.0, TRAY_TOP_Z])

# 런타임에 link_6과 두 흡착 접촉점의 높이 차이로 자동 계산한다.
DEFAULT_EE_OFFSET = np.array([0.0, 0.0, 0.20])

TOOL_YAW_DEG = 90.0

def downward_tool_orientation(yaw_deg: float) -> np.ndarray:
    half = np.deg2rad(yaw_deg) * 0.5

    return np.array(
        [
            0.0,
            np.cos(half),
            np.sin(half),
            0.0,
        ],
        dtype=np.float64,
    )


TOOL_ORIENTATION = downward_tool_orientation(TOOL_YAW_DEG)

EVENTS_DT = [
    0.008,   # 0 접근
    0.005,   # 1 하강
    0.02,    # 2 두 gripper close
    0.15,    # 3 흡착 안정화
    0.0025,  # 4 상승
    0.01,    # 5 place 이동
    0.0025,  # 6 하강
    1.0,     # 7 두 gripper open
    0.008,   # 8 상승
    0.08,    # 9 복귀
]


def find_prim_path_by_name(root_path: str, name: str):
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None

    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def get_world_position(prim_path: str) -> np.ndarray:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim을 찾지 못했습니다: {prim_path}")

    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translation = matrix.ExtractTranslation()
    return np.array(translation, dtype=np.float64)


def calculate_ee_offset(robot) -> np.ndarray:
    """현재 자세에서 link_6과 두 흡착점 평균 높이 차이를 계산한다."""
    ee_pos, _ = robot.end_effector.get_world_pose()
    contact_positions = [
        get_world_position(path) for path in SUCTION_CONTACT_PATHS
    ]
    contact_mid = np.mean(contact_positions, axis=0)

    offset_z = float(ee_pos[2] - contact_mid[2])
    if not 0.03 <= abs(offset_z) <= 0.50:
        print(
            "[경고] 자동 계산된 EE offset이 비정상적입니다:",
            offset_z,
            "-> 기본값 사용",
        )
        return DEFAULT_EE_OFFSET.copy()

    # PickPlaceController의 offset은 월드 Z 위쪽 거리로 사용한다.
    result = np.array([0.0, 0.0, abs(offset_z)])
    print(f"[OK] 자동 EE_OFFSET = {result}")
    print(f"     link_6={ee_pos}, suction_mid={contact_mid}")
    return result


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
    robot.gripper.open()


class M0609DualTrayTask(BaseTask):
    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._task_achieved = False

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._load_robot_usd()
        self._discover_links()
        self._setup_physics()
        self._register_robot(scene)
        self._create_scene(scene)
        print("\n[완료] 듀얼 흡착 + 트레이 씬 구성 성공!\n")

    def _load_robot_usd(self):
        print("\n" + "=" * 60)
        print("[1.LOAD] 듀얼 흡착 로봇 USD 로드")
        print("=" * 60)

        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath("/World").IsValid():
            UsdGeom.Xform.Define(stage, "/World")

        stage.GetPrimAtPath("/World").GetReferences().AddReference(
            ROBOT_USD_PATH
        )
        for _ in range(15):
            simulation_app.update()

        print(f"  [OK] {ROBOT_USD_PATH}")

    def _discover_links(self):
        print("\n" + "=" * 60)
        print("[2.DISCOVER] link_6/듀얼 SurfaceGripper 확인")
        print("=" * 60)

        self._ee_path = find_prim_path_by_name(
            ROBOT_PRIM_PATH, EE_LINK_NAME
        )
        if self._ee_path is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found")

        stage = omni.usd.get_context().get_stage()
        for path in SURFACE_GRIPPER_PATHS:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"SurfaceGripper가 없습니다: {path}")
            print(f"  SurfaceGripper = {path}")

        print(f"  EE = {self._ee_path}")

    def _setup_physics(self):
        print("\n" + "=" * 60)
        print("[3.PHYSICS] 로봇 drive 설정")
        print("=" * 60)

        stage = omni.usd.get_context().get_stage()
        drive_count = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM_PATH)):
            for drive_type in ("angular", "linear"):
                drive = UsdPhysics.DriveAPI.Get(prim, drive_type)
                if drive:
                    drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                    drive.GetDampingAttr().Set(DRIVE_DAMPING)
                    drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                    drive_count += 1

        print(f"  [OK] drive updated: {drive_count}")

    def _register_robot(self, scene):
        print("\n" + "=" * 60)
        print("[4.REGISTER] 로봇/듀얼 흡착 그리퍼 등록")
        print("=" * 60)

        gripper = DualSurfaceGripperAdapter(
            end_effector_prim_path=self._ee_path,
            surface_gripper_prim_paths=SURFACE_GRIPPER_PATHS,
            write_status_to_usd=True,
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
        print("\n" + "=" * 60)
        print("[5.SCENE] 동적 트레이 생성")
        print("=" * 60)

        add_reference_to_stage(
            usd_path=TRAY_USD_PATH,
            prim_path=TRAY_REFERENCE_PRIM_PATH,
        )
        for _ in range(5):
            simulation_app.update()

        self._tray = scene.add(
            SingleRigidPrim(
                prim_path=TRAY_RIGID_PRIM_PATH,
                name="target_tray",
                position=TRAY_INIT_POS,
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )

        scene.add(
            VisualCuboid(
                prim_path="/World/goal_marker",
                name="goal_marker",
                position=GOAL_FLOOR_POS,
                scale=np.array([0.20, 0.16, 0.001]),
                color=np.array([0.0, 1.0, 0.0]),
            )
        )

        print(f"  [OK] tray asset = {TRAY_USD_PATH}")
        print(f"  [OK] tray rigid prim = {TRAY_RIGID_PRIM_PATH}")
        print(f"  [OK] tray init = {TRAY_INIT_POS}")
        print(f"  [OK] goal floor = {GOAL_FLOOR_POS}")

    def get_observations(self):
        tray_pos, _ = self._tray.get_world_pose()
        pick_contact_pos = tray_pos + np.array([0.0, 0.0, TRAY_TOP_Z])

        return {
            self._robot.name: {
                "joint_positions": self._robot.get_joint_positions(),
            },
            self._tray.name: {
                "position": tray_pos,
                "picking_position": pick_contact_pos,
                "goal_position": GOAL_FLOOR_POS,
            },
        }

    def pre_step(self, control_index, simulation_time):
        tray_pos, _ = self._tray.get_world_pose()
        xy_error = np.mean(np.abs(GOAL_FLOOR_POS[:2] - tray_pos[:2]))
        if not self._task_achieved and xy_error < 0.03:
            print("[완료 판정] 트레이가 목표 XY에 도착했습니다.")
            self._task_achieved = True

    def post_reset(self):
        if hasattr(self, "_robot"):
            self._robot.gripper.post_reset()
        self._task_achieved = False


def main():
    my_world = World(stage_units_in_meters=1.0)

    task = M0609DualTrayTask(name="m0609_dual_tray_task")
    my_world.add_task(task)
    my_world.reset()

    robot = my_world.scene.get_object("m0609_robot")
    initialize_robot(robot, my_world)

    for _ in range(30):
        my_world.step(render=True)

    ee_offset = calculate_ee_offset(robot)

    controller = PickPlaceController(
        name="m0609_dual_tray_pick_place_controller",
        gripper=robot.gripper,
        robot_articulation=robot,
        end_effector_initial_height=0.35,
        events_dt=EVENTS_DT,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
    )

    print("\n[듀얼 흡착 트레이 Pick & Place 시작]\n")

    was_playing = False
    task_done = False
    previous_event = None

    while simulation_app.is_running():
        my_world.step(render=True)
        time.sleep(0.01)

        is_playing = my_world.is_playing()

        if is_playing and not was_playing:
            my_world.reset()
            initialize_robot(robot, my_world)
            controller.reset()
            robot.gripper.open()
            ee_offset = calculate_ee_offset(robot)
            task_done = False
            previous_event = None

        if is_playing and not task_done:
            obs = task.get_observations()
            tray_obs = obs["target_tray"]
            current_joints = obs["m0609_robot"]["joint_positions"]
            event = controller.get_current_event()

            current_ee_offset = ee_offset.copy()

            if event in (1, 2, 3):
                current_ee_offset[2] -= 0.042

            if event in (6, 7):
                current_ee_offset[2] -= 0.037

            actions = controller.forward(
                picking_position=tray_obs["picking_position"],
                placing_position=GOAL_CONTACT_POS,
                current_joint_positions=current_joints,
                end_effector_offset=current_ee_offset,
                end_effector_orientation=TOOL_ORIENTATION,
            )
            robot.apply_action(actions)

            if event != previous_event:
                try:
                    print(
                        f"[event {previous_event} -> {event}] "
                        f"status={robot.gripper.get_status()}, "
                        f"objects={robot.gripper.get_gripped_objects()}"
                    )
                except Exception as exc:
                    print("[듀얼 그리퍼 상태 조회 실패]", repr(exc))
                previous_event = event

            if controller.is_done():
                print("[완료] 듀얼 흡착 트레이 Pick & Place 완료")
                print("[최종 상태]", robot.gripper.get_status())
                print("[흡착 물체]", robot.gripper.get_gripped_objects())
                task_done = True
                my_world.pause()

            tray_pos = tray_obs["position"]
            ee_pos, _ = robot.end_effector.get_world_pose()
            print(
                f"[event={event}] "
                f"tray=({tray_pos[0]:.3f}, {tray_pos[1]:.3f}, {tray_pos[2]:.3f}) "
                f"ee_z={ee_pos[2]:.3f}"
            )

        was_playing = is_playing

    simulation_app.close()


if __name__ == "__main__":
    main()
