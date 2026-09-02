from __future__ import annotations

from typing import Optional

import numpy as np
import omni.usd

from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.surface_gripper import _surface_gripper


class SurfaceGripperAdapter:
    """Isaac Sim 5.1 Surface Gripper를 PickPlaceController gripper 인터페이스에 맞춘 어댑터.

    PickPlaceController는 event 2에서 ``gripper.forward("close")``,
    event 7에서 ``gripper.forward("open")``을 호출한다. 이 어댑터는
    해당 호출을 SurfaceGripperInterface로 전달하고, 로봇 관절에는
    아무 명령도 주지 않는 빈 ArticulationAction을 반환한다.
    """

    def __init__(
        self,
        end_effector_prim_path: str,
        surface_gripper_prim_path: str,
        *,
        write_status_to_usd: bool = True,
    ) -> None:
        self.end_effector_prim_path = end_effector_prim_path
        self.surface_gripper_prim_path = surface_gripper_prim_path
        self._write_status_to_usd = write_status_to_usd
        self._interface = None
        self._initialized = False

        # 기존 ParallelGripper 사용부와 호환하기 위한 빈 배열
        self.joint_opened_positions = np.empty(0, dtype=np.float64)
        self.joint_closed_positions = np.empty(0, dtype=np.float64)
        self.joint_dof_indicies = np.empty(0, dtype=np.int64)

    def initialize(
        self,
        physics_sim_view=None,
        articulation_apply_action_func=None,
        get_joint_positions_func=None,
        set_joint_positions_func=None,
        dof_names=None,
        **kwargs,
    ) -> None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.surface_gripper_prim_path)
        if not prim.IsValid():
            raise RuntimeError(
                f"SurfaceGripper prim을 찾지 못했습니다: "
                f"{self.surface_gripper_prim_path}"
            )

        self._interface = (
            _surface_gripper.acquire_surface_gripper_interface()
        )
        if self._interface is None:
            raise RuntimeError("SurfaceGripperInterface 획득 실패")

        self._interface.set_write_to_usd(self._write_status_to_usd)
        self._initialized = True

        print(
            "[SurfaceGripper] initialized:",
            self.surface_gripper_prim_path,
            f"(type={prim.GetTypeName()})",
        )

    def _require_initialized(self) -> None:
        if not self._initialized or self._interface is None:
            raise RuntimeError(
                "SurfaceGripperAdapter.initialize()가 먼저 호출되어야 합니다."
            )

    def forward(self, action: str) -> ArticulationAction:
        """PickPlaceController가 호출하는 open/close 진입점."""
        action_name = str(action).strip().lower()

        if action_name == "close":
            self.close()
        elif action_name == "open":
            self.open()
        else:
            raise ValueError(
                f"지원하지 않는 gripper action: {action!r}. "
                "'open' 또는 'close'만 사용할 수 있습니다."
            )

        # 흡착 명령은 SurfaceGripperInterface가 처리한다.
        # 로봇 articulation에 추가로 적용할 finger joint 명령은 없다.
        return ArticulationAction()

    def close(self) -> bool:
        self._require_initialized()
        result = bool(
            self._interface.close_gripper(self.surface_gripper_prim_path)
        )
        print(
            f"[SurfaceGripper] CLOSE -> {result}, "
            f"status={self.get_status()}, "
            f"objects={self.get_gripped_objects()}"
        )
        return result

    def open(self) -> bool:
        self._require_initialized()
        result = bool(
            self._interface.open_gripper(self.surface_gripper_prim_path)
        )
        print(
            f"[SurfaceGripper] OPEN -> {result}, "
            f"status={self.get_status()}"
        )
        return result

    def get_status(self) -> str:
        self._require_initialized()
        status = self._interface.get_gripper_status(
            self.surface_gripper_prim_path
        )
        # pybind enum은 str()이 가장 버전 호환성이 좋다.
        return str(status)

    def get_gripped_objects(self) -> list[str]:
        self._require_initialized()
        return list(
            self._interface.get_gripped_objects(
                self.surface_gripper_prim_path
            )
        )

    def set_joint_positions(self, positions) -> None:
        """ParallelGripper 호환용 no-op."""
        return None

    def get_joint_positions(self) -> np.ndarray:
        return np.empty(0, dtype=np.float64)

    def post_reset(self) -> None:
        if self._initialized:
            self.open()

    def reset(self) -> None:
        self.post_reset()
