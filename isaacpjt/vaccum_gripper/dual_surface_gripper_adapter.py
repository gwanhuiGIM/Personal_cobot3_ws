from __future__ import annotations

from typing import Iterable

import numpy as np
from isaacsim.core.utils.types import ArticulationAction

from surface_gripper_adapter import SurfaceGripperAdapter


class DualSurfaceGripperAdapter:
    """두 개의 Isaac SurfaceGripper를 하나의 PickPlace gripper처럼 제어한다."""

    def __init__(
        self,
        end_effector_prim_path: str,
        surface_gripper_prim_paths: Iterable[str],
        *,
        write_status_to_usd: bool = True,
    ) -> None:
        paths = list(surface_gripper_prim_paths)
        if len(paths) != 2:
            raise ValueError(f"SurfaceGripper 경로는 정확히 2개여야 합니다: {paths}")

        self.end_effector_prim_path = end_effector_prim_path
        self.surface_gripper_prim_paths = paths
        self._grippers = [
            SurfaceGripperAdapter(
                end_effector_prim_path=end_effector_prim_path,
                surface_gripper_prim_path=path,
                write_status_to_usd=write_status_to_usd,
            )
            for path in paths
        ]
        self._initialized = False

        # PickPlaceController/SingleManipulator 호환용
        self.joint_opened_positions = np.empty(0, dtype=np.float64)
        self.joint_closed_positions = np.empty(0, dtype=np.float64)
        self.joint_dof_indicies = np.empty(0, dtype=np.int64)

    def initialize(self, **kwargs) -> None:
        for gripper in self._grippers:
            gripper.initialize(**kwargs)
        self._initialized = True
        print("[DualSurfaceGripper] initialized:", self.surface_gripper_prim_paths)

    def forward(self, action: str) -> ArticulationAction:
        action_name = str(action).strip().lower()
        if action_name == "close":
            self.close()
        elif action_name == "open":
            self.open()
        else:
            raise ValueError(f"지원하지 않는 gripper action: {action!r}")
        return ArticulationAction()

    def close(self) -> bool:
        results = [gripper.close() for gripper in self._grippers]
        print("[DualSurfaceGripper] CLOSE:", results)
        return all(results)

    def open(self) -> bool:
        results = [gripper.open() for gripper in self._grippers]
        print("[DualSurfaceGripper] OPEN:", results)
        return all(results)

    def get_status(self) -> list[str]:
        return [gripper.get_status() for gripper in self._grippers]

    def get_gripped_objects(self) -> list[list[str]]:
        return [gripper.get_gripped_objects() for gripper in self._grippers]

    def set_joint_positions(self, positions) -> None:
        return None

    def get_joint_positions(self) -> np.ndarray:
        return np.empty(0, dtype=np.float64)

    def post_reset(self) -> None:
        if self._initialized:
            self.open()

    def reset(self) -> None:
        self.post_reset()
