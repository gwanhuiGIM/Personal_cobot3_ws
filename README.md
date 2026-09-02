# cobot3_ws

⚠️ 개인 개발 진행 중 워크스페이스입니다. 정리되지 않은 코드, 미완성 노드, 빈 스켈레톤 패키지가 섞여 있으며 **패키지 간 연동 플로우가 아직 없습니다.** 아래 내용은 코드 존재 여부만 확인한 것이며 빌드/실행 검증은 하지 않았습니다.

## 구조 (`src/`)

| 패키지 | 상태 | 비고 |
|---|---|---|
| `cobot3` | 스켈레톤 | `ros2 pkg create` 기본 talker/listener 예제만 있음. 실제 로직 없음 |
| `M0609` | 스켈레톤 | entry_points 비어있음, 노드 파일 없음 |
| `nova_carter/commander` | 부분 구현 | `nav_to_poses.py` — Nav2 `BasicNavigator` 기반 웨이포인트 이동 스크립트 |
| `rokeyf2_cobot3` | ROS 패키지 아님 | Isaac Sim standalone 스크립트 모음 (`standalone_basic.py`, `standalone_mission1~6.py`). ROS2 노드가 아니라 Isaac Sim 시뮬레이터를 직접 구동하는 독립 실행 파일 |
| `yolo_ros` | 서드파티 | 외부 vendored 패키지 (`.github/` 포함) — 직접 작성한 코드 아님 |

워크스페이스 루트에는 별도 YOLO 실험 디렉토리(`0622_v2_yolov8/`, `0624_YOLOv1_pt/`, `0625_seg_pt/`)와 `isaacpjt/`, `isaac_ros_manipultation_config/`가 있으나 어느 것도 `src/` 패키지와 연결되어 있지 않습니다.

## 알려진 공백
- `cobot3`, `M0609`는 아직 목적이 정해지지 않은 빈 패키지입니다.
- `rokeyf2_cobot3`의 standalone 스크립트들과 `nova_carter`, `yolo_ros`를 잇는 실행 플로우(누가 무엇을 호출하는지)가 없습니다.
- 루트의 `docs/`, `isaacpjt/`, `src/nova_carter/`, `src/rokeyf2_cobot3/`, `src/yolo_ros/`는 git 추적 여부가 아직 결정되지 않았습니다 (`docs/state.md` 참고).

## 빌드
```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```
⚠️ 미검증 — 이 세션에서 실행하지 않았습니다.
