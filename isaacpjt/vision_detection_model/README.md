# vision_detection_model

Isaac Sim의 RGB 카메라 영상을 입력받아 YOLO로 수술 도구를 인식하고, 각 도구가 위치한 트레이 번호와 빈 트레이 상태를 ROS2 토픽으로 발행하는 비전 인식 모듈입니다.

## 1. 주요 기능

```text
Isaac Sim RGB 이미지 수신
→ YOLO 기반 수술 도구 6종 검출
→ Bounding Box 중심점 계산
→ 트레이 ROI 매핑
→ 최근 프레임 기반 검출 안정화
→ 빈 트레이 포함 6개 트레이 상태 생성
→ ROS2 토픽 지속 발행
```

인식 대상 클래스:

| Class ID | Class name |
|---:|---|
| 0 | caliper |
| 1 | clamps |
| 2 | handsaws |
| 3 | knife |
| 4 | ligature_needle |
| 5 | mallet |

---

## 2. 트레이 번호

카메라 화면 기준 트레이 번호는 다음과 같습니다.

```text
┌─────────┬─────────┬─────────┐
│ Tray 5  │ Tray 3  │ Tray 1  │
├─────────┼─────────┼─────────┤
│ Tray 4  │ Tray 2  │ Tray 0  │
└─────────┴─────────┴─────────┘
```

```text
Tray 0: 오른쪽 아래
Tray 1: 오른쪽 위
Tray 2: 가운데 아래
Tray 3: 가운데 위
Tray 4: 왼쪽 아래
Tray 5: 왼쪽 위
```

---

## 3. 저장소 구조

```text
vision_detection_model/
├── README.md
├── vision_tool_detection_node.py
├── tray_roi_calibrator.py
├── tray_rois.json
└── weights/
    └── best.pt
```

### `vision_tool_detection_node.py`

메인 ROS2 비전 노드입니다.

- `/rgb` 이미지 구독
- YOLO 추론
- 트레이 ROI 매핑
- 최근 프레임 기반 검출 안정화
- `/yolo/tracking_image` 발행
- `/m0609/tool_detection` 발행

### `tray_roi_calibrator.py`

RGB 영상에서 트레이 네 모서리를 클릭하여 6개 트레이 ROI를 설정합니다.

### `tray_rois.json`

트레이별 픽셀 좌표가 저장된 설정 파일입니다.

카메라 위치, 각도 또는 이미지 해상도가 바뀌면 다시 생성해야 합니다.

### `weights/best.pt`

3,000장의 Isaac Sim 합성 데이터로 학습한 최종 YOLO 가중치입니다.

---

## 4. 실행 환경

기준 환경:

```text
Ubuntu 22.04
ROS2 Humble
Python 3.10
Isaac Sim 5.1
CUDA 지원 NVIDIA GPU
```

필요한 Python 패키지:

```bash
python3 -m pip install ultralytics opencv-python numpy
```

ROS 패키지:

```bash
sudo apt update
sudo apt install -y \
  ros-humble-cv-bridge \
  ros-humble-rqt \
  ros-humble-rqt-image-view
```

---

## 5. 저장소 Clone

원하는 작업 디렉터리로 이동합니다.

```bash
cd ~
```

저장소를 clone합니다.

```bash
git clone https://github.com/ROKEY-Project-F2/vision_detection_model.git
```

프로젝트 폴더로 이동합니다.

```bash
cd vision_detection_model
```

파일 구조를 확인합니다.

```bash
find . -maxdepth 2 -type f | sort
```

다음 파일이 있어야 합니다.

```text
./README.md
./tray_roi_calibrator.py
./tray_rois.json
./vision_tool_detection_node.py
./weights/best.pt
```

---

## 6. ROS2 환경 설정

각 터미널에서 다음 환경을 설정합니다.

```bash
source /opt/ros/humble/setup.bash

export ROS_DOMAIN_ID=140
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
```

환경값 확인:

```bash
echo $ROS_DOMAIN_ID
echo $RMW_IMPLEMENTATION
echo $ROS_LOCALHOST_ONLY
```

예상 출력:

```text
140
rmw_fastrtps_cpp
0
```

환경 변수를 매번 입력하지 않으려면 `~/.bashrc`에 추가할 수 있습니다.

```bash
nano ~/.bashrc
```

파일 마지막에 다음 내용을 추가합니다.

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=140
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
```

적용:

```bash
source ~/.bashrc
```

---

## 7. Isaac Sim 실행

Isaac Sim과 ROS2가 같은 환경 변수를 사용해야 합니다.

```bash
source /opt/ros/humble/setup.bash

export ROS_DOMAIN_ID=140
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
```

Isaac Sim 실행 예시:

```bash
/home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/isaac-sim.sh \
--enable isaacsim.ros2.bridge \
/home/rokey/surgical_tool_dataset/datasetscene.usda
```

> Isaac Sim 설치 경로와 USD 경로는 사용자 환경에 맞게 수정해야 합니다.

Isaac Sim이 실행되면 반드시 상단의 **Play** 버튼을 누릅니다.

RGB 토픽 확인:

```bash
ros2 topic list | grep rgb
```

전송 주기 확인:

```bash
ros2 topic hz /rgb
```

입력 토픽:

```text
/rgb
sensor_msgs/msg/Image
```

---

## 8. 트레이 ROI 설정

기존 `tray_rois.json`이 현재 카메라 환경과 맞으면 이 단계는 생략할 수 있습니다.

카메라 위치, 카메라 각도 또는 이미지 해상도가 바뀌었다면 ROI를 다시 설정합니다.

```bash
cd ~/vision_detection_model

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=140
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0

python3 tray_roi_calibrator.py
```

클릭 순서:

```text
0: 오른쪽 아래
1: 오른쪽 위
2: 가운데 아래
3: 가운데 위
4: 왼쪽 아래
5: 왼쪽 위
```

각 트레이의 모서리 네 개를 시계 방향으로 클릭합니다.

키 입력:

```text
r: 이전 트레이 다시 지정
s: 저장
q: 종료
```

저장 결과:

```text
tray_rois.json
```

---

## 9. 비전 노드 실행

```bash
cd ~/vision_detection_model

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=140
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0

python3 vision_tool_detection_node.py
```

정상 실행 시 다음 토픽이 생성됩니다.

```text
/yolo/tracking_image
/m0609/tool_detection
```

토픽 목록 확인:

```bash
ros2 topic list
```

---

## 10. 검출 화면 확인

RQT 실행:

```bash
rqt
```

RQT에서 다음 메뉴를 엽니다.

```text
Plugins
→ Visualization
→ Image View
```

토픽 선택:

```text
/yolo/tracking_image
```

영상에는 다음 정보가 표시됩니다.

```text
도구 클래스
confidence
tray_id
트레이 ROI
빈 트레이 상태
```

---

## 11. ROS2 토픽

### 입력 이미지

```text
Topic: /rgb
Type: sensor_msgs/msg/Image
```

### 검출 결과 이미지

```text
Topic: /yolo/tracking_image
Type: sensor_msgs/msg/Image
```

### 도구 및 트레이 상태

```text
Topic: /m0609/tool_detection
Type: std_msgs/msg/String
```

출력 확인:

```bash
ros2 topic echo /m0609/tool_detection
```

도구가 있는 트레이 예시:

```json
{
  "tool_id": "knife",
  "x": 0.0,
  "y": 0.0,
  "z": 0.0,
  "tray_id": 0,
  "timestamp": 1782432000.0
}
```

빈 트레이 예시:

```json
{
  "tool_id": "EMPTY",
  "x": 0.0,
  "y": 0.0,
  "z": 0.0,
  "tray_id": 2,
  "timestamp": 1782432000.0
}
```

비전 노드는 한 주기마다 6개 트레이 상태를 각각 발행합니다.

기본 발행 주기:

```text
5 Hz
```

---

## 12. 검출 안정화

한 프레임의 순간적인 오검출이나 미검출로 상태가 바로 바뀌지 않도록 최근 프레임을 사용합니다.

현재 기준:

```text
최근 5프레임 저장
3프레임 이상 동일한 결과
→ 해당 트레이 상태 확정
```

---

## 13. RobotManager 연동

비전 노드는 카메라가 관측한 상태를 전달합니다.

```text
도구가 있음
→ tool_id와 tray_id 발행

도구가 없음
→ tool_id = EMPTY 발행
```

RobotManager는 비전 결과와 로봇 내부 상태를 조합하여 최종 상태를 판단해야 합니다.

```text
Tray가 EMPTY
+
도구 상태가 HELD_BY_ROBOT
→ 로봇이 정상적으로 도구를 들고 있음

Tray가 EMPTY
+
관련 로봇 작업이 없음
→ 도구 상태 UNKNOWN

PICK_APPROACH 중 EMPTY
→ 로봇 또는 그리퍼 가림 가능성
→ 즉시 UNKNOWN으로 변경하지 않음
```

향후 `m0609_ros_bridge.py`에서 다음 토픽을 구독해야 합니다.

```text
/m0609/tool_detection
std_msgs/msg/String
```

---

## 14. 좌표값

현재 비전 단독 테스트에서는 README 메시지 형식을 유지하기 위해 다음 임시 좌표를 발행합니다.

```text
x = 0.0
y = 0.0
z = 0.0
```

실제 로봇 제어에서는 `tray_id`를 기준으로 RobotManager 또는 ToolStateManager 내부의 기존 트레이 좌표를 사용해야 합니다.

```text
비전 노드
→ tool_id + tray_id 전달

RobotManager
→ tray_id로 실제 트레이 좌표 조회

StateMachine
→ 기존 PICK/PLACE pose 사용
```

---

## 15. 모델 성능

데이터 구성:

```text
Isaac Sim 합성 이미지: 3,000장
Train: 2,400장
Validation: 300장
Test: 300장
```

검증 결과:

```text
Precision: 0.999
Recall: 1.000
mAP50: 0.995
mAP50-95: 0.984
Inference time: 약 6.4 ms/image
```

해당 수치는 Isaac Sim 합성 환경 기준입니다.

---

## 16. 테스트 시나리오

### 전체 도구 검출

```text
6개 트레이의 도구가 올바른 tool_id와 tray_id로 발행되는지 확인
```

### 빈 트레이 확인

```text
도구 제거
→ 해당 tray_id에서 tool_id = EMPTY 발행 확인
```

### 도구 이동 확인

```text
도구를 다른 트레이로 이동
→ 기존 트레이 EMPTY
→ 새 트레이에서 해당 tool_id 발행
```

### RGB 입력 확인

```bash
ros2 topic hz /rgb
```

### 검출 결과 확인

```bash
ros2 topic echo /m0609/tool_detection
```

### 검출 이미지 확인

```bash
rqt
```

---

## 17. 문제 해결

### `/rgb`가 보이지만 이미지가 들어오지 않는 경우

```bash
ros2 topic hz /rgb
```

아무 출력이 없다면 다음을 확인합니다.

```text
Isaac Sim Play 상태
ROS2 Camera Action Graph 연결
ROS2 Bridge 활성화
ROS_DOMAIN_ID 일치
RMW_IMPLEMENTATION 일치
```

### `/yolo/tracking_image`가 나오지 않는 경우

```bash
ros2 topic hz /rgb
```

먼저 `/rgb` 입력이 실제로 들어오는지 확인합니다.

### 모델 파일 오류

```text
Model not found
```

다음 파일이 있는지 확인합니다.

```bash
ls -lh weights/best.pt
```

### ROI 파일 오류

```text
ROI file not found
```

다음 파일을 확인합니다.

```bash
ls -lh tray_rois.json
```

필요하면 ROI를 다시 생성합니다.

```bash
python3 tray_roi_calibrator.py
```

---

## 18. 현재 구현 범위

구현 완료:

```text
YOLO 기반 수술 도구 6종 검출
6개 트레이 ROI 설정
도구와 트레이 매핑
최근 프레임 기반 상태 안정화
빈 트레이 상태 생성
검출 이미지 ROS2 발행
6개 트레이 상태 ROS2 발행
```

추후 RobotManager 측 구현:

```text
/m0609/tool_detection Subscriber
JSON 파싱
ToolStateManager 갱신
EMPTY와 HELD_BY_ROBOT 구분
비정상 미검출 시 UNKNOWN 처리
tray_id 기반 실제 PICK 좌표 조회
```
