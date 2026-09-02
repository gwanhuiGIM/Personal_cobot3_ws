# whisper_voice_node

수술실 협동로봇(M0609)을 위한 음성 인식 ROS2 노드.  
집도의의 음성을 실시간으로 인식해 도구 요청·반납 명령을 ROS2 토픽으로 발행한다.

---

## 개요

```
마이크 입력
  │
  ▼
Whisper (STT, 한국어)
  │  텍스트
  ▼
Gemini 2.5 Flash (의도 분류)
  │  도구 요청 / 특정 반납 / 최근 반납 / 미인식
  ▼
ROS2 Publisher
  ├─ /m0609/pick_command   (String)  ← 도구 요청
  ├─ /m0609/return_tool    (String)  ← 특정 도구 반납
  └─ /m0609/return_recent  (Empty)   ← 최근 작업 반납
```

- **STT**: OpenAI Whisper `base` 모델, 수술 도구 용어 힌트 프롬프트 적용
- **의도 분류**: Gemini가 도구 요청·반납·문맥 부정을 구분, 대화 문맥(직전 요청) 주입
- **ROS2**: `RELIABLE / KEEP_LAST / depth=10` QoS, 두 반납 토픽 동시 발행 금지

---

## 지원 도구

| tool_id | 도구명 | 인식 동의어 |
|:---:|---|---|
| 0 | 메스 | scalpel, 스칼펠, 메쓰, 칼, 절개도 |
| 1 | 가위 | scissors, 시저스, 시져스 |
| 2 | 포셉 | forceps, 겸자, 헤모스탯, 지혈겸자, 핀셋, 집게 |
| 3 | 클램프 | clamp, 클램프형겸자 |
| 4 | 석션 | suction, 흡인기, 빨아들이는거, 흡입기 |
| 5 | 니들홀더 | needle holder, 바늘잡이, 봉합기 |

---

## 토픽 규격

### 발행 (Publisher)

| 기능 | 토픽 | 타입 | 발행 조건 |
|---|---|---|---|
| 도구 요청 | `/m0609/pick_command` | `std_msgs/String` | 도구명 인식 시 |
| 특정 도구 반납 | `/m0609/return_tool` | `std_msgs/String` | 도구명을 명시한 반납 발화 시 |
| 최근 작업 반납 | `/m0609/return_recent` | `std_msgs/Empty` | 직전 요청을 부정·정정하는 발화 시 |

> `/m0609/return_tool`과 `/m0609/return_recent`는 동시에 발행되지 않는다.

### 페이로드 예시

```bash
# 도구 요청
/m0609/pick_command  →  data: "메스"

# 특정 도구 반납
/m0609/return_tool   →  data: "가위"

# 최근 작업 반납
/m0609/return_recent →  (Empty)
```

---

## 의도 분류 규칙

Gemini가 STT 텍스트를 아래 4가지로 분류한다.

| Gemini 출력 | 발화 예시 | 발행 토픽 |
|---|---|---|
| `메스` / `가위` / … | "메스 줘", "가위 가져와" | `/m0609/pick_command` |
| `CANCEL:0` / `CANCEL:1` / … | "가위 반납해", "메스 취소해" | `/m0609/return_tool` |
| `CANCEL_LAST` | "그거 아니야", "잘못 들었어", "방금 거 취소해" | `/m0609/return_recent` |
| `NONE` | 도구 무관 발화, 판단 불가 | 발행 없음 |

**직전 요청 문맥 추적**: 마지막으로 요청된 도구명이 Gemini 프롬프트에 자동 주입된다.  
"그거 아니야"처럼 지시어만 있는 발화도 올바르게 `CANCEL_LAST`로 분류된다.

---

## 설치

### 요구 환경

- Python 3.10+
- ROS2 Humble (Ubuntu 22.04)
- CUDA (선택, CPU도 동작)

### 패키지 설치

```bash
# ROS2 환경 소싱
source /opt/ros/humble/setup.bash

# Python 패키지
pip install openai-whisper torch sounddevice numpy pynput
pip install google-generativeai
```

---

## 실행

### 1. API 키 설정

`whisper_voice_node.py` 47번째 줄에 Gemini API 키를 입력한다.

```python
genai.configure(api_key="YOUR_GEMINI_API_KEY")  # pragma: allowlist secret
```

### 2. 노드 실행

```bash
source /opt/ros/humble/setup.bash
python3 whisper_voice_node.py
```

### 3. 조작 방법

| 입력 | 동작 |
|---|---|
| `스페이스` 누르는 동안 | 마이크 녹음 |
| `스페이스` 떼기 | 녹음 종료 → STT → 분류 → 토픽 발행 |
| `q` | 노드 종료 |

---

## 터미널 출력 예시

```
[ 스페이스 눌러서 녹음 시작 ]

[🎤 듣고있습니다... 스페이스 떼면 종료]
[녹음 완료]
[STT     312ms] → '메스 줘'
분류 중...
[Gemini  540ms] → '메스'
----------------------------------------
총 소요: 852ms  (STT 312 + Gemini 540)
[분류] 도구 요청 → '메스'
[PUBLISH] /m0609/pick_command  '메스'
----------------------------------------

[ 스페이스 눌러서 다시 녹음 ]
```

```
[🎤 듣고있습니다... 스페이스 떼면 종료]
[녹음 완료]
[STT     298ms] → '그거 아니야'
분류 중...
[Gemini  480ms] → 'CANCEL_LAST'
----------------------------------------
총 소요: 778ms  (STT 298 + Gemini 480)
[분류] 최근 작업 반납 (return_recent)
[PUBLISH] /m0609/return_recent
----------------------------------------
```

---

## 토픽 수동 확인

```bash
# 도구 요청 모니터링
ros2 topic echo /m0609/pick_command

# 반납 모니터링
ros2 topic echo /m0609/return_tool
ros2 topic echo /m0609/return_recent

# 테스트 발행 (수동)
ros2 topic pub --once /m0609/pick_command std_msgs/msg/String "{data: '메스'}"
ros2 topic pub --once /m0609/return_tool  std_msgs/msg/String "{data: '가위'}"
ros2 topic pub --once /m0609/return_recent std_msgs/msg/Empty "{}"
```

---

## 주요 설계 결정

**STT 중간 결과 발행 금지**  
스페이스를 뗀 시점, 최종 Whisper 결과에서만 한 번 발행한다. 인식 중간에 반복 발행하지 않는다.

**반납 토픽 동시 발행 금지**  
`handle_gemini_result()` 내부가 `if / elif` 분기로 구성되어 있어 구조적으로 두 반납 토픽이 동일 발화에서 동시 발행되는 경우가 없다.

**ROS spin 스레드 분리**  
`rclpy.spin()`은 별도 데몬 스레드에서 실행된다. `pynput` 키보드 리스너와 블로킹 충돌을 방지하며, `process_audio()`도 스레드로 분리해 키 입력이 지연되지 않는다.

**QoS**  
도구 요청·반납 토픽 모두 `RELIABLE / KEEP_LAST / depth=10 / VOLATILE`을 적용해 명령 유실을 방지한다.
