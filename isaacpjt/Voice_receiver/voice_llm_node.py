# whisper_voice_node.py
#
# [변경 요약]
#
# 1. 토픽명 및 메시지 타입 변경 (m0609 규격):
#      /tool/request     Int32  → /m0609/pick_command   String  (도구명 문자열)
#      /tool/cancel      Int32  → /m0609/return_tool    String  (도구명 문자열)
#      /tool/cancel_last Empty  → /m0609/return_recent  Empty   (유지)
#
# 2. 퍼블리시 페이로드 변경:
#      도구 요청·특정 반납 모두 tool_id(Int32) 대신 도구명 String으로 발행
#
# 3. Gemini 프롬프트 용어 변경:
#      "취소" → "반납" (도구를 돌려보내는 동작으로 의미 통일)
#
# 4. 대화 문맥 추적 (last_requested_tool) 유지:
#      직전 요청 도구명을 Gemini 프롬프트에 주입해 "그거 아니야" 등 처리
#
# 5. 두 반납 토픽 동시 발행 금지 - 분기 구조로 보장

import time
import threading
import whisper
import torch
import sounddevice as sd
import numpy as np
import google.generativeai as genai
from pynput import keyboard

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String, Empty

# ── 도구 테이블 (규격 §3) ─────────────────────
TOOL_TABLE = {
    "메스":      0,
    "가위":      1,
    "톱":      2,
    "캘리퍼":    3,
    "망치":      4,
    "니들홀더":  5,
}
TOOL_NAME_BY_ID = {v: k for k, v in TOOL_TABLE.items()}

# ── Gemini 설정 ───────────────────────────────
genai.configure(api_key="")  # API 키 입력

gemini_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={
        "temperature": 0,
        "max_output_tokens": 20,
    },
    system_instruction="""
너는 수술실에서 집도의의 음성 지시를 해석하는 AI 어시스턴트야.

[배경]
- 집도의가 수술 중 필요한 도구를 음성으로 요청하면, 협동로봇이
  해당 도구를 트레이에서 집어 전달한다.
- 음성은 한국어 STT를 거쳐 들어오므로 발음 오인식, 줄임말,
  영어/한국어 혼용, 반말/존댓말이 섞일 수 있다.
- 너는 의미를 추론해서 아래 6가지 도구 중 하나로 분류하거나,
  취소 의도를 판별한다.

[도구 후보 6종 및 동의어/유사 발음]
0. 메스      ← scalpel, knife, 스칼펠, 메쓰, 칼, 절개도,
1. 가위      ← scissors, 시저스, 시져스
2. 톱      ← handsaw, saw, 쏘우, 톱, 
3. 캘리퍼    ← caliper, ruler, measure, 캘리퍼, 자, 측정기
4. 망치      ← mallet, hammer, 망치, 해머
5. 니들홀더  ← needle holder, 바늘잡이, 봉합기

[반납 의도 분류]
- 반납 대상 도구가 명확하게 언급된 경우
  → CANCEL:<tool_id>  (숫자만, 예: CANCEL:1)
  예) "가위 반납해" → CANCEL:1
      "가위 요청 취소해" → CANCEL:1
      "메스 아니야" → CANCEL:0

- 반납 대상 도구가 언급되지 않았지만 직전 요청을 부정·정정하는 경우
  → CANCEL_LAST
  예) "그거 아니야", "아니 그거 말고", "잘못 들었어",
      "방금 거 반납해", "방금 거 취소해", "내가 말한 거 아니야",
      "됐어 하지 마", "취소해", "아니야"

[출력 규칙]
- 도구 요청    → 도구명 한국어 한 단어 (예: "메스", "가위", "포셉")
- 특정 반납    → CANCEL:<숫자>          (예: CANCEL:1)
- 직전 부정    → CANCEL_LAST
- 판단 불가    → NONE
- 출력은 반드시 위 형식 중 하나. 설명·문장·따옴표·구두점 금지.

[직전 요청 문맥]
프롬프트 마지막에 "직전요청:<tool_id>" 형식으로 제공된다.
이 정보를 참고해 "그거", `"방금 거" 같은 지시어를 해석하라.
직전 요청이 없으면 "없음"으로 표시된다.
"""
)

# ── ROS2 노드 ─────────────────────────────────
class VoicePublisherNode(Node):
    def __init__(self):
        super().__init__("surgical_voice_node")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.pub_pick_command   = self.create_publisher(String, "/m0609/pick_command",  qos)
        self.pub_return_tool    = self.create_publisher(String, "/m0609/return_tool",   qos)
        self.pub_return_recent  = self.create_publisher(Empty,  "/m0609/return_recent", qos)

    def publish_pick_command(self, tool_name: str):
        msg = String()
        msg.data = tool_name
        self.pub_pick_command.publish(msg)
        self.get_logger().info(f"[PUBLISH] /m0609/pick_command  '{tool_name}'")

    def publish_return_tool(self, tool_name: str):
        msg = String()
        msg.data = tool_name
        self.pub_return_tool.publish(msg)
        self.get_logger().info(f"[PUBLISH] /m0609/return_tool   '{tool_name}'")

    def publish_return_recent(self):
        self.pub_return_recent.publish(Empty())
        self.get_logger().info("[PUBLISH] /m0609/return_recent")


# ── 전역 상태 ─────────────────────────────────
is_recording      = False
audio_frames      = []
SAMPLE_RATE       = 16000
last_requested_tool_id: int | None = None  # 대화 문맥 추적용

ros_node: VoicePublisherNode | None = None

# ── Whisper 로드 ──────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Whisper 로딩 중... (device: {device})")
stt_model = whisper.load_model("base", device=device)
print("완료!\n")


def audio_callback(indata, frames, time_info, status):
    if is_recording:
        audio_frames.append(indata.copy())


def classify_with_gemini(stt_text: str) -> str:
    """
    Gemini로 발화를 분류한다.
    반환값: 도구명 / CANCEL:<id> / CANCEL_LAST / NONE
    """
    context = (
        f"없음" if last_requested_tool_id is None
        else f"{last_requested_tool_id}({TOOL_NAME_BY_ID[last_requested_tool_id]})"
    )
    prompt = f"{stt_text}\n직전요청:{context}"

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini 오류: {e}")
        return "NONE"


def handle_gemini_result(raw: str):
    """
    Gemini 출력을 파싱해 ROS2 토픽을 발행한다.
    두 반납 토픽을 동시에 발행하지 않도록 분기로 보장.
    """
    global last_requested_tool_id

    # ── 케이스 1: 특정 도구 반납 ──────────────
    if raw.startswith("CANCEL:"):
        try:
            tool_id = int(raw.split(":")[1])
        except (IndexError, ValueError):
            print(f"[WARN] CANCEL 파싱 실패: '{raw}' → 무시")
            return

        if tool_id not in TOOL_NAME_BY_ID:
            print(f"[WARN] 유효하지 않은 tool_id={tool_id} → 무시")
            return

        tool_name = TOOL_NAME_BY_ID[tool_id]
        print(f"[분류] 특정 도구 반납 → '{tool_name}'")
        ros_node.publish_return_tool(tool_name)
        return

    # ── 케이스 2: 최근 작업 반납 ──────────────
    if raw == "CANCEL_LAST":
        print("[분류] 최근 작업 반납 (return_recent)")
        ros_node.publish_return_recent()
        return

    # ── 케이스 3: 도구 요청 ───────────────────
    if raw in TOOL_TABLE:
        tool_id = TOOL_TABLE[raw]
        print(f"[분류] 도구 요청 → '{raw}'")
        ros_node.publish_pick_command(raw)
        last_requested_tool_id = tool_id  # 문맥 갱신
        return

    # ── 케이스 4: 판단 불가 ───────────────────
    print(f"[분류] NONE 또는 미인식 출력 '{raw}' → 발행 없음")


def process_audio():
    if not audio_frames:
        print("녹음 데이터 없음\n")
        return

    # 오디오 합치기
    audio = np.concatenate(audio_frames, axis=0).flatten()

    # ── STT ──────────────────────────────────
    t0 = time.perf_counter()
    result = stt_model.transcribe(
        audio,
        language="ko",
        initial_prompt=(
            "수술 도구를 요청하거나 취소하는 음성입니다. "
            "메스, 가위, 포셉, 클램프, 석션, 니들홀더, "
            "스칼펠, suction, forceps, clamp, needle holder."
        ),
    )
    stt_ms = (time.perf_counter() - t0) * 1000
    text = result["text"].strip()
    print(f"[STT  {stt_ms:6.0f}ms] → '{text}'")

    if not text:
        print("인식 실패\n")
        return

    # ── Gemini 분류 ───────────────────────────
    print("분류 중...")
    t1 = time.perf_counter()
    raw = classify_with_gemini(text)
    gem_ms = (time.perf_counter() - t1) * 1000
    print(f"[Gemini {gem_ms:6.0f}ms] → '{raw}'")

    # ── 타이밍 요약 ───────────────────────────
    print("-" * 40)
    print(f"총 소요: {stt_ms + gem_ms:.0f}ms  (STT {stt_ms:.0f} + Gemini {gem_ms:.0f})")

    # ── ROS2 퍼블리시 (최종 결과 한 번만) ─────
    handle_gemini_result(raw)

    print("-" * 40)
    print("\n[ 스페이스 눌러서 다시 녹음 ]")


def on_press(key):
    global is_recording, audio_frames
    if key == keyboard.Key.space and not is_recording:
        is_recording = True
        audio_frames = []
        print("\n[🎤 듣고있습니다... 스페이스 떼면 종료]")


def on_release(key):
    global is_recording
    if key == keyboard.Key.space and is_recording:
        is_recording = False
        print("[녹음 완료]")
        # ROS spin과 충돌 방지: 별도 스레드에서 처리
        threading.Thread(target=process_audio, daemon=True).start()

    if hasattr(key, "char") and key.char == "q":
        print("수고하셨습니다")
        return False


# ── 메인 ──────────────────────────────────────
def main():
    global ros_node

    rclpy.init()
    ros_node = VoicePublisherNode()

    # ROS2 spin을 별도 스레드로 실행 (pynput 블로킹과 분리)
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(ros_node,), daemon=True
    )
    spin_thread.start()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    )
    stream.start()

    print("=" * 50)
    print("수술 도구 음성 인식 노드")
    print("스페이스 누르는 동안 녹음 / 떼면 인식")
    print("도구 6종: 메스·가위·포셉·클램프·석션·니들홀더")
    print("반납 발화: '가위 반납해' / '그거 아니야' / '방금 거 취소해'")
    print("q → 종료")
    print("=" * 50)
    print("\n[ 스페이스 눌러서 녹음 시작 ]")

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    stream.stop()
    stream.close()
    ros_node.destroy_node()
    rclpy.shutdown()
    print("종료")


if __name__ == "__main__":
    main()