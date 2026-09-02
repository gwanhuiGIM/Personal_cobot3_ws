# llmtest.py
import os
import time
import whisper
import torch
import sounddevice as sd
import numpy as np
import google.generativeai as genai
from pynput import keyboard

# .env를 os.environ으로 로드 (표준 라이브러리만 사용)
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# ── Gemini 설정 ───────────────────────────────
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={
        "temperature": 0,
        "max_output_tokens": 10,
    },
    system_instruction="""
너는 수술실에서 집도의의 음성 지시를 해석하는 AI 어시스턴트야.

[배경]
- 집도의가 수술 중 필요한 도구를 음성으로 요청하면, 협동로봇이
  해당 도구를 트레이에서 집어 전달한다.
- 음성은 한국어 STT를 거쳐 들어오므로 발음 오인식, 줄임말,
  영어/한국어 혼용, 반말/존댓말이 섞일 수 있다.
- 너는 의미를 추론해서 아래 7가지 도구 중 하나로 분류한다.

[도구 후보 7종 및 동의어/유사 발음]
1. 메스    ← scalpel, 스칼펠, 메쓰, 칼, 절개도
2. 가위    ← scissors, 시저스, 시져스
3. 겸자    ← forceps, 포셉, 헤모스탯, 지혈겸자, 클램프형겸자
4. 핀셋    ← tweezers, 트위저, 집게, 포셉팁
5. 견인기  ← retractor, 리트랙터, 당기는거, 벌리는거
6. 거즈    ← gauze, 가제, 솜, 패드
7. 흡인기  ← suction, 석션, 빨아들이는거, 흡입기

[출력 규칙]
- 입력이 7종 중 하나를 가리키면 → 도구명 한국어 한 단어만
  (예: "메스", "가위", "겸자", "핀셋", "견인기", "거즈", "흡인기")
- 도구와 무관하거나 판단 불가 → "0"
- 출력은 반드시 한 단어. 설명·문장·따옴표·구두점 금지.
    """
)                        

# ── Whisper 로드 ──────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Whisper 로딩 중... (device: {device})")
stt_model = whisper.load_model("base", device=device)
print("완료!\n")

# ── 녹음 상태 ────────────────────────────────
is_recording = False
audio_frames = []
SAMPLE_RATE  = 16000

# 7종 도구 (Gemini 출력 검증용)
VALID_TOOLS = {"메스", "가위", "겸자", "핀셋", "견인기", "거즈", "흡인기"}

def audio_callback(indata, frames, time, status):
    if is_recording:
        audio_frames.append(indata.copy())

def process_audio():
    if not audio_frames:
        print("녹음 데이터 없음\n")
        return

    # 오디오 합치기
    audio = np.concatenate(audio_frames, axis=0).flatten()

    # STT
    t0 = time.perf_counter()
    result = stt_model.transcribe(
        audio,
        language="ko",
        initial_prompt="수술 도구를 요청하는 음성입니다. 메스, 가위, 겸자, 핀셋, 견인기, 거즈, 흡인기, 스칼펠, 석션, 포셉, 리트랙터, 트위저."
    )
    stt_ms = (time.perf_counter() - t0) * 1000
    text = result["text"].strip()
    print(f"[STT  {stt_ms:6.0f}ms] → '{text}'")

    if not text:
        print("인식 실패\n")
        return

    # Gemini
    print("도구 찾는중")
    t1 = time.perf_counter()
    try:
        response = model.generate_content(text)
        gem_ms = (time.perf_counter() - t1) * 1000
        raw = response.text.strip()
        tool = raw if raw in VALID_TOOLS else "0"
    except Exception as e:
        gem_ms = (time.perf_counter() - t1) * 1000
        print(f"Gemini 오류: {e}")
        tool = "0"

    print(f"[Gemini {gem_ms:6.0f}ms] → {tool}")
    print("-" * 30)
    print(f"총 소요: {(stt_ms + gem_ms):.0f}ms (STT {stt_ms:.0f} + Gemini {gem_ms:.0f})")
    # TODO: ROS publish 자리 — 현재는 터미널 출력만
    print(f"결과: {tool}")
    print("-" * 30)
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
        process_audio()

    if hasattr(key, 'char') and key.char == 'q':
        print("수고하셨습니다")
        return False

# ── 메인 ──────────────────────────────────────
print("=" * 40)
print("수술 도구 음성 인식 테스트")
print("스페이스 누르는 동안 녹음")
print("도구 7종: 메스 / 가위 / 겸자 / 핀셋 / 견인기 / 거즈 / 흡인기")
print("q → 수고하셨습니다")
print("=" * 40)
print("\n[ 스페이스 눌러서 녹음 시작 ]")

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32',
    callback=audio_callback
)
stream.start()

with keyboard.Listener(
    on_press=on_press,
    on_release=on_release
) as listener:
    listener.join()

stream.stop()
stream.close()
print("종료")