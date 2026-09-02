"""
whisperllmtest_audio.py
마이크 없이 시스템 오디오(TTS 생성 파일)로 STT → Gemini 파이프라인 테스트
"""
import time
import whisper
import torch
import numpy as np
import google.generativeai as genai
from gtts import gTTS
from pydub import AudioSegment
import io
import os
import tempfile

# .env를 os.environ으로 로드 (표준 라이브러리만 사용)
_env_path = os.path.join(os.path.dirname(__file__), "Voice_receiver", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# ── Gemini 설정 ───────────────────────────────
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
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

VALID_TOOLS = {"메스", "가위", "겸자", "핀셋", "견인기", "거즈", "흡인기"}

# ── 테스트 케이스 ─────────────────────────────
# (발화 텍스트, 예상 결과)
TEST_CASES = [
    ("메스 주세요",   "메스"),
    ("가위",          "가위"),
    ("포셉",          "겸자"),
    ("scissors",      "가위"),
    ("석션",          "흡인기"),
    ("리트랙터",      "견인기"),
    ("트위저 주세요", "핀셋"),
    ("거즈",          "거즈"),
    ("날씨 어때요",   "0"),    # 무관한 발화 → 0 기대
]


def tts_to_numpy(text: str, lang: str = "ko") -> np.ndarray:
    """gTTS로 텍스트 → 오디오를 numpy float32 배열(16kHz)로 변환"""
    tts = gTTS(text=text, lang=lang, slow=False)
    mp3_buf = io.BytesIO()
    tts.write_to_fp(mp3_buf)
    mp3_buf.seek(0)

    # pydub으로 mp3 → PCM 16kHz mono
    seg = AudioSegment.from_file(mp3_buf, format="mp3")
    seg = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    samples = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def run_pipeline(audio: np.ndarray) -> tuple[str, float, float]:
    """STT → Gemini 분류. (tool, stt_ms, gem_ms) 반환"""
    t0 = time.perf_counter()
    result = stt_model.transcribe(
        audio,
        language="ko",
        initial_prompt="수술 도구를 요청하는 음성입니다. 메스, 가위, 겸자, 핀셋, 견인기, 거즈, 흡인기, 스칼펠, 석션, 포셉, 리트랙터, 트위저."
    )
    stt_ms = (time.perf_counter() - t0) * 1000
    text = result["text"].strip()

    if not text:
        return "인식실패", stt_ms, 0.0

    t1 = time.perf_counter()
    try:
        response = model.generate_content(text)
        gem_ms = (time.perf_counter() - t1) * 1000
        raw = response.text.strip()
        tool = raw if raw in VALID_TOOLS else "0"
    except Exception as e:
        gem_ms = (time.perf_counter() - t1) * 1000
        print(f"  Gemini 오류: {e}")
        tool = "0"

    return tool, stt_ms, gem_ms, text


# ── 메인 테스트 루프 ──────────────────────────
print("=" * 55)
print("  수술 도구 음성 인식 — 시스템 오디오(TTS) 자동 테스트")
print("=" * 55)

pass_count = 0
for i, (phrase, expected) in enumerate(TEST_CASES, 1):
    print(f"\n[{i}/{len(TEST_CASES)}] 발화: \"{phrase}\"  (기대: {expected})")
    try:
        audio = tts_to_numpy(phrase)
    except Exception as e:
        print(f"  TTS 오류: {e}")
        continue

    result = run_pipeline(audio)
    if len(result) == 4:
        tool, stt_ms, gem_ms, recognized = result
    else:
        tool, stt_ms, gem_ms = result
        recognized = ""

    ok = "✓ PASS" if tool == expected else "✗ FAIL"
    if tool == expected:
        pass_count += 1

    print(f"  STT 인식: '{recognized}'  ({stt_ms:.0f}ms)")
    print(f"  Gemini  : {tool}  ({gem_ms:.0f}ms)")
    print(f"  → {ok}  (기대={expected}, 실제={tool})")

print("\n" + "=" * 55)
print(f"  결과: {pass_count} / {len(TEST_CASES)} 통과")
print("=" * 55)
