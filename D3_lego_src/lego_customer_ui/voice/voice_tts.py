import logging
import os
from gtts import gTTS

log = logging.getLogger(__name__)

# 고정 경로에 덮어쓰기 — 동시 호출 시 파일 충돌 가능하나 단일 사용자 환경에서 무관
_VOICE_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'voice.mp3')


def speak(text: str) -> None:
    """텍스트를 한국어 TTS로 변환하여 즉시 재생.

    처리 흐름:
    1. gTTS로 MP3 생성 후 _VOICE_PATH에 저장
    2. os.system('mpg123 ...')으로 재생 — blocking 호출

    Args:
        text: 읽을 한국어 문자열

    Note:
        os.system은 재생 완료까지 blocking — 연속 호출 시 이전 재생이 끝난 후 다음 실행
    """
    log.info("[VOICE] %s", text)
    tts = gTTS(text=text, lang='ko')
    tts.save(_VOICE_PATH)
    os.system(f"mpg123 {_VOICE_PATH}")


if __name__ == "__main__":
    print("타자 입력 (exit 종료)")
    while True:
        text = input(">>> ")
        if text == "exit":
            break
        speak(text)
