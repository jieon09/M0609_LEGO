import logging
import time
from typing import Generator

from flask import Blueprint, Response, jsonify

from ros.cam_state import cam_state, cam_lock

log = logging.getLogger(__name__)

camera_bp = Blueprint("camera", __name__)


def _generate_frames() -> Generator[bytes, None, None]:
    """
    카메라 프레임을 multipart/x-mixed-replace 스트림으로 생성한다.

    처리 흐름:
    1. cam_state["frame"] 폴링 (락 최소화를 위해 스냅샷 후 해제)
    2. 새 프레임이 없으면 10 ms 슬립 → CPU 점유 방지
    3. 새 프레임이면 MJPEG 청크로 yield

    Note:
        락 구간을 최소화해 카메라 노드 콜백과의 경합을 줄임
    """
    last: bytes | None = None
    while True:
        with cam_lock:
            frame: bytes = cam_state["frame"]
        if frame and frame is not last:
            last = frame
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame +
                b"\r\n"
            )
        else:
            # 새 프레임이 도착할 때까지 짧게 대기해 불필요한 busy-loop 방지
            time.sleep(0.01)


@camera_bp.route("/video_feed")
def video_feed() -> Response:
    """MJPEG 스트리밍 엔드포인트. 브라우저 <img src> 태그에서 직접 소비한다."""
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@camera_bp.route("/api/yolo/toggle", methods=["POST"])
def yolo_toggle() -> Response:
    """
    YOLO 추론 활성화 여부를 토글한다.

    처리 흐름:
    1. yolo_on 플래그 반전
    2. OFF 전환 시 detections 초기화 (낡은 결과 제거)

    Returns:
        ok=True, yolo_on: 변경 후 상태(bool)
    """
    with cam_lock:
        cam_state["yolo_on"] = not cam_state["yolo_on"]
        state: bool = cam_state["yolo_on"]
        if not state:
            # OFF 전환 시 이전 프레임의 탐지 결과가 UI에 남지 않도록 즉시 비움
            cam_state["detections"] = []
    log.info("[YOLO] 상태 변경 → %s", "ON" if state else "OFF")
    return jsonify({"ok": True, "yolo_on": state})


@camera_bp.route("/api/detections")
def api_detections() -> Response:
    """
    현재 YOLO 탐지 결과 목록을 반환한다.

    Returns:
        yolo_on: bool, detections: [{color, shape, conf}] 리스트
        conf 값은 소수점 1자리로 반올림
    """
    with cam_lock:
        detections: list[dict] = list(cam_state["detections"])
        yolo_on: bool          = cam_state["yolo_on"]
    return jsonify({
        "yolo_on": yolo_on,
        "detections": [
            {"color": d["color"], "shape": d["shape"], "conf": round(d["conf"], 1)}
            for d in detections
        ],
    })
