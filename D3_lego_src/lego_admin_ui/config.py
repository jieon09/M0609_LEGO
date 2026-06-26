import os
import subprocess

# workspace/src 루트 경로: 절대경로 기반으로 계산해 실행 위치에 무관하게 동작
WS_SRC: str   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 환경변수 미설정 시 기본 로봇 ID 사용 (단일 로봇 환경 대상)
ROBOT_ID: str = os.getenv("ROBOT_ID", "dsr01")

SCRIPT_DIR: str         = os.path.join(WS_SRC, "pick_block/pick_block")
PICK_READY_SCRIPT: str  = os.path.join(SCRIPT_DIR, "moving_pick_place.py")
MOVING_PICK_SCRIPT: str = os.path.join(SCRIPT_DIR, "moving_pick_yolo.py")
BLOCK_PLACE_DIR: str    = os.path.join(WS_SRC, "place_block")

# ROS2 환경 소싱 명령 — subprocess shell=True 호출 시 앞에 붙여 사용
ROS_SOURCE: str = (
    "source /opt/ros/humble/setup.bash && "
    "source /home/shin/cobot_ws/install/setup.bash"
)

YOLO_MODEL_PATH: str = os.path.join(WS_SRC, "pick_block", "resource", "best_obb_2.pt")


_ROS_ENV_CACHE: dict[str, str] | None = None


def get_ros_env() -> dict[str, str]:
    """ROS_SOURCE를 1회만 실행하고 결과 env를 캐싱한다.

    매 subprocess 호출마다 setup.bash를 source하면 AMENT_PREFIX_PATH 계산 등으로
    1회당 ~0.5초 비용이 누적된다. 캐싱된 env를 subprocess에 그대로 넘기면
    bash가 source할 필요가 없다.
    """
    global _ROS_ENV_CACHE
    if _ROS_ENV_CACHE is not None:
        return _ROS_ENV_CACHE

    result = subprocess.run(
        ["bash", "-c", f"{ROS_SOURCE} && env -0"],
        capture_output=True,
        check=True,
    )
    env: dict[str, str] = {}
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        k, _, v = entry.decode("utf-8", errors="replace").partition("=")
        if k:
            env[k] = v
    _ROS_ENV_CACHE = env
    return env
