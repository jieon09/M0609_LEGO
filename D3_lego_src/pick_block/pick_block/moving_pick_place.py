#!/usr/bin/env python3

import sys
import rclpy
import DR_init

# =========================================================
# 기본 설정
# =========================================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

VELOCITY = 100  # 로봇 이동 속도 (%)
ACC = 60        # 로봇 가속도 (%)

# 펌웨어 33버전 기준 실측 초기 조인트값 (단위: deg)
JREADY = {
    0: [11.07, 1.16, 88.84, 0, 90.01, 11.07],
    1: [23.06,-13.05,102.59,1.17,89.59,26.27]
}

# =========================================================
# 초기 위치 이동 함수
# =========================================================
def pick_up_ready(index: int = 0) -> None:
    """
    로봇을 지정한 JReady 위치로 이동

    Args:
        index: 이동할 JReady 번호 (0~3). 기본값 0.
    """
    target = JREADY.get(index)
    if target is None:
        print(f"유효하지 않은 JReady 번호: {index} (0~{len(JREADY)-1} 범위)")
        return

    print(f"JReady{index}로 이동 중... {target}")
    movej(target, vel=VELOCITY, acc=ACC)
    mwait()
    print("이동 완료")


# =========================================================
# MAIN
# =========================================================
def main(args: list | None = None) -> None:
    """
    ROS2 노드 초기화 후 로봇을 지정 JReady 위치로 이동하고 종료

    사용법:
        ros2 run pick_block move_pick_place        # JReady0
        ros2 run pick_block move_pick_place 1      # JReady1
        ros2 run pick_block move_pick_place 2      # JReady2
        ros2 run pick_block move_pick_place 3      # JReady3
    """
    # 커맨드라인 첫 번째 인자로 JReady 번호 수신 (없으면 0)
    argv = sys.argv[1:]
    try:
        jready_index = int(argv[0]) if argv else 0
    except ValueError:
        print(f"인자 오류: '{argv[0]}'는 숫자여야 합니다. 0으로 실행합니다.")
        jready_index = 0

    rclpy.init(args=args)

    # DR_init에 로봇 ID/모델 등록 후 노드 연결해야 DSR_ROBOT2 import가 동작한다
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    dsr_node = rclpy.create_node("move_pick_place", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    global movej, mwait

    try:
        from DSR_ROBOT2 import movej, mwait

    except ImportError as e:
        print(f"DSR_ROBOT2 import 실패: {e}")
        return

    try:
        pick_up_ready(jready_index)

    except KeyboardInterrupt:
        print("사용자 종료")

    finally:
        dsr_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()