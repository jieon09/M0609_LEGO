"""
DB 초기화 스크립트
실행: python database/init_db.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import init_db, Session
from database.models import Task, Inventory, RobotLog, RobotAction, LogStatus

from datetime import datetime, timezone


def seed_inventory() -> None:
    """
    재고 초기 행이 없으면 기본값(전 항목 0개)으로 1행 삽입한다.

    처리 흐름:
    1. Inventory 첫 번째 행 존재 확인
    2. 없으면 0개짜리 기본 행 삽입 후 commit
    3. 있으면 건너뜀

    Note:
        재고 테이블은 단일 행(싱글톤) 구조로 설계됨
    """
    db = Session()
    try:
        if not db.query(Inventory).first():
            db.add(Inventory(
                red_2x2= 0, red_2x3=0,
                blue_2x2= 0, blue_2x3=0,
                yellow_2x2=0, yellow_2x3=0,
                total_blocks=0,
            ))
            db.commit()
            print("✅ 재고 초기 데이터 삽입 완료")
        else:
            print("ℹ️  재고 데이터 이미 존재 — 건너뜀")
    except Exception as e:
        db.rollback()
        print(f"❌ 재고 삽입 실패: {e}")
    finally:
        db.close()


def example_full_flow() -> None:
    """
    Task 생성 → 로그 기록 → 완료 흐름 예시.

    처리 흐름:
    1. 테스트용 Task 생성 (customer_id="test_customer_001")
    2. 3스텝 분량의 RobotLog(success) 삽입
    3. Task.end_at 설정으로 완료 처리

    Note:
        실제 운영 코드가 아닌 동작 검증용 예시 함수
    """
    db = Session()
    try:
        task = Task(customer_id="test_customer_001")
        db.add(task)
        db.flush()
        print(f"✅ Task 생성: {task.id}")

        for step in range(1, 4):
            entry = RobotLog(
                task_id=task.id,
                robot_action=RobotAction.place,
                status=LogStatus.success,
                step_order=step,
            )
            db.add(entry)

        task.end_at = datetime.now(timezone.utc)
        db.commit()
        print(f"✅ 전체 플로우 완료! Task: {task.id}")

    except Exception as e:
        db.rollback()
        print(f"❌ 플로우 실패: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    print("=== DB 초기화 시작 ===")
    init_db()
    print("\n=== 재고 초기 데이터 삽입 ===")
    seed_inventory()
    print("\n=== 전체 플로우 예시 실행 ===")
    example_full_flow()
