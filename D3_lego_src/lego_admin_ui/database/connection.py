import os
from typing import Generator

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session as OrmSession

# 환경변수 미설정 시 로컬 개발용 기본값 사용
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:1234@localhost:5432/lego_robot"
)

engine: Engine  = create_engine(DATABASE_URL, echo=False)
Session         = sessionmaker(bind=engine)
Base            = declarative_base()


def get_db() -> Generator[OrmSession, None, None]:
    """
    Flask/FastAPI Depends용 DB 세션 generator.

    처리 흐름:
    1. 세션 생성
    2. 호출자에게 yield
    3. finally에서 세션 close (정상/예외 모두 보장)

    Yields:
        열린 SQLAlchemy 세션 — finally 블록에서 자동 close
    """
    db: OrmSession = Session()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    등록된 ORM 모델 기반으로 DB 테이블을 생성한다.

    처리 흐름:
    1. 모델 모듈 import (Base.metadata에 등록 트리거)
    2. create_all로 미존재 테이블만 생성

    Note:
        이미 존재하는 테이블은 건너뜀 (CREATE TABLE IF NOT EXISTS 동작)
    """
    from database.models import Task, Inventory, RobotLog  # noqa: F401
    Base.metadata.create_all(bind=engine)
    print("✅ 모든 테이블 생성 완료")
