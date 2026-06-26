import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session as SASession

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:1234@localhost:5432/lego_robot"
)

engine  = create_engine(DATABASE_URL, echo=False)
Session = sessionmaker(bind=engine)
Base    = declarative_base()


def get_db() -> Generator[SASession, None, None]:
    """Flask/FastAPI Depends용 DB 세션 generator.

    Yields:
        열린 SQLAlchemy 세션 — finally 블록에서 자동 close
    """
    db = Session()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """등록된 ORM 모델 기반으로 DB 테이블을 생성한다.

    Note:
        이미 존재하는 테이블은 건너뜀 (CREATE TABLE IF NOT EXISTS 동작)
    """
    from database.models import Task, Inventory, RobotLog  # noqa: F401
    Base.metadata.create_all(bind=engine)
    print("✅ 모든 테이블 생성 완료")
