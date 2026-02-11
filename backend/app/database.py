from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import DATABASE_URL

# Determine engine kwargs based on database backend
_engine_kwargs = {"echo": False}

if DATABASE_URL.startswith("sqlite"):
    # SQLite-specific: allow multi-threaded access
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # PostgreSQL/MySQL: use connection pooling
    _engine_kwargs["pool_size"] = 10
    _engine_kwargs["max_overflow"] = 20
    _engine_kwargs["pool_recycle"] = 1800  # recycle connections every 30min
    _engine_kwargs["pool_pre_ping"] = True  # verify connection health

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """Dependency to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all tables on startup."""
    Base.metadata.create_all(bind=engine)
