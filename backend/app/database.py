import logging

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import DATABASE_URL

logger = logging.getLogger("meowllm.database")

# Determine engine kwargs based on database backend
_engine_kwargs = {"echo": False}

if DATABASE_URL.startswith("sqlite"):
    # SQLite-specific: allow multi-threaded access
    _engine_kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    # Use WAL mode for better concurrency + enable foreign keys
    _engine_kwargs["pool_size"] = 1  # SQLite handles its own locking
    _engine_kwargs["pool_pre_ping"] = True
else:
    # PostgreSQL/MySQL: use connection pooling
    _engine_kwargs["pool_size"] = 10
    _engine_kwargs["max_overflow"] = 20
    _engine_kwargs["pool_recycle"] = 1800  # recycle connections every 30min
    _engine_kwargs["pool_pre_ping"] = True  # verify connection health
    _engine_kwargs["pool_timeout"] = 30  # wait max 30s for a connection

engine = create_engine(DATABASE_URL, **_engine_kwargs)


# Enable WAL mode and foreign keys for SQLite on every new connection
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """Dependency to get a database session.

    Automatically rolls back on exception and always closes.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_tables():
    """Create all tables on startup."""
    # Import all models so their tables get registered with Base.metadata
    import app.models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")
