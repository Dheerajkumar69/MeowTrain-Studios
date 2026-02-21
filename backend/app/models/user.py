from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=True)
    password_hash = Column(String, nullable=True)
    display_name = Column(String, default="User")
    is_guest = Column(Boolean, default=False)
    role = Column(String, default="member")  # admin | member | guest
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    password_reset_token = Column(String, nullable=True)
    password_reset_expires = Column(DateTime, nullable=True)

    # Email verification
    email_verified = Column(Boolean, default=False)
    email_verification_token = Column(String, nullable=True)

    # OAuth
    oauth_provider = Column(String, nullable=True)  # google | github | None
    oauth_id = Column(String, nullable=True)

    # JWT revocation — increment to invalidate all existing tokens
    token_version = Column(Integer, default=0, nullable=False, server_default="0")

    # Account lockout — progressive lockout after failed login attempts
    failed_login_attempts = Column(Integer, default=0, nullable=False, server_default="0")
    locked_until = Column(DateTime, nullable=True)

    projects = relationship("Project", back_populates="user", cascade="all, delete-orphan")
