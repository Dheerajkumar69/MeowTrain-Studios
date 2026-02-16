from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(String, default="")
    intended_use = Column(String, default="custom")  # chatbot, code, qa, custom
    status = Column(String, default="created")  # created, training, trained, error
    hardware_snapshot = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="projects")
    datasets = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")
    model_configs = relationship("ModelConfig", back_populates="project", cascade="all, delete-orphan")
    training_runs = relationship("TrainingRun", back_populates="project", cascade="all, delete-orphan")
    prompt_templates = relationship("PromptTemplate", back_populates="project", cascade="all, delete-orphan")
