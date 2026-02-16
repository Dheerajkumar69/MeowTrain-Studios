from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database import Base


class ModelConfig(Base):
    __tablename__ = "model_configs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    base_model = Column(String, nullable=False)
    training_method = Column(String, default="lora")  # lora, qlora, full
    hyperparameters = Column(JSON, default=dict)
    status = Column(String, default="configured")  # configured, training, trained, error
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    project = relationship("Project", back_populates="model_configs")
    training_runs = relationship("TrainingRun", back_populates="model_config", cascade="all, delete-orphan")
