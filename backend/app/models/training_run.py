from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database import Base


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    model_config_id = Column(Integer, ForeignKey("model_configs.id"), nullable=False)
    status = Column(String, default="pending")  # pending, running, paused, completed, stopped, error
    current_loss = Column(Float, nullable=True)
    best_loss = Column(Float, nullable=True)
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=3)
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    tokens_per_sec = Column(Float, default=0.0)
    checkpoint_path = Column(String, nullable=True)
    output_path = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    learning_rate_current = Column(Float, nullable=True)
    validation_loss = Column(Float, nullable=True)
    log_history = Column(JSON, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    project = relationship("Project", back_populates="training_runs")
    model_config = relationship("ModelConfig", back_populates="training_runs")
