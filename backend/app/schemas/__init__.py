from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from typing import Literal, Optional
from datetime import datetime

# Valid project statuses
VALID_PROJECT_STATUSES = {"created", "training", "trained", "error"}


# ===== Auth Schemas =====
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    display_name: str = Field(default="User", max_length=100)


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user: "UserResponse"


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: Optional[str] = None
    display_name: str
    is_guest: bool
    created_at: datetime


# ===== Project Schemas =====
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    intended_use: str = Field(default="custom")

    @field_validator("name")
    @classmethod
    def trim_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Project name cannot be blank")
        return v


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)
    intended_use: Optional[str] = None
    status: Optional[Literal["created", "training", "trained", "error"]] = None

    @field_validator("name")
    @classmethod
    def trim_name(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("Project name cannot be blank")
        return v


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    name: str
    description: str
    intended_use: str
    status: str
    hardware_snapshot: Optional[dict] = None
    created_at: datetime
    updated_at: datetime
    dataset_count: int = 0
    model_config_count: int = 0


# ===== Dataset Schemas =====
class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    filename: str
    original_name: str
    file_type: str
    file_size: int
    token_count: int
    chunk_count: int
    status: str
    uploaded_at: datetime


class DatasetPreview(BaseModel):
    id: int
    original_name: str
    total_tokens: int
    total_chunks: int
    chunks: list[dict]  # [{text, token_count, index}]


# ===== Model Schemas =====
class ModelInfo(BaseModel):
    model_id: str
    name: str
    description: str
    parameters: str
    size_gb: float
    ram_required_gb: int
    vram_required_gb: int
    recommended_hardware: str
    estimated_train_minutes: int
    icon: str
    is_cached: bool = False
    compatibility: str = "unknown"  # compatible, may_be_slow, too_large


# ===== Training Schemas =====
class TrainingConfigRequest(BaseModel):
    base_model: str
    method: Literal["lora", "qlora", "full"] = Field(default="lora")
    epochs: int = Field(default=3, ge=1, le=50)
    batch_size: int = Field(default=4, ge=1, le=64)
    learning_rate: float = Field(default=2e-4, gt=0, le=1.0)
    max_tokens: int = Field(default=512, ge=64, le=4096)
    train_split: float = Field(default=0.9, ge=0.5, le=0.99)
    lora_rank: int = Field(default=16, ge=4, le=128)
    lora_alpha: int = Field(default=32, ge=4, le=256)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    warmup_steps: int = Field(default=10, ge=0, le=1000)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=32)


class TrainingStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    current_loss: Optional[float] = None
    best_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    learning_rate_current: Optional[float] = None
    current_epoch: int
    total_epochs: int
    current_step: int
    total_steps: int
    tokens_per_sec: float
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TrainingLogEntry(BaseModel):
    step: int
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: Optional[float] = None
    timestamp: Optional[float] = None


# ===== Inference Schemas =====
class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=50000)
    system_prompt: str = Field(default="You are a helpful assistant.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    context_dataset_ids: list[int] = Field(default_factory=list)
    lmstudio_model: Optional[str] = Field(default=None, description="LM Studio model ID to use for inference")


class ChatResponse(BaseModel):
    response: str
    tokens_used: int
    generation_time_ms: float
    model_used: Optional[str] = None


class PromptTemplateCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    system_prompt: str = Field(default="")
    user_prompt: str = Field(default="")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class PromptTemplateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    name: str
    system_prompt: str
    user_prompt: str
    temperature: float
    created_at: datetime


# ===== Hardware Schemas =====
class HardwareStatus(BaseModel):
    cpu_name: str
    cpu_cores: int
    cpu_usage_percent: float
    ram_total_gb: float
    ram_available_gb: float
    ram_used_gb: float = 0.0
    ram_usage_percent: float
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_vram_total_gb: Optional[float] = None
    gpu_vram_available_gb: Optional[float] = None
    gpu_vram_used_gb: Optional[float] = None
    gpu_usage_percent: Optional[float] = None
    gpu_memory_utilization_percent: Optional[float] = None
    gpu_temp_celsius: Optional[int] = None
    gpu_power_watts: Optional[float] = None
    gpu_power_limit_watts: Optional[float] = None
    gpu_fan_percent: Optional[int] = None
    disk_total_gb: float
    disk_free_gb: float
    model_cache_size_gb: float


# ===== User Profile Schemas =====
class ProfileUpdateRequest(BaseModel):
    display_name: Optional[str] = Field(default=None, min_length=1, max_length=100)

    @field_validator("display_name")
    @classmethod
    def trim_display_name(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("Display name cannot be blank")
        return v


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6)
