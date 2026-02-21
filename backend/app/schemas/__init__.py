from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator
from typing import Literal, Optional
from datetime import datetime

# Valid project statuses
VALID_PROJECT_STATUSES = {"created", "training", "trained", "error"}


# ===== Auth Schemas =====
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    display_name: str = Field(default="User", min_length=1, max_length=100)

    @field_validator("display_name")
    @classmethod
    def sanitize_display_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            return "User"
        return v


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=1, max_length=128)


class AuthResponse(BaseModel):
    token: str
    user: "UserResponse"


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: Optional[str] = None
    display_name: str
    is_guest: bool
    role: str = "member"
    email_verified: bool = False
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
    # NOTE: 'status' intentionally excluded — project status is managed
    # internally by the training pipeline, not by client requests.

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
    base_model: str = Field(..., min_length=3, max_length=200)
    method: Literal["lora", "qlora", "full", "dpo", "orpo"] = Field(default="lora")
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
    # Best-practice additions
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    lr_scheduler_type: Literal["cosine", "linear", "constant", "cosine_with_restarts"] = Field(default="cosine")
    early_stopping_patience: int = Field(default=3, ge=0, le=20)
    early_stopping_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
    gradient_checkpointing: bool = Field(default=True)
    eval_steps: int = Field(default=50, ge=10, le=1000)
    # Alignment training (DPO/ORPO)
    dpo_beta: float = Field(default=0.1, ge=0.01, le=1.0, description="DPO KL penalty coefficient")
    orpo_alpha: float = Field(default=1.0, ge=0.1, le=10.0, description="ORPO odds ratio weight")
    # Multi-GPU / DeepSpeed
    multi_gpu: bool = Field(default=False, description="Enable multi-GPU via DeepSpeed")
    deepspeed_stage: Literal[2, 3] = Field(default=2, description="DeepSpeed ZeRO stage")
    # Resume from checkpoint
    resume_from_checkpoint: bool = Field(default=False, description="Resume training from latest checkpoint")

    @model_validator(mode="after")
    def _cross_field_checks(self):
        """Validate hyperparameter combinations that individual field constraints cannot catch."""
        # LoRA alpha should be >= rank (standard best practice; alpha < rank degrades performance)
        if self.method in ("lora", "qlora") and self.lora_alpha < self.lora_rank:
            raise ValueError(
                f"lora_alpha ({self.lora_alpha}) should be >= lora_rank ({self.lora_rank}). "
                f"A common default is alpha = 2 × rank."
            )
        return self


class TrainingStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    current_loss: Optional[float] = None
    best_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    perplexity: Optional[float] = None
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
    eval_loss: Optional[float] = None
    perplexity: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: Optional[float] = None
    timestamp: Optional[float] = None


# ===== Inference Schemas =====
class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=50000)
    system_prompt: str = Field(default="You are a helpful assistant.", max_length=10000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    context_dataset_ids: list[int] = Field(default_factory=list, max_length=50)
    lmstudio_model: Optional[str] = Field(default=None, max_length=500, description="LM Studio model ID to use for inference")


class ChatResponse(BaseModel):
    response: str
    tokens_used: int
    generation_time_ms: float
    model_used: Optional[str] = None


class PromptTemplateCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    system_prompt: str = Field(default="", max_length=10000)
    user_prompt: str = Field(default="", max_length=50000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator("name")
    @classmethod
    def trim_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Template name cannot be blank")
        return v


class PromptTemplateUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    system_prompt: Optional[str] = Field(default=None, max_length=10000)
    user_prompt: Optional[str] = Field(default=None, max_length=50000)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)

    @field_validator("name")
    @classmethod
    def trim_name(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("Template name cannot be blank")
        return v


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
    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)


# ===== Generic Responses =====
class DetailResponse(BaseModel):
    detail: str


class DetailWithIdResponse(BaseModel):
    detail: str
    run_id: Optional[int] = None


# ===== Paginated Responses =====
class PaginatedProjectsResponse(BaseModel):
    items: list[ProjectResponse]
    total: int
    page: int
    per_page: int


class PaginatedDatasetsResponse(BaseModel):
    items: list[DatasetResponse]
    total: int
    page: int
    per_page: int


class PaginatedPromptsResponse(BaseModel):
    items: list[PromptTemplateResponse]
    total: int
    page: int
    per_page: int


# ===== Password Reset =====
class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=1, max_length=512)
    new_password: str = Field(..., min_length=8, max_length=128)


# ===== Training History & Compare =====
class TrainingHistoryResponse(BaseModel):
    runs: list[TrainingStatusResponse]
    total: int
    limit: int
    offset: int


class ConfigDiffEntry(BaseModel):
    key: str
    run_a: Optional[object] = None
    run_b: Optional[object] = None
    different: bool


class RunComparison(BaseModel):
    run_id: int
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    config: dict
    metrics: dict
    log_history: list = Field(default_factory=list)


class RunComparisonResponse(BaseModel):
    runs: list[RunComparison]
    config_diff: list[ConfigDiffEntry]


# ===== Inference Context =====
class DatasetSummary(BaseModel):
    id: int
    name: str
    tokens: int


class InferenceContextResponse(BaseModel):
    datasets: list[DatasetSummary]
    model: Optional[dict] = None


# ===== Dataset Preview-Training =====
class SamplePreview(BaseModel):
    raw: str = ""
    has_messages: bool = False
    token_count: int = 0
    messages: Optional[list[dict]] = None
    templated: Optional[str] = None
    templated_tokens: Optional[int] = None


class DatasetFormatInfo(BaseModel):
    dataset_id: int
    filename: str
    format: str
    format_description: str = ""
    sample_count: int = 0
    samples: list[SamplePreview] = Field(default_factory=list)


class TokenSummary(BaseModel):
    total_examples: int
    instruction_examples: int
    text_only_examples: int
    avg_tokens: int
    min_tokens: int
    max_tokens: int
    total_tokens: int


class TrainingPreviewResponse(BaseModel):
    datasets: list[DatasetFormatInfo]
    summary: TokenSummary


# ===== Augmentation =====
class AugmentStats(BaseModel):
    original_count: int
    final_count: int
    duplicates_removed: int
    filtered_out: int
    filter_reasons: dict = Field(default_factory=dict)
    reduction_percent: float


class AugmentSample(BaseModel):
    index: int
    before: str
    after: str
    changed: bool


class AugmentDatasetInfo(BaseModel):
    id: int
    name: str
    format: str
    example_count: int


class CreatedDataset(BaseModel):
    id: int
    filename: str


class AugmentationResponse(BaseModel):
    preview_only: bool
    datasets: list[AugmentDatasetInfo]
    stats: AugmentStats
    samples: list[AugmentSample]
    created_dataset: Optional[CreatedDataset] = None


# ===== Model Status / Download =====
class DownloadInfo(BaseModel):
    status: str
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    local_path: Optional[str] = None


class ModelStatusResponse(BaseModel):
    model_id: str
    is_cached: bool
    size_gb: Optional[float] = None
    download: Optional[DownloadInfo] = None


class DownloadStartResponse(BaseModel):
    detail: str
    status: str
    progress: Optional[float] = None
    estimated_size_gb: Optional[float] = None


class DownloadProgressResponse(BaseModel):
    status: str
    progress: float = 0.0
    message: str = ""
    is_cached: bool = False
    elapsed_seconds: Optional[int] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    local_path: Optional[str] = None


# ===== GGUF =====
class GGUFExportResponse(BaseModel):
    detail: str
    status: str


class GGUFStatusResponse(BaseModel):
    step: str
    progress: int = 0
    message: str = ""
    error: Optional[str] = None
    gguf_filename: Optional[str] = None
    gguf_size_mb: Optional[float] = None


# ===== LM Studio =====
class LMStudioConfigResponse(BaseModel):
    host: str
    port: int
    enabled: bool
    url: Optional[str] = None


class LMStudioConnectionResponse(BaseModel):
    connected: bool
    url: Optional[str] = None
    error: Optional[str] = None
    models: Optional[list[dict]] = None


class LMStudioModelsResponse(BaseModel):
    models: list[dict] = Field(default_factory=list)
    enabled: bool = True
    url: Optional[str] = None
    message: Optional[str] = None


# ===== Health =====
class HealthResponse(BaseModel):
    status: str
    version: str
    db_connected: bool = True
    db_latency_ms: Optional[float] = None


# ===== Email Verification =====
class VerifyEmailRequest(BaseModel):
    token: str = Field(..., min_length=1, max_length=512)


class ResendVerificationRequest(BaseModel):
    email: EmailStr


# ===== OAuth =====
class OAuthCallbackRequest(BaseModel):
    code: str = Field(..., min_length=1)
    state: Optional[str] = None


# ===== Admin =====
class SystemStatsResponse(BaseModel):
    users: dict
    projects: dict
    datasets: dict
    training: dict
    models: dict
    tasks: dict
    disk: dict


# ===== Backup =====
class BackupImportResponse(BaseModel):
    detail: str
    project_id: int
    datasets_restored: int


# ===== Lineage =====
class LineageMetrics(BaseModel):
    current_loss: Optional[float] = None
    best_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    perplexity: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    total_epochs: Optional[int] = None
    total_steps: Optional[int] = None
    learning_rate_current: Optional[float] = None
