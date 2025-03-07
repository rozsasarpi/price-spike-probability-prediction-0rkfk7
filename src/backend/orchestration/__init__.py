"""
Entry point for the orchestration module of the ERCOT RTLMP spike prediction system.
Exposes key components for task management, scheduling, pipeline execution, and error
recovery to provide a unified interface for coordinating the system's operations.
"""

__version__ = "0.1.0"

from .task_management import (
    TaskManager,
    Task,
    TaskStatus,
    TaskPriority,
    TaskResult,
    create_task_id,
    format_task_result,
    execute_with_timeout,
    resolve_dependencies,
)
from .scheduler import (
    Scheduler,
    DailyScheduler,
    ScheduledTask,
    ScheduleFrequency,
    parse_cron_expression,
    get_next_run_time,
    is_time_to_run,
    create_data_fetch_task,
    create_inference_task,
    create_training_task,
    handle_task_error,
)
from .pipeline import (
    Pipeline,
    PipelineExecutor,
    PipelineStage,
    execute_pipeline,
    schedule_pipeline,
    setup_daily_pipeline,
    DEFAULT_PIPELINE_CONFIG,
)
from .error_recovery import (
    ErrorRecoveryManager,
    RecoveryContext,
    RecoveryStrategy,
    PipelineStage as ErrorPipelineStage,
    with_recovery,
    get_error_type_category,
    create_recovery_context,
    DataRecoveryHandler,
    ModelRecoveryHandler,
)

__all__ = [
    "TaskManager",
    "Task",
    "TaskStatus",
    "TaskPriority",
    "TaskResult",
    "create_task_id",
    "format_task_result",
    "execute_with_timeout",
    "resolve_dependencies",
    "Scheduler",
    "DailyScheduler",
    "ScheduledTask",
    "ScheduleFrequency",
    "parse_cron_expression",
    "get_next_run_time",
    "is_time_to_run",
    "create_data_fetch_task",
    "create_inference_task",
    "create_training_task",
    "handle_task_error",
    "Pipeline",
    "PipelineExecutor",
    "PipelineStage",
    "execute_pipeline",
    "schedule_pipeline",
    "setup_daily_pipeline",
    "DEFAULT_PIPELINE_CONFIG",
    "ErrorRecoveryManager",
    "RecoveryContext",
    "RecoveryStrategy",
    "ErrorPipelineStage",
    "with_recovery",
    "get_error_type_category",
    "create_recovery_context",
    "DataRecoveryHandler",
    "ModelRecoveryHandler",
    "__version__"
]