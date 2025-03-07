"""
Core scheduler module for the ERCOT RTLMP spike prediction system.

This module provides functionality for scheduling tasks at specific times,
handling recurring tasks, and ensuring critical operations like data fetching,
model training, and inference run according to required schedules.
"""

import typing
from typing import Dict, List, Optional, Callable, Any, Tuple, Union, Set, Type
import enum
import datetime
import time
import threading

from croniter import croniter  # version 1.0+

from .task_management import TaskManager, Task, TaskStatus, TaskPriority, TaskResult
from ..utils.logging import get_logger, log_execution_time, PerformanceLogger
from ..utils.error_handling import retry_with_backoff, handle_errors, ErrorHandler, RetryContext
from ..utils.type_definitions import DataFrameType, ModelType, PathType
from ..config.schema import Config, DataConfig, FeatureConfig, ModelConfig, InferenceConfig

# Set up logger
logger = get_logger(__name__)

# Default values
DEFAULT_CHECK_INTERVAL = 1.0  # seconds
DEFAULT_TIMEZONE = "America/Chicago"  # ERCOT is in Central Time
DEFAULT_RETRY_COUNT = 3


class ScheduleFrequency(enum.Enum):
    """Enum representing the frequency of scheduled tasks."""
    DAILY = "DAILY"
    HOURLY = "HOURLY"
    MINUTELY = "MINUTELY"
    CUSTOM = "CUSTOM"


@handle_errors(ValueError, None, True, 'Invalid cron expression format')
def parse_cron_expression(cron_expression: str) -> croniter:
    """
    Parses a cron expression and validates its format.
    
    Args:
        cron_expression: Cron expression string
        
    Returns:
        Croniter object for the parsed expression
        
    Raises:
        ValueError: If the cron expression is invalid
    """
    # Validate that the cron expression is a string
    if not isinstance(cron_expression, str):
        raise ValueError(f"Cron expression must be a string, got {type(cron_expression)}")
    
    # Create a croniter object to validate the expression
    now = datetime.datetime.now()
    cron = croniter(cron_expression, now)
    
    return cron


@handle_errors(ValueError, datetime.datetime.now() + datetime.timedelta(minutes=5), False, 'Error calculating next run time')
def get_next_run_time(cron_expression: str, base_time: Optional[datetime.datetime] = None) -> datetime.datetime:
    """
    Calculates the next run time based on a cron expression.
    
    Args:
        cron_expression: Cron expression string
        base_time: Base time for calculation, defaults to current time
        
    Returns:
        Next scheduled run time
        
    Raises:
        ValueError: If the cron expression is invalid
    """
    # Use current time if base_time is not provided
    if base_time is None:
        base_time = datetime.datetime.now()
    
    # Parse the cron expression
    cron = parse_cron_expression(cron_expression)
    
    # Calculate the next run time
    next_time = cron.get_next(datetime.datetime)
    
    return next_time


def is_time_to_run(scheduled_time: datetime.datetime, 
                  current_time: Optional[datetime.datetime] = None,
                  tolerance_seconds: float = 1.0) -> bool:
    """
    Checks if it's time to run a scheduled task.
    
    Args:
        scheduled_time: The scheduled execution time
        current_time: Current time, defaults to now
        tolerance_seconds: Tolerance in seconds
        
    Returns:
        True if it's time to run, False otherwise
    """
    if current_time is None:
        current_time = datetime.datetime.now()
    
    # Calculate the time difference in seconds
    time_diff = (current_time - scheduled_time).total_seconds()
    
    # If the scheduled time is within the tolerance range of current time,
    # or if the scheduled time has passed, it's time to run
    return -tolerance_seconds <= time_diff <= tolerance_seconds or time_diff > 0


def create_data_fetch_task(fetch_function: Callable, 
                          fetch_params: Dict[str, Any],
                          task_name: Optional[str] = None) -> Task:
    """
    Creates a task for fetching data.
    
    Args:
        fetch_function: Function to call for data fetching
        fetch_params: Parameters for the fetch function
        task_name: Optional task name
        
    Returns:
        Created data fetch task
    """
    if task_name is None:
        task_name = f"data_fetch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    task = Task(
        func=fetch_function,
        name=task_name,
        args=(),
        kwargs=fetch_params,
        priority=TaskPriority.HIGH,
        metadata={"type": "data_fetch"}
    )
    
    return task


def create_inference_task(inference_function: Callable, 
                         inference_params: Dict[str, Any],
                         task_name: Optional[str] = None) -> Task:
    """
    Creates a task for running inference.
    
    Args:
        inference_function: Function to call for inference
        inference_params: Parameters for the inference function
        task_name: Optional task name
        
    Returns:
        Created inference task
    """
    if task_name is None:
        task_name = f"inference_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    task = Task(
        func=inference_function,
        name=task_name,
        args=(),
        kwargs=inference_params,
        priority=TaskPriority.CRITICAL,  # Inference has the highest priority
        metadata={"type": "inference"}
    )
    
    return task


def create_training_task(training_function: Callable, 
                        training_params: Dict[str, Any],
                        task_name: Optional[str] = None) -> Task:
    """
    Creates a task for training a model.
    
    Args:
        training_function: Function to call for model training
        training_params: Parameters for the training function
        task_name: Optional task name
        
    Returns:
        Created training task
    """
    if task_name is None:
        task_name = f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    task = Task(
        func=training_function,
        name=task_name,
        args=(),
        kwargs=training_params,
        priority=TaskPriority.HIGH,
        metadata={"type": "training"}
    )
    
    return task


def handle_task_error(error: Exception, task_id: str, task_name: str, context: Dict[str, Any]) -> bool:
    """
    Handles errors during task execution with appropriate recovery actions.
    
    Args:
        error: The exception that occurred
        task_id: ID of the task that failed
        task_name: Name of the task that failed
        context: Additional context about the task
        
    Returns:
        True if recovery was successful, False otherwise
    """
    logger.error(f"Error executing task {task_id} ({task_name}): {error}", 
                extra={"context": context, "task_id": task_id, "error": str(error)})
    
    # Determine the error type and severity
    from ..utils.error_handling import is_retryable_error
    
    if is_retryable_error(error):
        logger.info(f"Error is retryable, recommending retry for task {task_id}")
        return True  # Recommend retry
    else:
        logger.critical(f"Critical error for task {task_id}, cannot recover automatically")
        # Update task execution statistics here if needed
        return False  # Don't retry


class ScheduledTask:
    """Class representing a task scheduled for periodic execution."""
    
    def __init__(self, 
                task: Task, 
                schedule: Union[str, ScheduleFrequency],
                next_run_time: Optional[datetime.datetime] = None,
                enabled: bool = True,
                metadata: Dict[str, Any] = None):
        """
        Initialize a new ScheduledTask instance.
        
        Args:
            task: The task to schedule
            schedule: Cron expression or ScheduleFrequency
            next_run_time: Next execution time, calculated if not provided
            enabled: Whether the task is enabled
            metadata: Additional metadata about the scheduled task
        """
        self.id = str(uuid.uuid4())
        self.name = task.name
        self.task = task
        self.schedule = schedule
        
        if next_run_time is None:
            self.next_run_time = self.update_next_run_time()
        else:
            self.next_run_time = next_run_time
        
        self.enabled = enabled
        self.last_run_time = None
        self.last_result = None
        self.run_count = 0
        self.metadata = metadata or {}
    
    def update_next_run_time(self, base_time: Optional[datetime.datetime] = None) -> datetime.datetime:
        """
        Updates the next run time based on the schedule.
        
        Args:
            base_time: Base time for calculation, defaults to current time or last run time
            
        Returns:
            Updated next run time
        """
        if base_time is None:
            base_time = self.last_run_time if self.last_run_time else datetime.datetime.now()
        
        if isinstance(self.schedule, str):
            # It's a cron expression
            self.next_run_time = get_next_run_time(self.schedule, base_time)
        else:
            # It's a ScheduleFrequency
            if self.schedule == ScheduleFrequency.DAILY:
                self.next_run_time = base_time + datetime.timedelta(days=1)
            elif self.schedule == ScheduleFrequency.HOURLY:
                self.next_run_time = base_time + datetime.timedelta(hours=1)
            elif self.schedule == ScheduleFrequency.MINUTELY:
                self.next_run_time = base_time + datetime.timedelta(minutes=1)
            else:
                # Default to daily if unknown frequency
                self.next_run_time = base_time + datetime.timedelta(days=1)
        
        return self.next_run_time
    
    def is_due(self, current_time: Optional[datetime.datetime] = None, 
              tolerance_seconds: float = 1.0) -> bool:
        """
        Checks if the task is due for execution.
        
        Args:
            current_time: Current time, defaults to now
            tolerance_seconds: Tolerance in seconds
            
        Returns:
            True if the task is due, False otherwise
        """
        if not self.enabled:
            return False
        
        return is_time_to_run(self.next_run_time, current_time, tolerance_seconds)
    
    def mark_executed(self, result: TaskResult) -> None:
        """
        Marks the task as executed and updates statistics.
        
        Args:
            result: Result of the task execution
        """
        self.last_run_time = datetime.datetime.now()
        self.last_result = result
        self.run_count += 1
        self.update_next_run_time(self.last_run_time)
        
        logger.info(f"Task '{self.name}' executed with status {result.status.name}, "
                   f"next run scheduled at {self.next_run_time}")
    
    def enable(self) -> None:
        """Enables the scheduled task."""
        self.enabled = True
        logger.info(f"Task '{self.name}' enabled")
    
    def disable(self) -> None:
        """Disables the scheduled task."""
        self.enabled = False
        logger.info(f"Task '{self.name}' disabled")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the scheduled task to a dictionary representation.
        
        Returns:
            Dictionary representation of the scheduled task
        """
        return {
            "id": self.id,
            "name": self.name,
            "task_id": self.task.id,
            "schedule": self.schedule.value if isinstance(self.schedule, ScheduleFrequency) else self.schedule,
            "next_run_time": self.next_run_time.isoformat() if self.next_run_time else None,
            "enabled": self.enabled,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "run_count": self.run_count,
            "metadata": self.metadata,
            "task": self.task.to_dict()
        }


class ErrorRecoveryHandler:
    """Simple error recovery handler for scheduler operations."""
    
    def __init__(self):
        """Initialize a new ErrorRecoveryHandler."""
        self._error_counts = {}  # Track error counts by task ID
        self._last_errors = {}  # Track timestamps of last errors by task ID
        self._recovery_strategies = {}  # Strategies for different error types
        
        # Set up default recovery strategies
        self._setup_default_strategies()
    
    def _setup_default_strategies(self) -> None:
        """Sets up default recovery strategies for common error types."""
        # For connection errors, retry with backoff
        self._recovery_strategies[ConnectionError] = {
            "retryable": True,
            "max_retries": 3,
            "backoff_factor": 2.0,
            "log_level": "WARNING"
        }
        
        # For timeout errors, retry with longer timeout
        self._recovery_strategies[TimeoutError] = {
            "retryable": True,
            "max_retries": 2,
            "increase_timeout": True,
            "log_level": "WARNING"
        }
        
        # For data format errors, don't retry
        self._recovery_strategies[ValueError] = {
            "retryable": False,
            "log_level": "ERROR",
            "fallback": "use_previous_result"
        }
    
    def recover_from_error(self, error: Exception, task_id: str, context: Dict[str, Any]) -> bool:
        """
        Attempt to recover from an error during task execution.
        
        Args:
            error: The exception that occurred
            task_id: ID of the task that failed
            context: Additional context about the task
            
        Returns:
            True if recovery was successful, False otherwise
        """
        # Log the error
        logger.error(f"Error in task {task_id}: {error}", extra={"context": context})
        
        # Update error statistics
        if task_id not in self._error_counts:
            self._error_counts[task_id] = 0
        self._error_counts[task_id] += 1
        self._last_errors[task_id] = datetime.datetime.now()
        
        # Determine error type and find appropriate strategy
        error_type = type(error)
        strategy = None
        
        # Look for a matching strategy based on error type or its base classes
        for err_type, strat in self._recovery_strategies.items():
            if isinstance(error, err_type):
                strategy = strat
                break
        
        # If no specific strategy found, use a default strategy
        if strategy is None:
            from ..utils.error_handling import is_retryable_error
            retryable = is_retryable_error(error)
            strategy = {
                "retryable": retryable,
                "max_retries": 1 if retryable else 0,
                "log_level": "WARNING" if retryable else "ERROR"
            }
        
        # Check if we've exceeded max retries
        if self._error_counts[task_id] > strategy.get("max_retries", 0):
            logger.error(f"Max retries exceeded for task {task_id}")
            
            # If there's a fallback strategy, suggest it
            if "fallback" in strategy:
                logger.info(f"Suggesting fallback strategy: {strategy['fallback']}")
                return False
            
            return False  # Don't retry
        
        # If the error is retryable, return True to indicate retry
        if strategy.get("retryable", False):
            logger.warning(f"Retrying task {task_id} ({self._error_counts[task_id]}/{strategy.get('max_retries', 0)})")
            return True
        
        return False  # Don't retry by default
    
    def register_recovery_strategy(self, error_type: Type[Exception], strategy: Dict[str, Any]) -> None:
        """
        Register a custom recovery strategy for an error type.
        
        Args:
            error_type: Type of exception
            strategy: Strategy dictionary with recovery options
        """
        self._recovery_strategies[error_type] = strategy
        logger.info(f"Registered recovery strategy for {error_type.__name__}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics for monitoring.
        
        Returns:
            Dictionary with error statistics
        """
        stats = {
            "error_counts": self._error_counts.copy(),
            "last_errors": {task_id: ts.isoformat() for task_id, ts in self._last_errors.items()},
            "total_errors": sum(self._error_counts.values()),
            "tasks_with_errors": len(self._error_counts)
        }
        
        return stats
    
    def reset_error_stats(self, task_id: Optional[str] = None) -> None:
        """
        Reset error statistics.
        
        Args:
            task_id: Optional task ID to reset stats for just one task
        """
        if task_id:
            if task_id in self._error_counts:
                del self._error_counts[task_id]
            if task_id in self._last_errors:
                del self._last_errors[task_id]
            logger.debug(f"Reset error statistics for task {task_id}")
        else:
            self._error_counts = {}
            self._last_errors = {}
            logger.debug("Reset all error statistics")


class Scheduler:
    """Main scheduler class for managing and executing scheduled tasks."""
    
    def __init__(self, 
                task_manager: Optional[TaskManager] = None,
                recovery_handler: Optional[Any] = None,
                check_interval: float = DEFAULT_CHECK_INTERVAL,
                timezone: str = DEFAULT_TIMEZONE):
        """
        Initialize a new Scheduler instance.
        
        Args:
            task_manager: TaskManager instance, created if not provided
            recovery_handler: Error recovery handler, created if not provided
            check_interval: Interval in seconds to check for due tasks
            timezone: Timezone for schedule calculations
        """
        self._tasks = {}  # Dictionary of scheduled tasks by ID
        self._task_manager = task_manager or TaskManager()
        self._recovery_handler = recovery_handler or ErrorRecoveryHandler()
        self._check_interval = check_interval
        self._timezone = timezone
        
        # Thread management
        self._scheduler_thread = None
        self._stop_event = threading.Event()
        self._running = False
        
        # Performance tracking
        self._performance_logger = PerformanceLogger()
    
    def add_task(self, 
                task: Task, 
                schedule: Union[str, ScheduleFrequency],
                next_run_time: Optional[datetime.datetime] = None,
                enabled: bool = True,
                metadata: Dict[str, Any] = None) -> ScheduledTask:
        """
        Adds a scheduled task to the scheduler.
        
        Args:
            task: Task to schedule
            schedule: Cron expression or ScheduleFrequency
            next_run_time: Next execution time, calculated if not provided
            enabled: Whether the task is enabled
            metadata: Additional metadata about the scheduled task
            
        Returns:
            The added scheduled task
        """
        # First add the task to the task manager
        self._task_manager.add_task(task)
        
        # Create a scheduled task
        scheduled_task = ScheduledTask(
            task=task, 
            schedule=schedule,
            next_run_time=next_run_time,
            enabled=enabled,
            metadata=metadata or {}
        )
        
        # Add to the tasks dictionary
        self._tasks[scheduled_task.id] = scheduled_task
        
        logger.info(f"Added scheduled task '{scheduled_task.name}' with ID {scheduled_task.id}, "
                   f"next run at {scheduled_task.next_run_time}")
        
        return scheduled_task
    
    def remove_task(self, task_id: str) -> bool:
        """
        Removes a scheduled task from the scheduler.
        
        Args:
            task_id: ID of the task to remove
            
        Returns:
            True if task was removed, False if not found
        """
        if task_id in self._tasks:
            task = self._tasks[task_id]
            del self._tasks[task_id]
            logger.info(f"Removed scheduled task '{task.name}' with ID {task_id}")
            return True
        
        logger.warning(f"Task with ID {task_id} not found")
        return False
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """
        Retrieves a scheduled task by ID.
        
        Args:
            task_id: ID of the task to retrieve
            
        Returns:
            The scheduled task if found, None otherwise
        """
        return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, ScheduledTask]:
        """
        Retrieves all scheduled tasks.
        
        Returns:
            Dictionary of all scheduled tasks
        """
        return self._tasks.copy()
    
    def get_due_tasks(self, 
                     current_time: Optional[datetime.datetime] = None,
                     tolerance_seconds: float = 1.0) -> List[ScheduledTask]:
        """
        Retrieves all tasks that are due for execution.
        
        Args:
            current_time: Current time, defaults to now
            tolerance_seconds: Tolerance in seconds
            
        Returns:
            List of tasks due for execution
        """
        if current_time is None:
            current_time = datetime.datetime.now()
        
        due_tasks = []
        
        for task_id, task in self._tasks.items():
            if task.is_due(current_time, tolerance_seconds):
                due_tasks.append(task)
        
        return due_tasks
    
    @log_execution_time(logger, 'INFO')
    def run_task_now(self, task_id: str, with_retry: bool = True) -> TaskResult:
        """
        Executes a scheduled task immediately.
        
        Args:
            task_id: ID of the task to execute
            with_retry: Whether to use retry logic
            
        Returns:
            Result of the task execution
            
        Raises:
            ValueError: If the task is not found
        """
        # Get the scheduled task
        scheduled_task = self.get_task(task_id)
        if scheduled_task is None:
            raise ValueError(f"Task with ID {task_id} not found")
        
        logger.info(f"Executing task '{scheduled_task.name}' (ID: {task_id}) immediately")
        
        # Start timer for performance tracking
        self._performance_logger.start_timer(scheduled_task.name, "task_execution")
        
        # Execute the task with or without retry
        if with_retry:
            result = self._task_manager.execute_task_with_retry(scheduled_task.task.id)
        else:
            result = self._task_manager.execute_task(scheduled_task.task.id)
        
        # Stop timer and record metrics
        execution_time = self._performance_logger.stop_timer(scheduled_task.name, "task_execution")
        
        # Update the scheduled task
        scheduled_task.mark_executed(result)
        
        return result
    
    def _handle_task_error(self, error: Exception, task: ScheduledTask) -> bool:
        """
        Handles errors during task execution.
        
        Args:
            error: The exception that occurred
            task: The scheduled task that failed
            
        Returns:
            True if recovery was successful, False otherwise
        """
        # Create error context
        context = {
            "scheduler": self.__class__.__name__,
            "task_name": task.name,
            "task_id": task.id,
            "schedule": task.schedule.value if isinstance(task.schedule, ScheduleFrequency) else task.schedule,
            "run_count": task.run_count,
            "last_run_time": task.last_run_time.isoformat() if task.last_run_time else None,
            "metadata": task.metadata
        }
        
        # Use the recovery handler if available
        if self._recovery_handler:
            return self._recovery_handler.recover_from_error(error, task.id, context)
        
        # Otherwise, use a simple error handling approach
        logger.error(f"Error executing task '{task.name}': {error}", extra={"context": context})
        
        from ..utils.error_handling import is_retryable_error
        if is_retryable_error(error):
            logger.warning(f"Error is retryable, will retry task '{task.name}'")
            return True  # Retry
        else:
            logger.critical(f"Critical error for task '{task.name}', cannot recover automatically")
            return False  # Don't retry
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop that checks for and executes due tasks."""
        logger.info("Scheduler loop started")
        
        while not self._stop_event.is_set():
            try:
                # Get tasks that are due for execution
                due_tasks = self.get_due_tasks()
                
                if due_tasks:
                    logger.info(f"Found {len(due_tasks)} tasks due for execution")
                
                # Execute each due task
                for task in due_tasks:
                    try:
                        self.run_task_now(task.id, with_retry=True)
                    except Exception as e:
                        logger.error(f"Error executing task '{task.name}': {e}")
                        self._handle_task_error(e, task)
                
                # Sleep for the check interval
                time.sleep(self._check_interval)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                # Sleep briefly to avoid tight loop in case of persistent errors
                time.sleep(1.0)
        
        logger.info("Scheduler loop stopped")
    
    def start(self) -> bool:
        """
        Starts the scheduler in a background thread.
        
        Returns:
            True if started successfully, False if already running
        """
        if self._running:
            logger.warning("Scheduler is already running")
            return False
        
        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self._scheduler_thread.daemon = True
        self._scheduler_thread.start()
        self._running = True
        
        logger.info("Scheduler started")
        return True
    
    def stop(self, wait: bool = True) -> bool:
        """
        Stops the scheduler.
        
        Args:
            wait: Whether to wait for the scheduler thread to terminate
            
        Returns:
            True if stopped successfully, False if not running
        """
        if not self._running:
            logger.warning("Scheduler is not running")
            return False
        
        self._stop_event.set()
        
        if wait and self._scheduler_thread:
            self._scheduler_thread.join()
        
        self._running = False
        logger.info("Scheduler stopped")
        return True
    
    def is_running(self) -> bool:
        """
        Checks if the scheduler is currently running.
        
        Returns:
            True if running, False otherwise
        """
        return self._running
    
    def add_data_fetch_schedule(self, 
                              fetch_function: Callable, 
                              fetch_params: Dict[str, Any],
                              schedule: Union[str, ScheduleFrequency],
                              task_name: Optional[str] = None,
                              next_run_time: Optional[datetime.datetime] = None) -> ScheduledTask:
        """
        Adds a scheduled data fetching task.
        
        Args:
            fetch_function: Function to call for data fetching
            fetch_params: Parameters for the fetch function
            schedule: Cron expression or ScheduleFrequency
            task_name: Optional task name
            next_run_time: Next execution time, calculated if not provided
            
        Returns:
            The scheduled task
        """
        task = create_data_fetch_task(fetch_function, fetch_params, task_name)
        return self.add_task(task, schedule, next_run_time, True, {"type": "data_fetch"})
    
    def add_inference_schedule(self, 
                             inference_function: Callable, 
                             inference_params: Dict[str, Any],
                             schedule: Union[str, ScheduleFrequency],
                             task_name: Optional[str] = None,
                             next_run_time: Optional[datetime.datetime] = None) -> ScheduledTask:
        """
        Adds a scheduled inference task.
        
        Args:
            inference_function: Function to call for inference
            inference_params: Parameters for the inference function
            schedule: Cron expression or ScheduleFrequency
            task_name: Optional task name
            next_run_time: Next execution time, calculated if not provided
            
        Returns:
            The scheduled task
        """
        task = create_inference_task(inference_function, inference_params, task_name)
        return self.add_task(task, schedule, next_run_time, True, {"type": "inference"})
    
    def add_training_schedule(self, 
                            training_function: Callable, 
                            training_params: Dict[str, Any],
                            schedule: Union[str, ScheduleFrequency],
                            task_name: Optional[str] = None,
                            next_run_time: Optional[datetime.datetime] = None) -> ScheduledTask:
        """
        Adds a scheduled model training task.
        
        Args:
            training_function: Function to call for model training
            training_params: Parameters for the training function
            schedule: Cron expression or ScheduleFrequency
            task_name: Optional task name
            next_run_time: Next execution time, calculated if not provided
            
        Returns:
            The scheduled task
        """
        task = create_training_task(training_function, training_params, task_name)
        return self.add_task(task, schedule, next_run_time, True, {"type": "training"})
    
    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Retrieves performance metrics for scheduler operations.
        
        Returns:
            Dictionary of performance metrics
        """
        return self._performance_logger.get_average_metrics()
    
    def reset(self) -> None:
        """Resets the scheduler state."""
        if self._running:
            self.stop()
        
        self._tasks = {}
        self._performance_logger.reset_metrics()
        logger.info("Scheduler reset")


class DailyScheduler(Scheduler):
    """Specialized scheduler for daily operations with predefined schedules."""
    
    def __init__(self, 
                task_manager: Optional[TaskManager] = None,
                recovery_handler: Optional[Any] = None,
                inference_cron: str = "0 6 * * *",  # 6:00 AM every day
                training_cron: str = "0 1 */2 * *",  # 1:00 AM every 2 days
                data_fetch_cron: str = "0 0 * * *"):  # Midnight every day
        """
        Initialize a new DailyScheduler instance.
        
        Args:
            task_manager: TaskManager instance, created if not provided
            recovery_handler: Error recovery handler, created if not provided
            inference_cron: Cron expression for inference schedule
            training_cron: Cron expression for training schedule
            data_fetch_cron: Cron expression for data fetching schedule
        """
        super().__init__(task_manager, recovery_handler)
        
        self._daily_tasks = {}
        self._inference_cron = inference_cron
        self._training_cron = training_cron
        self._data_fetch_cron = data_fetch_cron
    
    def setup_daily_inference(self, 
                            inference_function: Callable, 
                            inference_params: Dict[str, Any],
                            task_name: Optional[str] = None) -> ScheduledTask:
        """
        Sets up the daily inference schedule.
        
        Args:
            inference_function: Function to call for inference
            inference_params: Parameters for the inference function
            task_name: Optional task name
            
        Returns:
            The scheduled inference task
        """
        if task_name is None:
            task_name = "daily_inference"
        
        task = create_inference_task(inference_function, inference_params, task_name)
        scheduled_task = self.add_task(task, self._inference_cron, None, True, {"type": "inference", "recurrence": "daily"})
        
        self._daily_tasks["inference"] = scheduled_task
        return scheduled_task
    
    def setup_bidaily_retraining(self, 
                               training_function: Callable, 
                               training_params: Dict[str, Any],
                               task_name: Optional[str] = None) -> ScheduledTask:
        """
        Sets up the bi-daily model retraining schedule.
        
        Args:
            training_function: Function to call for model training
            training_params: Parameters for the training function
            task_name: Optional task name
            
        Returns:
            The scheduled training task
        """
        if task_name is None:
            task_name = "bidaily_training"
        
        task = create_training_task(training_function, training_params, task_name)
        scheduled_task = self.add_task(task, self._training_cron, None, True, {"type": "training", "recurrence": "bidaily"})
        
        self._daily_tasks["training"] = scheduled_task
        return scheduled_task
    
    def setup_data_fetching(self, 
                          fetch_function: Callable, 
                          fetch_params: Dict[str, Any],
                          task_name: Optional[str] = None) -> ScheduledTask:
        """
        Sets up the data fetching schedule.
        
        Args:
            fetch_function: Function to call for data fetching
            fetch_params: Parameters for the fetch function
            task_name: Optional task name
            
        Returns:
            The scheduled data fetch task
        """
        if task_name is None:
            task_name = "daily_data_fetch"
        
        task = create_data_fetch_task(fetch_function, fetch_params, task_name)
        scheduled_task = self.add_task(task, self._data_fetch_cron, None, True, {"type": "data_fetch", "recurrence": "daily"})
        
        self._daily_tasks["data_fetch"] = scheduled_task
        return scheduled_task
    
    def start_all(self) -> bool:
        """
        Starts all daily scheduled tasks.
        
        Returns:
            True if started successfully
        """
        # Make sure all required tasks are set up
        required_tasks = ["inference", "training", "data_fetch"]
        for task_name in required_tasks:
            if task_name not in self._daily_tasks:
                logger.warning(f"Required daily task '{task_name}' has not been set up")
        
        # Start the scheduler
        result = self.start()
        logger.info("All daily scheduled tasks started")
        
        return result
    
    def get_daily_task(self, task_name: str) -> Optional[ScheduledTask]:
        """
        Retrieves a daily task by name.
        
        Args:
            task_name: Name of the task to retrieve
            
        Returns:
            The scheduled task if found, None otherwise
        """
        return self._daily_tasks.get(task_name)
    
    def update_schedule(self, task_name: str, new_cron: str) -> bool:
        """
        Updates the schedule for a daily task.
        
        Args:
            task_name: Name of the task to update
            new_cron: New cron expression
            
        Returns:
            True if updated successfully, False if task not found
        """
        # Get the task
        task = self.get_daily_task(task_name)
        if task is None:
            logger.warning(f"Daily task '{task_name}' not found")
            return False
        
        # Update the schedule
        task.schedule = new_cron
        task.next_run_time = get_next_run_time(new_cron)
        
        logger.info(f"Updated schedule for task '{task_name}' to '{new_cron}', "
                   f"next run at {task.next_run_time}")
        
        return True