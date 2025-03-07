"""
Core task management module for the ERCOT RTLMP spike prediction system.

This module provides abstractions for defining, executing, and managing tasks
with dependencies, priorities, and retry capabilities. Serves as the foundation
for orchestrating the various components of the prediction pipeline.
"""

import typing
from typing import Dict, List, Optional, Callable, Any, Tuple, Union, Set, TypeVar, Generic, cast
import enum
import uuid
import datetime
import time
import functools
import concurrent.futures

from ..utils.logging import get_logger, log_execution_time, PerformanceLogger
from ..utils.error_handling import retry_with_backoff, handle_errors, ErrorHandler
from ..utils.type_definitions import DataFrameType, ModelType, PathType

# Set up logger
logger = get_logger(__name__)

# Default values
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_TIMEOUT = 3600  # 1 hour in seconds

# Type variable for generic result type
T = TypeVar('T')


def create_task_id() -> str:
    """
    Generates a unique ID for a task.
    
    Returns:
        str: Unique task ID
    """
    return str(uuid.uuid4())


def format_task_result(result: 'TaskResult') -> str:
    """
    Formats a task result for logging and display.
    
    Args:
        result: Task execution result
        
    Returns:
        str: Formatted task result string
    """
    task_str = f"Task '{result.task_name}' (ID: {result.task_id}) - Status: {result.status.name}"
    time_str = f"Execution time: {result.execution_time:.2f}s"
    
    if result.error:
        error_str = f"Error: {type(result.error).__name__}: {str(result.error)}"
        return f"{task_str} - {time_str} - {error_str}"
    return f"{task_str} - {time_str}"


@handle_errors(concurrent.futures.TimeoutError, None, True, 'Task execution timed out')
def execute_with_timeout(func: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any], 
                        timeout: Optional[float]) -> Any:
    """
    Executes a function with a timeout.
    
    Args:
        func: Function to execute
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        timeout: Maximum execution time in seconds, or None for no timeout
        
    Returns:
        Result of the function execution
        
    Raises:
        TimeoutError: If the function execution exceeds the timeout
    """
    if timeout is None:
        return func(*args, **kwargs)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            # Cancel the future if possible
            future.cancel()
            raise


def resolve_dependencies(tasks: Dict[str, 'Task']) -> List[str]:
    """
    Resolves task dependencies to determine execution order.
    
    Args:
        tasks: Dictionary of tasks keyed by task ID
        
    Returns:
        List of task IDs in dependency-resolved order
        
    Raises:
        ValueError: If circular dependencies are detected
    """
    # Set to track visited nodes (permanent marks)
    visited = set()
    
    # Set to track nodes in current recursion stack (temporary marks)
    temp_marks = set()
    
    # List to store the result in reverse order
    result = []
    
    def visit(task_id: str) -> None:
        """
        Depth-first search for topological sorting.
        
        Args:
            task_id: ID of the task to visit
            
        Raises:
            ValueError: If circular dependencies are detected
        """
        if task_id in visited:
            return
        
        if task_id in temp_marks:
            raise ValueError(f"Circular dependency detected involving task {task_id}")
        
        temp_marks.add(task_id)
        
        task = tasks[task_id]
        for dep_id in task.dependencies:
            if dep_id in tasks:
                visit(dep_id)
        
        temp_marks.remove(task_id)
        visited.add(task_id)
        result.append(task_id)
    
    # Visit all tasks
    for task_id in tasks:
        if task_id not in visited:
            visit(task_id)
    
    # Return the result in correct order (reverse the DFS result)
    return result[::-1]


class TaskStatus(enum.Enum):
    """Enum representing the status of a task."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    TIMEOUT = "TIMEOUT"


class TaskPriority(enum.Enum):
    """Enum representing the priority of a task."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class TaskResult(Generic[T]):
    """Class representing the result of a task execution."""
    
    def __init__(
        self,
        task_id: str,
        task_name: str,
        status: TaskStatus,
        result: Optional[T] = None,
        error: Optional[Exception] = None,
        execution_time: float = 0.0,
        start_time: datetime.datetime = datetime.datetime.now(),
        end_time: datetime.datetime = datetime.datetime.now(),
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new TaskResult instance.
        
        Args:
            task_id: Unique ID of the task
            task_name: Name of the task
            status: Execution status of the task
            result: Task execution result, if any
            error: Exception raised during execution, if any
            execution_time: Time taken to execute the task in seconds
            start_time: When task execution started
            end_time: When task execution ended
            metadata: Additional metadata about the task execution
        """
        self.task_id = task_id
        self.task_name = task_name
        self.status = status
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.start_time = start_time
        self.end_time = end_time
        self.metadata = metadata or {}
    
    def is_successful(self) -> bool:
        """
        Checks if the task execution was successful.
        
        Returns:
            bool: True if the task completed successfully, False otherwise
        """
        return self.status == TaskStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the task result to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the task result
        """
        result_dict = {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "status": self.status.name,
            "execution_time": self.execution_time,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "metadata": self.metadata
        }
        
        if self.result is not None:
            result_dict["result"] = self.result
        
        if self.error is not None:
            result_dict["error"] = {
                "type": type(self.error).__name__,
                "message": str(self.error)
            }
        
        return result_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskResult':
        """
        Creates a TaskResult instance from a dictionary.
        
        Args:
            data: Dictionary containing task result data
            
        Returns:
            TaskResult: TaskResult instance created from the dictionary
        """
        task_id = data["task_id"]
        task_name = data["task_name"]
        status = TaskStatus[data["status"]]
        
        result = data.get("result")
        error = data.get("error")
        if error:
            error = Exception(f"{error['type']}: {error['message']}")
        
        execution_time = data["execution_time"]
        start_time = datetime.datetime.fromisoformat(data["start_time"])
        end_time = datetime.datetime.fromisoformat(data["end_time"])
        metadata = data.get("metadata", {})
        
        return cls(
            task_id=task_id,
            task_name=task_name,
            status=status,
            result=result,
            error=error,
            execution_time=execution_time,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata
        )


class Task(Generic[T]):
    """Class representing a task to be executed."""
    
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        task_id: Optional[str] = None,
        args: Tuple[Any, ...] = (),
        kwargs: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        dependencies: Set[str] = None,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a new Task instance.
        
        Args:
            func: Function to be executed as the task
            name: Name of the task (defaults to function name)
            task_id: Unique ID for the task (auto-generated if None)
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            priority: Priority level of the task
            dependencies: Set of task IDs that this task depends on
            timeout: Maximum execution time in seconds, or None for no timeout
            max_retries: Maximum number of retry attempts for the task
            retry_delay: Initial delay between retries in seconds
            backoff_factor: Factor to increase delay between retries
            metadata: Additional metadata about the task
        """
        self.id = task_id or create_task_id()
        self.name = name or func.__name__
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.status = TaskStatus.PENDING
        self.priority = priority
        self.dependencies = dependencies or set()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.metadata = metadata or {}
    
    def add_dependency(self, dependency: Union[str, 'Task']) -> None:
        """
        Adds a dependency to the task.
        
        Args:
            dependency: Task ID or Task object to add as a dependency
        """
        if isinstance(dependency, Task):
            self.dependencies.add(dependency.id)
        else:
            self.dependencies.add(dependency)
    
    def remove_dependency(self, dependency: Union[str, 'Task']) -> bool:
        """
        Removes a dependency from the task.
        
        Args:
            dependency: Task ID or Task object to remove
            
        Returns:
            bool: True if dependency was removed, False if not found
        """
        dep_id = dependency.id if isinstance(dependency, Task) else dependency
        
        if dep_id in self.dependencies:
            self.dependencies.remove(dep_id)
            return True
        return False
    
    def set_status(self, status: TaskStatus) -> None:
        """
        Sets the status of the task.
        
        Args:
            status: New status for the task
        """
        self.status = status
        logger.debug(f"Task '{self.name}' (ID: {self.id}) status changed to {status.name}")
    
    @log_execution_time(logger, 'INFO')
    def execute(self) -> TaskResult[T]:
        """
        Executes the task function with the specified arguments.
        
        Returns:
            TaskResult[T]: Result of the task execution
        """
        start_time = datetime.datetime.now()
        self.set_status(TaskStatus.RUNNING)
        
        try:
            result = execute_with_timeout(self.func, self.args, self.kwargs, self.timeout)
            self.set_status(TaskStatus.COMPLETED)
        except Exception as e:
            self.set_status(TaskStatus.FAILED)
            logger.error(f"Task '{self.name}' failed with error: {e}")
            end_time = datetime.datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return TaskResult(
                task_id=self.id,
                task_name=self.name,
                status=self.status,
                error=e,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                metadata=self.metadata
            )
        
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        return TaskResult(
            task_id=self.id,
            task_name=self.name,
            status=self.status,
            result=result,
            execution_time=execution_time,
            start_time=start_time,
            end_time=end_time,
            metadata=self.metadata
        )
    
    @log_execution_time(logger, 'INFO')
    def execute_with_retry(self) -> TaskResult[T]:
        """
        Executes the task with retry logic for handling transient failures.
        
        Returns:
            TaskResult[T]: Result of the task execution
        """
        retry_count = 0
        delay = self.retry_delay
        
        start_time = datetime.datetime.now()
        self.set_status(TaskStatus.RUNNING)
        
        result = None
        error = None
        
        while retry_count <= self.max_retries:
            try:
                result = execute_with_timeout(self.func, self.args, self.kwargs, self.timeout)
                self.set_status(TaskStatus.COMPLETED)
                break
            except Exception as e:
                retry_count += 1
                error = e
                
                if retry_count > self.max_retries:
                    self.set_status(TaskStatus.FAILED)
                    logger.error(f"Task '{self.name}' failed after {retry_count-1} retries with error: {e}")
                    break
                
                logger.warning(f"Task '{self.name}' failed, retrying ({retry_count}/{self.max_retries}): {e}")
                time.sleep(delay)
                delay *= self.backoff_factor
        
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        return TaskResult(
            task_id=self.id,
            task_name=self.name,
            status=self.status,
            result=result,
            error=error,
            execution_time=execution_time,
            start_time=start_time,
            end_time=end_time,
            metadata=self.metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the task to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the task
        """
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.name,
            "priority": self.priority.name,
            "dependencies": list(self.dependencies),
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "backoff_factor": self.backoff_factor,
            "metadata": self.metadata
        }


class TaskManager:
    """Class for managing and executing tasks with dependencies."""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize a new TaskManager instance.
        
        Args:
            max_workers: Maximum number of worker threads for parallel execution
        """
        self._tasks: Dict[str, Task] = {}
        self._results: Dict[str, TaskResult] = {}
        self._performance_logger = PerformanceLogger()
        self._max_workers = max_workers
    
    def add_task(self, task: Task) -> str:
        """
        Adds a task to the manager.
        
        Args:
            task: Task to add
            
        Returns:
            str: ID of the added task
        """
        self._tasks[task.id] = task
        logger.debug(f"Added task '{task.name}' with ID {task.id}")
        return task.id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Retrieves a task by ID.
        
        Args:
            task_id: ID of the task to retrieve
            
        Returns:
            Optional[Task]: The task if found, None otherwise
        """
        return self._tasks.get(task_id)
    
    def remove_task(self, task_id: str) -> bool:
        """
        Removes a task from the manager.
        
        Args:
            task_id: ID of the task to remove
            
        Returns:
            bool: True if task was removed, False if not found
        """
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False
    
    def get_result(self, task_id: str) -> Optional[TaskResult]:
        """
        Retrieves the result of a task by ID.
        
        Args:
            task_id: ID of the task result to retrieve
            
        Returns:
            Optional[TaskResult]: The task result if found, None otherwise
        """
        return self._results.get(task_id)
    
    @log_execution_time(logger, 'INFO')
    def execute_task(self, task_id: str) -> TaskResult:
        """
        Executes a single task by ID.
        
        Args:
            task_id: ID of the task to execute
            
        Returns:
            TaskResult: Result of the task execution
            
        Raises:
            ValueError: If the task with the specified ID is not found
        """
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task with ID {task_id} not found")
        
        # Check if all dependencies have been successfully executed
        for dep_id in task.dependencies:
            dep_result = self.get_result(dep_id)
            if not dep_result or not dep_result.is_successful():
                logger.warning(f"Dependency {dep_id} for task {task_id} has not completed successfully")
                
                # Set the task as skipped due to dependency failure
                task.set_status(TaskStatus.SKIPPED)
                result = TaskResult(
                    task_id=task.id,
                    task_name=task.name,
                    status=TaskStatus.SKIPPED,
                    error=Exception(f"Dependency {dep_id} failed or was not executed"),
                    execution_time=0.0,
                    start_time=datetime.datetime.now(),
                    end_time=datetime.datetime.now(),
                    metadata=task.metadata
                )
                
                self._results[task_id] = result
                return result
        
        # Execute the task
        self._performance_logger.start_timer(task.name, category="task_execution")
        result = task.execute()
        self._performance_logger.stop_timer(task.name, category="task_execution")
        
        # Store the result
        self._results[task_id] = result
        
        # Log the result
        logger.info(format_task_result(result))
        
        return result
    
    @log_execution_time(logger, 'INFO')
    def execute_task_with_retry(self, task_id: str) -> TaskResult:
        """
        Executes a single task with retry logic.
        
        Args:
            task_id: ID of the task to execute
            
        Returns:
            TaskResult: Result of the task execution
            
        Raises:
            ValueError: If the task with the specified ID is not found
        """
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task with ID {task_id} not found")
        
        # Check if all dependencies have been successfully executed
        for dep_id in task.dependencies:
            dep_result = self.get_result(dep_id)
            if not dep_result or not dep_result.is_successful():
                logger.warning(f"Dependency {dep_id} for task {task_id} has not completed successfully")
                
                # Set the task as skipped due to dependency failure
                task.set_status(TaskStatus.SKIPPED)
                result = TaskResult(
                    task_id=task.id,
                    task_name=task.name,
                    status=TaskStatus.SKIPPED,
                    error=Exception(f"Dependency {dep_id} failed or was not executed"),
                    execution_time=0.0,
                    start_time=datetime.datetime.now(),
                    end_time=datetime.datetime.now(),
                    metadata=task.metadata
                )
                
                self._results[task_id] = result
                return result
        
        # Execute the task with retry
        self._performance_logger.start_timer(task.name, category="task_execution")
        result = task.execute_with_retry()
        self._performance_logger.stop_timer(task.name, category="task_execution")
        
        # Store the result
        self._results[task_id] = result
        
        # Log the result
        logger.info(format_task_result(result))
        
        return result
    
    @log_execution_time(logger, 'INFO')
    def execute_all(self, parallel: bool = True, with_retry: bool = True) -> Dict[str, TaskResult]:
        """
        Executes all tasks in dependency order.
        
        Args:
            parallel: Whether to execute independent tasks in parallel
            with_retry: Whether to use retry logic for task execution
            
        Returns:
            Dict[str, TaskResult]: Dictionary of task results by task ID
        """
        # Resolve dependencies
        try:
            ordered_task_ids = resolve_dependencies(self._tasks)
        except ValueError as e:
            logger.error(f"Failed to resolve task dependencies: {e}")
            return {}
        
        logger.info(f"Executing {len(ordered_task_ids)} tasks in dependency order")
        
        # Choose execution function based on retry preference
        exec_func = self.execute_task_with_retry if with_retry else self.execute_task
        
        if not parallel:
            # Sequential execution
            for task_id in ordered_task_ids:
                try:
                    exec_func(task_id)
                except Exception as e:
                    logger.error(f"Error executing task {task_id}: {e}")
        else:
            # Parallel execution with dependency constraints
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                # Group tasks by their dependency level
                task_levels: Dict[int, List[str]] = {}
                for task_id in ordered_task_ids:
                    # Calculate the level based on dependencies
                    task = self._tasks[task_id]
                    level = 0
                    for dep_id in task.dependencies:
                        if dep_id in self._tasks:
                            level = max(level, task_levels.get(dep_id, 0) + 1)
                    
                    if level not in task_levels:
                        task_levels[level] = []
                    task_levels[level].append(task_id)
                
                # Execute tasks level by level
                for level in sorted(task_levels.keys()):
                    futures = {}
                    for task_id in task_levels[level]:
                        # Skip tasks whose dependencies failed
                        if any(
                            dep_id in self._results and not self._results[dep_id].is_successful()
                            for dep_id in self._tasks[task_id].dependencies
                        ):
                            self._tasks[task_id].set_status(TaskStatus.SKIPPED)
                            self._results[task_id] = TaskResult(
                                task_id=task_id,
                                task_name=self._tasks[task_id].name,
                                status=TaskStatus.SKIPPED,
                                error=Exception("Dependency failed"),
                                execution_time=0.0,
                                start_time=datetime.datetime.now(),
                                end_time=datetime.datetime.now(),
                                metadata=self._tasks[task_id].metadata
                            )
                            continue
                        
                        futures[executor.submit(exec_func, task_id)] = task_id
                    
                    # Wait for all tasks at this level to complete
                    for future in concurrent.futures.as_completed(futures):
                        task_id = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Error executing task {task_id}: {e}")
        
        # Log overall execution statistics
        success_count = sum(1 for r in self._results.values() if r.is_successful())
        total_count = len(self._results)
        logger.info(f"Task execution completed: {success_count}/{total_count} tasks successful")
        
        return self._results
    
    @log_execution_time(logger, 'INFO')
    def execute_by_priority(self, with_retry: bool = True) -> Dict[str, TaskResult]:
        """
        Executes tasks in order of priority.
        
        Args:
            with_retry: Whether to use retry logic for task execution
            
        Returns:
            Dict[str, TaskResult]: Dictionary of task results by task ID
        """
        # Group tasks by priority
        priority_groups: Dict[TaskPriority, List[str]] = {}
        
        for task_id, task in self._tasks.items():
            if task.priority not in priority_groups:
                priority_groups[task.priority] = []
            priority_groups[task.priority].append(task_id)
        
        # Execute tasks in priority order (CRITICAL -> HIGH -> MEDIUM -> LOW)
        for priority in sorted(priority_groups.keys(), key=lambda p: p.value, reverse=True):
            # Create a sub-manager for this priority group
            sub_manager = TaskManager(max_workers=self._max_workers)
            
            # Add tasks to sub-manager
            for task_id in priority_groups[priority]:
                sub_manager.add_task(self._tasks[task_id])
            
            # Execute tasks in this priority group
            results = sub_manager.execute_all(parallel=True, with_retry=with_retry)
            
            # Merge results
            self._results.update(results)
        
        # Log overall execution statistics
        success_count = sum(1 for r in self._results.values() if r.is_successful())
        total_count = len(self._results)
        logger.info(f"Priority-based execution completed: {success_count}/{total_count} tasks successful")
        
        return self._results
    
    def clear_results(self) -> None:
        """
        Clears all task results.
        """
        self._results.clear()
        logger.debug("Cleared all task results")
    
    def reset(self) -> None:
        """
        Resets the task manager state.
        """
        self._tasks.clear()
        self._results.clear()
        self._performance_logger.reset_metrics()
        logger.debug("Reset task manager state")
    
    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Retrieves performance metrics for task execution.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of performance metrics
        """
        return self._performance_logger.get_average_metrics()
    
    def create_task(
        self,
        func: Callable,
        name: Optional[str] = None,
        args: Tuple[Any, ...] = (),
        kwargs: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        dependencies: Set[str] = None,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        metadata: Dict[str, Any] = None
    ) -> Task:
        """
        Creates and adds a new task to the manager.
        
        Args:
            func: Function to be executed as the task
            name: Name of the task (defaults to function name)
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            priority: Priority level of the task
            dependencies: Set of task IDs that this task depends on
            timeout: Maximum execution time in seconds, or None for no timeout
            max_retries: Maximum number of retry attempts for the task
            retry_delay: Initial delay between retries in seconds
            backoff_factor: Factor to increase delay between retries
            metadata: Additional metadata about the task
            
        Returns:
            Task: The created task
        """
        task = Task(
            func=func,
            name=name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            dependencies=dependencies,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            backoff_factor=backoff_factor,
            metadata=metadata
        )
        
        self.add_task(task)
        return task