"""
Core error recovery module for the ERCOT RTLMP spike prediction system.

Implements strategies for handling and recovering from various types of errors
that occur during pipeline execution, ensuring system resilience and maintaining
forecast availability even in the presence of failures.
"""

import typing
from typing import Dict, List, Optional, Callable, Any, Tuple, Union, Type, Set, Protocol, cast
import enum
import datetime
import time
import functools
import pandas as pd  # version 2.0+

from .task_management import TaskStatus, TaskResult, Task
from ..utils.logging import get_logger, log_execution_time
from ..utils.error_handling import (
    retry_with_backoff, handle_errors, ErrorHandler, RetryContext, is_retryable_error
)
from ..utils.error_handling import (
    BaseError, DataError, ConnectionError, ModelError, InferenceError
)
from ..utils.type_definitions import DataFrameType, ModelType, PathType

# Set up logger
logger = get_logger(__name__)

# Default values
DEFAULT_MAX_RECOVERY_ATTEMPTS = 3
DEFAULT_RECOVERY_DELAY = 5
DEFAULT_BACKOFF_FACTOR = 2.0


class PipelineStage(enum.Enum):
    """Enum representing different stages of the prediction pipeline."""
    DATA_FETCH = "DATA_FETCH"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    MODEL_TRAINING = "MODEL_TRAINING"
    INFERENCE = "INFERENCE"


class RecoveryStrategy(enum.Enum):
    """Enum representing different error recovery strategies."""
    RETRY = "RETRY"             # Retry the operation
    FALLBACK = "FALLBACK"       # Use a fallback option (previous data/model)
    SKIP = "SKIP"               # Skip the operation and continue
    REPAIR = "REPAIR"           # Attempt to repair the data/model
    RESTART = "RESTART"         # Restart the component


def get_error_type_category(error: Exception) -> str:
    """
    Categorizes an error based on its type and attributes.
    
    Args:
        error: Exception to categorize
        
    Returns:
        Category of the error (data, connection, model, inference, system)
    """
    if isinstance(error, DataError) or any(isinstance(error, t) for t in DataError.__subclasses__()):
        return "data"
    elif isinstance(error, ConnectionError):
        return "connection"
    elif isinstance(error, ModelError) or any(isinstance(error, t) for t in ModelError.__subclasses__()):
        return "model"
    elif isinstance(error, InferenceError):
        return "inference"
    
    # For other error types, check the error message and attributes for classification
    error_message = str(error).lower()
    error_type = type(error).__name__.lower()
    
    if any(term in error_message or term in error_type for term in ['data', 'dataframe', 'value', 'format']):
        return "data"
    elif any(term in error_message or term in error_type for term in ['connect', 'http', 'network', 'request', 'api']):
        return "connection"
    elif any(term in error_message or term in error_type for term in ['model', 'predict', 'train']):
        return "model"
    elif any(term in error_message or term in error_type for term in ['inference', 'forecast', 'prediction']):
        return "inference"
    
    # Default to system error
    return "system"


def create_recovery_context(
    error: Exception,
    pipeline_stage: Optional[PipelineStage] = None,
    additional_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Creates a context dictionary for error recovery.
    
    Args:
        error: Exception that occurred
        pipeline_stage: Stage of the pipeline where the error occurred
        additional_context: Additional context information
        
    Returns:
        Recovery context dictionary
    """
    # Create base context
    context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Add error category
    context["error_category"] = get_error_type_category(error)
    
    # Add pipeline stage information if provided
    if pipeline_stage:
        context["pipeline_stage"] = pipeline_stage.value
    
    # Add traceback information if available
    if hasattr(error, "__traceback__") and error.__traceback__:
        import traceback
        context["traceback"] = traceback.format_exception(
            type(error), error, error.__traceback__
        )
    
    # Add retryable flag
    context["retryable"] = is_retryable_error(error)
    
    # Add additional context if provided
    if additional_context:
        context.update(additional_context)
    
    return context


def log_recovery_attempt(
    error: Exception,
    strategy: RecoveryStrategy,
    attempt_number: int,
    context: Dict[str, Any],
    success: bool
) -> None:
    """
    Logs information about a recovery attempt.
    
    Args:
        error: Exception being recovered from
        strategy: Recovery strategy being used
        attempt_number: Current attempt number
        context: Recovery context
        success: Whether the recovery attempt was successful
    """
    # Determine log level based on success and attempt number
    if success:
        level = "INFO"
    elif attempt_number >= DEFAULT_MAX_RECOVERY_ATTEMPTS:
        level = "ERROR"
    else:
        level = "WARNING"
    
    # Get the appropriate logging function
    log_func = getattr(logger, level.lower())
    
    # Format the message
    message = f"Recovery attempt {attempt_number} using {strategy.value} strategy for {type(error).__name__}: {str(error)}"
    if success:
        message += " - SUCCESS"
    else:
        message += " - FAILED"
    
    # Log with context
    log_func(message, extra={"context": context})


def with_recovery(strategy: RecoveryStrategy, recovery_params: Optional[Dict[str, Any]] = None):
    """
    Decorator that applies recovery strategies to a function.
    
    Args:
        strategy: Recovery strategy to use
        recovery_params: Parameters for the recovery strategy
        
    Returns:
        Decorated function with recovery capabilities
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as error:
                # Create recovery context
                context = create_recovery_context(error)
                
                # Apply recovery strategy
                recovery_params_to_use = recovery_params or {}
                recovery_category = context["error_category"]
                recovery_func = get_recovery_function(recovery_category)
                
                try:
                    # Attempt recovery
                    recovery_result = recovery_func(error, context, recovery_params_to_use)
                    
                    # Log the recovery attempt
                    log_recovery_attempt(error, strategy, 1, context, recovery_result is not None)
                    
                    if recovery_result is not None:
                        return recovery_result
                    else:
                        # Recovery failed, re-raise the original error
                        raise
                except Exception:
                    # Recovery itself failed, re-raise the original error
                    raise error
                
        return wrapper
    return decorator


@log_execution_time(logger, 'INFO')
def recover_data(
    error: Exception,
    context: Dict[str, Any],
    recovery_params: Dict[str, Any]
) -> Optional[Any]:
    """
    Recovers from data-related errors.
    
    Args:
        error: Data error to recover from
        context: Recovery context
        recovery_params: Parameters for recovery
        
    Returns:
        Recovered data or None if recovery failed
    """
    error_type = context.get("error_type", "")
    
    if "missing" in error_type.lower() or "missing" in str(error).lower():
        # Missing data errors - try to use cached data
        if "cached_data" in recovery_params:
            logger.info(f"Recovering from missing data error using cached data")
            return recovery_params["cached_data"]
    
    elif "format" in error_type.lower() or "format" in str(error).lower():
        # Data format errors - try to repair/transform the data
        if "repair_func" in recovery_params and "data" in recovery_params:
            logger.info(f"Attempting to repair data format")
            repair_func = recovery_params["repair_func"]
            return repair_func(recovery_params["data"])
    
    elif "validation" in error_type.lower() or "validation" in str(error).lower():
        # Data validation errors - try to apply defaults or imputation
        if "default_values" in recovery_params and "data" in recovery_params:
            logger.info(f"Applying default values to invalid data")
            data = recovery_params["data"]
            defaults = recovery_params["default_values"]
            
            if isinstance(data, pd.DataFrame):
                for col, value in defaults.items():
                    if col in data.columns:
                        is_null = data[col].isnull()
                        data.loc[is_null, col] = value
                return data
    
    # Last resort - use fallback data if provided
    if "fallback_data" in recovery_params:
        logger.info(f"Using fallback data as last resort")
        return recovery_params["fallback_data"]
    
    # Recovery failed
    logger.error(f"Data recovery failed for error: {error}")
    return None


@log_execution_time(logger, 'INFO')
def recover_connection(
    error: Exception,
    context: Dict[str, Any],
    recovery_params: Dict[str, Any]
) -> bool:
    """
    Recovers from connection-related errors.
    
    Args:
        error: Connection error to recover from
        context: Recovery context
        recovery_params: Parameters for recovery
        
    Returns:
        True if recovery was successful, False otherwise
    """
    # Get retry parameters
    max_attempts = recovery_params.get("max_attempts", DEFAULT_MAX_RECOVERY_ATTEMPTS)
    delay = recovery_params.get("delay", DEFAULT_RECOVERY_DELAY)
    backoff_factor = recovery_params.get("backoff_factor", DEFAULT_BACKOFF_FACTOR)
    
    # Implement exponential backoff retry
    for attempt in range(1, max_attempts + 1):
        wait_time = delay * (backoff_factor ** (attempt - 1))
        logger.info(f"Connection recovery attempt {attempt}/{max_attempts} - waiting {wait_time} seconds")
        
        time.sleep(wait_time)
        
        # Attempt to re-establish connection
        if "connect_func" in recovery_params:
            try:
                connect_func = recovery_params["connect_func"]
                connect_func()
                logger.info(f"Connection re-established successfully after {attempt} attempts")
                return True
            except Exception as e:
                logger.warning(f"Connection recovery attempt {attempt} failed: {e}")
    
    # All retries failed, try fallback endpoint if available
    if "fallback_endpoint" in recovery_params:
        try:
            if "connect_func" in recovery_params:
                connect_func = recovery_params["connect_func"]
                connect_func(endpoint=recovery_params["fallback_endpoint"])
                logger.info(f"Connection established to fallback endpoint")
                return True
        except Exception as e:
            logger.error(f"Fallback connection failed: {e}")
    
    # Recovery failed
    logger.error(f"Connection recovery failed after {max_attempts} attempts")
    return False


@log_execution_time(logger, 'INFO')
def recover_model(
    error: Exception,
    context: Dict[str, Any],
    recovery_params: Dict[str, Any]
) -> Optional[ModelType]:
    """
    Recovers from model-related errors.
    
    Args:
        error: Model error to recover from
        context: Recovery context
        recovery_params: Parameters for recovery
        
    Returns:
        Recovered model or None if recovery failed
    """
    error_type = context.get("error_type", "")
    
    if "load" in error_type.lower() or "load" in str(error).lower():
        # Model loading errors - try to load a previous version
        if "previous_model_paths" in recovery_params:
            previous_paths = recovery_params["previous_model_paths"]
            load_func = recovery_params.get("load_func")
            
            if load_func and previous_paths:
                for path in previous_paths:
                    try:
                        logger.info(f"Attempting to load previous model from {path}")
                        model = load_func(path)
                        return model
                    except Exception as e:
                        logger.warning(f"Failed to load previous model from {path}: {e}")
    
    elif "training" in error_type.lower() or "train" in str(error).lower():
        # Model training errors - try to use a simpler model configuration
        if "simple_config" in recovery_params and "train_func" in recovery_params:
            try:
                logger.info(f"Attempting to train model with simplified configuration")
                simple_config = recovery_params["simple_config"]
                train_func = recovery_params["train_func"]
                model = train_func(config=simple_config)
                return model
            except Exception as e:
                logger.warning(f"Failed to train model with simplified configuration: {e}")
    
    # Last resort - use fallback model if provided
    if "fallback_model_path" in recovery_params and "load_func" in recovery_params:
        try:
            logger.info(f"Using fallback model as last resort")
            load_func = recovery_params["load_func"]
            model = load_func(recovery_params["fallback_model_path"])
            return model
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
    
    # Recovery failed
    logger.error(f"Model recovery failed for error: {error}")
    return None


@log_execution_time(logger, 'INFO')
def recover_inference(
    error: Exception,
    context: Dict[str, Any],
    recovery_params: Dict[str, Any]
) -> Optional[DataFrameType]:
    """
    Recovers from inference-related errors.
    
    Args:
        error: Inference error to recover from
        context: Recovery context
        recovery_params: Parameters for recovery
        
    Returns:
        Recovered forecast or None if recovery failed
    """
    error_type = context.get("error_type", "")
    
    if "prediction" in error_type.lower() or "prediction" in str(error).lower():
        # Prediction errors - try using a simpler model or fewer features
        if "simple_model" in recovery_params and "features" in recovery_params:
            try:
                logger.info(f"Attempting prediction with simplified model")
                simple_model = recovery_params["simple_model"]
                features = recovery_params["features"]
                return simple_model.predict_proba(features)
            except Exception as e:
                logger.warning(f"Failed prediction with simplified model: {e}")
    
    elif "threshold" in error_type.lower() or "threshold" in str(error).lower():
        # Threshold configuration errors - try using default thresholds
        if "default_thresholds" in recovery_params and "model" in recovery_params and "features" in recovery_params:
            try:
                logger.info(f"Using default thresholds for inference")
                model = recovery_params["model"]
                features = recovery_params["features"]
                default_thresholds = recovery_params["default_thresholds"]
                
                # Generate predictions
                probs = model.predict_proba(features)
                
                # Format results with default thresholds
                result = pd.DataFrame()
                for threshold in default_thresholds:
                    df = pd.DataFrame({
                        "threshold": threshold,
                        "probability": probs
                    })
                    result = pd.concat([result, df])
                
                return result
            except Exception as e:
                logger.warning(f"Failed inference with default thresholds: {e}")
    
    # Try using previous forecast if available
    if "previous_forecast" in recovery_params:
        logger.info(f"Using previous forecast as fallback")
        return recovery_params["previous_forecast"]
    
    # Last resort - generate a conservative forecast based on historical data
    if "historical_data" in recovery_params:
        try:
            logger.info(f"Generating conservative forecast from historical data")
            historical_data = recovery_params["historical_data"]
            
            # Simple implementation - use historical average probabilities
            if "thresholds" in recovery_params:
                thresholds = recovery_params["thresholds"]
                result = pd.DataFrame()
                
                for threshold in thresholds:
                    if f"probability_{threshold}" in historical_data.columns:
                        avg_prob = historical_data[f"probability_{threshold}"].mean()
                    else:
                        avg_prob = 0.1  # Conservative default
                    
                    # Create a forecast with the average probability
                    horizon = recovery_params.get("horizon", 72)
                    now = datetime.datetime.now()
                    dates = [now + datetime.timedelta(hours=i) for i in range(horizon)]
                    
                    df = pd.DataFrame({
                        "timestamp": dates,
                        "threshold": threshold,
                        "probability": avg_prob
                    })
                    result = pd.concat([result, df])
                
                return result
        except Exception as e:
            logger.error(f"Failed to generate conservative forecast: {e}")
    
    # Recovery failed
    logger.error(f"Inference recovery failed for error: {error}")
    return None


@log_execution_time(logger, 'INFO')
def recover_system(
    error: Exception,
    context: Dict[str, Any],
    recovery_params: Dict[str, Any]
) -> bool:
    """
    Recovers from system-level errors.
    
    Args:
        error: System error to recover from
        context: Recovery context
        recovery_params: Parameters for recovery
        
    Returns:
        True if recovery was successful, False otherwise
    """
    error_type = context.get("error_type", "")
    
    if "memory" in error_type.lower() or "memory" in str(error).lower():
        # Memory errors - try to free memory or reduce batch size
        if "reduce_batch_size" in recovery_params and "process_func" in recovery_params:
            try:
                logger.info(f"Attempting recovery with reduced batch size")
                process_func = recovery_params["process_func"]
                reduced_batch_size = recovery_params["reduce_batch_size"]
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Retry with reduced batch size
                process_func(batch_size=reduced_batch_size)
                return True
            except Exception as e:
                logger.warning(f"Failed recovery with reduced batch size: {e}")
    
    elif "timeout" in error_type.lower() or "timeout" in str(error).lower():
        # Timeout errors - try to extend timeout or optimize the operation
        if "extended_timeout" in recovery_params and "operation_func" in recovery_params:
            try:
                logger.info(f"Retrying operation with extended timeout")
                operation_func = recovery_params["operation_func"]
                extended_timeout = recovery_params["extended_timeout"]
                
                operation_func(timeout=extended_timeout)
                return True
            except Exception as e:
                logger.warning(f"Failed retry with extended timeout: {e}")
    
    # For unexpected errors, log detailed diagnostics
    try:
        import platform
        import psutil
        
        diagnostics = {
            "system": platform.system(),
            "python_version": platform.python_version(),
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "disk_percent": psutil.disk_usage('/').percent
        }
        
        logger.info(f"System diagnostics: {diagnostics}")
    except ImportError:
        logger.warning("Unable to collect system diagnostics - psutil not available")
    
    # Try to restart the component if specified
    if "restart_component" in recovery_params:
        try:
            logger.info(f"Attempting to restart component")
            component = recovery_params["restart_component"]
            if hasattr(component, "restart") and callable(getattr(component, "restart")):
                component.restart()
                return True
        except Exception as e:
            logger.error(f"Failed to restart component: {e}")
    
    # Recovery failed
    logger.error(f"System recovery failed for error: {error}")
    return False


def get_recovery_function(error_category: str) -> Callable[[Exception, Dict[str, Any], Dict[str, Any]], Any]:
    """
    Gets the appropriate recovery function based on error category.
    
    Args:
        error_category: Category of the error
        
    Returns:
        Recovery function for the specified category
    """
    recovery_functions = {
        "data": recover_data,
        "connection": recover_connection,
        "model": recover_model,
        "inference": recover_inference,
        "system": recover_system
    }
    
    return recovery_functions.get(error_category, recover_system)


class RecoveryContext:
    """Context manager for error recovery operations."""
    
    def __init__(
        self,
        strategy: RecoveryStrategy,
        recovery_params: Dict[str, Any],
        pipeline_stage: Optional[PipelineStage] = None,
        max_attempts: int = DEFAULT_MAX_RECOVERY_ATTEMPTS
    ):
        """
        Initialize a new RecoveryContext instance.
        
        Args:
            strategy: Recovery strategy to use
            recovery_params: Parameters for the recovery strategy
            pipeline_stage: Stage of the pipeline where recovery is applied
            max_attempts: Maximum number of recovery attempts
        """
        self._strategy = strategy
        self._recovery_params = recovery_params
        self._pipeline_stage = pipeline_stage
        self._attempt = 0
        self._max_attempts = max_attempts
        self._context = {}
    
    def __enter__(self) -> 'RecoveryContext':
        """
        Enter the recovery context.
        
        Returns:
            Self reference
        """
        return self
    
    def __exit__(
        self,
        exc_type: Optional[Type[Exception]],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any]
    ) -> bool:
        """
        Exit the recovery context, handling any exceptions.
        
        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
            
        Returns:
            True if exception was handled, False otherwise
        """
        if exc_val is None:
            # No exception occurred
            return False
        
        # Increment attempt counter
        self._attempt += 1
        
        # Create recovery context
        self._context = create_recovery_context(exc_val, self._pipeline_stage)
        
        # Get the appropriate recovery function
        recovery_category = self._context["error_category"]
        recovery_func = get_recovery_function(recovery_category)
        
        try:
            # Attempt recovery
            recovery_result = recovery_func(exc_val, self._context, self._recovery_params)
            
            # Log the recovery attempt
            success = recovery_result is not None
            log_recovery_attempt(exc_val, self._strategy, self._attempt, self._context, success)
            
            # If recovery was successful and we haven't exceeded max attempts, suppress the exception
            if success and self._attempt <= self._max_attempts:
                return True
        except Exception as e:
            logger.error(f"Error during recovery attempt: {e}")
        
        # Recovery failed or max attempts exceeded, let the exception propagate
        return False
    
    def get_attempt_count(self) -> int:
        """
        Gets the current attempt count.
        
        Returns:
            Current attempt count
        """
        return self._attempt
    
    def reset(self) -> None:
        """
        Resets the attempt counter.
        """
        self._attempt = 0
        self._context.clear()
    
    def get_context(self) -> Dict[str, Any]:
        """
        Gets the current recovery context.
        
        Returns:
            Current recovery context
        """
        return self._context.copy()


class ErrorRecoveryManager:
    """Manager class for handling error recovery across the pipeline."""
    
    def __init__(self, max_recovery_attempts: int = DEFAULT_MAX_RECOVERY_ATTEMPTS):
        """
        Initialize a new ErrorRecoveryManager instance.
        
        Args:
            max_recovery_attempts: Maximum number of recovery attempts
        """
        self._recovery_stats = {}  # Stats about recovery operations
        self._recovery_strategies = {}  # Registered recovery strategies
        self._recovery_params = {}  # Parameters for recovery strategies
        self._fallbacks = {}  # Registered fallbacks
        self._max_recovery_attempts = max_recovery_attempts
        
        # Set up default recovery strategies
        self._recovery_strategies = {
            "data": {},
            "connection": {},
            "model": {},
            "inference": {},
            "system": {}
        }
        
        self._recovery_params = {
            "data": {},
            "connection": {},
            "model": {},
            "inference": {},
            "system": {}
        }
    
    def register_recovery_strategy(
        self,
        error_category: str,
        pipeline_stage: PipelineStage,
        strategy: RecoveryStrategy,
        params: Dict[str, Any]
    ) -> None:
        """
        Registers a recovery strategy for a specific error type and pipeline stage.
        
        Args:
            error_category: Category of the error
            pipeline_stage: Stage of the pipeline
            strategy: Recovery strategy to use
            params: Parameters for the recovery strategy
        """
        if error_category not in self._recovery_strategies:
            self._recovery_strategies[error_category] = {}
            self._recovery_params[error_category] = {}
        
        self._recovery_strategies[error_category][pipeline_stage] = strategy
        self._recovery_params[error_category][pipeline_stage] = params
        
        logger.info(f"Registered {strategy.value} strategy for {error_category} errors in {pipeline_stage.value} stage")
    
    def register_fallback(self, pipeline_stage: PipelineStage, fallback_object: Any) -> None:
        """
        Registers a fallback object for a specific pipeline stage.
        
        Args:
            pipeline_stage: Stage of the pipeline
            fallback_object: Fallback object to use in recovery
        """
        self._fallbacks[pipeline_stage] = fallback_object
        logger.info(f"Registered fallback for {pipeline_stage.value} stage")
    
    def get_recovery_strategy(
        self,
        error: Exception,
        pipeline_stage: Optional[PipelineStage] = None
    ) -> Tuple[RecoveryStrategy, Dict[str, Any]]:
        """
        Gets the recovery strategy for a specific error and pipeline stage.
        
        Args:
            error: Exception to recover from
            pipeline_stage: Stage of the pipeline where the error occurred
            
        Returns:
            Tuple of recovery strategy and parameters
        """
        error_category = get_error_type_category(error)
        
        # If no pipeline stage is specified, use a default strategy
        if pipeline_stage is None:
            if "connection" == error_category:
                return RecoveryStrategy.RETRY, {"max_attempts": self._max_recovery_attempts}
            elif "data" == error_category:
                return RecoveryStrategy.FALLBACK, {}
            elif "model" == error_category:
                return RecoveryStrategy.FALLBACK, {}
            elif "inference" == error_category:
                return RecoveryStrategy.FALLBACK, {}
            else:
                return RecoveryStrategy.RETRY, {"max_attempts": self._max_recovery_attempts}
        
        # Get the registered strategy for this error category and pipeline stage
        strategy = self._recovery_strategies.get(error_category, {}).get(
            pipeline_stage, RecoveryStrategy.RETRY
        )
        
        # Get the registered parameters
        params = self._recovery_params.get(error_category, {}).get(
            pipeline_stage, {}
        ).copy()
        
        # Add fallback if available
        if pipeline_stage in self._fallbacks:
            if "fallback" not in params:
                params["fallback"] = self._fallbacks[pipeline_stage]
        
        return strategy, params
    
    @log_execution_time(logger, 'INFO')
    def recover_from_error(
        self,
        error: Exception,
        pipeline_stage: Optional[PipelineStage] = None,
        context: Dict[str, Any] = None
    ) -> Tuple[bool, Optional[Any]]:
        """
        Attempts to recover from an error during pipeline execution.
        
        Args:
            error: Exception to recover from
            pipeline_stage: Stage of the pipeline where the error occurred
            context: Additional context for recovery
            
        Returns:
            Tuple of recovery success flag and recovered object if applicable
        """
        # Get the recovery strategy and parameters
        strategy, recovery_params = self.get_recovery_strategy(error, pipeline_stage)
        
        # Create recovery context
        recovery_context = create_recovery_context(error, pipeline_stage, context)
        recovery_category = recovery_context["error_category"]
        
        # Get the appropriate recovery function
        recovery_func = get_recovery_function(recovery_category)
        
        # Initialize variables for recovery loop
        attempt = 0
        success = False
        recovered_object = None
        
        # Try recovery up to max_attempts times
        while attempt < self._max_recovery_attempts and not success:
            attempt += 1
            
            try:
                # Attempt recovery
                recovered_object = recovery_func(error, recovery_context, recovery_params)
                
                # Check if recovery was successful
                success = recovered_object is not None
                
                # Log the recovery attempt
                log_recovery_attempt(error, strategy, attempt, recovery_context, success)
                
                if success:
                    break
            except Exception as e:
                logger.error(f"Error during recovery attempt {attempt}: {e}")
            
            # Apply backoff delay before next attempt
            if attempt < self._max_recovery_attempts:
                delay = DEFAULT_RECOVERY_DELAY * (DEFAULT_BACKOFF_FACTOR ** (attempt - 1))
                time.sleep(delay)
        
        # Update recovery statistics
        if recovery_category not in self._recovery_stats:
            self._recovery_stats[recovery_category] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0
            }
        
        self._recovery_stats[recovery_category]["attempts"] += attempt
        if success:
            self._recovery_stats[recovery_category]["successes"] += 1
        else:
            self._recovery_stats[recovery_category]["failures"] += 1
        
        return success, recovered_object
    
    def create_recovery_context(
        self,
        strategy: RecoveryStrategy,
        params: Dict[str, Any],
        pipeline_stage: Optional[PipelineStage] = None
    ) -> RecoveryContext:
        """
        Creates a RecoveryContext for a specific strategy.
        
        Args:
            strategy: Recovery strategy to use
            params: Parameters for the recovery strategy
            pipeline_stage: Stage of the pipeline
            
        Returns:
            Created RecoveryContext instance
        """
        return RecoveryContext(
            strategy=strategy,
            recovery_params=params,
            pipeline_stage=pipeline_stage,
            max_attempts=self._max_recovery_attempts
        )
    
    def get_recovery_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Gets statistics about recovery operations.
        
        Returns:
            Dictionary with recovery statistics
        """
        return self._recovery_stats.copy()
    
    def reset_stats(self) -> None:
        """
        Resets recovery statistics.
        """
        self._recovery_stats.clear()
        logger.info("Recovery statistics reset")


class RecoveryProtocol(Protocol):
    """Protocol defining the interface for custom recovery handlers."""
    
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        Checks if the handler can handle a specific error.
        
        Args:
            error: Exception to check
            context: Error context
            
        Returns:
            True if the handler can handle the error, False otherwise
        """
        ...
    
    def handle(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Optional[Any]]:
        """
        Handles the error and attempts recovery.
        
        Args:
            error: Exception to handle
            context: Error context
            
        Returns:
            Tuple of recovery success flag and recovered object if applicable
        """
        ...


class DataRecoveryHandler:
    """Handler for recovering from data-related errors."""
    
    def __init__(self):
        """Initialize a new DataRecoveryHandler instance."""
        self._cached_data = {}  # Cache of data for recovery
        self._backup_paths = {}  # Backup file paths for data sources
    
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        Checks if the handler can handle a specific error.
        
        Args:
            error: Exception to check
            context: Error context
            
        Returns:
            True if the handler can handle the error, False otherwise
        """
        error_category = context.get("error_category", get_error_type_category(error))
        return error_category == "data"
    
    @log_execution_time(logger, 'INFO')
    def handle(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Optional[DataFrameType]]:
        """
        Handles data errors and attempts recovery.
        
        Args:
            error: Exception to handle
            context: Error context
            
        Returns:
            Tuple of recovery success flag and recovered data if applicable
        """
        # Extract data source information
        data_source = context.get("data_source", "unknown")
        
        # Try to recover using cached data
        if data_source in self._cached_data:
            logger.info(f"Recovering using cached data for source: {data_source}")
            return True, self._cached_data[data_source]
        
        # Try to load from backup path
        if data_source in self._backup_paths:
            backup_path = self._backup_paths[data_source]
            try:
                logger.info(f"Recovering using backup data from: {backup_path}")
                if isinstance(backup_path, str) and backup_path.endswith(".parquet"):
                    data = pd.read_parquet(backup_path)
                elif isinstance(backup_path, str) and backup_path.endswith(".csv"):
                    data = pd.read_csv(backup_path)
                else:
                    # Assume it's a Path object or string path to a parquet file
                    data = pd.read_parquet(backup_path)
                
                return True, data
            except Exception as e:
                logger.warning(f"Failed to load backup data from {backup_path}: {e}")
        
        # Try to repair or transform the data if we have a repair function
        repair_func = context.get("repair_func")
        data_to_repair = context.get("data")
        
        if repair_func and data_to_repair is not None:
            try:
                logger.info(f"Attempting to repair data")
                repaired_data = repair_func(data_to_repair)
                return True, repaired_data
            except Exception as e:
                logger.warning(f"Failed to repair data: {e}")
        
        # Recovery failed
        logger.error(f"Data recovery failed for error: {error}")
        return False, None
    
    def cache_data(self, data_key: str, data: DataFrameType) -> None:
        """
        Caches data for potential recovery use.
        
        Args:
            data_key: Key to identify the data
            data: Data to cache
        """
        self._cached_data[data_key] = data.copy()
    
    def register_backup_path(self, data_key: str, backup_path: PathType) -> None:
        """
        Registers a backup file path for a data source.
        
        Args:
            data_key: Key to identify the data source
            backup_path: Path to backup file
        """
        self._backup_paths[data_key] = backup_path
    
    def clear_cache(self) -> None:
        """
        Clears the data cache.
        """
        self._cached_data.clear()


class ModelRecoveryHandler:
    """Handler for recovering from model-related errors."""
    
    def __init__(self, fallback_model: Optional[ModelType] = None):
        """
        Initialize a new ModelRecoveryHandler instance.
        
        Args:
            fallback_model: Optional fallback model to use when all recovery attempts fail
        """
        self._model_versions = {}  # Previous model versions
        self._model_paths = {}  # Paths to model files
        self._fallback_model = fallback_model
    
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        Checks if the handler can handle a specific error.
        
        Args:
            error: Exception to check
            context: Error context
            
        Returns:
            True if the handler can handle the error, False otherwise
        """
        error_category = context.get("error_category", get_error_type_category(error))
        return error_category == "model"
    
    @log_execution_time(logger, 'INFO')
    def handle(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Optional[ModelType]]:
        """
        Handles model errors and attempts recovery.
        
        Args:
            error: Exception to handle
            context: Error context
            
        Returns:
            Tuple of recovery success flag and recovered model if applicable
        """
        # Extract model information
        model_key = context.get("model_key", "default")
        
        # Try to use a previous model version
        if model_key in self._model_versions:
            logger.info(f"Recovering using previous model version for: {model_key}")
            return True, self._model_versions[model_key]
        
        # Try to load from a model path
        if model_key in self._model_paths:
            model_path = self._model_paths[model_key]
            load_func = context.get("load_func")
            
            if load_func:
                try:
                    logger.info(f"Recovering by loading model from: {model_path}")
                    model = load_func(model_path)
                    return True, model
                except Exception as e:
                    logger.warning(f"Failed to load model from {model_path}: {e}")
        
        # Use the fallback model if available
        if self._fallback_model is not None:
            logger.info(f"Using fallback model as last resort")
            return True, self._fallback_model
        
        # Recovery failed
        logger.error(f"Model recovery failed for error: {error}")
        return False, None
    
    def register_model_version(self, model_key: str, model: ModelType) -> None:
        """
        Registers a model version for potential recovery use.
        
        Args:
            model_key: Key to identify the model
            model: Model to register
        """
        self._model_versions[model_key] = model
    
    def register_model_path(self, model_key: str, model_path: PathType) -> None:
        """
        Registers a model file path for recovery.
        
        Args:
            model_key: Key to identify the model
            model_path: Path to model file
        """
        self._model_paths[model_key] = model_path
    
    def set_fallback_model(self, model: ModelType) -> None:
        """
        Sets a fallback model to use when all recovery attempts fail.
        
        Args:
            model: Fallback model
        """
        self._fallback_model = model