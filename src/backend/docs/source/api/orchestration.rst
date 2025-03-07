Orchestration API
=================

The orchestration module provides functionality for coordinating the various components
of the ERCOT RTLMP spike prediction system. It includes task management, scheduling,
pipeline execution, and error recovery capabilities to ensure reliable operation of the system.

Overview
--------

The orchestration module provides functionality for coordinating the various components of the ERCOT RTLMP spike prediction system. It includes task management, scheduling, pipeline execution, and error recovery capabilities to ensure reliable operation of the system.

Task Management
---------------

.. automodule:: backend.orchestration.task_management
   :members:
   :undoc-members:
   :show-inheritance:

Scheduler
---------

.. automodule:: backend.orchestration.scheduler
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline
--------

.. automodule:: backend.orchestration.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Error Recovery
--------------

.. automodule:: backend.orchestration.error_recovery
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Setting up a Daily Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example of setting up a daily pipeline for RTLMP spike prediction

.. code-block:: python

    from backend.orchestration import PipelineExecutor, DailyScheduler

    # Create a pipeline executor
    pipeline_executor = PipelineExecutor()

    # Set up daily operations
    pipeline_executor.setup_daily_operations(
        data_fetch_time="00:00",
        inference_time="06:00",
        training_time="01:00",
        training_frequency="0 1 */2 * *"  # Every second day at 1:00 AM
    )

    # Start the scheduler
    pipeline_executor.start_scheduler()

Creating and Executing Tasks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example of creating and executing tasks with dependencies

.. code-block:: python

    from backend.orchestration import TaskManager, Task, TaskPriority

    # Create a task manager
    task_manager = TaskManager(max_workers=4)

    # Create tasks
    fetch_task = task_manager.create_task(
        func=fetch_data,
        name="fetch_data",
        args=(start_date, end_date),
        priority=TaskPriority.HIGH
    )

    feature_task = task_manager.create_task(
        func=engineer_features,
        name="engineer_features",
        args=(raw_data,),
        priority=TaskPriority.MEDIUM
    )

    # Add dependency
    feature_task.add_dependency(fetch_task)

    # Execute tasks in dependency order
    results = task_manager.execute_all(parallel=True, with_retry=True)

Error Recovery
^^^^^^^^^^^^^^

Example of implementing error recovery strategies

.. code-block:: python

    from backend.orchestration import ErrorRecoveryManager, RecoveryStrategy

    # Create an error recovery manager
    recovery_manager = ErrorRecoveryManager()

    # Register custom recovery strategies
    recovery_manager.register_recovery_strategy(
        ConnectionError,
        {
            "strategy": RecoveryStrategy.RETRY,
            "max_attempts": 3,
            "backoff_factor": 2.0
        }
    )

    # Use in a pipeline
    pipeline = Pipeline(recovery_manager=recovery_manager)
    pipeline.execute_pipeline()