Models API
==========

The models module provides a comprehensive set of tools for training, evaluating, and using machine learning models for RTLMP spike prediction in the ERCOT market.

Base Models
-----------

Abstract base classes and interfaces for model implementations.

base_model
~~~~~~~~~~

Abstract base class for all prediction models.

.. class:: BaseModel

   Abstract base class that defines the common interface for all model implementations.

   .. method:: train(features: DataFrame, targets: Series, params: Dict[str, Any] = None) -> BaseModel

      Train the model on provided features and targets.

      :param features: Feature data for training
      :param targets: Target values for training
      :param params: Optional training parameters
      :returns: Self for method chaining

   .. method:: predict(features: DataFrame) -> ndarray

      Generate binary predictions using the trained model.

      :param features: Feature data for prediction
      :returns: Binary predictions (0 or 1)

   .. method:: predict_proba(features: DataFrame) -> ndarray

      Generate probability predictions using the trained model.

      :param features: Feature data for prediction
      :returns: Probability predictions (values between 0 and 1)

   .. method:: get_feature_importance() -> Dict[str, float]

      Get feature importance scores from the trained model.

      :returns: Dictionary mapping feature names to importance scores

   .. method:: save(path: Path = None) -> Path

      Save the model to a file with metadata.

      :param path: Optional path to save the model
      :returns: Path to the saved model directory

   .. method:: load(path: Path) -> BaseModel

      Load the model from a file with metadata.

      :param path: Path to the model directory
      :returns: Self for method chaining

   .. method:: is_trained() -> bool

      Check if the model has been trained.

      :returns: True if the model is trained, False otherwise

   .. method:: validate_features(features: DataFrame) -> bool

      Validate that features contain required columns.

      :param features: Feature data to validate
      :returns: True if features are valid, False otherwise

   .. method:: get_model_config() -> Dict[str, Any]

      Get the model configuration as a dictionary.

      :returns: Model configuration dictionary

   .. method:: set_performance_metrics(metrics: Dict[str, float]) -> BaseModel

      Set the performance metrics for the model.

      :param metrics: Performance metrics dictionary
      :returns: Self for method chaining

   .. method:: get_performance_metrics() -> Dict[str, float]

      Get the performance metrics for the model.

      :returns: Performance metrics dictionary

Model Implementations
--------------------

Concrete model implementations for RTLMP spike prediction.

xgboost_model
~~~~~~~~~~~~

XGBoost implementation for RTLMP spike prediction.

.. data:: DEFAULT_XGBOOST_PARAMS

   Default hyperparameters for XGBoost models.

.. class:: XGBoostModel

   XGBoost implementation for RTLMP spike prediction.

   .. method:: train(features: DataFrame, targets: Series, params: Dict[str, Any] = None) -> XGBoostModel

      Train the XGBoost model on provided features and targets.

      :param features: Feature data for training
      :param targets: Target values for training
      :param params: Optional training parameters
      :returns: Self for method chaining

   .. method:: evaluate(X_test: DataFrame, y_test: Series, threshold: float = 0.5) -> Dict[str, float]

      Evaluate the XGBoost model performance on test data.

      :param X_test: Test feature data
      :param y_test: Test target values
      :param threshold: Optional probability threshold for binary classification
      :returns: Dictionary of performance metrics

lightgbm_model
~~~~~~~~~~~~~

LightGBM implementation for RTLMP spike prediction.

.. data:: DEFAULT_LIGHTGBM_PARAMS

   Default hyperparameters for LightGBM models.

.. class:: LightGBMModel

   LightGBM implementation for RTLMP spike prediction.

   .. method:: train(features: DataFrame, targets: Series, params: Dict[str, Any] = None) -> LightGBMModel

      Train the LightGBM model on provided features and targets.

      :param features: Feature data for training
      :param targets: Target values for training
      :param params: Optional training parameters
      :returns: Self for method chaining

   .. method:: evaluate(X_test: DataFrame, y_test: Series, threshold: float = 0.5) -> Dict[str, float]

      Evaluate the LightGBM model performance on test data.

      :param X_test: Test feature data
      :param y_test: Test target values
      :param threshold: Optional probability threshold for binary classification
      :returns: Dictionary of performance metrics

ensemble
~~~~~~~~

Ensemble model implementation that combines multiple base models.

.. class:: EnsembleModel

   Ensemble model implementation that combines multiple base models.

   .. method:: add_model(model: BaseModel, weight: float = 1.0) -> EnsembleModel

      Add a model to the ensemble.

      :param model: Model to add to the ensemble
      :param weight: Optional weight for the model
      :returns: Self for method chaining

   .. method:: train(features: DataFrame, targets: Series, params: Dict[str, Any] = None) -> EnsembleModel

      Train all models in the ensemble on provided features and targets.

      :param features: Feature data for training
      :param targets: Target values for training
      :param params: Optional training parameters
      :returns: Self for method chaining

   .. method:: compare_model_performance(X_test: DataFrame, y_test: Series, threshold: float = 0.5) -> DataFrame

      Compare performance of individual models and the ensemble.

      :param X_test: Test feature data
      :param y_test: Test target values
      :param threshold: Optional probability threshold for binary classification
      :returns: DataFrame with performance comparison

.. class:: EnsembleMethod

   Enum defining different ensemble methods.

   .. data:: VOTING

      Voting ensemble method.

   .. data:: STACKING

      Stacking ensemble method.

   .. data:: BAGGING

      Bagging ensemble method.

   .. data:: BOOSTING

      Boosting ensemble method.

Model Training
-------------

Functions and classes for model training and hyperparameter optimization.

training
~~~~~~~~

Core module for training machine learning models.

.. function:: create_model(model_id: str, model_type: str, version: str = None, hyperparameters: Dict[str, Any] = None) -> BaseModel

   Factory function to create a model instance of the specified type.

   :param model_id: Unique identifier for the model
   :param model_type: Type of model to create (e.g., 'xgboost', 'lightgbm')
   :param version: Optional version string for the model
   :param hyperparameters: Optional hyperparameters for the model
   :returns: Initialized model instance

.. function:: train_model(model_type: str, features: DataFrame, targets: Series, hyperparameters: Dict[str, Any] = None, model_id: str = None, model_path: Path = None) -> Tuple[BaseModel, Dict[str, float]]

   Train a model with the specified parameters.

   :param model_type: Type of model to train
   :param features: Feature data for training
   :param targets: Target values for training
   :param hyperparameters: Optional hyperparameters for the model
   :param model_id: Optional unique identifier for the model
   :param model_path: Optional path to save the trained model
   :returns: Trained model and performance metrics

.. function:: train_and_evaluate(model_type: str, features: DataFrame, targets: Series, hyperparameters: Dict[str, Any] = None, cv_folds: int = 5, cv_strategy: str = 'time') -> Tuple[BaseModel, Dict[str, List[float]]]

   Train a model and evaluate its performance with cross-validation.

   :param model_type: Type of model to train
   :param features: Feature data for training
   :param targets: Target values for training
   :param hyperparameters: Optional hyperparameters for the model
   :param cv_folds: Number of cross-validation folds
   :param cv_strategy: Cross-validation strategy
   :returns: Trained model and cross-validation metrics

.. function:: optimize_and_train(model_type: str, features: DataFrame, targets: Series, param_grid: Dict[str, Any], optimization_method: str = 'bayesian', n_iterations: int = 50, model_path: Path = None) -> Tuple[BaseModel, Dict[str, float]]

   Optimize hyperparameters and train a model.

   :param model_type: Type of model to train
   :param features: Feature data for training
   :param targets: Target values for training
   :param param_grid: Parameter grid for optimization
   :param optimization_method: Method for hyperparameter optimization
   :param n_iterations: Number of iterations for optimization
   :param model_path: Optional path to save the trained model
   :returns: Optimized model and performance metrics

.. function:: load_model(model_identifier: Union[str, Path], model_type: str = None, version: str = None, registry_path: Path = None) -> BaseModel

   Load a model from the registry or a file path.

   :param model_identifier: Model ID or file path
   :param model_type: Optional type of model to load
   :param version: Optional version of the model to load
   :param registry_path: Optional path to the model registry
   :returns: Loaded model instance

.. function:: get_latest_model(model_type: str, registry_path: Path = None) -> Optional[BaseModel]

   Get the latest model of a specific type from the registry.

   :param model_type: Type of model to retrieve
   :param registry_path: Optional path to the model registry
   :returns: Latest model instance or None if not found

.. function:: compare_models(models: List[BaseModel], features: DataFrame, targets: Series, threshold: float = 0.5) -> Dict[str, Dict[str, float]]

   Compare multiple models on the same test data.

   :param models: List of models to compare
   :param features: Test feature data
   :param targets: Test target values
   :param threshold: Optional probability threshold for binary classification
   :returns: Dictionary of model IDs mapped to their performance metrics

.. function:: select_best_model(models: List[BaseModel], features: DataFrame, targets: Series, metric: str = 'auc', higher_is_better: bool = True) -> Tuple[BaseModel, float]

   Select the best model from a list based on a specific metric.

   :param models: List of models to compare
   :param features: Test feature data
   :param targets: Test target values
   :param metric: Metric to use for comparison
   :param higher_is_better: Whether higher metric values are better
   :returns: Best model and its metric value

.. function:: retrain_model(model: BaseModel, features: DataFrame, targets: Series, hyperparameters: Dict[str, Any] = None, model_path: Path = None, increment_type: str = 'patch') -> Tuple[BaseModel, Dict[str, float]]

   Retrain an existing model with new data.

   :param model: Model to retrain
   :param features: Feature data for training
   :param targets: Target values for training
   :param hyperparameters: Optional hyperparameters for the model
   :param model_path: Optional path to save the retrained model
   :param increment_type: Type of version increment for the retrained model
   :returns: Retrained model and performance metrics

.. class:: ModelTrainer

   Class for managing model training workflows.

   .. method:: train(features: DataFrame, targets: Series, model_id: str = None) -> Tuple[BaseModel, Dict[str, float]]

      Train a new model with the configured settings.

      :param features: Feature data for training
      :param targets: Target values for training
      :param model_id: Optional unique identifier for the model
      :returns: Trained model and performance metrics

   .. method:: optimize_and_train(features: DataFrame, targets: Series, param_grid: Dict[str, Any], optimization_method: str = 'bayesian', n_iterations: int = 50) -> Tuple[BaseModel, Dict[str, float]]

      Optimize hyperparameters and train a model.

      :param features: Feature data for training
      :param targets: Target values for training
      :param param_grid: Parameter grid for optimization
      :param optimization_method: Method for hyperparameter optimization
      :param n_iterations: Number of iterations for optimization
      :returns: Optimized model and performance metrics

cross_validation
~~~~~~~~~~~~~~~

Implements time-based cross-validation strategies.

.. function:: time_series_split(features: DataFrame, targets: Series, n_splits: int = 5, test_size: float = 0.2) -> List[Tuple[DataFrame, DataFrame, Series, Series]]

   Creates time series cross-validation splits respecting temporal order.

   :param features: Feature data to split
   :param targets: Target values to split
   :param n_splits: Number of splits to create
   :param test_size: Proportion of data to use for testing
   :returns: List of (X_train, X_test, y_train, y_test) tuples

.. function:: cross_validate_model(model: BaseModel, features: DataFrame, targets: Series, n_splits: int = 5, cv_strategy: str = 'time', metrics: List[str] = None, cv_params: Dict[str, Any] = None) -> Dict[str, List[float]]

   Performs cross-validation for a model using time series splits.

   :param model: Model to cross-validate
   :param features: Feature data for cross-validation
   :param targets: Target values for cross-validation
   :param n_splits: Number of splits to use
   :param cv_strategy: Cross-validation strategy
   :param metrics: Metrics to calculate
   :param cv_params: Additional parameters for cross-validation
   :returns: Dictionary of metric names mapped to lists of scores

.. class:: TimeSeriesCV

   Class for managing time series cross-validation.

   .. method:: split(features: DataFrame, targets: Series) -> List[Tuple[DataFrame, DataFrame, Series, Series]]

      Generate cross-validation splits based on strategy.

      :param features: Feature data to split
      :param targets: Target values to split
      :returns: List of (X_train, X_test, y_train, y_test) tuples

   .. method:: cross_validate(model: BaseModel, features: DataFrame, targets: Series) -> Dict[str, List[float]]

      Perform cross-validation for a model.

      :param model: Model to cross-validate
      :param features: Feature data for cross-validation
      :param targets: Target values for cross-validation
      :returns: Dictionary of metric names mapped to lists of scores

hyperparameter_tuning
~~~~~~~~~~~~~~~~~~~~

Module for hyperparameter optimization.

.. function:: optimize_hyperparameters(model_type: str, features: DataFrame, targets: Series, param_grid: Dict[str, List[Any]], optimization_method: str = 'bayesian', n_iterations: int = 50, cv_folds: int = 5, scoring: str = 'roc_auc', verbose: bool = False, n_jobs: int = -1) -> Tuple[Dict[str, Any], Dict[str, float]]

   Optimize hyperparameters using the specified method.

   :param model_type: Type of model to optimize
   :param features: Feature data for optimization
   :param targets: Target values for optimization
   :param param_grid: Parameter grid for optimization
   :param optimization_method: Method for hyperparameter optimization
   :param n_iterations: Number of iterations for optimization
   :param cv_folds: Number of cross-validation folds
   :param scoring: Scoring metric for optimization
   :param verbose: Whether to print verbose output
   :param n_jobs: Number of parallel jobs
   :returns: Best parameters and corresponding scores

.. function:: grid_search_cv(model_type: str, features: DataFrame, targets: Series, param_grid: Dict[str, List[Any]], cv_folds: int = 5, scoring: str = 'roc_auc', verbose: bool = False, n_jobs: int = -1) -> Tuple[Dict[str, Any], Dict[str, float]]

   Perform grid search cross-validation for hyperparameter optimization.

   :param model_type: Type of model to optimize
   :param features: Feature data for optimization
   :param targets: Target values for optimization
   :param param_grid: Parameter grid for optimization
   :param cv_folds: Number of cross-validation folds
   :param scoring: Scoring metric for optimization
   :param verbose: Whether to print verbose output
   :param n_jobs: Number of parallel jobs
   :returns: Best parameters and corresponding scores

.. function:: random_search_cv(model_type: str, features: DataFrame, targets: Series, param_grid: Dict[str, List[Any]], n_iterations: int = 20, cv_folds: int = 5, scoring: str = 'roc_auc', verbose: bool = False, n_jobs: int = -1) -> Tuple[Dict[str, Any], Dict[str, float]]

   Perform random search cross-validation for hyperparameter optimization.

   :param model_type: Type of model to optimize
   :param features: Feature data for optimization
   :param targets: Target values for optimization
   :param param_grid: Parameter grid for optimization
   :param n_iterations: Number of iterations for optimization
   :param cv_folds: Number of cross-validation folds
   :param scoring: Scoring metric for optimization
   :param verbose: Whether to print verbose output
   :param n_jobs: Number of parallel jobs
   :returns: Best parameters and corresponding scores

.. function:: bayesian_optimization(model_type: str, features: DataFrame, targets: Series, param_grid: Dict[str, List[Any]], n_trials: int = 50, cv_folds: int = 5, scoring: str = 'roc_auc', verbose: bool = False) -> Tuple[Dict[str, Any], Dict[str, float]]

   Perform Bayesian optimization for hyperparameter tuning using Optuna.

   :param model_type: Type of model to optimize
   :param features: Feature data for optimization
   :param targets: Target values for optimization
   :param param_grid: Parameter grid for optimization
   :param n_trials: Number of trials for optimization
   :param cv_folds: Number of cross-validation folds
   :param scoring: Scoring metric for optimization
   :param verbose: Whether to print verbose output
   :returns: Best parameters and corresponding scores

.. class:: HyperparameterOptimizer

   Class for managing hyperparameter optimization workflows.

   .. method:: optimize(features: DataFrame, targets: Series, verbose: bool = False, n_jobs: int = -1) -> Dict[str, Any]

      Perform hyperparameter optimization.

      :param features: Feature data for optimization
      :param targets: Target values for optimization
      :param verbose: Whether to print verbose output
      :param n_jobs: Number of parallel jobs
      :returns: Best hyperparameters

   .. method:: optimize_and_train(features: DataFrame, targets: Series, model_id: str = None) -> Tuple[BaseModel, Dict[str, float]]

      Optimize hyperparameters and train a model.

      :param features: Feature data for training
      :param targets: Target values for training
      :param model_id: Optional unique identifier for the model
      :returns: Trained model and performance metrics

Model Evaluation
---------------

Functions and classes for model evaluation and performance assessment.

evaluation
~~~~~~~~~

Module for evaluating model performance.

.. function:: evaluate_model_performance(model: BaseModel, features: DataFrame, targets: Series, metrics: List[str] = None, threshold: float = 0.5) -> Dict[str, float]

   Evaluates model performance using various metrics.

   :param model: Model to evaluate
   :param features: Feature data for evaluation
   :param targets: Target values for evaluation
   :param metrics: Metrics to calculate
   :param threshold: Probability threshold for binary classification
   :returns: Dictionary of performance metrics

.. function:: calculate_confusion_matrix(y_true: Series, y_pred: ndarray) -> Dict[str, int]

   Calculates confusion matrix for binary predictions.

   :param y_true: True target values
   :param y_pred: Predicted values
   :returns: Dictionary with confusion matrix values

.. function:: calculate_roc_curve(y_true: Series, y_prob: ndarray) -> Tuple[ndarray, ndarray, float]

   Calculates ROC curve points and AUC.

   :param y_true: True target values
   :param y_prob: Predicted probabilities
   :returns: FPR, TPR, and AUC value

.. function:: calculate_precision_recall_curve(y_true: Series, y_prob: ndarray) -> Tuple[ndarray, ndarray, float]

   Calculates precision-recall curve points and average precision.

   :param y_true: True target values
   :param y_prob: Predicted probabilities
   :returns: Precision, recall, and average precision

.. function:: compare_models(models: List[BaseModel], features: DataFrame, targets: Series, metrics: List[str] = None) -> DataFrame

   Compares performance of multiple models.

   :param models: List of models to compare
   :param features: Feature data for evaluation
   :param targets: Target values for evaluation
   :param metrics: Metrics to calculate
   :returns: DataFrame with metrics for each model

.. function:: generate_evaluation_report(model: BaseModel, features: DataFrame, targets: Series, output_path: str = None) -> Dict[str, Any]

   Generates comprehensive evaluation report for a model.

   :param model: Model to evaluate
   :param features: Feature data for evaluation
   :param targets: Target values for evaluation
   :param output_path: Optional path to save the report
   :returns: Dictionary with evaluation results and report metadata

.. class:: ModelEvaluator

   Class for comprehensive model evaluation.

   .. method:: evaluate(model: BaseModel, features: DataFrame, targets: Series, metrics: List[str] = None) -> Dict[str, float]

      Evaluate a model with specified metrics.

      :param model: Model to evaluate
      :param features: Feature data for evaluation
      :param targets: Target values for evaluation
      :param metrics: Metrics to calculate
      :returns: Dictionary of evaluation metrics

   .. method:: compare_models(models: List[BaseModel], features: DataFrame, targets: Series) -> DataFrame

      Compare multiple models on the same dataset.

      :param models: List of models to compare
      :param features: Feature data for evaluation
      :param targets: Target values for evaluation
      :returns: DataFrame with metrics for each model

   .. method:: generate_report(model: BaseModel, features: DataFrame, targets: Series, output_path: str = None) -> Dict[str, Any]

      Generate comprehensive evaluation report.

      :param model: Model to evaluate
      :param features: Feature data for evaluation
      :param targets: Target values for evaluation
      :param output_path: Optional path to save the report
      :returns: Evaluation report dictionary

.. class:: ThresholdOptimizer

   Class for finding optimal probability threshold for binary classification.

   .. method:: find_optimal_threshold(y_true: Series, y_prob: ndarray, thresholds: List[float] = None) -> float

      Find the optimal threshold that maximizes the specified metric.

      :param y_true: True target values
      :param y_prob: Predicted probabilities
      :param thresholds: List of thresholds to evaluate
      :returns: Optimal threshold value

   .. method:: optimize_for_model(model: BaseModel, features: DataFrame, targets: Series, thresholds: List[float] = None) -> float

      Find the optimal threshold for a specific model.

      :param model: Model to optimize threshold for
      :param features: Feature data for evaluation
      :param targets: Target values for evaluation
      :param thresholds: List of thresholds to evaluate
      :returns: Optimal threshold value

Model Persistence
----------------

Functions and classes for model persistence and versioning.

persistence
~~~~~~~~~~

Implements model persistence functionality.

.. function:: save_model_to_path(model: Any, path: Path) -> Path

   Saves a model object to a specified file path using joblib.

   :param model: Model object to save
   :param path: Path to save the model
   :returns: Path to the saved model file

.. function:: load_model_from_path(path: Path) -> Any

   Loads a model object from a specified file path using joblib.

   :param path: Path to the model file
   :returns: Loaded model object

.. function:: save_metadata(metadata: Dict[str, Any], path: Path) -> Path

   Saves model metadata to a JSON file.

   :param metadata: Metadata to save
   :param path: Path to save the metadata
   :returns: Path to the saved metadata file

.. function:: load_metadata(path: Path) -> Dict[str, Any]

   Loads model metadata from a JSON file.

   :param path: Path to the metadata file
   :returns: Loaded metadata dictionary

.. function:: list_models(model_type: str = None, base_path: Path = None) -> Dict[str, Dict[str, Dict[str, Any]]]

   Lists all available models in the model directory.

   :param model_type: Optional model type to filter by
   :param base_path: Optional base path to the model directory
   :returns: Dictionary of available models with their metadata

.. class:: ModelPersistence

   Class that handles model persistence operations with a consistent interface.

   .. method:: save_model(model: Any, metadata: Dict[str, Any], custom_path: Path = None) -> Path

      Saves a model and its metadata to storage.

      :param model: Model object to save
      :param metadata: Model metadata
      :param custom_path: Optional custom path to save the model
      :returns: Path to the saved model directory

   .. method:: load_model(model_id: str, model_type: str, version: str = None, custom_path: Path = None) -> Tuple[Any, Dict[str, Any]]

      Loads a model and its metadata from storage.

      :param model_id: Model ID to load
      :param model_type: Model type to load
      :param version: Optional version to load
      :param custom_path: Optional custom path to load the model from
      :returns: Tuple of (model, metadata)

   .. method:: list_models(model_type: str = None) -> Dict[str, Dict[str, Dict[str, Any]]]

      Lists all available models in the model directory.

      :param model_type: Optional model type to filter by
      :returns: Dictionary of available models with their metadata

versioning
~~~~~~~~~

Implements semantic versioning functionality for machine learning models.

.. function:: parse_version(version: str) -> Tuple[int, int, int]

   Parses a version string into its major, minor, and patch components.

   :param version: Version string to parse
   :returns: Tuple of (major, minor, patch) version components

.. function:: increment_version(current_version: str, increment_type: str = 'patch') -> str

   Increments a version string according to semantic versioning rules.

   :param current_version: Current version string
   :param increment_type: Type of increment ('major', 'minor', or 'patch')
   :returns: Incremented version string

.. function:: compare_versions(version1: str, version2: str) -> int

   Compares two version strings according to semantic versioning rules.

   :param version1: First version string
   :param version2: Second version string
   :returns: 1 if version1 > version2, -1 if version1 < version2, 0 if equal

.. function:: get_latest_version(versions: List[str]) -> Optional[str]

   Determines the latest version from a list of version strings.

   :param versions: List of version strings
   :returns: Latest version string or None if the list is empty

.. class:: VersionManager

   Class that manages version information for models.

   .. method:: get_latest_version(model_type: str, model_id: str) -> Optional[str]

      Gets the latest version for a specific model.

      :param model_type: Model type
      :param model_id: Model ID
      :returns: Latest version string or None if no versions exist

   .. method:: create_new_version(model_type: str, model_id: str, base_version: str = None, increment_type: str = 'patch') -> str

      Creates a new version for a model based on an existing version.

      :param model_type: Model type
      :param model_id: Model ID
      :param base_version: Optional base version
      :param increment_type: Type of increment ('major', 'minor', or 'patch')
      :returns: New version string

   .. method:: list_versions(model_type: str, model_id: str) -> List[str]

      Lists all versions for a specific model.

      :param model_type: Model type
      :param model_id: Model ID
      :returns: List of version strings