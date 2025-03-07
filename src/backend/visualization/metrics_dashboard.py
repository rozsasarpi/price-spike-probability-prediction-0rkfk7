# src/backend/visualization/metrics_dashboard.py
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import datetime
from pathlib import Path

import pandas  # version 2.0+
import numpy  # version 1.24+
import plotly  # version 5.14+
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash  # version 2.9+
import dash.dcc as dcc
import dash.html as html
from dash.dependencies import Input, Output, State

from ..utils.type_definitions import DataFrameType, SeriesType, ArrayType, PathType, ModelType
from ..utils.logging import get_logger
from ..utils.error_handling import handle_errors, VisualizationError
from .forecast_plots import ForecastPlotter, create_forecast_dashboard
from .performance_plots import ModelPerformancePlotter, create_performance_dashboard
from .feature_importance import FeatureImportancePlotter, create_feature_importance_dashboard
from .export import export_plotly_figure_to_file, figure_to_base64, plotly_figure_to_base64
from ..data.storage.forecast_repository import ForecastRepository
from ..data.storage.model_registry import ModelRegistry

logger = get_logger(__name__)

DEFAULT_PORT = 8050
DEFAULT_HOST = "127.0.0.1"
DEFAULT_DEBUG = False
DEFAULT_THEME = "plotly"
DEFAULT_TITLE = "ERCOT RTLMP Spike Prediction Dashboard"
DEFAULT_REFRESH_INTERVAL = 300  # seconds

@handle_errors(logger, VisualizationError)
def create_metrics_dashboard(
    forecast: Optional[Union[DataFrameType, datetime.datetime, str, None]] = None,
    model_id: Optional[str] = None,
    version: Optional[str] = None,
    dashboard_config: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    Creates a comprehensive dashboard combining forecast, model performance, and feature importance visualizations
    """
    # Initialize dashboard configuration with defaults or provided values
    config = {
        'title': DEFAULT_TITLE,
        'theme': DEFAULT_THEME,
        'show_metrics': True,
        'show_feature_importance': True
    }
    if dashboard_config:
        config.update(dashboard_config)

    # Create a subplot figure with appropriate layout using make_subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Forecast Visualization", "Model Performance", "Feature Importance"),
        vertical_spacing=0.1
    )

    # Initialize ForecastPlotter and load forecast data
    forecast_plotter = ForecastPlotter()
    forecast_plotter.load_forecast(forecast=forecast)
    forecast_fig = forecast_plotter.create_interactive_timeline()
    fig.add_trace(forecast_fig.data[0], row=1, col=1)

    # If model_id is provided, initialize ModelPerformancePlotter and load model data
    if model_id:
        performance_plotter = ModelPerformancePlotter()
        performance_plotter.load_model_data(model_id=model_id, version=version)
        performance_fig = performance_plotter.create_interactive_roc_curve()
        fig.add_trace(performance_fig.data[0], row=2, col=1)

        feature_importance_plotter = FeatureImportancePlotter()
        feature_importance_plotter.load_model_data(model_id=model_id, version=version)
        feature_importance_fig = feature_importance_plotter.create_interactive_feature_importance()
        fig.add_trace(feature_importance_fig.data[0], row=3, col=1)

    # Configure layout with appropriate titles, spacing, and theme
    fig.update_layout(title_text=config['title'], template=config['theme'])

    # Return the Plotly figure object
    return fig

@handle_errors(logger, VisualizationError)
def save_dashboard_to_html(
    fig: go.Figure,
    output_path: PathType,
    include_plotlyjs: Optional[bool] = True,
    full_html: Optional[bool] = True
) -> PathType:
    """
    Saves a dashboard figure to an HTML file
    """
    # Validate the figure object is not None
    if fig is None:
        raise VisualizationError("Figure object is None")

    # Ensure the output directory exists, create if necessary
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Call export_plotly_figure_to_file with the figure and output path
    export_plotly_figure_to_file(fig, output_path, include_plotlyjs, full_html)

    # Return the path to the saved HTML file
    return Path(output_path)

@handle_errors(logger, VisualizationError)
def dashboard_to_base64(
    fig: go.Figure,
    format: str
) -> str:
    """
    Converts a dashboard figure to a base64-encoded string
    """
    # Validate the figure object is not None
    if fig is None:
        raise VisualizationError("Figure object is None")

    # Call plotly_figure_to_base64 with the figure and format
    base64_string = plotly_figure_to_base64(fig, format)

    # Return the base64-encoded string
    return base64_string

class MetricsDashboard:
    """
    Class for creating and managing interactive metrics dashboards
    """
    def __init__(
        self,
        forecast_repository_path: Optional[PathType] = None,
        model_registry_path: Optional[PathType] = None,
        dashboard_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the MetricsDashboard with repositories and configuration
        """
        # Initialize ForecastRepository with forecast_repository_path
        self._forecast_repository = ForecastRepository(forecast_repository_path)

        # Initialize ModelRegistry with model_registry_path
        self._model_registry = ModelRegistry(model_registry_path)

        # Initialize ForecastPlotter with forecast_repository_path
        self._forecast_plotter = ForecastPlotter(forecast_repository_path)

        # Initialize ModelPerformancePlotter with model_registry_path
        self._performance_plotter = ModelPerformancePlotter(model_registry_path)

        # Initialize FeatureImportancePlotter with model_registry_path
        self._feature_plotter = FeatureImportancePlotter(model_registry_path)

        # Initialize dashboard configuration with defaults
        self._dashboard_config = {
            'title': DEFAULT_TITLE,
            'theme': DEFAULT_THEME,
            'refresh_interval': DEFAULT_REFRESH_INTERVAL
        }

        # If dashboard_config is provided, update the default configuration
        if dashboard_config:
            self._dashboard_config.update(dashboard_config)

        # Initialize _app as None (will be created when run_dashboard is called)
        self._app = None

        # Initialize _figures as an empty dictionary
        self._figures = {}

    def create_dashboard(
        self,
        forecast: Optional[Union[DataFrameType, datetime.datetime, str, None]] = None,
        model_id: Optional[str] = None,
        version: Optional[str] = None
    ) -> go.Figure:
        """
        Creates a comprehensive dashboard with all visualizations
        """
        # Call create_metrics_dashboard with the provided parameters and _dashboard_config
        dashboard_figure = create_metrics_dashboard(
            forecast=forecast,
            model_id=model_id,
            version=version,
            dashboard_config=self._dashboard_config
        )

        # Store the resulting figure in _figures dictionary
        self._figures['comprehensive'] = dashboard_figure

        # Return the dashboard figure
        return dashboard_figure

    def create_web_dashboard(self) -> dash.Dash:
        """
        Creates an interactive web dashboard using Dash
        """
        # Initialize a new Dash application with the title from _dashboard_config
        app = dash.Dash(__name__, title=self._dashboard_config['title'])

        # Create layout with tabs for different dashboard sections
        app.layout = html.Div([
            html.H1(self._dashboard_config['title']),
            dcc.Tabs([
                dcc.Tab(label='Forecast', children=[
                    html.Div([
                        html.H3("Forecast Visualization"),
                        dcc.Graph(id='forecast-graph'),
                        html.Div([
                            html.Label("Threshold:"),
                            dcc.Dropdown(
                                id='threshold-dropdown',
                                options=[{'label': str(th), 'value': str(th)} for th in [50, 100, 200]],
                                value='100'
                            ),
                            html.Label("Node:"),
                            dcc.Dropdown(
                                id='node-dropdown',
                                options=[{'label': node, 'value': node} for node in ['HB_NORTH', 'HB_SOUTH']],
                                value='HB_NORTH'
                            )
                        ])
                    ])
                ]),
                dcc.Tab(label='Model Performance', children=[
                    html.Div([
                        html.H3("Model Performance"),
                        dcc.Graph(id='performance-graph'),
                        html.Div([
                            html.Label("Model:"),
                            dcc.Dropdown(
                                id='model-dropdown',
                                options=[{'label': model, 'value': model} for model in ['XGBoost', 'LightGBM']],
                                value='XGBoost'
                            )
                        ])
                    ])
                ]),
                dcc.Tab(label='Feature Importance', children=[
                    html.Div([
                        html.H3("Feature Importance"),
                        dcc.Graph(id='feature-graph')
                    ])
                ]),
                dcc.Tab(label='System Metrics', children=[
                    html.Div([
                        html.H3("System Metrics"),
                        html.P("Time Range Selector")
                    ])
                ])
            ])
        ])

        # Set up callbacks for interactive elements
        @app.callback(
            Output('forecast-graph', 'figure'),
            [Input('threshold-dropdown', 'value'),
             Input('node-dropdown', 'value')]
        )
        def update_forecast_graph(threshold, node):
            # Load the specified forecast data into _forecast_plotter
            self._forecast_plotter.load_forecast()
            # Create forecast dashboard with the specified thresholds and nodes
            return self._forecast_plotter.create_interactive_timeline(threshold=threshold, node_id=node)

        # Store the Dash app in _app
        self._app = app

        # Return the Dash app
        return app

    def run_dashboard(self, host: Optional[str] = None, port: Optional[int] = None, debug: Optional[bool] = None):
        """
        Runs the interactive web dashboard
        """
        # If _app is None, call create_web_dashboard to create it
        if self._app is None:
            self.create_web_dashboard()

        # Set host to provided value or DEFAULT_HOST
        use_host = host if host is not None else DEFAULT_HOST

        # Set port to provided value or DEFAULT_PORT
        use_port = port if port is not None else DEFAULT_PORT

        # Set debug to provided value or DEFAULT_DEBUG
        use_debug = debug if debug is not None else DEFAULT_DEBUG

        # Run the Dash app with the specified host, port, and debug settings
        self._app.run_server(host=use_host, port=use_port, debug=use_debug)

    def update_forecast_visualization(
        self,
        forecast: Union[DataFrameType, datetime.datetime, str, None],
        thresholds: Optional[List[float]] = None,
        nodes: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Updates the forecast visualization in the dashboard
        """
        # Load the specified forecast data into _forecast_plotter
        self._forecast_plotter.load_forecast(forecast=forecast, thresholds=thresholds, nodes=nodes)

        # Create forecast dashboard with the specified thresholds and nodes
        forecast_figure = self._forecast_plotter.create_interactive_timeline()

        # Update the forecast figure in _figures dictionary
        self._figures['forecast'] = forecast_figure

        # Return the updated figure
        return forecast_figure

    def update_model_performance_visualization(self, model_id: str, version: Optional[str] = None) -> go.Figure:
        """
        Updates the model performance visualization in the dashboard
        """
        # Load the specified model data into _performance_plotter
        self._performance_plotter.load_model_data(model_id=model_id, version=version)

        # Create performance dashboard for the specified model
        performance_figure = self._performance_plotter.create_interactive_roc_curve()

        # Update the performance figure in _figures dictionary
        self._figures['performance'] = performance_figure

        # Return the updated figure
        return performance_figure

    def update_feature_importance_visualization(
        self,
        model_id: str,
        version: Optional[str] = None,
        n_features: Optional[int] = None
    ) -> go.Figure:
        """
        Updates the feature importance visualization in the dashboard
        """
        # Load the specified model data into _feature_plotter
        self._feature_plotter.load_model_data(model_id=model_id, version=version)

        # Create feature importance dashboard with the specified number of features
        feature_importance_figure = self._feature_plotter.create_interactive_feature_importance()

        # Update the feature importance figure in _figures dictionary
        self._figures['feature_importance'] = feature_importance_figure

        # Return the updated figure
        return feature_importance_figure

    def save_dashboard_to_html(
        self,
        output_path: PathType,
        dashboard_type: Optional[str] = None,
        include_plotlyjs: Optional[bool] = True,
        full_html: Optional[bool] = True
    ) -> PathType:
        """
        Saves the dashboard to an HTML file
        """
        # If dashboard_type is specified, get the corresponding figure from _figures
        if dashboard_type:
            fig = self._figures.get(dashboard_type)
            if fig is None:
                raise VisualizationError(f"Dashboard type '{dashboard_type}' not found")
        else:
            # If dashboard_type is not specified, create a comprehensive dashboard
            fig = self.create_dashboard()

        # Call save_dashboard_to_html function with the figure and output path
        return save_dashboard_to_html(fig, output_path, include_plotlyjs, full_html)

    def get_dashboard_as_base64(self, dashboard_type: Optional[str] = None, format: str = 'png') -> str:
        """
        Gets the dashboard as a base64-encoded string
        """
        # If dashboard_type is specified, get the corresponding figure from _figures
        if dashboard_type:
            fig = self._figures.get(dashboard_type)
            if fig is None:
                raise VisualizationError(f"Dashboard type '{dashboard_type}' not found")
        else:
            # If dashboard_type is not specified, create a comprehensive dashboard
            fig = self.create_dashboard()

        # Call dashboard_to_base64 function with the figure and format
        return dashboard_to_base64(fig, format)

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Gets a list of available models from the registry
        """
        # Query the model registry for available models
        models = self._model_registry.list_models()

        # Format the model information into a list of dictionaries
        model_list = []
        for model_type, model_ids in models.items():
            for model_id, versions in model_ids.items():
                model_list.append({
                    'model_type': model_type,
                    'model_id': model_id,
                    'versions': versions
                })

        # Return the list of model information
        return model_list

    def get_available_forecasts(self, start_date: Optional[datetime.datetime] = None, end_date: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
        """
        Gets a list of available forecasts from the repository
        """
        # Query the forecast repository for available forecasts within the date range
        forecasts = self._forecast_repository.list_available_forecasts()

        # Format the forecast information into a list of dictionaries
        forecast_list = []
        for forecast in forecasts:
            forecast_list.append({
                'file_path': forecast['file_path'],
                'forecast_timestamp': forecast['forecast_timestamp']
            })

        # Return the list of forecast information
        return forecast_list