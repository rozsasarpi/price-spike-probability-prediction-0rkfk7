"""
Implements the 'fetch-data' command for the ERCOT RTLMP spike prediction system CLI.
This command retrieves data from external sources including ERCOT market data, weather forecasts,
and grid conditions with options for date ranges, data types, and output formats.
"""
import click
from typing import Dict, List, Optional, Any, Union, cast
from pathlib import Path
from datetime import datetime
import pandas  # version 2.0+

# Internal imports
from ..cli_types import FetchDataParamsDict
from ..cli_types import DataType
from ..cli_types import OutputFormat
from ..utils.validators import validate_fetch_data_params
from ..utils.output_handlers import handle_command_output
from ..utils.output_handlers import export_dataframe
from ..utils.error_handlers import ErrorHandler
from ..logger import get_cli_logger
from ..exceptions import DataFetchError
from ...backend.api.data_api import DataAPI

# Initialize logger
logger = get_cli_logger('fetch_data_command')

# Initialize error handler
error_handler = ErrorHandler(verbose=True, exit_on_error=True)

# Initialize DataAPI
data_api = DataAPI()


@click.command('fetch-data', help='Fetch data from external sources')
@click.option('--data-type', '-t', type=click.Choice(['rtlmp', 'weather', 'grid_conditions', 'all']), required=True,
              help='Type of data to fetch')
@click.option('--start-date', '-s', type=click.DateTime(formats=['%Y-%m-%d']), required=True,
              help='Start date for data retrieval (YYYY-MM-DD)')
@click.option('--end-date', '-e', type=click.DateTime(formats=['%Y-%m-%d']), required=True,
              help='End date for data retrieval (YYYY-MM-DD)')
@click.option('--nodes', '-n', multiple=True, help='Node IDs to fetch data for (can specify multiple)')
@click.option('--output-path', '-o', type=Path, help='Path to save the fetched data')
@click.option('--output-format', '-f', type=click.Choice(['text', 'json', 'csv']), help='Output format')
@click.option('--force-refresh/--use-cache', default=False, help='Force refresh data from source instead of using cache')
@click.option('--verbose/--no-verbose', default=False, help='Show detailed output')
@error_handler.command_error_handler
def fetch_data_command(data_type: str, start_date: datetime.date, end_date: datetime.date, nodes: List[str],
                       output_path: Optional[Path], output_format: Optional[str], force_refresh: bool, verbose: bool) -> Dict[str, Any]:
    """
    Main function for the fetch-data command that retrieves data from external sources
    """
    logger.info(f"Starting fetch-data command execution with data_type={data_type}, start_date={start_date}, end_date={end_date}, "
                f"nodes={nodes}, output_path={output_path}, output_format={output_format}, force_refresh={force_refresh}, verbose={verbose}")

    # Validate the command parameters
    params: FetchDataParamsDict = {
        'data_type': cast(DataType, data_type),
        'start_date': start_date,
        'end_date': end_date,
        'nodes': nodes,
        'output_path': output_path,
        'output_format': cast(Optional[OutputFormat], output_format),
        'force_refresh': force_refresh
    }
    validated_params = validate_fetch_data_params(params)

    # Convert start_date and end_date to datetime.date objects if they are datetime objects
    if isinstance(validated_params['start_date'], datetime):
        validated_params['start_date'] = validated_params['start_date'].date()
    if isinstance(validated_params['end_date'], datetime):
        validated_params['end_date'] = validated_params['end_date'].date()

    # Fetch the requested data based on data_type parameter
    if validated_params['data_type'] == 'rtlmp':
        data = fetch_rtlmp_data(validated_params['start_date'], validated_params['end_date'], validated_params['nodes'], validated_params['force_refresh'])
    elif validated_params['data_type'] == 'weather':
        data = fetch_weather_data(validated_params['start_date'], validated_params['end_date'], validated_params['force_refresh'])
    elif validated_params['data_type'] == 'grid_conditions':
        data = fetch_grid_conditions(validated_params['start_date'], validated_params['end_date'], validated_params['force_refresh'])
    elif validated_params['data_type'] == 'all':
        data = fetch_all_data(validated_params['start_date'], validated_params['end_date'], validated_params['nodes'], validated_params['force_refresh'])
    else:
        raise ValueError(f"Invalid data_type: {validated_params['data_type']}")

    # Format the fetched data for output
    result = format_data_result(data, validated_params['data_type'])

    # Handle command output with appropriate formatting
    handle_command_output(result, validated_params['output_path'], validated_params['output_format'], verbose)

    # Return the fetched data and status
    return result


def fetch_rtlmp_data(start_date: datetime.date, end_date: datetime.date, nodes: List[str], force_refresh: bool) -> pandas.DataFrame:
    """
    Fetches historical RTLMP data for the specified date range and nodes
    """
    logger.info(f"Fetching RTLMP data for start_date={start_date}, end_date={end_date}, nodes={nodes}, force_refresh={force_refresh}")
    try:
        use_cache = not force_refresh
        df = data_api.get_historical_rtlmp(start_date=start_date, end_date=end_date, nodes=nodes, use_cache=use_cache)
        return df
    except Exception as e:
        raise DataFetchError(f"Failed to fetch RTLMP data: {e}", data_source="ERCOT API") from e


def fetch_weather_data(start_date: datetime.date, end_date: datetime.date, force_refresh: bool) -> pandas.DataFrame:
    """
    Fetches historical weather data for the specified date range
    """
    logger.info(f"Fetching weather data for start_date={start_date}, end_date={end_date}, force_refresh={force_refresh}")
    try:
        use_cache = not force_refresh
        df = data_api.get_historical_weather(start_date=start_date, end_date=end_date, use_cache=use_cache)
        return df
    except Exception as e:
        raise DataFetchError(f"Failed to fetch weather data: {e}", data_source="Weather API") from e


def fetch_grid_conditions(start_date: datetime.date, end_date: datetime.date, force_refresh: bool) -> pandas.DataFrame:
    """
    Fetches historical grid condition data for the specified date range
    """
    logger.info(f"Fetching grid condition data for start_date={start_date}, end_date={end_date}, force_refresh={force_refresh}")
    try:
        use_cache = not force_refresh
        df = data_api.get_historical_grid_conditions(start_date=start_date, end_date=end_date, use_cache=use_cache)
        return df
    except Exception as e:
        raise DataFetchError(f"Failed to fetch grid condition data: {e}", data_source="ERCOT API") from e


def fetch_all_data(start_date: datetime.date, end_date: datetime.date, nodes: List[str], force_refresh: bool) -> pandas.DataFrame:
    """
    Fetches combined historical data including RTLMP, weather, and grid conditions
    """
    logger.info(f"Fetching all data for start_date={start_date}, end_date={end_date}, nodes={nodes}, force_refresh={force_refresh}")
    try:
        use_cache = not force_refresh
        df = data_api.get_combined_historical_data(start_date=start_date, end_date=end_date, nodes=nodes, use_cache=use_cache)
        return df
    except Exception as e:
        raise DataFetchError(f"Failed to fetch all data: {e}", data_source="Multiple") from e


def format_data_result(data: pandas.DataFrame, data_type: str) -> Dict[str, Any]:
    """
    Formats the fetched data result for output
    """
    result = {
        'data_type': data_type,
        'row_count': len(data)
    }

    # Add summary statistics for the data
    if not data.empty:
        result['summary'] = data.describe().to_dict()

    # Add timestamp range information
    if 'timestamp' in data.columns:
        result['start_timestamp'] = str(data['timestamp'].min())
        result['end_timestamp'] = str(data['timestamp'].max())

    # Add the actual data in a format suitable for output
    result['data'] = data.to_dict(orient='records')

    return result