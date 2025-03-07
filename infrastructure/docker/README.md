# ERCOT RTLMP Spike Prediction System - Docker Deployment

This directory contains Docker configuration files for deploying the ERCOT RTLMP spike prediction system. The containerized deployment provides a consistent, isolated environment for running the prediction system with appropriate resource allocations and monitoring capabilities.

## System Components

The Docker deployment includes the following components:

- **Backend Service**: Core prediction system that handles data fetching, feature engineering, model training, and inference
- **CLI Service**: Command-line interface for interacting with the prediction system
- **Prometheus**: Monitoring system for collecting and querying metrics
- **Grafana**: Dashboard visualization platform for monitoring metrics
- **Node Exporter**: Exports host system metrics to Prometheus

### Backend Service

The backend service is the core component of the ERCOT RTLMP spike prediction system. It handles data fetching from ERCOT and weather APIs, feature engineering, model training, and inference operations. The service is configured to run scheduled tasks for daily inference and bi-daily model retraining.

### CLI Service

The CLI service provides a command-line interface for interacting with the prediction system. It allows users to manually trigger data fetching, model training, inference, and backtesting operations, as well as view results and generate visualizations.

### Monitoring Stack

The monitoring stack consists of Prometheus, Grafana, and Node Exporter. Prometheus collects metrics from the backend service and Node Exporter, while Grafana provides dashboards for visualizing these metrics. The monitoring stack helps track system health, performance, and model quality.

## Prerequisites

Before deploying the system, ensure you have the following prerequisites installed:

- Docker Engine (20.10+)
- Docker Compose (2.0+)
- At least 16GB of RAM available for containers
- At least 100GB of disk space for data, models, and logs
- Network connectivity to ERCOT and weather APIs

## Quick Start

To quickly deploy the system with default settings:

1. Copy `.env.example` to `.env` and update the required environment variables
2. Run `docker-compose up -d` to start all services
3. Access Grafana at http://localhost:3000 (default credentials: admin/change_me_in_production)
4. Access Prometheus at http://localhost:9090

```bash
# Clone the repository (if not already done)
git clone https://github.com/your-org/ercot-rtlmp-prediction.git
cd ercot-rtlmp-prediction/infrastructure/docker

# Create .env file from example
cp .env.example .env

# Edit .env file to set required variables
nano .env

# Start the services
docker-compose up -d

# Check service status
docker-compose ps
```

## Configuration

The system is configured through environment variables defined in the `.env` file. This file contains settings for all services, including API credentials, resource limits, and monitoring configuration.

### Environment Variables

The `.env.example` file provides a template with default values for all required environment variables. Copy this file to `.env` and update the values as needed. Key variables include:

- **ERCOT_API_KEY** and **ERCOT_API_SECRET**: Credentials for accessing ERCOT data
- **WEATHER_API_KEY**: API key for weather data access
- **DEFAULT_NODE_IDS**: Comma-separated list of ERCOT node IDs to monitor
- **DEFAULT_PRICE_THRESHOLDS**: Comma-separated list of price thresholds for spike prediction
- **BACKEND_CPU_LIMIT** and **BACKEND_MEMORY_LIMIT**: Resource limits for the backend service
- **GRAFANA_ADMIN_PASSWORD**: Admin password for Grafana dashboard access

### Volume Mounts

The Docker Compose configuration mounts several volumes to persist data across container restarts:

- **./data**: Raw data from ERCOT and weather APIs
- **./models**: Trained model artifacts
- **./forecasts**: Generated forecasts
- **./logs**: Application and system logs
- **prometheus-data**: Prometheus time series data
- **grafana-data**: Grafana dashboards and configurations

Ensure these directories exist and have appropriate permissions before starting the services.

### Resource Allocation

Each service has configurable CPU and memory limits defined in the `.env` file. Adjust these values based on your available resources and workload requirements. The backend service requires the most resources, especially during model training operations.

## Usage

Once the system is deployed, you can interact with it through the CLI service or monitor its operation through Grafana dashboards.

### Running CLI Commands

You can execute CLI commands using Docker Compose:

```bash
# Show available commands
docker-compose exec rtlmp-cli python -m src.cli.main --help

# Fetch data manually
docker-compose exec rtlmp-cli python -m src.cli.main fetch-data --start-date 2023-01-01 --end-date 2023-01-31

# Train a model manually
docker-compose exec rtlmp-cli python -m src.cli.main train --start-date 2020-01-01 --end-date 2023-06-30

# Generate a forecast manually
docker-compose exec rtlmp-cli python -m src.cli.main predict --threshold 100 --node HB_NORTH

# Run backtesting
docker-compose exec rtlmp-cli python -m src.cli.main backtest --start-date 2022-01-01 --end-date 2022-12-31 --threshold 100
```

### Monitoring with Grafana

Grafana provides several dashboards for monitoring the system:

- **System Metrics**: Host and container resource utilization
- **Data Quality**: Metrics related to data completeness and validation
- **Model Performance**: Model accuracy, calibration, and prediction quality

Access Grafana at http://localhost:3000 and log in with the credentials defined in the `.env` file.

### Scheduled Operations

The backend service is configured to run scheduled operations:

- **Data Fetching**: Daily at 00:00
- **Inference**: Daily at 06:00 (before day-ahead market closure)
- **Model Retraining**: Every 2 days at 01:00

These schedules can be adjusted in the backend service configuration.

## Maintenance

Regular maintenance tasks for the containerized deployment include:

### Updating the System

To update the system to a new version:

```bash
# Pull the latest code
git pull

# Rebuild and restart the services
docker-compose down
docker-compose build
docker-compose up -d
```

### Backup and Recovery

To backup the system data:

```bash
# Stop the services
docker-compose down

# Backup the data directories
tar -czf rtlmp_backup_$(date +%Y%m%d).tar.gz data models forecasts logs

# Restart the services
docker-compose up -d
```

To restore from a backup:

```bash
# Stop the services
docker-compose down

# Restore the data directories
tar -xzf rtlmp_backup_YYYYMMDD.tar.gz

# Restart the services
docker-compose up -d
```

### Log Management

Container logs can be viewed using Docker Compose:

```bash
# View logs for all services
docker-compose logs

# View logs for a specific service
docker-compose logs rtlmp-backend

# Follow logs in real-time
docker-compose logs -f rtlmp-backend
```

Application logs are stored in the `./logs` directory and can be accessed directly.

### Monitoring Alerts

Prometheus is configured with alerting rules for various system conditions, including:

- Task execution failures
- Data fetch failures
- Model performance degradation
- Inference delays
- Data completeness issues

Alerts can be viewed in the Prometheus UI at http://localhost:9090/alerts.

## Troubleshooting

Common issues and their solutions:

### Container Startup Failures

If containers fail to start, check the logs for error messages:

```bash
docker-compose logs rtlmp-backend
```

Common issues include:

- Missing or incorrect environment variables in `.env`
- Insufficient system resources
- Network connectivity problems to external APIs

### Data Fetch Failures

If data fetching fails, check:

- API credentials in `.env`
- Network connectivity to ERCOT and weather APIs
- API rate limits and quotas

You can manually trigger data fetching to test the connection:

```bash
docker-compose exec rtlmp-cli python -m src.cli.main fetch-data --test-connection
```

### Model Training Issues

If model training fails, check:

- Available memory for the backend service
- Data completeness and quality
- Training configuration parameters

You can adjust the training parameters in the `.env` file or through the CLI.

### Monitoring Stack Issues

If Prometheus or Grafana are not working correctly, check:

- Container status with `docker-compose ps`
- Logs with `docker-compose logs prometheus` or `docker-compose logs grafana`
- Network connectivity between containers
- Volume permissions for prometheus-data and grafana-data

## Advanced Configuration

For advanced users, additional configuration options are available:

### Custom Prometheus Rules

You can add custom alerting rules by modifying the files in the `infrastructure/monitoring/rules` directory. After making changes, restart Prometheus:

```bash
docker-compose restart prometheus
```

### Custom Grafana Dashboards

You can add custom Grafana dashboards by placing JSON dashboard definitions in the `infrastructure/monitoring/grafana/dashboards` directory. After making changes, restart Grafana:

```bash
docker-compose restart grafana
```

### Scaling Resources

For larger workloads, you can adjust the resource allocations in the `.env` file. Key parameters include:

- **BACKEND_CPU_LIMIT** and **BACKEND_MEMORY_LIMIT**: Resource limits for the backend service
- **BACKEND_PARALLEL_JOBS**: Number of parallel jobs for data processing and model training
- **PROMETHEUS_RETENTION_TIME**: Data retention period for Prometheus

## References

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [ERCOT API Documentation](https://www.ercot.com/services/api)