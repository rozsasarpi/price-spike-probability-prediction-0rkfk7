# Scheduled Execution

## Overview

This document provides instructions for setting up and managing scheduled execution of the ERCOT RTLMP spike prediction system. Proper scheduling is critical to ensure that daily inference runs complete before the day-ahead market closure and that model retraining occurs on the required bi-daily cadence.

> **Note:** Before configuring scheduled execution, ensure you have completed the initial system setup. For environment setup instructions, refer to the local setup documentation in the deployment section.

## Scheduling Requirements

The ERCOT RTLMP spike prediction system requires the following scheduled tasks:

| Task | Schedule | Execution Window | Priority | Purpose |
|------|----------|------------------|----------|--------|
| Data Fetching | Daily, 00:00 | 1 hour | High | Retrieve latest ERCOT and weather data |
| Inference | Daily, 06:00 | 2 hours | Critical | Generate 72-hour price spike forecasts |
| Model Retraining | Every 2 days, 01:00 | 4 hours | Medium | Update prediction models with latest data |
| Backtesting | Weekly, Sunday 02:00 | 6 hours | Low | Evaluate model performance on historical data |

## Scheduling Methods

The system supports two approaches for scheduling execution:

1. **Python-based Scheduler**: Using the built-in scheduler module
2. **System Scheduler**: Using cron (Linux/macOS) or Task Scheduler (Windows)

### Python-based Scheduler

The system includes a Python-based scheduler implemented using the `scheduler.py` module in the `orchestration` package. This approach is recommended for development environments or when you need more complex scheduling logic.

#### Configuration

1. Edit the scheduler configuration in `config/hydra/config.yaml`:

```yaml
scheduler:
  data_fetching:
    enabled: true
    cron_expression: "0 0 * * *"  # Daily at midnight
    timeout_minutes: 60
    retry_count: 3
    priority: high
  
  inference:
    enabled: true
    cron_expression: "0 6 * * *"  # Daily at 6 AM
    timeout_minutes: 120
    retry_count: 2
    priority: critical
    
  model_retraining:
    enabled: true
    cron_expression: "0 1 */2 * *"  # Every 2 days at 1 AM
    timeout_minutes: 240
    retry_count: 1
    priority: medium
    
  backtesting:
    enabled: true
    cron_expression: "0 2 * * 0"  # Every Sunday at 2 AM
    timeout_minutes: 360
    retry_count: 1
    priority: low
```

2. Start the scheduler as a long-running process:

```bash
python -m rtlmp_predict.orchestration.scheduler
```

3. For production environments, configure the scheduler to run as a service using systemd or similar.

### System Scheduler (Cron)

For production environments, using the system's native scheduler is recommended for reliability.

#### Linux/macOS (Cron)

1. Create shell scripts for each task in the `infrastructure/scripts/` directory:
   - `daily_data_fetch.sh`
   - `daily_inference.sh`
   - `model_retraining.sh`
   - `weekly_backtesting.sh`

2. Configure crontab by running `crontab -e` and adding:

```
# ERCOT RTLMP Spike Prediction System Schedule
0 0 * * * /path/to/infrastructure/scripts/daily_data_fetch.sh
0 6 * * * /path/to/infrastructure/scripts/daily_inference.sh
0 1 */2 * * /path/to/infrastructure/scripts/model_retraining.sh
0 2 * * 0 /path/to/infrastructure/scripts/weekly_backtesting.sh
```

#### Windows (Task Scheduler)

1. Create batch scripts for each task in the `infrastructure/scripts/` directory:
   - `daily_data_fetch.bat`
   - `daily_inference.bat`
   - `model_retraining.bat`
   - `weekly_backtesting.bat`

2. Use the Windows Task Scheduler to create tasks that run these scripts on the required schedule.

## Error Handling and Retry Logic

The system implements robust error handling to ensure critical tasks complete successfully:

1. **Retry Logic**: Failed tasks automatically retry according to the configured retry_count.
2. **Fallback Mechanisms**: For inference tasks, if data fetching fails, the system falls back to cached data.
3. **Alerting**: Critical failures trigger notifications via email or logging systems.

### Configuring Error Handling

Edit the error recovery settings in `config/hydra/config.yaml`:

```yaml
error_recovery:
  data_fetching:
    retry_delay_seconds: 300  # 5 minutes
    max_retries: 3
    fallback: "use_cached_data"
    
  inference:
    retry_delay_seconds: 600  # 10 minutes
    max_retries: 2
    fallback: "use_previous_forecast"
    
  model_retraining:
    retry_delay_seconds: 1800  # 30 minutes
    max_retries: 1
    fallback: "retain_current_model"
```

## Monitoring Scheduled Execution

Monitor the execution of scheduled tasks using logs and the monitoring dashboard:

1. **Logging**: All scheduled tasks log their execution status to `logs/scheduler.log`
2. **Monitoring Dashboard**: The system includes a monitoring dashboard that shows:
   - Task execution status and history
   - Execution times and resource usage
   - Success/failure rates
   - Alert history

### Accessing the Monitoring Dashboard

If you're using the provided monitoring infrastructure with Grafana:

1. Start the monitoring services:

```bash
cd infrastructure/monitoring
docker-compose up -d
```

2. Access the Grafana dashboard at http://localhost:3000

## Handling DAM Closure Deadlines

The critical requirement is that forecasts must be available before the Day-Ahead Market (DAM) closure. The system ensures this by:

1. Scheduling the inference run at 06:00, well before the typical DAM closure time
2. Implementing monitoring to alert if the inference run is delayed
3. Using fallback mechanisms if the primary inference fails

### Configuring DAM Deadline Alerts

Edit the alert configuration in `config/hydra/config.yaml`:

```yaml
deadlines:
  dam_closure:
    time: "10:00"  # 10 AM
    buffer_minutes: 60  # 1 hour buffer
    alert_on_risk: true
    fallback_on_miss: true
```

## Best Practices

1. **Stagger Scheduling**: Start data fetching well before inference to ensure data availability
2. **Resource Allocation**: Schedule resource-intensive tasks like model retraining during off-hours
3. **Monitoring**: Continuously monitor execution times to detect performance degradation
4. **Redundancy**: Implement redundant alerting for critical failures
5. **Logging**: Maintain detailed logs for troubleshooting scheduling issues

## Troubleshooting

### Common Issues

1. **Task Not Running**:
   - Check if the scheduler service is running
   - Verify crontab entries or Windows Task Scheduler configurations
   - Check file permissions on scripts

2. **Execution Failures**:
   - Check `logs/error.log` for detailed error messages
   - Verify that prerequisites are available (data, storage space, etc.)
   - Test running the task manually to identify issues

3. **Missed Deadlines**:
   - Review task execution time trends for performance degradation
   - Consider allocating more resources or optimizing the code
   - Adjust schedule to provide more buffer time