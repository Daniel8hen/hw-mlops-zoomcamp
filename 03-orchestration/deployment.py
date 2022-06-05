from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule


DeploymentSpec(
    name="cron-schedule-deployment-daniel",
    flow_location="/home/ubuntu/mlops-zoomcamp/03-orchestration/homework.py",
    schedule=CronSchedule(cron="0 9 15 * *"),
    tags=['flow_test_daniel_cron']
)
