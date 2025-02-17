
import src.configs.config
from src.configs.config import yaml_configs
from google.cloud import bigquery
from loguru import logger
import pandas as pd
import uuid



client = bigquery.Client()


credentials = client._credentials
logger.info(f"Client project: {client.project}")
logger.info(f"Credentials: {credentials}")
logger.info(f"Service account email: {credentials.service_account_email}")

gcp_project = yaml_configs['gcp']['project']
logger.info(f"project: {gcp_project}")

table_id = yaml_configs['modeling']['model1']['table1']
logger.info(f"table_id: {table_id}")

bq_table = gcp_project + "." + table_id


data = {
    'area': [750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200],
    'price': [150000, 160000, 170000, 180000, 190000, 200000, 210000, 220000, 230000, 240000]
}
df = pd.DataFrame(data)


list_uuid = [str(uuid.uuid4()) for _ in range(len(df))]
logger.info(f"list_uuid: {list_uuid}, type: {type(list_uuid)}")

df['id'] = list_uuid
logger.info(f"df: {df}")

job = client.load_table_from_dataframe(df, bq_table)
job.result()

logger.info(f"Loaded {job.output_rows} rows into {table_id}.")