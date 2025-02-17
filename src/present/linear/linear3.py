from src.mathutils.linear.mes import mes
from src.mlflow.LinearFlow import LinearFlow
from src.pltutils.printfuncs import add_straight_line_to_plt
import src.configs.config
from src.configs.config import yaml_configs, project_path
from google.cloud import bigquery
from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.model1.model1 import LinearRegressionModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# get area and house price data from bigquery
def get_data():
    csv_path = project_path + "/src/present/linear/data2.csv"
    df = pd.read_csv(csv_path)
    return df

df = get_data()
# logger.info("columns: {}".format(df.columns))


mlflow = LinearFlow(slope=110)
mlflow.loss_function = mes
mlflow.train()