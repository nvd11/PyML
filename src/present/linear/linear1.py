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
    csv_path = project_path + "/src/present/linear/data.csv"
    df = pd.read_csv(csv_path)
    return df

df = get_data()
logger.info("columns: {}".format(df.columns))


x_domain = np.linspace(750, 980, 100)
def line_fun1(x, w):
    return w * x



# present the points to plt
plt.scatter(df['area'], df['price'], label='Original data')
plt.plot(x_domain, line_fun1(x_domain, 180), label='y=wx')
plt.plot(x_domain, line_fun1(x_domain, 177), label='y=wx')
plt.plot(x_domain, line_fun1(x_domain, 187), label='y=wx')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend() # show the label
plt.show()