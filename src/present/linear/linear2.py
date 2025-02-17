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
logger.info("columns: {}".format(df.columns))


x_domain = np.linspace(750, 870, 100)
def line_fun1(x, w):
    return w * x



# present the points to plt
plt.scatter(df['area'], df['price'], label='Original data')
plt.plot(x_domain, line_fun1(x_domain, 180), label='y=wx')

# Draw perpendicular lines from each point to the graph of the prediction function
for x in df['area']:
    # plt.plot([x, x], [line_fun1(x, 180), df['price'][0]], color='red', linestyle='--')
    point_a = [x, df[df['area'] == x]['price'].values[0]]
    point_b = [x, line_fun1(x, 180)]
    color = "green"
    if point_a[1] < point_b[1]:
        color = "red"
    add_straight_line_to_plt(plt, point_a, point_b, color, linestyle='--')





plt.xlabel('Area')
plt.ylabel('Price')
plt.legend() # show the label
plt.show()