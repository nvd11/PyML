
import src.configs.config
from src.configs.config import yaml_configs
from google.cloud import bigquery
from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.model1.model1 import LinearRegressionModel
import matplotlib.pyplot as plt

# get area and house price data from bigquery
def get_data():
    client = bigquery.Client()
    gcp_project = yaml_configs['gcp']['project']
    table_id = yaml_configs['modeling']['model1']['table1']
    bq_table = gcp_project + "." + table_id

    query = f"SELECT * FROM {bq_table}"
    df = client.query(query).to_dataframe()
    return df




df = get_data()
logger.info("columns: {}".format(df.columns))

df = df[['area', 'price']]
logger.info(f"df: {df}")


# df['col'] return a ndarray
logger.info(f"type(df['area'].values): {type(df['area'].values)}")


# data preparation
# view(-1, 1) is used to reshape the tensor, 
# -1 means the number of rows is unknown,
# 1 means the number of columns is 1
X = torch.tensor(df['area'].values, dtype=torch.float32).view(-1, 1)
Y = torch.tensor(df['price'].values, dtype=torch.float32).view(-1, 1)

# 标准化数据
# X = (X - X.mean()) / X.std()
# Y = (y - y.mean()) / y.std()


logger.info(f"X: {X}") 
logger.info(f"Y: {Y}") 
logger.info(f"type(X): {type(X)}") # torch.Tensor


model = LinearRegressionModel()


# define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0000001)

# train the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 绘制结果
model.eval()
predicted = model(X).detach().numpy() 

logger.info(f"predicted: {predicted}")
logger.info("type(predicted): {}".format(type(predicted)))

plt.scatter(df['area'], df['price'], label='Original data')
plt.plot(df['area'], predicted, label='Fitted line', color='r')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()
plt.show()