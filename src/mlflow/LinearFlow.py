import src.configs.config
from loguru import logger
from src.mlflow.MLFlow import MLFlow


class LinearFlow(MLFlow):
    def __init__(self, training_data=None, model=None, predict_function=None,
                 loss_function=None, cost_function=None, learning_rate=None, slope=None):
        super().__init__(training_data, model, predict_function,
                        loss_function, cost_function, learning_rate)
       
        self.slope = slope


    

    def train(self):
        self.loss_function(4)