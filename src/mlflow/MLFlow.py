class MLFlow:

    def __init__(self, training_data=None, model=None, predict_function=None,
                  loss_function=None, cost_function=None, learning_rate=None):
        self.training_data = training_data
        self.model = model
        self.predict_function = predict_function
        self.loss_function = loss_function
        self.cost_function = cost_function
        self.learning_rate = learning_rate


    

    # Getter for training_data
    @property
    def training_data(self):
        return self._training_data

    # Setter for training_data
    @training_data.setter
    def training_data(self, value):
        self._training_data = value

    # Getter for model
    @property
    def model(self):
        return self._model

    # Setter for model
    @model.setter
    def model(self, value):
        self._model = value

    # Getter for predict_function
    @property
    def predict_function(self):
        return self._predict_function

    # Setter for predict_function
    @predict_function.setter
    def predict_function(self, value):
        self._predict_function = value

    # Getter for loss_function
    @property
    def loss_function(self):
        return self._loss_function

    # Setter for loss_function
    @loss_function.setter
    def loss_function(self, value):
        self._loss_function = value

    # Getter for cost_function
    @property
    def cost_function(self):
        return self._cost_function

    # Setter for cost_function
    @cost_function.setter
    def cost_function(self, value):
        self._cost_function = value

    # Getter for learning_rate
    @property
    def learning_rate(self):
        return self._learning_rate

    # Setter for learning_rate
    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    def train(self):
        pass