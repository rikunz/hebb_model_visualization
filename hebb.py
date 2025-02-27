import numpy as np

class Hebb:
    def __init__(self, training_data, target_data, threshold=0):
        """
        Initializes a Hebb (Hebbian Extended Perceptron Network) model with training data and target.
            :param training_data: Input data for training the model, will be converted to numpy array if not already
            :param target_data: Target/output data corresponding to the training data
            :param threshold: The activation threshold for the perceptron, defaults to 0
        """
        self.model_name = 'Hebb'
        # Model Section
        self.training_data = training_data if type(training_data) == np.ndarray else np.array(training_data)
        self.target_data = target_data
        self.weights = np.zeros((25,))
        self.bias = 0
        self.threshold = threshold
        # Training Section
        self.epochs = 1
        self.current_data = 0
        self.num_weight_not_change = 0
    def get_weights(self):
        return self.weights
    def get_bias(self):
        return self.bias
    def get_model_parameter(self):
        return self.weights, self.bias
    def train_step(self):
        if self.num_weight_not_change == len(self.training_data):
            return f"Model sudah konvergen pada epoch ke-{self.epochs}"
        if self.epochs >= 10:
            return f"Epoch sudah mencapai batas maksimal"
        data = self.training_data[self.current_data]
        target = self.target_data[self.current_data]
        bias = 1
        activation = np.dot(data, self.weights) + self.bias * bias
        if activation >= self.threshold:
            activation = 1
        else:
            activation = -1
        if (activation == target):
            self.num_weight_not_change += 1
            return
        deltaData = data * target
        deltaBias = bias * target
        self.weights += deltaData
        self.bias += deltaBias
        self.current_data = (self.current_data + 1) % len(self.training_data)
        if (self.current_data == 0):
            self.epochs += 1
        self.num_weight_not_change = 0
        return
    def train(self):
        while self.num_weight_not_change < len(self.training_data) and self.epochs < 10:
            self.train_step()
        return f"Model sudah konvergen pada epoch ke-{self.epochs}"
    def get_epoch(self):
        return self.epochs
    def predict(self, input_data):
        activation = np.dot(input_data, self.weights) + self.bias
        return 1 if activation >= self.threshold else -1
    
if __name__ =="__main__":
    hebbmmodel = Hebb(np.array
    ([[1,1,1,1,1,
        0,0,1,0,0,
        0,0,1,0,0,
        0,0,1,0,0,
        0,0,1,0,0], 
      [1,1,1,1,1,
        1,0,0,0,0,
        1,1,1,1,1,
        1,0,0,0,0,
        1,1,1,1,1]]), [1, -1])   
    print("before training")
    print(hebbmmodel.get_epoch())
    print(hebbmmodel.get_weights())
    print(hebbmmodel.get_bias())
    print(hebbmmodel.get_model_parameter())
    
    hebbmmodel.train_step()
    print(hebbmmodel.get_epoch())
    print(hebbmmodel.get_weights())
    print(hebbmmodel.get_bias())
    print(hebbmmodel.get_model_parameter())