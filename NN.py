import numpy as np

class NN():
    def __init__(self,input_size,output_size,layers):
        self.weights = []
        input_matrix = np.random.rand(input_size,layers[0])
        self.weights.append(input_matrix)
        for i in range(len(layers)-1):
            first_layer = layers[i]
            second_layer = layers[i+1]
            weight_matrix =  np.random.rand(first_layer,second_layer)
            self.weights.append(weight_matrix)
        output_matrix = np.random.rand(layers[-1],output_size)
        self.weights.append(output_matrix)
    def softmax(self,x):
        return np.exp(x)/sum(np.exp(x))
    def predict(self,input):
        output = input
        for w in self.weights:
           output = np.maximum(output.dot(w), 0)
        softmax_output = self.softmax(output[0])
        return softmax_output
    def flatten(self):
        w_flattend = []
        for w in self.weights:
            w_flattend += list(w.flatten())
        return w_flattend
    def reconstruct(self,weights):
        new_weights = []
        start = 0
        for w in self.weights:
            shape = w.shape
            size_needed = shape[0]*shape[1]
            reshaped_weight = np.reshape(weights[start:start+size_needed],shape)
            start += size_needed
            new_weights.append(reshaped_weight)
        self.weights = new_weights
nn = NN(3,2,[5,3])
nn.predict(np.array([[1,2,3]]))
weights = nn.flatten()
nn.reconstruct(weights)
