import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.random.randn(1, self.hidden_size)
        
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.random.randn(1, self.output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def feedforward(self, inputs):
        # Calculate output from input to hidden layer
        hidden_inputs = np.dot(inputs, self.weights_input_hidden) + self.bias_input_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)
        
        # Calculate final output
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output) + self.bias_hidden_output
        final_outputs = self.sigmoid(final_inputs)
        
        return final_outputs
    
    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            # Forward pass
            hidden_inputs = np.dot(inputs, self.weights_input_hidden) + self.bias_input_hidden
            hidden_outputs = self.sigmoid(hidden_inputs)
            
            final_inputs = np.dot(hidden_outputs, self.weights_hidden_output) + self.bias_hidden_output
            final_outputs = self.sigmoid(final_inputs)
            
            # Backpropagation
            output_errors = targets - final_outputs
            output_delta = output_errors * self.sigmoid_derivative(final_outputs)
            
            hidden_errors = np.dot(output_delta, self.weights_hidden_output.T)
            hidden_delta = hidden_errors * self.sigmoid_derivative(hidden_outputs)
            
            # Update weights and biases
            self.weights_hidden_output += np.dot(hidden_outputs.T, output_delta)
            self.bias_hidden_output += np.sum(output_delta, axis=0, keepdims=True)
            
            self.weights_input_hidden += np.dot(inputs.T, hidden_delta)
            self.bias_input_hidden += np.sum(hidden_delta, axis=0, keepdims=True)
            
            # Print error
            error = np.mean(np.abs(output_errors))
            print(f'Epoch {epoch+1}/{epochs}, Error: {error}')
            
# Example usage
if __name__ == "__main__":
    # Define neural network parameters
    input_size = 2
    hidden_size = 3
    output_size = 1

    # Create neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size)

    # Example training data
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])

    # Train the neural network
    nn.train(inputs, targets, epochs=1000)

    # Test the trained neural network
    test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = nn.feedforward(test_inputs)
    print("Predictions after training:")
    print(predictions)
