import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative(x):
    return x * (1 - x)


error = None  # output error
predicted_output = None
final_prediction = None  # prediction with values 0 or 1

training_input = np.array([[0, 0, 1],
                           [1, 1, 1],
                           [0, 1, 1],
                           [1, 0, 1]])

training_output = np.array([[0, 0, 1, 1]]).T

np.random.seed(1)
weights_h = np.random.random((3, 2)) * 2 - 1
weights_o = np.random.random((2, 1)) * 2 - 1

output_bias = np.random.random((1, 1)) * 2 - 1

for i in range(50000):
    # forward feeding
    hidden_layer = np.dot(training_input, weights_h)
    hidden_layer_output = sigmoid(hidden_layer)

    output_layer = np.dot(hidden_layer_output, weights_o)
    output_layer += output_bias
    predicted_output = sigmoid(output_layer)

    # backpropagation
    error = training_output - predicted_output
    predicted_output_delta = error * derivative(predicted_output)

    error_h = predicted_output_delta.dot(weights_o.T)
    hidden_layer_delta = error_h * derivative(hidden_layer_output)

    # updating weigths
    weights_o += hidden_layer_output.T.dot(predicted_output_delta)
    output_bias += np.sum(predicted_output_delta, axis=0, keepdims=True)
    weights_h += training_input.T.dot(hidden_layer_delta)

    if i % 10000 == 0:
        print("Output error after {} iterations: ".format(str(i)))
        print(error)

print("Output error after 50000 iterations: ")
print(error)
final_prediction = np.copy(predicted_output)
final_prediction[final_prediction >= 0.4] = 1
final_prediction[final_prediction < 0.4] = 0

data = {'Inputs': training_input.tolist(), 'Outputs': training_output.tolist(),
        'Network prediction': predicted_output.tolist(), 'Network output': map(int, final_prediction)}


df = pd.DataFrame(data)
df['Outputs'] = df['Outputs'].str[0]


print(df)



