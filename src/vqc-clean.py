import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import QuantumCircuit, ParameterVector

# Set the theme for seaborn
sns.set_theme()

# Importing and pre-processing the MNIST dataset
def load_preprocess_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
    return x_train, y_train, x_test, y_test

# Function to filter dataset for digits 0 and 1
def filter_01(x, y):
    keep = (y == 0) | (y == 1)
    return x[keep], y[keep]

# Normalize data
def normalize(data):
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return data / norm

# Resize images
def resize_images(x, size=(8, 8)):
    return tf.image.resize(x, size, method='nearest', preserve_aspect_ratio=True).numpy()

# Remove empty examples
def remove_empty(x, y):
    non_empty = np.sum(x, axis=1) != 0
    return x[non_empty], y[non_empty]

# Quantum circuit functions
def create_initial_circuit(n, inputs):
    circuit = QuantumCircuit(n, 1)
    circuit.initialize(inputs, range(n))
    circuit.barrier()
    return circuit

def create_variational_circuit(n, num_layers, params):
    parameters = ParameterVector('θ', 10 * num_layers)
    circuit = QuantumCircuit(n, 1)
    for layer in range(num_layers):
        for i in range(n):
            circuit.ry(parameters[layer * 10 + i], i)
        circuit.barrier()
        circuit.barrier()
    circuit.assign_parameters({p: val for p, val in zip(parameters, params)})
    return circuit

def create_swap_test_circuit(n):
    qc = QuantumCircuit(2 * n + 1, 1)
    qc.h(0)
    for i in range(n):
        qc.cswap(0, i + 1, 2 * n - i)
    qc.h(0)
    qc.barrier()
    return qc

# Main execution block
def main():
    x_train, y_train, x_test, y_test = load_preprocess_mnist()
    x_train, y_train = filter_01(x_train, y_train)
    x_test, y_test = filter_01(x_test, y_test)
    
    x_train = resize_images(x_train)
    x_test = resize_images(x_test)
    
    x_train = normalize(x_train.reshape(len(x_train), -1))
    x_test = normalize(x_test.reshape(len(x_test), -1))
    
    x_test, y_test = remove_empty(x_test, y_test)
    
    num_qubits = 6
    num_layers = 1
    params = np.random.random(10 * num_layers)
    
    initial_circuit = create_initial_circuit(num_qubits, x_train[0])
    variational_circuit = create_variational_circuit(num_qubits, num_layers, params)
    swap_test_circuit = create_swap_test_circuit(2)


if __name__ == "__main__":
    main()
