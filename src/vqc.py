import numpy as np
# Quiskit libraries 
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, execute, QuantumRegister, ClassicalRegister
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.circuit import Parameter, ParameterVector

#MNIST set libraries for the acquisition and pre-processing data.
import tensorflow as tf

#Graph libraries
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

#Loading the MNIST set divided by a train set and a test set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

#Showing the length of the train and test sets
print("Number of images in the training set:", len(x_train))
print("Number of images in the test set:", len(x_test))

#Function to filter the 0 and 1 labels of the MNIST set
'''
Input = x_label and y_label sets
Output = x_label and y_label sets filtered
'''
def filter_01(x, y):
    keep = (y == 0) | (y == 1)
    x, y = x[keep], y[keep]
    return x,y

x_train, y_train = filter_01(x_train, y_train) #Filter the train set
x_test, y_test = filter_01(x_test, y_test) #Filter the test set

#Showing the length of the train and test sets after filtering the data
print("Number of images in the training set:", len(x_train))
print("Number of images in the test set:", len(x_test))

#Plotting the first element of the train set
plt.imshow(x_train[0, :, :, 0])
plt.colorbar()

#resizing the image from 28x28 to 8x8 by the nearest method
x_train_small = tf.image.resize(x_train, (8,8), method='nearest', preserve_aspect_ratio=True).numpy()
x_test_small = tf.image.resize(x_test, (8,8), method='nearest', preserve_aspect_ratio=True).numpy()

#Plotting the first element of the train set after the resizing
plt.imshow(x_train_small[0,:,:,0], vmin=0, vmax=1)
plt.colorbar()

#Reshaping the train and test test to a 64x1 matriz
x_train = x_train_small.reshape(len(x_train_small), 64)
x_test = x_test_small.reshape(len(x_test_small), 64)

x_train = (x_train)
x_test = (x_test)

#Deleting no valuable information for the test set
k = 0

while k < len(x_test): #Deleting no valuable information for the training set
    a = x_test[k].copy()
    #Verfify if it has some valuable data
    if np.sum(a) == 0.:
        #If not has valuable data
        print(k,x_test[k])
        x_test = np.delete(x_test, k, axis=0) #Delete the actual element from the x_label
        y_test = np.delete(y_test, k, axis=0) #Delete the actual element from the y_label
        k -= 1 #Take back one value of the counter to match the new set length
    k+=1

import cmath
#Funtion to normalize the data of an array
'''
Input = Array with n values
Output = Array with normalized valued
'''
def Normalize(row):
    #We calculate the squareroot of the sum of the square values of the row
    suma = np.sqrt(np.sum(row**2)) 
    if suma == 0.:
        #If the sum is zero we return a 0
        return 0.0
    #Else we divide each value between the sum value above
    row = row/suma
    return row 

#Normalize the training set data
for i in range(len(x_train)):
    x_train[i] = Normalize(x_train[i])

#Normalize the test set data
for i in range(len(x_test)):
    x_test[i] = Normalize(x_test[i])
    
#Showing the state sum of the training set    
print("The sum of the states from the training set 0",np.sum(x_train[0]**2))

n=6 #Number of qubits 
num_layers = 1 #Number of layers
#Making a ndarray of floats based on the number of layers
params = np.random.random(10*(num_layers))

#Function to create a quantum circuit based on the number of qubit and a
#vector of complex amplitudes to initialize to
'''
Input: Number of qubits, vector of complex amplitudes
Output: Quantum Circuit
'''
def input_data(n,inputs):
    circuit = QuantumCircuit(n,1) #create the quantum circuit with n qubits
    #initialization of the circuit with the vector of amplitudes
    circuit.initialize(inputs,range(0,n,1)) 
    circuit.barrier() #Draw a barrier
    return circuit

#Example of a quantum circuit with the first row of te trainig set
input_data(n,x_train[0]).draw(output="mpl")

#Function to create a quantum variational circuit
'''
Input: number of qubits, number of layers, parameters to initialized the circuit
Output: Quantum Circuit
'''
def vqc(n, num_layers,params):
    #Set the number of layers and qubits
    #ParameterVectors are initialized with a string identifier and an integer specifying the vector length
    parameters = ParameterVector('θ', 10*(num_layers))
    len_p = len(parameters)
    circuit = QuantumCircuit(n, 1) #create the quantum circuit with n qubits
    

    #Creating the circuit for each layer
    for layer in range(num_layers):
        #Applying a ry gate in each qubit
        for i in range(n):
            #the rotation of the ry gate is defined in the parameters list
            #based on the layer
            circuit.ry(parameters[(layer)+i], i)
        circuit.barrier() #Create a barrier

        circuit.cx(2,0) #Apply a CNOT gate between the qubit 2 and 0
        circuit.cx(3,1) #Apply a CNOT gate between the qubit 3 and 1
        circuit.cx(5,4) #Apply a CNOT gate between the qubit 5 and 4
        circuit.barrier() #Create a barrier
        
        #Apply a RY gate in the qubit 0 with the rotation specified in the parameter list
        circuit.ry(parameters[6+(layer)],0)
        #Apply a RY gate in the qubit 1 with the rotation specified in the parameter list
        circuit.ry(parameters[7+(layer)],1)
        #Apply a RY gate in the qubit 4 with the rotation specified in the parameter list
        circuit.ry(parameters[8+(layer)],4)
        circuit.barrier() #Create a barrier
        
        circuit.cx(4,1) #Apply a CNOT gate between the qubit 4 and 1
        circuit.barrier() #Create a barrier
        
        #Apply a RY gate in the qubit 1 with the rotation specified in the parameter list
        circuit.ry(parameters[9+(layer)], 1)
        circuit.barrier() #Create a barrier
        

    #Creating a parameters dictionary
    params_dict = {}
    i = 0
    for p in parameters:
        #The name of the value will be the string identifier and an integer specifying the vector length
        params_dict[p] = params[i] 
        i += 1
    #Assign parameters using the assign_parameters method
    circuit = circuit.assign_parameters(parameters = params_dict)
    return circuit

#An example with 6 quibits, one layer and 10 parameters
vqc(n,num_layers,params).draw(output="mpl")

'''
Input: Number of qubits
Output: Quantum circuit
'''
def swap_test(n):
    qubits_values = 2*n+1 #Create a new qubit value to create our circuit
    qc = QuantumCircuit(qubits_values) #Create the quantum circuit with the qubits value
    qc.h(0) #Applying a H gate to the first qubit
    for i in range(n):
        #Applying a cswap gate between the first quibit and the i+1 and 2*n-i qubits
        qc.cswap(0,i+1,2*n-i) 
    qc.h(0) #Applying a H gate to the first qubit
    qc.barrier() #Create a barrier
    return qc
#Example of a swap test with 2 quibits
swap_test(2).draw(output="mpl")

size_reduce = 2 #Number of qubits we want to reduce
circuit_init = input_data(n,x_train[0]) #Create a inicial circuit
circuit_vqc = vqc(n,num_layers,params) #Create a quantum variational circuit
circuit_swap_test = swap_test(size_reduce) #Create a swap test circuit

#Create a new circuit based on the size of the initial circuit and the desired qubits to reduce
circuit_full = QuantumCircuit(n+size_reduce+1,1)

#Combine the initial circuit, the quantum variatinal circuit and the swap test
#For the initial circuit and QVC we start at the qubit size_reduce + 1
#For the swap test we start at the qubit 0
circuit_full = circuit_full.compose(circuit_init,[i for i in range(size_reduce+1,n+size_reduce+1)])
circuit_full = circuit_full.compose(circuit_vqc,[i for i in range(size_reduce+1,n+size_reduce+1)])
circuit_full = circuit_full.compose(circuit_swap_test,[i for i in range(2*size_reduce+1)])
circuit_full.draw(output="mpl")