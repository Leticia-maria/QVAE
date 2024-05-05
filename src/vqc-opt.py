from random import shuffle
from scipy.optimize import minimize 

import numpy as np
# Quiskit libraries 
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, execute, QuantumRegister, ClassicalRegister
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.circuit import Parameter, ParameterVector

#Function to identify a function cost
'''
Input: An array of parameters(vector of complex amplitudes)
Output: Function cost
'''
def objective_function(params):
    costo = 0
    shuffle(x_train) #reorganize the order of the train set items
    lenght= 5 #We only will consider the first five elements of the taining set
    #For each item of the trainig set
    for i in range(lenght):

        circuit_init = input_data(n,x_train[i])#Create a inicial circuit
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
        circuit_full.measure(0, 0) #Measure the first qubit
        #qc.draw()
        shots= 8192 #Number of shots
        #Execute the circuit in the qasm_simulator
        job = execute( circuit_full, Aer.get_backend('qasm_simulator'),shots=shots )
        counts = job.result().get_counts() #Count the results of the execution
        probs = {} #Calculate the probabilities of 0 and 1 state
        for output in ['0','1']:
            if output in counts:
                probs[output] = counts[output]/shots #Calculate the average of a state
            else:
                probs[output] = 0
        costo += (1 +probs['1'] -  probs['0']) #Update the actual function cost
    
    return costo/lenght

for i in range(1):
    #Minimization of the objective_fucntion by a COBYLA method
    minimum = minimize(objective_function, params, method='COBYLA', tol=1e-6)
    params = minimum.x #Get the solution array
    #Show the cost of the solution array
    print(" cost: ",objective_function(params))
    print(params)

#Function to compress the test set values
'''
Input: An array of parameters(vector of complex amplitudes)
Output: Array with compress values
'''
def compress_result_test(params):
    reduce = [] #List to save the compress values
    #For each row in the test set we will
    for i in range(len(x_test)):
        
        circuit_init = input_data(n,x_test[i]) #Create a inicial circuit
        circuit_vqc = vqc(n,num_layers,params) #Create a quantum variational circuit
 
        #Create a new circuit based on the size of the initial circuit and the desired qubits to reduce
        circuit_full = QuantumCircuit(n,n-size_reduce)

        #Combine the initial circuit, the quantum variatinal circuit
        circuit_full = circuit_full.compose(circuit_init,[i for i in range(n)])
        circuit_full = circuit_full.compose(circuit_vqc,[i for i in range(n)])
        len_cf = len(circuit_full) #Known the length of the circuit
        #For each n - the the desired qubits to reduce we will
        for i in range(n-size_reduce):
            circuit_full.measure(size_reduce+i, i) #Measure the circuit in the position size_reduce+i 
        #We will execute the full circuit in the qasm simulator
        job = execute( circuit_full, Aer.get_backend('qasm_simulator'),shots=8192 )
        result = job.result().get_counts() #Get the results of the execution
        #Get the probabilities of each state
        probs = {k: np.sqrt(v / 8192) for k, v in result.items()}
        reduce.append(probs) #Save the probabilities
        
    return reduce

#Call the compress_result_test function with the parameters defined above
reduce_img =compress_result_test(params)
test_reduce = [] #List to save the new values of the image reduction
#for each value in the reduce_img list
for i in reduce_img:
    index_image = [] #List to save the reduction values
    #We now take in count we want a 4X4 image
    for j in range(16):
        bin_index = bin(j)[2:] #We take the binary value of j from the 2 position to the end
        while len(bin_index) <4: #While bin_index is less than 4
            bin_index = '0'+bin_index #We concatenate a 0 string at the beginnig
        try:   
            #We try to save the element of the row in the position bin_index
            index_image.append(i[bin_index]) 
        except:
            index_image.append(0) #If we can't, we only save a 0
    
    #We save the new imagen values in the test_recuce list
    test_reduce.append(np.array(index_image))

#Function to compress the training set values
'''
Input: An array of parameters(vector of complex amplitudes)
Output: Array with compress values
'''
def compress_result_train(params):
    reduce = [] #List to save the compress values
    #For each row in the training set we will
    for i in range(len(x_train)):
        circuit_init = input_data(n,x_train[i]) #Create a inicial circuit
        circuit_vqc = vqc(n,num_layers,params) #Create a quantum variational circuit
        
        #Create a new circuit based on the size of the initial circuit and the desired qubits to reduce
        circuit_full = QuantumCircuit(n,n-size_reduce)

        #Combine the initial circuit, the quantum variatinal circuit
        circuit_full = circuit_full.compose(circuit_init,[i for i in range(n)])
        circuit_full = circuit_full.compose(circuit_vqc,[i for i in range(n)])
        len_cf = len(circuit_full) #Known the length of the circuit
        #For each n - the the desired qubits to reduce we will
        for i in range(n-size_reduce):
            circuit_full.measure(size_reduce+i, i) #Measure the circuit in the position size_reduce+i 
        #We will execute the full circuit in the qasm simulator
        job = execute( circuit_full, Aer.get_backend('qasm_simulator'),shots=8192 )
        result = job.result().get_counts() #Get the results of the execution
        #Get the probabilities of each state
        probs = {k: np.sqrt(v / 8192) for k, v in result.items()}
        reduce.append(probs) #Save the probabilities
        
    return reduce
        
#Call the compress_result_train function with the parameters defined above
reduce_img =compress_result_train(params)
train_reduce = [] #List to save the new values of the image reduction
#for each value in the reduce_img list
for i in reduce_img:
    index_image = [] #List to save the reduction values
    #We now take in count we want a 4X4 image
    for j in range(16):
        bin_index = bin(j)[2:] #We take the binary value of j from the 2 position to the end
        while len(bin_index) <4: #While bin_index is less than 4
            bin_index = '0'+bin_index #We concatenate a 0 string at the beginnig
        try:  
            #We try to save the element of the row in the position bin_index
            index_image.append(i[bin_index])
        except:
            index_image.append(0) #If we can't, we only save a 0
            
    #We save the new imagen values in the train_recuce list
    train_reduce.append(np.array(index_image))

#Function to decode the test set values compressed
'''
Input: An array of parameters(vector of complex amplitudes)
Output: Array with decode values
'''

def decoder_result_test(params):
    reduce = [] #List to save the decoded values
    #For each row in the test set reduced we will
    for i in range(len(test_reduce)):

        #Create a initial circuit with 6 qubits and a list of 48 zeros and the i row of the test reduced values
        circuit_init = input_data(6,np.concatenate((np.zeros(48), test_reduce[i]), axis=0))
        #Create the inverse VQC 
        circuit_vqc = vqc(n,num_layers,params).inverse()
        
        #Create a new circuit to combine the inicial circuit and the VQC
        circuit_full = QuantumCircuit(n,n)
        
        #Combine the initial circuit, the quantum variatinal circuit
        circuit_full = circuit_full.compose(circuit_init,[i for i in range(n)])
        circuit_full = circuit_full.compose(circuit_vqc,[i for i in range(n)])
        #We will execute the full circuit in the qasm simulator
        job = execute( circuit_full, Aer.get_backend('statevector_simulator') )
        result = job.result().get_statevector() #Get the results of the execution
        reduce.append(result) #Save the results
    return reduce
        
#Call the decoder_result_test function
decoder =decoder_result_test(params)

#Function to decode the training set values compressed
'''
Input: An array of parameters(vector of complex amplitudes)
Output: Array with decode values
'''
def decoder_result_train(params):
    reduce = [] #List to save the decoded values
    #For each row in the test set reduced we will
    for i in range(len(train_reduce)):
        #Create a initial circuit with 6 qubits and a list of 48 zeros and the i row of the test reduced values
        circuit_init = input_data(n,np.concatenate((np.zeros(48), train_reduce[i]), axis=0))
        #Create the inverse VQC 
        circuit_vqc = vqc(n,num_layers,params).inverse()

        #Create a new circuit to combine the inicial circuit and the VQC
        circuit_full = QuantumCircuit(n,n)
        
        #Combine the initial circuit, the quantum variatinal circuit
        circuit_full = circuit_full.compose(circuit_init,[i for i in range(n)])
        circuit_full = circuit_full.compose(circuit_vqc,[i for i in range(n)])
        #We will execute the full circuit in the qasm simulator
        job = execute( circuit_full, Aer.get_backend('statevector_simulator') )
        result = job.result().get_statevector() #Get the results of the execution
        reduce.append(result) #Save the results
    return reduce
     
#Call the decoder_result_train function    
decoder_train =decoder_result_train(params)

#Function to calculate Mean square error
'''
Input: 2 list with the images values
Output: the mean square error
'''
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

from skimage.metrics import structural_similarity as ssim

import math 
ssim_list = [] #List to save the structural similarity index measure
mse_list = [] #List to save the Mean square error
psnr_list = [] #List to save the Peak signal-to-noise ratio

#For each row of the training set we will
for i in range(len(x_train)):
    #Reshape to a 8X8 image of the training set
    test_img = x_train[i].reshape(8,8)*255 
    #Reshape to a 8X8 image of the decoded trainig set
    decoded_img = decoder_train[i].real.reshape(8,8)*255 
    #Calculate the MSE between the reshaped decoded image and the trainig set image
    Y = float(mse(decoded_img,test_img)) 
    #Calculate the SSIM between the reshaped decoded image and the trainig set image
    ssim_list.append(ssim(decoded_img.astype("float"),test_img.astype("float")))
    mse_list.append(Y) #Save the MSE value
    aux = (64**2)/Y #Calculate the PSNR
    psnr_list.append(10*math.log10(aux)) #Save the PSRN value

ssim_list = [] #List to save the structural similarity index measure
mse_list = [] #List to save the Mean square error
psnr_list = [] #List to save the Peak signal-to-noise ratio

#For each row of the test set we will
for i in range(len(x_test)):
    #Reshape to a 8X8 image of the training set
    test_img = x_test[i].reshape(8,8)*255
    #Reshape to a 8X8 image of the decoded trainig set
    decoded_img = decoder[i].real.reshape(8,8)*255
    #Calculate the MSE between the reshaped decoded image and the test set image
    Y = float(mse(decoded_img,test_img))
    #Calculate the SSIM between the reshaped decoded image and the test set image
    ssim_list.append(ssim(decoded_img.astype("float"),test_img.astype("float")))
    mse_list.append(Y) #Save the MSE value
    aux = (64**2)/Y #Calculate the PSNR
    psnr_list.append(10*math.log10(aux)) #Save the PSRN value

#Normalize the training set data
for i in range(len(x_train_c)):
    x_train_c[i] = Normalize(x_train_c[i])

#Normalize the test set data
for i in range(len(x_test)):
    x_test_c[i] = Normalize(x_test_c[i])