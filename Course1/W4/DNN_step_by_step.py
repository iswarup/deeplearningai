import numpy as np


layer_dims = [4,4,5,3]

def inititialize_parameters_deep(layer_dims):
	np.random.seed(3)
	parameters = {}
	for l in range(1,len(layer_dims)):
		parameters["W"+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
		parameters["b"+str(l)] = np.zeros((layer_dims[l],1))
	return parameters

# print(inititialize_parameters_deep(layer_dims))

def sigmoid(Z):
	pass
def relu(Z):
	pass

def linear_forward(A,W,b):

	Z = np.dot(W,A)+b
	cache = (A,W,b)

	return Z,cache

def linear_activation_forward(A_prev,W,b,activation):

	if activation == "sigmoid":
		Z,linear_cache = linear_forward(A_prev,W,b)
		A,activation_cache = sigmoid(Z)

	if activation == "relu":
		Z,linear_cache = linear_forward(A_prev,W,b)
		A,activation_cache = relu(Z)

	cache = (linear_cache,activation_cache)

	return A, cache

def L_model_forward(X,parameters):
	caches = []
	A = X
	L = len(parameters) // 2

	for l in range(L):
		A_prev = A
		A,cache = linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
		caches.append(cache)
	AL,cache = linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
	caches.append(cache)

	return AL, caches

def compute_cost(AL,Y):
	m = Y.shape[1]
	cost = (-1/m)*np.sum( np.dot(Y,np.log(AL).T) + np.dot((1-Y),np.log(1-AL).T) )
	return cost

##########---Backward Pass---###########
def relu_backward(dA,activation_cache):
	dZl = dAl ∗ g′(Zl)
	return dZ
def sigmoid_backward(dA,activation_cache):
	pass

def linear_backward(dZ,cache):
	A_prev,W,b = cache
	m = A_prev.shape[1]

	dW = 1/m* np.dot(dZ,A_prev.T)
	db = 1/m* np.sum(dZ,axis=1,keepdims=True)
	dA_prev = np.dot(W.T,dZ)

	return dA_prev,dW,db

def linear_activation_backward(dA,cache,activation):
	linear_cache, activation_cache = cache

	if activation == "relu":
		dZ = relu_backward(dA,activation_cache)
		dA_prev, dW, db = linear_backward(dZ,linear_cache)

	if activation == "sigmoid":
		dZ = sigmoid_backward(dA,activation_cache)
		dA_prev,dW,db = linear_backward(dZ,linear_cache)

	return dA_prev, dW, db

def L_model_backward(AL,Y,caches):
	grads = {}
	L = len(caches)  # the number of layers
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)

	dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

	current_cache = caches[-1]
	grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")

	for l in reversed(range(L-1)):
		current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)],current_cache,"relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters,grads,learning_rate):
	L = len(parameters) // 2

	for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate*grads["dW"+str(l+1)] 
        parameters["b" + str(l+1)] -= learning_rate*grads["db"+str(l+1)]

    return parameters