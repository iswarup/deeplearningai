
What You Need To Remember ?

C1W2 Pr Ass

- np.exp() works elementwise
- np.sum(x,axis=0/1,keepdims=True), np.dot, np.multiply, np.maximum,


C1W3? Ass
Common steps for pre-processing a new dataset are:
- Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
- Reshape the datasets such that each example is now a vector of size (num_px \* num_px \* 3, 1)
- "Standardize" the data  
	(We did it by /255 here.)
	But generally you sub the mean of the whole np array from each ex and then divide each ex by the std deviation of the whole np array.

- Preprocessing the dataset is important.
- You implemented each function separately: initialize(), propagate(), optimize(). Then you built a model().
- Tuning the learning rate (which is an example of a "hyperparameter") can make a big difference to the algorithm.

parameters -- W, b
cache --      Z, A
grads --      dW, db

backprop---
	dZ[L] = A[L] - Y
	dW[]



C1W4 Ass1
functions:
	def initialize_parameters(nx,nh,ny):
		return parameters							{W1b1W2b2}
	def initialize_parameters_deep(layer_dims):
		return parameters							{W1,b1.....WL-1,bL-1}

	def linear_forward(A,W,b):
		return Z,cache								(A,W,b)
	def linear_activation_forward(A_prev,W,b,activation):
		return A,cache								(linear_Cache,activation_Cache)
	def L_model_forward(X,parameters):
		return AL, caches							every cache of LAF, L-1 of them.

	def computeCost(AL,y):
		return cost

	def linear_backward(dZ,cache=Aprev,W,b)
		return dAprev,dW,db
	def linear_activation_backward(dA,cache,activation):
		return dAprev,dW,db
	def L_model_backward(AL,Y,caches):
		return grads								{dA,dW,db}

	def update_parameters(param,grads,LR):
		return parameters							make predictions using these parameters

C1W4 Ass2

	def initialize_parameters_deep(layers_dims):
	    ...
	    return parameters 
	def L_model_forward(X, parameters):
	    ...
	    return AL, caches
	def compute_cost(AL, Y):
	    ...
	    return cost
	def L_model_backward(AL, Y, caches):
	    ...
	    return grads
	def update_parameters(parameters, grads, learning_rate):
	    ...
	    return parameters


## Course 2 Week 1

Train Dev Test sets. ------> Bigger Data 98-2-2, Smaller 60-20-20
Bias/Variance. ------------> High Bias = Underfiting, High Variance = Overfiting
Basic Recipe: "On Loop"
	1.High Bias?
		Bigger network, layers/hiddenUnits
		Train longer (doesn't always help, but certainly doesn't hurt)
		[NN Architectures]  (not always work)
	2.High Variance?
		More Data
		Regularization
		[NN Archis]  keep trying, not always work

	Regularized Bigger Networks almost never hurts

Regularization: add lambd/2m(sum(W**W)) to the cost  ## Square of norm of W     # sum over nL,nL-1  # Frobenius Norm
				add lambd/m (sum(W))
Dropout Regularization

	if Overfiting: lower down the keep_prob, so that it randomly kills more of the units
	then the cost_function J is now less well defined so harder to debug by plotting it on a graph with no of iterations on X-axis
	to go around this problem, turn off the drop out, set keep_prob = 1.0 then debug plot.
Other Regularization methods: Data Augmentation, Early Stopping(Orthogonalization:-it couples the two tasks-HB,HV)
		Substitute for ES is L2 but computationally expensive.


Normalizing------ setting up your opt prob to train NN quickly
		Use same values for both test/train data
	1.Subtract mean		X -=   mu:= 1/m* sum(X)
	2.Normalize Variance   X /= sigma    sigma= 1/m* sum(X**2)

		So X= (X-mu)/sigma

		
Initialize weights to avoid them exploding/vanishing, also to speed up the training
	#doesn't completely help but good to some extent

	Z = w1x1 + w2x2 + ...... + WnXn
		large n ----->>  Smaller W(i)
		if the activation is RelU here.
			Var(Wi) = 2/n
			Wl = np.random.randn(shape)* np.sqrt(2/n(l-1))		# a less imp hyperpara can be inserted in the numerator 
		tanh	by Xavier init
			sqrt(1/n(l-1))
		????	by Yoshua Bengio
			sqrt(2/n(l-1)+nl)

Grad Check = Debugging