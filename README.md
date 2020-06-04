
What You Need To Remember ?

C1W2 Pr Ass

- np.exp() works elementwise
- np.sum(x,axis=0/1,keepdims=True), np.dot, np.multiply, np.maximum,


C1W2 Ass
Common steps for pre-processing a new dataset are:
- Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
- Reshape the datasets such that each example is now a vector of size (num_px \* num_px \* 3, 1)
- "Standardize" the data  
	(We did it by /255 here.)
	But generally you sub the mean of the whole np array from each ex and then divide each ex by the std deviation of the whole np array.

- Preprocessing the dataset is important.
- You implemented each function separately: initialize(), propagate(), optimize(). Then you built a model().
- Tuning the learning rate (which is an example of a "hyperparameter") can make a big difference to the algorithm.

C1W3

