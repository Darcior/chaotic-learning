# chaotic-learning
Predicting Chaos through Machine Learning
data.txt has the maximum lyapunov exponent for the set of parameters given in param.txt. There are 2048 set of parameters. Each set of parameter has 5 values, each corresponding to alpha, beta , gamma, delta and omega in the duffing oscillator DE. 
data_gen.py is the program used for generating data.
NN.py is used for classification using Neural Networks.
RK2.py is used for runge-kutta method to obtain time-series data.
NN_regression.py is used for regression using neural networks.
Logistic_regression.py is used for implementing logistic regression for binary calssification.
pyESN folder contains all the ESN implementation code for pyESN library. You need the whole folder to run the program.
pyESN.ipynb is the program we use for our training.
