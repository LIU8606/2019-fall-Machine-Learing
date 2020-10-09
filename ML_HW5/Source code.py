import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
from numpy.linalg import cholesky, det, lstsq
import csv


#### GP

from matplotlib import cm

def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples = None):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    #plt.plot(X, samples, lw=1, ls='--')
    plt.plot(X_train, Y_train, 'ro')
    plt.show()


# In[8]:


X = []
Y = []
f = open("input.data")
line = f.readline()
while line:
    x,y = line.split(" ")
    X.append(float(x))
    Y.append(float(y))
    line = f.readline()
f.close()

mean = 0
var = 0.2
n = 34

def kernel_function(x, y, sigma, a, l):
    kernel = (sigma**2) * ((1+ ((x-y)**2 / (2 * a * l**2)) )**(-a))
    return kernel


import itertools

def compute_cov_matrices(x, x_star, sigma, a , l):

    K = []
    K_star = []
    K_star2 = []
    n = x.shape[0]
    n_star = x_star.shape[0]

    for i in x:
        for j in x:
            K.append(kernel_function(i, j, sigma, a , l))

    K = np.array(K).reshape(n, n)

    for i in x:
        for j in x_star:
            K_star.append(kernel_function(i, j, sigma, a , l))

    K_star = np.array(K_star).reshape(n, n_star)

    for i in x_star:
        for j in x_star:
            K_star2.append(kernel_function(i, j, sigma, a , l))   

    K_star2 = np.array(K_star2).reshape(n_star, n_star)
    
    return K, K_star, K_star2


def predict(sigma, a, l):
    
    K, K_star, K_star2 = compute_cov_matrices(np.array(X), x_star, sigma, a, l)
    C = K + var*np.identity(n)
    k_predict  = K_star2 + var*np.identity(K_star2.shape[0])
    var_predict = k_predict - np.dot(np.dot(K_star.T, np.linalg.inv(C)), K_star)
    mean_predict = np.dot(np.dot(K_star.T, np.linalg.inv(C)), Y)
    prediction = np.random.multivariate_normal(mean_predict, var_predict)
    
    return  mean_predict, var_predict, prediction


x_star = np.linspace(-60,60,100)
mean_predict, var_predict, prediction = predict(1,1,1)
plot_gp(mean_predict, var_predict, x_star, X_train=np.array(X), Y_train=np.array(Y), samples=prediction)


def likelihood(params):
    sigma, a, l = params[0], params[1], params[2]
    K = [kernel_function(i, j, sigma, a, l) for (i, j) in itertools.product(np.array(X), np.array(X))]
    K = np.array(K).reshape(n, n)
    C = K + var*np.eye(n)
    L = (0.5 * np.log(np.linalg.det(C)) + 0.5 * np.dot(np.dot(np.array(Y).T, np.linalg.inv(C)), np.array(Y)) + (n/2)*np.log(2*(np.pi)))
    
    return L


guess = np.array([1,1,1])
results = minimize(likelihood, guess, method = 'L-BFGS-B', options={"disp": True})
params = results.x
print(params)


mean_predict, var_predict, prediction = predict(params[0],params[1],params[2])
plot_gp(mean_predict, var_predict, x_star, X_train=np.array(X), Y_train=np.array(Y), samples=prediction)


##### SVM

import sys
sys.path.append('./libsvm-3.24/python')

import svm
from svmutil import *

def read_data(filename):
    
    data = []
    with open(filename) as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            data.append(row)
            
    return np.array(data,dtype='double')


X_train = read_data('X_train.csv')
X_test = read_data('X_test.csv')
Y_train = read_data('Y_train.csv').reshape(5000)
Y_test = read_data('Y_test.csv').reshape(2500)


### linear, poly, rbf

prob = svm_problem(Y_train, X_train)
param_linear = svm_parameter('-t 0')
param_poly = svm_parameter('-t 1')
param_rbf = svm_parameter('-t 2')


m_linear = svm_train(prob, param_linear)
res_linear = svm_predict(Y_test, X_test, m_linear)

m_poly = svm_train(prob, param_poly)
res_poly = svm_predict(Y_test, X_test, m_poly)

m_rbf = svm_train(prob, param_rbf)
res_rbf = svm_predict(Y_test, X_test, m_rbf)


print("linear:")
print(res_linear[1])
print("poly:")
print(res_poly[1])
print("rbf:")
print(res_rbf[1])


DEGREE = [1,3,5]
COEF = [-1,0,1]
GAMMA = [1.0/784.0, 3.0/784.0, 5.0/784.0]

param_linear = svm_parameter('-t 0')
m_linear = svm_train(prob, param_linear)
res_linear = svm_predict(Y_test, X_test, m_linear)

max_poly = None 
max_poly_i = None
max_accu = 0

for d in DEGREE:
   for coef in COEF:
       for g in GAMMA:
           param_poly = svm_parameter('-t 1 -d '+ str(d)+ ' -r '+ str(coef)+ ' -g ' + str(g))
           m_poly = svm_train(prob, param_poly)
           res_poly = svm_predict(Y_test, X_test, m_poly)
           if max_accu< res_poly[1][0]:
               max_accu = res_poly[1][0]
               max_poly = res_poly
               max_poly_i = d,coef,g

max_accu = 0
max_rbf = None 
max_rbf_i  = 0

for g in GAMMA:
    param_rbf = svm_parameter('-t 2 -g ' + str(g))
    m_rbf = svm_train(prob, param_rbf)
    res_rbf = svm_predict(Y_test, X_test, m_rbf)
    if max_accu< res_rbf[1][0]:
        max_accu = res_rbf[1][0]
        max_rbf = res_rbf
        max_rbf_i = g


print("linear:")
print(res_linear[1])
print("poly:")
print(max_poly[1])
print("degree:{}, coef0:{}, gamma:{}".format(max_poly_i[0],max_poly_i[1],max_poly_i[2]))
print("rbf:")
print(max_rbf[1])
print("gamma:{}".format(max_rbf_i))


### C-SVC

sys.path.append('./libsvm-3.24/tools')
from grid import *

with open('./libsvm-3.24/tools/data','w') as f:
    for i in range(X_train.shape[0]):
        s = str(int(Y_train[i])) +  ' '
        for j in range(X_train.shape[1]):
            s = s  + '{}:{} '.format(j+1,X_train[i][j])
        f.write(s + '\n')
f.close()

rate, param = find_parameters('./libsvm-3.24/tools/data','-log2c -5,5,2 -log2g -5,5,2 -v 3 -out output')
print(rate)
print(param)

### linear+RBF

from scipy.spatial.distance import cdist

def get_kernel(X, Y, sigma, a):
    
    K_combine = np.zeros((X.shape[0], Y.shape[0] + 1))  
    
    D = cdist(X,Y, 'euclidean')
    K_rbf = np.exp(-sigma * D**2)
    K_linear = np.dot(X,Y.T) 
    
    K_combine[: , 1:] = K_linear + a*K_rbf 
    K_combine[: , 0] = [i+1 for i in range(X.shape[0])]
    
    return K_combine



sigma = 1.0/784.0
a = 1
K_combine_train = get_kernel(X_train, X_train, sigma, a)
K_combine_test = get_kernel(X_test, X_train, sigma, a)


prob = svm_problem(Y_train, K_combine_train, isKernel=True)
m = svm_train(prob, '-t 4')

res = svm_predict(Y_test, K_combine_test, m)

print(res[1])


max_accu = 0
max_combine = None 
max_combine_g  = 5.0/784.0
max_combine_a  = 0
weight = [200,300,400] 

for g in GAMMA:
    K_combine_train = get_kernel(X_train, X_train, g, a)
    K_combine_test = get_kernel(X_test, X_train, g, a)


    prob = svm_problem(Y_train, K_combine_train, isKernel=True)
    m = svm_train(prob, '-t 4')

    res = svm_predict(Y_test, K_combine_test, m)
    if max_accu< res[1][0]:
        max_accu = res[1][0]
        max_combine = res
        max_combine_g = g  

print(max_combine[1])
print("gamma:{}".format(max_combine_g))  

for a in weight:
    K_combine_train = get_kernel(X_train, X_train, max_combine_g, a)
    K_combine_test = get_kernel(X_test, X_train, max_combine_g, a)


    prob = svm_problem(Y_train, K_combine_train, isKernel=True)
    m = svm_train(prob, '-t 4')

    res = svm_predict(Y_test, K_combine_test, m)
    if max_accu< res[1][0]:
        max_accu = res[1][0]
        max_combine = res
        max_combine_a = a 

print(max_combine[1])
print("weight:{}".format(max_combine_a)) 





