#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import matplotlib.pyplot as plt
import random


# In[37]:


def Univariate_gaussian_data_generator(mean, var, size):

    U = np.random.uniform(0, 1, size)
    V = np.random.uniform(0, 1, size)

    z = (np.sqrt(-2.0 * np.log(U))) * (np.cos(2 * np.pi * V))
    
    X = z * np.sqrt(var) + mean
    
    return X


# In[38]:


def get_data(N,mx_1,my_1,mx_2,my_2,vx_1,vy_1,vx_2,vy_2):
    x_data1 = Univariate_gaussian_data_generator(mx_1, vx_1, N)
    y_data1 = Univariate_gaussian_data_generator(my_1, vy_1, N)
    x_data2 = Univariate_gaussian_data_generator(mx_2, vx_2, N)
    y_data2 = Univariate_gaussian_data_generator(my_2, vy_2, N)

    data = []
    y = []
    for i in range(N):
        data.append([x_data1[i],y_data1[i]])
        y.append(0)
    for i in range(N):
        data.append([x_data2[i],y_data2[i]])
        y.append(1)
    data = np.array(data)
    y = np.array(y)
    
    return data,y


# In[39]:


def gradient_descent(data, y):

    i = 0
    w = np.array([1,1,0]).reshape(3,1)
    
    while i<10000:
        X = np.array([[point[0],point[1],1] for point in data])
        f = 1/(1+ np.exp(- X @ w))
        delta = X.T @ (y.reshape(100,1) - f)
        last_w = w
        w = last_w + delta
        i += 1
        if (np.abs(delta) < 1e-3).all():
            break
    if i == 10000:
        print("can't converge")
            
    return w


# In[40]:


def Newton_method(data,y):
    w = np.array([1,1,0]).reshape(3,1)
    I = np.identity(data.shape[0])
    A = np.array([[point[0],point[1],1] for point in data])
    i = 0
    while i<10000:
        D = []
        for i in range(data.shape[0]):
            X = np.array([data[i][0],data[i][1],1])
            c = np.exp(- X @ w.reshape(3,1) / ((1 + np.exp(- X @ w.reshape(3,1)))**2))
            D.append(c * I[i])
        
        H = A.T @ np.array(D) @ A
        f = 1/(1+ np.exp(- A @ w))
        delta = A.T @ (y.reshape(2*N,1) - f)
        last_w = w

        try:
            w = last_w + np.linalg.inv(H) @ delta
        except:
            w = last_w + delta
    
        i+=1
        if (np.abs(w - last_w) < 1e-3).all():
            break
            
    if i == 10000:
        print("can't converge")
    
    return w


# In[41]:


def show(w,data,method):
    Blue = []
    Red = []
    P00  = P01 = P10 = P11 =0
    for point in data[:N]:
        if (w[0]*point[0] + w[1]*point[1] + w[2]) >0:
            Blue.append([point[0],point[1]])
            P01 +=1
        else:
            Red.append([point[0],point[1]])
            P00 +=1
    for point in data[N:]:
        if (w[0]*point[0] + w[1]*point[1] + w[2]) >0:
            Blue.append([point[0],point[1]])
            P11 +=1
        else:
            Red.append([point[0],point[1]])
            P10 +=1
    
    Blue = np.array(Blue)
    Red = np.array(Red)
    
    print(method+"\n")
    print("w:")
    print(w)
    
    print('\nConfusion Matrix:')
    print('                 Predict cluster 1 Predict cluster 2')
    print('Is cluster 1            {}                      {}'.format(P00,P01))
    print('Is cluster 2           {}                      {}'.format(P10,P11))
    print()
    
    print("Sensitivity (Successfully predict cluster 1):", P00/(P00+P01))
    print("Specificity (Successfully predict cluster 2):", P11/(P10+P11))
    print()
    
    return Blue, Red


# In[42]:


N= 50 
mx_1 = my_1 = 1
mx_2 = my_2 = 10
vx_1 = vy_1 = 2
vx_2 = vy_2 = 2


# In[43]:


data, y = get_data(N,mx_1,my_1,mx_2,my_2,vx_1,vy_1,vx_2,vy_2)
w = gradient_descent(data, y)
Blue_gd , Red_gd = show(w, data, "Gradient descent")
w = Newton_method(data, y)
Blue_N , Red_N = show(w,data, "Newton's method")


# In[44]:


plt.subplot(131)
plt.scatter(data[:N,0], data[:N,1], s = 60, c = 'r', marker = "o")
plt.scatter(data[N:,0], data[N:,1], s = 60, c = 'b', marker = "o")
plt.title("Ground truth")

plt.subplot(132)
plt.scatter(Blue_gd[:,0],Blue_gd[:,1], s = 60, c = 'b', marker = "o")
plt.scatter(Red_gd[:,0], Red_gd[:,1], s = 60, c = 'r', marker = "o")
plt.title("Gradient descent")

plt.subplot(133)
plt.scatter(Blue_N[:,0],Blue_N[:,1], s = 60, c = 'b', marker = "o")
plt.scatter(Red_N[:,0], Red_N[:,1], s = 60, c = 'r', marker = "o")
plt.title("Newton")
plt.show() 


# # EM

# In[65]:


import os                                                                       
import sys                                                                      
import struct                                                                   
import numpy as np                                                              
import matplotlib.pyplot as plt   
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')
#from PIL import Image         

def read_data(file_name):

    image_file_name = file_name
    image_file_object = open(image_file_name, 'rb')  
    
    raw_header = image_file_object.read(16)                                      
    image_header_data = struct.unpack(">4I", raw_header)  #> : big endian , I : unsigned int
    
    print(image_file_name,"header data:" , image_header_data)
    
    image_L = []
    
    for i in range(image_header_data[1]):
        img = image_file_object.read(28*28)                                               
        tp = struct.unpack(">784B",img) #28*28=784                                             
        image = np.array(tp)                                                      
        #image = image.reshape((28,28))   
        image_L.append(image)
    
    return np.array(image_L)
    
def read_label(file_name):
    
    label_file_name = file_name
    label_file_object = open(label_file_name, 'rb')  
    
    raw_header = label_file_object.read(8)                                      
    label_header_data = struct.unpack(">2I", raw_header) 
    
    print(label_file_name,"header data:" , label_header_data)
    
    label_L = []
    
    for i in range(label_header_data[1]):
        img = label_file_object.read(1)                                                   
        tp = struct.unpack(">B",img)
        label_L.append(tp)
    
    return np.array(label_L)
    
    
train_image_orginal = read_data("train-images-idx3-ubyte")
train_label = read_label("train-labels-idx1-ubyte")
test_image_orginal = read_data("t10k-images-idx3-ubyte")
test_label = read_label("t10k-labels-idx1-ubyte")  


# In[69]:


def bin_data(train_image, test_image):
    
    train_image = train_image//128
    train_image = train_image.astype(np.int)
        
    test_image = test_image//128
    test_image = test_image.astype(np.int)
        
    return train_image, test_image


# In[70]:


train_image, test_image = bin_data(train_image_orginal, test_image_orginal)


# In[427]:


def EM():
    pi = np.random.rand(10)
    pi /= pi.sum() # sum up to one
    mu = np.random.rand(10, train_image.shape[1])
    w = np.zeros((train_image.shape[0], 10))

    NO = 1
    num = 1
    last_error = 0
    while True:
        last_mu = np.copy(mu)
        #E step
        for n in range(train_image.shape[0]):
            for k in range(10):
                w[n][k] = np.log(pi[k]) + (train_image[n] * np.log(mu[k]) +   (1 -train_image[n]) * (np.log(1 - mu[k]))).sum()
            w[n] -= w[n].max()
            w[n] = np.exp(w[n])/np.exp(w[n]).sum()
    
        #M step
        for k in range(10):
            N_k = w[:,k].sum() 
            pi[k] = N_k/train_image.shape[0]
            pi[pi < 1e-13] = 1e-13
            for i in range(784):
                mu[k][i] = ((train_image[:,i] * w[:,k]).sum()  + 1e-13 ) / ( N_k + 1e-13 * train_image.shape[1])
              
    
        if NO < 3 :
            for k in range(10):
                L=[]
                print("class {}:".format(k))
                for i in range(784):
                    if mu[k][i] <=0.5:
                        L.append(0)
                    else:
                        L.append(1)
                print(np.array(L).reshape(28,28))
                print()
        
        new_error = abs(last_mu-mu).sum()
        if (abs(last_error-new_error) < 1):
            num+=1
            if num == 6:
                break
        else:
            num = 1
        last_error= new_error
    
        print("No. of Iteration: {}, Difference: {}".format(NO, new_error))
        print("------------------------------------------------------------\n")
    
        NO +=1
    
    
    for k in range(10):
        L=[]
        print("class {}:".format(k))
        for i in range(784):
            if mu[k][i] <0.5:
                L.append(0)
            else:
                L.append(1)
        print(np.array(L).reshape(28,28))
        print()
    
    print("No. of Iteration: {}, Difference: {}\n".format(NO, new_error))    
    print("------------------------------------------------------------")
    print("------------------------------------------------------------\n")
    
    return mu, w, NO


# In[428]:


def labeling(w, mu):
    
    pred = np.argmax(w, axis=1)
    mapping = np.zeros((10),dtype='int') -1
    for k in range(10):
        unique, counts = np.unique(train_label[np.where(pred == k)], return_counts=True)
        if unique.shape[0] > 0:
                mapping[k] = unique[counts.argmax()]
                
                
    for k  in range(10):
        print("label class {}:".format(k))
        label = np.where(mapping == k)
        if label[0].shape[0] >0:
            for l in label[0]:
                L = []
                for i in range(784):
                    if mu[l][i] <0.5:
                        L.append(0)
                    else:
                        L.append(1)
                print(np.array(L).reshape(28,28))
                print()
        else:
            print("Can not find label class{}\n".format(k))
            
    return pred, mapping


# In[429]:


def Confusion_Matrix(NO, pred, mapping):
    
    matrix = np.zeros((10,10),dtype='int')
    for i in range(train_image.shape[0]):
        matrix[train_label[i][0]][mapping[pred[i]]] +=1
    
    for i in range(10):
        TP = matrix[i][i]
        FN = matrix[i].sum() - matrix[i][i] 
        FP = matrix[:,i].sum() - matrix[i][i]
        TN = matrix.sum() - TP - FN - FP
        if (TP+FP) == 0:
            Sensitivity = 0
        else:
            Sensitivity = TP/(TP+FP)
        if (TN+FN) == 0:
            Specificity =0
        else:
            Specificity = TN/(TN+FN)
    
        print()
        print('Confusion Matrix {}:'.format(i))
        print('                Predict number {} Predict not number {}'.format(i, i))
        print('Is number    {}          {}               {}'.format(i,TP, FN))                                                                   
        print('Isn\'t number {}          {}               {}'.format(i,FP,TN))
        print()
        print('Sensitivity (Successfully predict number {})    : {:.5f}'.format(i,Sensitivity))
        print('Specificity (Successfully predict not number {}): {:.5f}'.format(i,Specificity))
        print()
        print('------------------------------------------------------------')

    print()
    print('Total iteration to converge: {}'.format(NO))

    error = 0
    for i in range(10):
        for j in range(10):
            if i != j:
                error += matric[i][j]
    print('Total error rate: {:.10f}'.format(error/train_image.shape[0]))


# In[432]:


mu, w, NO = EM()
pred, mapping = labeling(w, mu)
Confusion_Matrix(NO, pred, mapping)


# In[ ]:





# In[ ]:




