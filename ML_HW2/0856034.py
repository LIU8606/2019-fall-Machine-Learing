#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
    
    
train_image = read_data("train-images-idx3-ubyte")
train_label = read_label("train-labels-idx1-ubyte")
test_image = read_data("t10k-images-idx3-ubyte")
test_label = read_label("t10k-labels-idx1-ubyte")  

#plt.imshow(train_image[0].reshape(28,28))                                        
#plt.show()


# In[2]:


unique_label, label_count = np.unique(train_label, return_counts=True)
prior = label_count/label_count.sum()

print("label_count:",label_count)
print("prior:",prior)


# In[3]:


def bin_data(train_image, test_image , is_continuous):
    
    if is_continuous == 0:
        train_image = train_image/8
        train_image = train_image.astype(np.int)
        
        test_image = test_image/8
        test_image = test_image.astype(np.int)
    else:
        train_image = train_image.astype(np.int)
        test_image = test_image.astype(np.int)
        
    return train_image, test_image


# In[4]:


def count_data(train_image, is_continuous):
    
    #sort by label
    train_image = np.append(train_image, train_label, axis=1)
    train_image = train_image[np.argsort(train_image[:, 784])]

    sort_train_image = np.delete(train_image , 784, 1)

    if is_continuous == 0:
        data = np.zeros((10,784,32))
    else:
        data = np.zeros((10,784,256))
        
    pos = 0
    for label in range(10):
        #print(pos)
        for pixel in range(784):
            recounted = Counter(sort_train_image[pos : pos+label_count[label] , pixel])
            #print(list(sorted(recounted.items())))
            L = (list(sorted(recounted.items())))
            for i in range(len(L)):
                data[label][pixel][int(L[i][0])] = L[i][1]
        pos = pos+label_count[label]
        
    return data, sort_train_image


# In[5]:


def discrete(test_image, data):
    neg = 0
    for i in range(10000):
        Likelihood  = np.zeros((1,10))
        for pixel in range(784):
            Likelihood += -(np.log(data[: , pixel, (test_image[i][pixel])]/label_count + 1e-7))
        posterior = Likelihood + (-np.log(prior))
        posterior = posterior/posterior.sum()
        if i<3:
            print("Postirior (in log scale):")
            for index in range(10):
                print("{}: {}".format(index, posterior[0][index]))
            print("Prediction:",np.argmin(posterior),"Ans:",test_label[i])
            print()
        if np.argmin(posterior) != test_label[i]:
            neg +=1
    print("Error rate:", neg/10000)
    print()


# In[6]:


def discrete_draw(data):
    for label in range(10):
        L = []
        for i in range(784):
            if np.argmax(data[label][i])>=16:
                L.append(1)
            else:
                L.append(0)
            
        L = np.array(L)
        L = np.reshape(L,(28,28))
        print(label,":")
        print(L)
        print()


# In[7]:


def mean_var(sort_train_image):
    mean = np.zeros((10,784))
    var = np.zeros((10,784))
    pos = 0
    for label in range(10):
        mean[label] = np.mean(sort_train_image[pos:pos+label_count[label]],axis=0)
        var[label] = np.var(sort_train_image[pos:pos+label_count[label]],axis=0)
        pos = pos+label_count[label]
            
    return mean,var


# In[8]:


def continuous(test_image, mean , var):
    gau_prob = np.zeros((10,784)) 
    posterior = np.zeros((10,1)) 
    neg = 0
    for i in range(10000):
        for label in range(10):
            gau_prob[label] = np.exp(-0.5 * np.square(test_image[i] - mean[label]) / (var[label]+1e-7) ) / np.sqrt((var[label]+1e-7) *2. *np.pi)
        for label in range(10):
            posterior[label] = (-np.log(gau_prob[label] + 1e-7)).sum() - np.log(prior[label])
        posterior  = posterior/posterior.sum()
        if i<3:
            print("Postirior (in log scale):")
            for index in range(10):
                print("{}: {}".format(index, posterior[index]))
            print("Prediction:",np.argmin(posterior),"Ans:",test_label[i])
            print()
        if np.argmin(posterior) != test_label[i]:
            neg+=1
    print("Error rate:", neg/10000)
    print()


# In[9]:


def continuous_draw(mean , var):
    p_0 = np.zeros((128,784)) 
    p_1 = np.zeros((128,784)) 
    for label in range(10):
        for i in range(128):
            p_0[i] = np.exp(-0.5 * np.square(np.ones(784)*i - mean[label]) / (var[label]+1e-7) ) / np.sqrt((var[label]+1e-7) *2. *np.pi)
        for i in range(128,256):
            p_1[i-128] = np.exp(-0.5 * np.square(np.ones(784)*i - mean[label]) / (var[label]+1e-7) ) / np.sqrt((var[label]+1e-7) *2. *np.pi)   
        mean_0 = np.sum(p_0,axis = 0)
        mean_1 = np.sum(p_1,axis = 0) 
        L = np.array([mean_0 < mean_1],dtype = "int")
        L = np.reshape(L,(28,28))
        print(label,":")
        print(L)
        print()


# In[10]:


is_continuous = int(input("Toggle option (0 is discrete mode, 1 is continuous mode): "))

train_image, test_image = bin_data(train_image, test_image, is_continuous)
data, sort_train_image = count_data(train_image, is_continuous)

if is_continuous ==0:
    discrete(test_image, data)
    discrete_draw(data)
else:
    mean , var = mean_var(sort_train_image)
    continuous(test_image, mean , var)
    continuous_draw(mean , var)


# # Online testing

# In[11]:


s = []
f = open("testfile.txt")
line = f.readline()
while line:
    s.append(line.strip('\n'))
    line = f.readline()
f.close()
print(s)


# In[13]:


from collections import Counter
from scipy.special import comb, perm

a = int(input("a = "))
b = int(input("b = "))

for index in range(len(s)):
    num = list(Counter(s[index]).items())  
    num = sorted(num , key=lambda x:x[0] )
    
    head = num[1][1]
    tail = num[0][1]
    
    p = head / (head+tail)
    
    likelihood = (p**head) * ((1-p)**tail) * comb(head + tail , head)
    
    print("case" + str(index+1) + ": " + s[index])
    print("Likelihood: " + str(likelihood))
    print('Beta prior:       a = {}  b = {}'.format(a, b))
    a = a + head
    b = b + tail
    print('Beta posterior: a = {}  b = {}'.format(a, b))
    print()


# In[ ]:




