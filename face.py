#!/usr/bin/env python
# coding: utf-8

# In[1]:


import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import PIL.Image
from scipy.spatial.distance import cdist,pdist,squareform


# In[2]:


def load_data(foldername):

    filenames = glob.glob('./Yale_Face_Database/'+ foldername + '/*.pgm')

    for file in filenames:
        im = PIL.Image.open(file)
        im = im.resize((29,41),PIL.Image.ANTIALIAS)
        im.save(foldername+'/'+file.split('/')[-1])
        
    data = []
    
    for file in filenames:
        data.append(imageio.imread(file))
     
    
    filenames = glob.glob('./' + foldername+'/*.pgm')
    
    data_resize = []
    
    for file in filenames:
        data_resize.append(imageio.imread(file))
        
    return filenames,  np.array(data), np.array(data_resize)


# In[223]:


train_file, train, train_resize = load_data("Training")
test_file, test, test_resize = load_data("Testing")


# In[224]:


train_label = [int(filename.split('/')[-1][7:9]) for filename in train_file]
test_label = [int(filename.split('/')[-1][7:9]) for filename in test_file]


# In[225]:


train = train_resize.reshape((train.shape[0], -1))
test = test_resize.reshape((test.shape[0], -1))
train.shape


# In[6]:


def pca(data, cov, n=50):
    
    data = data - data.mean(axis = 0)
    cov = np.cov(data.T)
    eigenvalue, eigenvector = np.linalg.eig(cov)
    eigenface = (eigenvector[:, np.argsort(eigenvalue)[::-1]])[:,:n]
    
    return eigenface


# In[143]:


def draw(row, col, figsize, data, title):
    fig, axes = plt.subplots(row, col, sharex=True, sharey=True, figsize=figsize)
    for i, (fp, ax) in enumerate(zip(data, axes.flatten())):
        ax.imshow(fp.real, cmap='gray')
        ax.set_title(title+' - {:d}'.format(i))
    plt.show()


# In[8]:


mean = train.mean(axis = 0)
cov = np.cov((train-mean).T)


# In[9]:


n=40
eigenface = pca(train, cov,n)
draw(5, 5, (16,20), eigenface.T.reshape(n,41,29)[:25,:,:], 'eigenfaces')


# In[10]:


def reconstruct(data, eigenface, chosen_indices):
    
    r = []
    for i in chosen_indices:
        recontruct = (data[i]@eigenface@eigenface.T).T.reshape(1,41,29)
        r.append(recontruct)    
    
    return r


# In[11]:


chosen_indices = np.random.randint(train.shape[0], size=10)
reconstruct_list = reconstruct(train, eigenface, chosen_indices)
draw(2, 5, (16, 8), train_resize[chosen_indices],'original')
draw(2, 5, (16, 8), np.array(reconstruct_list).reshape((10,41,29)), 'reconstruct')


# In[12]:


def project(W, data):
    w = (W.T.real@data)
    return w


# In[147]:


def face_recognition(test_data, train_w, w):
    
    accu = 0
    for face_i in range(test_data.shape[0]):
        test = test_data[face_i]
        #test = test - test.mean()
        test_w = project(w, test.T)
        distance = []
        for i in range(train_w.shape[1]):
            distance.append(np.sum((test_w - train_w[:,i])**2))
            
        #print(np.argsort(distance)//9)
        cluster = (np.argsort(distance)//9)[0]
        if test_label[face_i] == cluster+1 :
            accu += 1
        
    return accu/test_data.shape[0]


# In[148]:


mean = train.mean(0)
train_w_pca = project(eigenface, train.T)


# In[207]:


pca_accu = face_recognition(test, train_w_pca, eigenface)
print(pca_accu)


# In[208]:


def get_gaussian_kernel(X, Y, sigma):
       
    D = cdist(X,Y, 'euclidean')
    K = np.exp(-sigma * D**2)

    return K


# In[209]:


def get_poly_kernel(X, Y, d):
    
    return (X@Y.T)**d


# In[296]:


method = "guassian"
if method == "guassian":
    K_train = get_gaussian_kernel((train-mean),(train-mean),0.00000015)
    K_new = get_gaussian_kernel(train-mean,test-test.mean(0),0.00000015)
elif method == "poly":
    K_train = get_poly_kernel((train-mean),(train-mean),2)
    K_new = get_poly_kernel(train-mean,test-test.mean(0),2)
    
N = (train-mean).shape[0]
one_N = np.ones((N, N))/N
K_train = K_train - one_N.dot(K_train) - K_train.dot(one_N) + one_N.dot(K_train).dot(one_N)


# In[297]:


kernel_eigenvalue, kernel_eigenvector = np.linalg.eig(K_train)
kernel_eigenface = (kernel_eigenvector[:, np.argsort(kernel_eigenvalue)[::-1]])[:,:n]


# In[298]:


train_w_kpca = project(kernel_eigenface, K_train)
test_w_kpca = project(kernel_eigenface, K_new)


# In[299]:


def kernel_face_recognition(test_w, train_w):
    
    accu = 0
    for face_i in range(30):
        distance = []
        for i in range(train_w.shape[1]):
            distance.append(np.sum((test_w[:,face_i] - train_w[:,i])**2))
        #print(np.argsort(distance)//9)
        cluster = (np.argsort(distance)//9)[0]
        #group, counts = np.unique(first_k, return_counts = True)
        if test_label[face_i] == cluster+1 :
            accu += 1
        
    return accu/test_resize.shape[0]


# In[300]:


kpca_accu = kernel_face_recognition(test_w_kpca, train_w_kpca)
kpca_accu


# # LDA 
# 

# In[133]:


def lda(X, y, rank=14):
    def calculate_sw(X, y, unique_y):
        def sj(X):
            mean = X.mean(axis=1)[..., np.newaxis] # (n_dim, 1)
            return (X - mean) @ (X - mean).T
        
        return np.array([sj(X[:, y == uy]) for uy in unique_y]).sum(axis=0)
    
    def calculate_sb(X, y, unique_y):
        means = np.array([X[:, y==uy].mean(axis=1) for uy in unique_y]) # (n_group, n_dim)
        return np.array([(-(means - means[i])).T @ (-(means - means[i])) for i in range(means.shape[0])]).sum(axis=0)
        
    
    X = X.reshape((X.shape[0], -1)).T # flatten and transpose # (n_dim, n_sample)
    X = X- X.mean(0)
    unique_y = np.unique(y)
    sw = calculate_sw(X, y, unique_y) + np.identity(X.shape[0]) # deal with singular matrix issue
    sb = calculate_sb(X, y, unique_y)
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(sw) @ sb)
    
    # get first `rank` eigenvectors
    eigenvectors = (eigenvectors[:, np.argsort(eigenvalues)[::-1]])[:, :rank]
    return eigenvectors


# In[134]:


n=14
fisherface = lda(train_resize, train_label, n)


# In[135]:


draw(5,5,(16,20),fisherface.T.reshape(n,41,29)[:25,:,:], 'fisherfaces')


# In[26]:


# pca_eigenvector = pca(train, cov,121)
# print(pca_eigenvector.shape)
# project= (train@pca_eigenvector)
# print(project.shape)
# fisherface, X_ldaed = lda(project, train_label,25)
# print(fisherface.shape)
# fisherface = pca_eigenvector@fisherface
# print(fisherface.shape)


# In[136]:


chosen_indices = np.random.randint(train.shape[0], size=10)
reconstruct_list = reconstruct(train, fisherface, chosen_indices)
draw(2, 5, (16, 8), train_resize[chosen_indices],'original')
draw(2, 5, (16, 8), np.array(reconstruct_list).reshape((10,41,29)), 'reconstruct')


# In[28]:


chosen_indices = np.random.randint(135, size=10)
r = []
for i in chosen_indices:
    #recontruct = (np.linalg.pinv(fisherface[:,0:14].T)@fisherface[:,0:14].T@train[i]).T.reshape(1,41,29)
    #recontruct = (fisherface[:,0:14]@fisherface[:,0:14].T@train[i].T).T.reshape(1,41,29)
    recontruct = (train[i]@fisherface@fisherface.T).T.reshape(1,41,29)
    r.append(recontruct)


# In[29]:


fig, axes = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(16, 8))
for i, (fp, ax) in enumerate(zip(train_resize[chosen_indices], axes.flatten())):
    ax.imshow(fp.real, cmap='gray')
    ax.set_title('original - {:d}'.format(i))
plt.show()

fig, axes = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(16, 8))
for i, (fp, ax) in enumerate(zip(np.array(r), axes.flatten())):
    ax.imshow(fp[0].real, cmap='gray')
    ax.set_title('reconstruct - {:d}'.format(i))
plt.show()


# In[295]:


train_w_lda = project(fisherface, (train-mean).T)
train_w_lda.shape
lda_accu = face_recognition(test, train_w_lda, fisherface)
print(lda_accu)


# In[321]:


method = "guassian"
if method == "guassian":
    K_train = get_gaussian_kernel(train,train,0.000001)
    K_new = get_gaussian_kernel(train,test,0.000001)
elif method == "poly":
    K_train = get_poly_kernel((train),(train),2)
    K_new = get_poly_kernel(train,test,2)
    
N = (train-mean).shape[0]
one_N = np.ones((N, N))/N
K_train = K_train - one_N.dot(K_train) - K_train.dot(one_N) + one_N.dot(K_train).dot(one_N)


# In[322]:


Z = np.zeros((train.shape[0],train.shape[0]))
for i in range(15):
    Z[(i*9):(i+1)*9,(i*9):(i+1)*9] = 1/9
    
K = np.zeros((train.shape[0],train.shape[0]))
for i in range(15):
    for j in range(15):
        x = train - mean
        if method =="guassian":
            k = get_gaussian_kernel(x[(i*9):(i+1)*9],x[(j*9):(j+1)*9],0.0000010)
        elif method == 'poly':
            k = get_poly_kernel(x[(i*9):(i+1)*9],x[(j*9):(j+1)*9],2)

        N = k.shape[0]
        one_N = np.ones((N, N))/N
        k = k - one_N.dot(k) - k.dot(one_N) + one_N.dot(k).dot(one_N)
        K[(i*9):(i+1)*9,(j*9):(j+1)*9] = k
        
K = K + np.identity(K.shape[0])


# In[323]:


S = np.linalg.inv(K)@Z@K


# In[324]:


kernel_fishervalue, kernel_fishervector = np.linalg.eig(S)


# In[325]:


kernel_fisherface = (kernel_fishervector[:, np.argsort(kernel_fishervalue)[::-1]])[:,:n]
train_w_klda = project(kernel_fisherface, K_train)
test_w_klda = project(kernel_fisherface, K_new)
klda_accu = kernel_face_recognition(test_w_klda, train_w_klda)
print(klda_accu)


# In[72]:


# accu = 0
# for face_i in range(30):
#     distance = []
#     for i in range(w.shape[1]):
#         distance.append((test_w_lda[:,face_i] - train_w_lda[:,i])@ (test_w_lda[:,face_i] - train_w_lda[:,i]).T)
#     #print(np.argsort(distance)//9)
#     cluster = (np.argsort(distance)//9)[0]
#     #group, counts = np.unique(first_k, return_counts = True)
#     if test_label[face_i] == cluster+1 :
#         accu += 1
        
# print(accu/test_resize.shape[0])


# In[ ]:





# In[ ]:





# In[ ]:




