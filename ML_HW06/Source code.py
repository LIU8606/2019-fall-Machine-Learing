#!/usr/bin/env python
# coding: utf-8

# In[1]:


import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import pdist, squareform,cdist


# In[2]:


image1 = imageio.imread('image1.png')
image2 = imageio.imread('image2.png')


# In[3]:


def kernel(data, gamma_s = 0.001, gamma_c = 0.001):
    
    
    def spatial_information(data):
        S = np.empty((data.shape[0]*data.shape[1],2))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                S[i*data.shape[0] + j] = [i,j]
        return S
    
    def color_information(data):
        return data.reshape(data.shape[0]*data.shape[1] , 3)
    
    S = spatial_information(data)
    C = color_information(data)
    
    S_dist = squareform(pdist(S,'sqeuclidean'))
    S_K = np.exp(-gamma_s * S_dist)
 
    C_dist = squareform(pdist(C,'sqeuclidean'))
    C_K = np.exp(-gamma_c * C_dist)
    
    K = S_K*C_K
    
    return K


# In[4]:


def get_spatial_merge_color(data):
        S = np.empty((data.shape[0]*data.shape[1],5))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                S[i*data.shape[0] + j] = [i,j,data[i][j][0],data[i][j][1],data[i][j][2]]
        return S


# In[35]:


def find_init_center(data,group):
    
    key = np.random.randint(0,data.shape[0])
    step =0
    center_list = []

    while step<10:
        if step == 0:
            seed = data[key]
        else:
            distance_list = np.sqrt(np.power(data - seed, 2).sum(axis=1)).tolist()
            dx = sum(distance_list)
            random = np.random.random()*dx
            for index in range(len(distance_list)):
                random -= distance_list[index]
                if random.real <=0:
                    seed  = distance_list[index]
                    break
            x_index  = index
            seed = data[x_index]
            center_list.append(seed)
        
            if len(center_list) == group:
                break

        step +=1
    
    center_list = np.array(center_list)
    
    return center_list


# In[36]:


def clustering(data, center,group):
    distances = np.zeros((data.shape[0], group))
    for gi in range(group):
        distances[:, gi] = np.power(data - center[gi], 2).sum(axis=1)
    group_indices = np.argmin(distances, axis=1)
    return group_indices


# In[37]:


def find_init_a(data,group):
    
    X = get_spatial_merge_color(data)
    center = find_init_center(X,group)
    group_indices = clustering(X, center,group)
    
    a = np.zeros((data.shape[0] * data.shape[1] , group))       
    for g in range(group):
        a[:, g] = (group_indices == g)
    
    return a


# In[ ]:





# In[38]:


def random_a(data,group):
    random_group = np.random.randint(group, size=data.shape[0] * data.shape[1])
    a = np.zeros((data.shape[0] * data.shape[1] , group))       
    for g in range(group):
        a[:, g] = (random_group == g)
    
    return a


# In[39]:


def distance(K,a,group):
    d = np.zeros((K.shape[0] , group))
    for g in range(group):
        a_k = a[:, g]
        ck_size = a_k.sum()
        d[:,g] = np.diagonal(K)
        d[:,g] -= (2 / ck_size) * (a_k@K)
        d[:,g] += (1 / (ck_size*ck_size)) * ((a_k[..., np.newaxis] @ a_k[np.newaxis, ...]).T * K).sum()
    
    return d


# In[40]:


def kernel_kmeans(data, K, group, plus, title):
    
    if plus:
        a = find_init_a(data, group)
    else:
        a = random_a(data, group)
    
    ims = []
    fig = plt.figure()
    iter_num = 0
    while True:
        d = distance(K, a , group)
        group_indices = np.argmin(d, axis=1)
        s = group_indices.reshape(100,100)

        plt.imshow(s,animated=True)
        plt.title(title + str(group) + 'group iter:{}'.format(iter_num+1))
        plt.savefig(title + "/" + title + str(group) + "group_" +str(iter_num) + ".png")
        plt.show()
    
        new_a = np.zeros((K.shape[0] , group))       
        for g in range(group):
            new_a[:, g] = (group_indices == g)
            
        
        if (a == new_a).all():
            break
        a = new_a
        iter_num+=1
        
    return iter_num
    


# In[41]:


def make_gif(iter_num, group, title):
    images = []
    for i in range(iter_num+1):
        im = title + "/" + title+ "{}group_{}.png".format(group, i)
        images.append(imageio.imread(im))
    imageio.mimsave( "gif/" + title + str(group) + "group" +".gif", images, duration = 0.3)


# In[42]:


def kernel_kmeans_draw(image, dataname,K, plus):
    if plus:
        title = "kernel_kmeans+_{}_".format(dataname)
    else:
        title = "kernel_kmeans_{}_".format(dataname)
    for group in [2,3,4]:
        iter_num = kernel_kmeans(image, K , group, plus, title)
        make_gif(iter_num, group, title)


# In[32]:


K = kernel(image1, 0.001, 0.001)
kernel_kmeans_draw(image1,"image1", K, plus = False)


# In[319]:


K = kernel(image1, 0.001, 0.001)
kernel_kmeans_draw(image1,"image1", K,plus = True)


# In[85]:


K = kernel(image2, 0.0005, 0.001)
kernel_kmeans_draw(image2,"image2", K, plus = True)


# In[89]:


K = kernel(image2, 0.0005, 0.001)
kernel_kmeans_draw(image2,"image2", K, plus = False)


# # Spectral Clustering - radio cut

# In[12]:


# def get_eig_radio(data,group):
    
# #     W = kernel(data,0.001,0.001)
# #     D = np.diag(W.sum(axis=1))
# #     L = D-W
# #     eigenvalues, eigenvectors = np.linalg.eig(L)
    
# #     U = eigenvectors[:, np.argsort((eigenvalues))[:group]]
    
#     if data == "image1":
#         eigenvectors = np.load("image1_ratio_eigenvectors.npy")
#         eigenvalues  = np.load("image1_ratio_eigenvalues.npy")
#     else:
#         eigenvectors = np.load("image2_ratio_eigenvectors.npy")
#         eigenvalues  = np.load("image2_ratio_eigenvalues.npy")        
    
#     U = eigenvectors[:, np.argsort((eigenvalues))[:group]]
    
#     return U


# In[ ]:


def get_eig_radio(data,group):
    
    if data == "image1":
        data = image1
    else: 
        data == image2
    
    W = kernel(data,0.001,0.001)
    D = np.diag(W.sum(axis=1))
    L = D-W
    eigenvalues, eigenvectors = np.linalg.eig(L)
    
    U = eigenvectors[:, np.argsort((eigenvalues))[:group]]
    
    return U


# In[63]:


# def get_eig_normalized(data,group):
    
# #     D = np.diag(W.sum(axis=1))
# #     D_sqrt = np.diag((W.sum(axis=1))**0.5)
# #     D_sqrt_n = np.diag((W.sum(axis=1))**(-0.5))
# #     L = D-W
# #     L_sym = D_sqrt_n@L@D_sqrt
    
# #     eigenvalues, eigenvectors = np.linalg.eig(L_sym)

#     if data == "image1":
#         eigenvectors = np.load("image1_normalize_eigenvectors.npy")
#         eigenvalues  = np.load("image1_normalize_eigenvalues.npy")
#     else:
#         eigenvectors = np.load("image2_normalize_eigenvectors.npy")
#         eigenvalues  = np.load("image2_normalize_eigenvalues.npy") 
#     U = eigenvectors[:, np.argsort((eigenvalues))[:group]]
    
#     return U


# In[ ]:


def get_eig_normalized(data,group):
    
    if data == "image1":
        data = image1
    else: 
        data == image2    
        
    W = kernel(data,0.0005,0.001)
    D = np.diag(W.sum(axis=1))
    D_sqrt = np.diag((W.sum(axis=1))**-0.5)
    L = D-W
    L_sym = D_sqrt@L@D_sqrt
    
    eigenvalues, eigenvectors = np.linalg.eig(L_sym)
    U = eigenvectors[:, np.argsort((eigenvalues))[:group]]
    
    return U


# In[45]:


def random_centor_init(data,group):
    #random init centors
    center_list = []
    for i in range(group):
        key = np.random.randint(0,data.shape[0])
        center_list.append(data[key])
        
    center_list = np.array(center_list)
                     
    return center_list


# In[46]:


def kmeans(data, group , plus, title):
    
    if plus:
        center = find_init_center(data, group)
    else:
        center = random_centor_init(data, group)
        
    iter_num = 0
    plt.figure()
    while True:
        group_indices = clustering(data, center, group)
    
        plt.imshow(group_indices.reshape(100,100))
        plt.title(title + str(group) + 'group iter:{}'.format(iter_num+1))
        plt.savefig(title + "/" + title + str(group) + "group_" +str(iter_num) + ".png")
        plt.show()
    
        new_center = []
        for g in range(group):
            index = np.where(group_indices == g)[0]
            new_center.append(data[index].mean(axis=0))
        new_center = np.array(new_center)
    
        if (new_center==center).all():
            break
        center = new_center
        
        iter_num+=1
        
    return iter_num


# In[47]:


def spectral_clustering_ratio_draw(image, plus):
    if plus:
        title = "ratio_cut+_{}_".format(image)
    else:
        title = "ratio_cut_{}_".format(image)
        
    for group in [2,3,4]:
        U = get_eig_radio(image, group)
        iter_num = kmeans(U, group, plus , title)
        make_gif(iter_num, group, title)


# In[16]:


spectral_clustering_ratio_draw("image1", plus =True)


# In[ ]:





# In[31]:


spectral_clustering_ratio_draw("image1", plus =False)


# In[59]:


spectral_clustering_ratio_draw("image2", plus =True)


# In[60]:


spectral_clustering_ratio_draw("image2", plus =False)


# In[27]:


def spectral_clustering_normalized_draw(image, plus):
    
    if plus:
        title = "normalized_cut+_{}_".format(image)
    else:
        title = "normalized_cut_{}_".format(image)
        
    for group in [2,3,4]:
        U = get_eig_normalized(image, group)
        T = np.zeros_like(U)
        for i in range(10000):
            T[i, :] = U[i, :] / np.sqrt((U[i, :] * U[i, :]).sum())
        iter_num = kmeans(T, group, plus, title)
        make_gif(iter_num, group, title)


# In[28]:


spectral_clustering_normalized_draw("image1", plus =True)


# In[28]:


spectral_clustering_normalized_draw("image1", plus =  False)


# In[65]:


spectral_clustering_normalized_draw("image2", plus = True)


# In[66]:


spectral_clustering_normalized_draw("image2", plus = False)


# # Check eigen

# In[48]:


def get_group(data, group):
    

    center = random_centor_init(data, group)
        
    iter_num = 0
    plt.figure()
    while True:
        group_indices = clustering(data, center, group)
    
        plt.imshow(group_indices.reshape(100,100))
    
        new_center = []
        for g in range(group):
            index = np.where(group_indices == g)[0]
            new_center.append(data[index].mean(axis=0))
        new_center = np.array(new_center)
    
        if (new_center==center).all():
            break
        center = new_center
        
        iter_num+=1
        
    return group_indices,new_center


# In[49]:


def spectral_clustering_ratio(image,group):
        
    U = get_eig_radio(image, group)
    group_indices,new_center = get_group(U, group)
          
    return U, group_indices,new_center


# In[50]:


def spectral_clustering_normalized(image, plus):
    
    U = get_eig_normalized(image, group)
    T = np.zeros_like(U)
    for i in range(10000):
        T[i, :] = U[i, :] / np.sqrt((U[i, :] * U[i, :]).sum())
        
    group_indices, new_center = get_group(T, group)
        
    return U, group_indices,new_center


# In[56]:


group = 3
U, group_indices,center = spectral_clustering_ratio("image1", group)

for i in range(group-1):

    plt.figure(figsize =(20,10))
    
    check = []
    for j in range(group):
        check.append(np.where(group_indices==j)[0])

    for j in range(group):
        plt.plot(check[j], U[check[j]][:,i+1],'o')
        
    plt.title("Eigenvector "+str(i+1))
    plt.plot(np.arange(0,10000,1)+1, np.zeros((10000)),"black")


# In[57]:


group = 3
U, group_indices,center = spectral_clustering_normalized("image1", group)

for i in range(group-1):

    plt.figure(figsize =(20,10))
    
    check = []
    for j in range(group):
        check.append(np.where(group_indices==j)[0])

    for j in range(group):
        plt.plot(check[j], U[check[j]][:,i+1],'o')
        
    plt.title("Eigenvector "+str(i+1))
    plt.plot(np.arange(0,10000,1)+1, np.zeros((10000)),"black")


# In[ ]:




