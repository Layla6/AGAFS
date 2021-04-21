# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:33:06 2019

@author: yanglei
"""

import numpy as np
import time
from scipy.io import loadmat

def load_data():
    path='./dataset/mnist.npz'
    f=np.load(path)
    x_train,y_train=f['x_train'],f['y_train']
    x_test,y_test=f['x_test'],f['y_test']
    f.close()    
    return (x_train,y_train),(x_test,y_test)

def pre_process(x):
    num=x.shape[0] #计算样本数量
    pixel=x.shape[1]#计算图像大小
    x=x.reshape(num,pixel*pixel)
    x=x.transpose()
    print(x.shape)
    x=x/255
    return x

def cosine_dis(x,y):
    s=(np.linalg.norm(x)*np.linalg.norm(y))
    if(s==0):
        return 0
    else:
        return np.dot(x,y)/s

def knn_graph(X,k):
    d,n=np.shape(X)
    A=np.zeros([n,n])
    L=np.zeros([n,n])
    D=np.zeros([n,n])
    k=k+1 # while i==j
    for i in range(n):
        distances=np.zeros([n])
        dis_index=0
        for j in range(n):
            #print(X[:,j],X[:,i])
            distances[dis_index]=cosine_dis(X[:,i],X[:,j])
            dis_index+=1
        sorted_index=np.argsort(-distances)    #ascending order
        Nk_index=sorted_index[:k]
        
        #print(Nk_index,sorted_index);print(distances);print()
        A[i,Nk_index]=distances[Nk_index]
        diagonal = np.diag_indices(n)   #waste  方到外面
        A[diagonal]=0
        #print(A)
        
        for i in range(n):
            D[i,i]=np.sum(A[i,:])
        #print(D)
        
        L=D-A
        #print(L)
    return L

def Adaptive_knnG(X,Y,k,gam):
    d,n=np.shape(X)
    L=np.zeros([n,n])
    D=np.zeros([n,n])
    G=np.zeros([n,n])
    #SavedG_ind=np.zeros([n,n])
    S=np.zeros([n,n])
    for i in range(n):
        distances=np.zeros([n])
        dis_index=0
        for j in range(n):
            distances[dis_index]=np.linalg.norm(X[:,i]-X[:,j])+gam*np.linalg.norm(Y[:,i]-Y[:,j])
            dis_index+=1
        sorted_index=np.argsort(distances)  #save index
        #SavedG_ind[i,:]=sorted_index#delete??
        G[i,:]=distances
        #g_0ksum=np.dot(np.ones([1,n]),G[i][:k])
        g_0ksum=np.sum(G[i][:k])
        for j in range(k):   #only update k entris for every si
            tmp=(G[i][k+1]-G[i][j])/(k*G[i][k+1]-g_0ksum)
            S[i][sorted_index[j]]=tmp
            S[sorted_index[j]][i]=tmp
    diagonal = np.diag_indices(n)
    S[diagonal]=0
    for i in range(n):
        D[i,i]=np.sum(S[i,:])        
    L=D-S
    return L
        
        
        

def xavier_init(fan_in,fan_out,constant=1):
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high=constant*np.sqrt(6.0/(fan_in+fan_out))
    return np.random.uniform(low=low,high=high,size=(fan_in,fan_out))

def y_encode(W1,b1,x,m):
    return 1/(1+np.exp(-(np.dot(W1,x).reshape([m,1])+b1)))

def Xre_decode(W2,b2,y,d):
    return 1/(1+np.exp(-(np.dot(W2,y).reshape([d,1])+b2)))

def sigmid(x):
    return 1/(1+np.exp(-x))

def objective_opt(X,m,gam,lam,k):
    d,n=np.shape(X)              # !!! data type
    print(n,d)
    e=0.0001
    max_iteration=300
    diff=0.00001
    fun_diff=1
    iteration=0
    prior_fun=10000
    
    W1=xavier_init(m,d)
    W2=xavier_init(d,m)
    b1=xavier_init(m,1)
    b2=xavier_init(d,1)
    
    Y=np.zeros([m,n])
    Xre=np.zeros([d,n])
    U=np.eye(d)
    L=knn_graph(X,k)
    #stop condition:(1) max iteration (2)the difference between two iteration of obecjive_fun less than threshold
    while((iteration<=max_iteration)and(fun_diff>=diff)):
        for i in range(n):
            Y[:,i]=y_encode(W1,b1,X[:,i],m).reshape(m,)
            Xre[:,i]=Xre_decode(W2,b2,Y[:,i],d).reshape(d,)
        L=Adaptive_knnG(X,Y,k,gam)
        #objective function
        L_fun=(1/(2*n))*np.power(np.linalg.norm((X-Xre),ord='fro'),2)
        R_fun=lam*np.linalg.norm( np.linalg.norm(W1,axis=0),ord=1 )    #先对列求2范数，再求1范数
        G_fun=gam*np.ndarray.trace( np.dot(np.dot(Y,L), Y.transpose()) )
        F_fun=L_fun+R_fun+G_fun
        
        fun_diff=abs(prior_fun-F_fun)
        prior_fun=F_fun
        
        
        delta3=np.multiply( np.multiply( (Xre-X),Xre ) , (np.ones([d,n])-Xre) )
        delta2=np.multiply( np.multiply( np.dot(W2.transpose(),delta3) ,Y) ,(np.ones([m,n])-Y) )
        
        #compute U matrix
        for i in range(d):
            nm=np.linalg.norm(W1[:,i])
            if(nm==0):
                U[i,i]=0
            else:
                U[i,i]=1/(nm+e)
                
        #the partial of F_fun 
        W1_partial=(1/n)*np.dot(delta2,X.transpose())+lam*np.dot(W1,U)+\
                    2*gam*np.dot( np.multiply(np.multiply(np.dot(Y,L),Y),(np.ones([m,n])-Y)) ,X.transpose())
        W2_partial=(1/n)*np.dot(delta3,Y.transpose())
        b1_partial=(1/n)*np.dot(delta2,np.ones([n,1]))+\
                    2*gam*np.dot( np.multiply(np.multiply(np.dot(Y,L),Y),(np.ones([m,n])-Y)) ,np.ones([n,1]))
        b2_partial=(1/n)*np.dot(delta3,np.ones([n,1]))
        
        W1=W1-0.1*W1_partial
        W2=W2-0.1*W2_partial
        b1=b1-0.1*b1_partial
        b2=b2-0.1*b2_partial
        
        print(iteration,F_fun,fun_diff)
        
        iteration+=1
    #print(W1)
    score=np.zeros([d,])
    for i in range(d):
        score[i]=np.linalg.norm(W1[:,i])
    index=np.argsort(score)
    index_fin=(index+1)
    np.save('./para/result_k(5)m(200)_iter(60)_gy.npy',index_fin)    
        


time_s=time.time()
(x_train,y_train),(x_test,y_test)=load_data()

#对样本进行reshape 归一化
x_train=pre_process(x_train)
x_test=pre_process(x_test)

#mnist_dis=loadmat('mnist_disorder.mat')['Mnist_random'] 
objective_opt(x_train[:,:100],m=300,gam=0.005,lam=0.01,k=5)
#knn_graph(X,k)

time_end=time.time()
print('total time:',time_end-time_s)



'''
Z=np.multiply(X,Y)  #矩阵对应元素相乘
W=np.dot(X,Y)   #矩阵的 乘积
'''