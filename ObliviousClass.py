#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:51:10 2020
"""
import math
from copy import copy
import numpy as np

class  ObliviousData:
    def build_O_discrete(self,K,S):
        n = math.floor(((np.shape(K))[1])/2) # n = half the samples
        self.n = n
        self.K=K[:2*n,:2*n] #remove one data point if n is odd
        
        #Bin S
        S_binned = S #assuming that S is discrete and starts at 0
        self.S_train = S_binned[0:n]
        self.S_cond = S_binned[n:2*n]
        
        self.S_max = int(max(S_binned)+1) 
   
        #precompute

        Ephi = np.mean(K[n:,n:])
        
        mean_iI = np.zeros((n,self.S_max))
        mean_i = np.zeros((n,1))
        for i in range(n):
            mean_i[i] = np.mean(K[i,n:])
            for u in range(self.S_max): 
                I = n + np.where(self.S_cond==u)[0] # all I in cond which correspond to sensitive value u
                mean_iI[i,u] = np.mean(K[i,I])

        mean_I = np.zeros((self.S_max,1))
        mean_IJ = np.zeros((self.S_max,self.S_max))
        for u in range(self.S_max):
            I = n + np.where(self.S_cond==u)[0] # all I in cond which correspond to sensitive value u
            mean_I[u] = np.mean(K[I,n:])
            for v in range(self.S_max):
                J = n + np.where(self.S_cond==v)[0] # all I in cond which correspond to sensitive value v
                mean_IJ[u,v] = np.mean((K[I,:])[:,J]) 


        #final loop
        O = copy(K[:n,:n]) #Kernel matrix for the first n elements
        for i in range(n):
            for j in range(i,n):
                u = int(self.S_train[i])
                v = int(self.S_train[j])
    
                O[i,j] = O[i,j] - mean_iI[i,v]  - mean_iI[j,u]  + mean_IJ[u,v] + mean_i[j] + mean_i[i] - mean_I[u] - mean_I[v]  + Ephi
                O[j,i] = O[i,j]
                
        self.O = O
        self.mean_i = mean_i
        self.mean_iI = mean_iI
        self.mean_I = mean_I
        self.mean_IJ = mean_IJ
        self.Ephi = Ephi
                
        return O
    
    
    def build_Ot_discrete(self,Kt,St): #Kt has size testsample x 2n 
        Ot = copy(Kt[:,:self.n])
        m = (np.shape(Kt))[0]
    
        mean_test_iI = np.zeros((m,self.S_max))
        mean_test_i = np.zeros((m,1))
        for i in range(m):
            mean_test_i[i] = np.mean(Kt[i,self.n:])
            for u in range(self.S_max):
                I = self.n + np.where(self.S_cond==u)[0] # all I in cond which correspond to sensitive value u
                mean_test_iI[i,u] = np.mean(Kt[i,I])

        for i in range(m):
            for j in range(self.n): #not symmetric anymore so we need to go through all
                u = int(St[i])
                v = int(self.S_train[j]) 
                Ot[i,j] = Ot[i,j] - mean_test_iI[i,v]- self.mean_iI[j,u]  + self.mean_IJ[u,v]  +  self.mean_i[j] + mean_test_i[i]  - self.mean_I[u] - self.mean_I[v]  + self.Ephi # i corresponds to test, j to train; 
        self.Ot = Ot
        return self.Ot 
    
    
    def predict_g(self,Kt, St): #for M-Oblivious 
        m = (np.shape(Kt))[0]#kt has size testsample x 2n 
        mean_test_iI = np.zeros((self.n,self.S_max))
        mean_test_i = np.zeros((self.n,1))
        for i in range(self.n):
            mean_test_i[i] = np.mean(self.K[i,self.n:])
            for u in range(self.S_max):
                I = self.n + np.where(self.S_cond==u)[0] # all i in cond which correspond to sensitive value u
                mean_test_iI[i,u] = np.mean(self.K[i,I])
        
        M_XS = np.matlib.repmat(mean_test_i, 1, self.S_max)-mean_test_iI
        gOt=np.zeros((m,self.n))
        for i in range(m):
            gOt[i,:]=M_XS[:,St[i].astype(int)]
        
        gOt= gOt+Kt[:,:self.n]

        return gOt 
    
    

    def build_K_rbf(self,X1,X2,sigma=1):
        # X1 has n rows = number of samples; X2 has m rows = number of samples 
        n = (np.shape(X1))[0]
        m = (np.shape(X2))[0]
        K = np.zeros((n,m))
    
        for i in range(n):
            for j in range(m):
                K[i,j] = np.exp((-1)*(np.dot(X1[i]-X2[j],X1[i]-X2[j]))/sigma)
                
        return K


    def build_K_lin(self,X1,X2):
        # X1 has n rows = number of samples; X2 has m rows = number of samples 
        n = (np.shape(X1))[0]
        m = (np.shape(X2))[0]
        K = np.zeros((n,m))
    
        for i in range(n):
            for j in range(m):
                K[i,j] = np.dot(X1[i],X2[j])
        

        return K

    
    def Omatrix(self):
        return self.O
    
    def Otmatrix(self):
        return self.Ot

    
    
    
