import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model.ridge import _solve_cholesky_kernel
import copy
from scipy import stats
import numpy.matlib
from ObliviousClass import ObliviousData



#%%
n = 500 # 500 or 1000 would be a good number
m = 100 # number of test samples and validation samples


gam = np.linspace(0.0, 1.0, 11)
repetitions = 20


MSE_obl = np.zeros((len(gam),repetitions))
MSE_ridge = np.zeros((len(gam),repetitions))
MSE_MC = np.zeros((len(gam),repetitions))

for reps in range(repetitions):
    for gam_ind in range(len(gam)):
        gamma = gam[gam_ind]
    
    
        S = np.asmatrix(np.random.rand(2*n,1)*10-5) # uniform between -5 and 5
        U = np.asmatrix(np.random.rand(2*n,1)*10-5) #uniform between -5 and 5
        X = gamma* U + (1-gamma)*S
        
        discr_level = 4 
        #bins = np.arange(2 ** discr_level)
        
        S_binned = np.floor(((S +5)/10) * 2 ** discr_level).astype(int) #normalize S first; gives number 0 to 2** discr level -1
        #bins = np.unique(S_binned)
        bins = S_binned[n:2*n] #bins of cond samples
        
        S_train = S[0:n]
        S_train_cond = S[n:2*n]
        S_binned_train = S_binned[0:n]
        S_binned_cond = S_binned[n:2*n]
        
        U_train = U[0:n]
        U_train_cond = U[n:2*n]
        
        X_train =  gamma* U_train + (1-gamma) *S_train
        X_train_cond = gamma * U_train_cond + (1-gamma)* S_train_cond  
        Y_train = np.power(X_train,2) + np.asmatrix([np.random.normal(0,0.1,n)]).T + np.power(S_train,2)
        
        S_val = np.asmatrix(np.random.rand(m,1)*10-5)
        U_val = np.asmatrix(np.random.rand(m,1)*10-5)
        X_val = gamma* U_val + (1-gamma)*S_val 
        Y_val = np.power(X_val,2) + np.asmatrix([np.random.normal(0,0.1,m)]).T + np.power(S_val,2)
        data = np.c_[X_val,S_val,Y_val]
        S_binned_val = np.floor(((S_val+5)/10) * (2 ** discr_level)).astype(int) #normalize S first
        
        S_test = np.asmatrix(np.random.rand(m,1)*10-5)
        U_test = np.asmatrix(np.random.rand(m,1)*10-5)
        X_test = gamma* U_test + (1-gamma)*S_test 
        Y_test = np.power(X_test,2) + np.asmatrix([np.random.normal(0,0.1,m)]).T + np.power(S_test,2)
        data = np.c_[X_test,S_test,Y_test]
        S_binned_test = np.floor(((S_test+5)/10) * (2 ** discr_level)).astype(int) #normalize S first


        #Oblivious SVM
        print('OBLIVIOUS SVM',gam[gam_ind],reps)
        obl = ObliviousData()
        K = obl.build_K_rbf(X,X)
        O = obl.build_O_discrete(K,S_binned)
        Kt_val = obl.build_K_rbf(X_val,X)
        Ot_val = obl.build_Ot_discrete(Kt_val,S_binned_val)
        MCOt_val=obl.predict_g(Kt_val, S_binned_val)
        Kt = obl.build_K_rbf(X_test,X)
        Ot = obl.build_Ot_discrete(Kt,S_binned_test)
        MCOt=obl.predict_g(Kt, S_binned_test)

        print('Regression')
        
        #%%
        alpha_list = [2**v for v in range(-5,5)]
        print("List of alpha tested:", alpha_list)
        MSE_obl_cv = np.zeros((len(alpha_list),1))
        MSE_ridge_cv = np.zeros((len(alpha_list),1))
        MSE_MC_cv = np.zeros((len(alpha_list),1))
        
        for alpha_ind in range(len(alpha_list)):
            alpha = alpha_list[alpha_ind]
            
            dual_coef = np.linalg.solve(O+alpha*np.asmatrix(np.eye(n)), Y_train)
            dual_kernel_ridge = np.linalg.solve(K[0:n,0:n] +alpha*np.asmatrix(np.eye(n)), Y_train)
        
            Y_pred_val = Ot_val.dot(dual_coef)
            Y_pred_ridge_val = (Kt_val[:,:n]).dot(dual_kernel_ridge)
            Y_pred_MC_val = MCOt_val.dot(dual_kernel_ridge)
            MSE_obl_cv[alpha_ind] = np.mean(np.power(Y_val - Y_pred_val,2))
            MSE_ridge_cv[alpha_ind] = np.mean(np.power(Y_val - Y_pred_ridge_val,2))             
            MSE_MC_cv[alpha_ind] = np.mean(np.power(Y_val - Y_pred_MC_val,2))
        
        alpha_orr = alpha_list[np.argmin(MSE_obl_cv)]
        alpha_krr = alpha_list[np.argmin(MSE_ridge_cv)]
        alpha_morr = alpha_list[np.argmin(MSE_MC_cv)]
        
        dual_coef = np.linalg.solve(O+alpha_orr*np.asmatrix(np.eye(n)), Y_train)
        dual_kernel_ridge = np.linalg.solve(K[0:n,0:n] +alpha_krr*np.asmatrix(np.eye(n)), Y_train)
        dual_morr = np.linalg.solve(K[0:n,0:n] +alpha_morr *np.asmatrix(np.eye(n)), Y_train)

        
        Y_pred = Ot.dot(dual_coef)
        Y_pred_ridge = (Kt[:,:n]).dot(dual_kernel_ridge)
        Y_pred_MC = MCOt.dot(dual_morr)
        MSE_obl[gam_ind,reps] = np.mean(np.power(Y_test - Y_pred,2))
        MSE_ridge[gam_ind,reps] = np.mean(np.power(Y_test - Y_pred_ridge,2))
        MSE_MC[gam_ind,reps] = np.mean(np.power(Y_test - Y_pred_MC,2))
        
MSE_mean_obl = np.mean(MSE_obl,axis = 1)
MSE_std_obl = np.std(MSE_obl,axis = 1)
MSE_mean_ridge = np.mean(MSE_ridge,axis = 1)
MSE_std_ridge = np.std(MSE_ridge,axis = 1)
MSE_mean_MC = np.mean(MSE_MC,axis = 1)
MSE_std_MC = np.std(MSE_MC,axis = 1)
   
#%%
fig = plt.figure()
ax=plt.subplot(111)
ax.set_ylabel('MSE')
ax.set_xlabel('$\gamma$')
plt.title('Ridge Regression (Experiment 2)')
plt.plot(gam,MSE_mean_ridge,label='KRR',color='red',linestyle=':')
plt.plot(gam,MSE_mean_obl,label='ORR',color='blue',linestyle='-')
plt.plot(gam,MSE_mean_MC,label='M-ORR',color='green',linestyle='--')

plt.errorbar(gam,MSE_mean_ridge,MSE_std_ridge,fmt='-o',color='red',linestyle=':')
plt.errorbar(gam,MSE_mean_obl,MSE_std_obl,fmt='-o',color='blue',linestyle='-')
plt.errorbar(gam,MSE_mean_MC,MSE_std_MC,fmt='-o',color='green',linestyle='--')
ax.legend()
fig.savefig('RidgePlotSq_New.pdf')        
