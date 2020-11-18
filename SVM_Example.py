import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import numpy.matlib
from ObliviousClass import ObliviousData
from sklearn import svm
from collections import namedtuple

def generate_truncnorm_samples(n_samples,lower,upper,mu,sigma):
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    values = X.rvs(n_samples)
    return values

def generate_toy_data(n_samples):
    sigma =0.5
    unique_sensitive_feature_values=[0,1]
    max_non_sensitive_feature_value=4.0
    min_non_sensitive_feature_value=1.0
    mu = 0.5*(max_non_sensitive_feature_value+min_non_sensitive_feature_value)
    sensitive_features = [unique_sensitive_feature_values[0]] * n_samples + [unique_sensitive_feature_values[1]] * n_samples 
    sensitive_features = np.array(sensitive_features)
    sensitive_features.shape = (len(sensitive_features), 1)
    
    Lower = generate_truncnorm_samples(n_samples,min_non_sensitive_feature_value,max_non_sensitive_feature_value,mu,sigma)
    Upper = generate_truncnorm_samples(n_samples,min_non_sensitive_feature_value,max_non_sensitive_feature_value,mu,sigma)
    non_sensitive_features0=[Lower]+[Upper]
    non_sensitive_features0 = np.array(np.hstack(non_sensitive_features0))
    non_sensitive_features0.shape=(len(non_sensitive_features0),1)
    
    
    non_sensitive_features=[Lower-stats.bernoulli(0.9).rvs(n_samples)*1]+[Upper+stats.bernoulli(0.9).rvs(n_samples)*1]
    non_sensitive_features = np.array(np.hstack(non_sensitive_features))
    non_sensitive_features.shape=(len(non_sensitive_features),1)
    
    X = np.hstack([non_sensitive_features, sensitive_features])
    
    threshold=mu
    Y_Bernoulli_params=np.array(non_sensitive_features0/max_non_sensitive_feature_value).flatten()
    Y=np.array([stats.bernoulli(Y_Bernoulli_params[i]).rvs(1) for i in range(len(Y_Bernoulli_params))]).flatten()
    True_Y=Y*(np.array((non_sensitive_features0>=threshold)*1).flatten())
    Y=Y*(np.array((non_sensitive_features+sensitive_features>=threshold)*1).flatten())
    sensitive_feature_id=np.shape(X)[1]-1

    return X, Y, sensitive_feature_id, True_Y





def estimate_beta_dependence(predicted_labels, sensitive_features, labels):
    estimated_beta=0
    n = np.size(predicted_labels)
    unique_label_freqs =[]
    for i in range(len(labels)):
        unique_label_freqs.append(np.mean(predicted_labels==labels[i]))

    unique_S_features=list(set(sensitive_features))
    unique_S_freqs =[]
    for i in range(len(unique_S_features)):
        unique_S_freqs.append(np.mean(sensitive_features==unique_S_features[i]))

    pred_feature_pairs = np.vstack((predicted_labels, sensitive_features)).T
     
    joint_freqs=[]       
    for i in range(len(labels)):
        for j in range(len(unique_S_features)):
            pattern=(labels[i],unique_S_features[j])
            joint_freq=np.size(np.where(np.sum(np.abs(pred_feature_pairs-np.matlib.repmat(pattern, n, 1)),axis=1)==0))/n
            joint_freqs.append(joint_freq)
            marginal_label_freq=unique_label_freqs[i]
            marginal_S_freq=unique_S_freqs[j]
            estimated_beta=estimated_beta+np.abs(marginal_label_freq * marginal_S_freq-joint_freq)
    return estimated_beta


if __name__ == "__main__":
    n_train = 500 #this means n=1000 samples
    n_cond = n_train
    n_test = n_train

    X_train,y_train,sensitive_feature_index, True_Y_train = generate_toy_data(n_train)
    X_cond,y_cond,sensitive_feature_index,True_Y_cond = generate_toy_data(n_cond)
    X_test,y_test,sensitive_feature_index,True_Y_test = generate_toy_data(n_test)
          
    X = np.vstack((X_train, X_cond))
    y = np.concatenate((y_train,y_cond))
    S_train = X_train[:, sensitive_feature_index]
    S_train_cond = X_cond[:, sensitive_feature_index]
    
    S = np.concatenate((S_train, S_train_cond))
    S_test = X_test[:,sensitive_feature_index]
    X_obl = X[:,:sensitive_feature_index]
    X_obl_test = X_test[:,:sensitive_feature_index]

    unique_labels=list(set(True_Y_test))



    #Plot data
    plt.hist(X_test[:,0])
    sensitive_feature_values = sorted(list(set(X[:, sensitive_feature_index])))

    

    C_list = [2**v for v in range (16,20)]    
    #Oblivious SVM
    print('OBLIVIOUS SVM')
    obl = ObliviousData()
    K = obl.build_K_lin(X,X)
    O = obl.build_O_discrete(K,S)
    Kt = obl.build_K_lin(X_test,X)
    Ot = obl.build_Ot_discrete(Kt,S_test)
    
    stdacc_list,true_acc_list,betas =[],[],[]
    
    for c in C_list:
        print('C: ', c)
        clf = svm.SVC(kernel='precomputed', C=c)
        print('Fitting OBLIVIOUS SVM')
        clf.fit(O,y_train)
        pred_oblv = clf.predict(Ot)
        standard_missclassification_error=np.mean((pred_oblv!=y_test)*1) # error with respect to observed labels 
        print('Standard Missclassification Error: ', standard_missclassification_error)
        true_missclassification_error=np.mean((pred_oblv!=True_Y_test)*1) # error with respect to ground-truth labels 
        print('True Missclassification Error: ', true_missclassification_error) 
        # Dependence measure
        beta = estimate_beta_dependence(pred_oblv,X_test[:, sensitive_feature_index],unique_labels)
        print('Beta-Dependence: ',beta)
        stdacc_list.append(standard_missclassification_error)
        true_acc_list.append(true_missclassification_error)
        betas.append(beta)
        
            
    stdind=np.where(stdacc_list==np.min(stdacc_list))[0][0]
    print('(standard error, dependence):')
    print((stdacc_list[stdind], betas[stdind]))        
    print('(true error, dependence):')
    print((true_acc_list[stdind], betas[stdind]))   
