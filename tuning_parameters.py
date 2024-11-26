import numpy as np

RANDOM_SEED = 42

lda={
        'solver': ['svd', 'lsqr', 'eigen']
    }

qda={ 'reg_param': [0.0, 0.1, 0.5, 1.0],  # Regularization parameter
        'store_covariance': [True, False],  # Whether to store covariance matrices
        'tol': [1e-4, 1e-3, 1e-2]}  # Tolerance for convergence

dummy={ 'strategy': ['stratified', 'most_frequent', 'prior', 'uniform', 'constant'],
        'constant': [0, 1],  
       'random_state': [RANDOM_SEED]}  


svc={ 'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf'],
        'degree': [3],
        'gamma': ['scale', 0.001, 0.01, 0.1, 1]}



#Nearest Neighbours
knn={ 'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'leaf_size': [10, 30, 50],
        'p': [1, 2]
        }


nc={'metric': ['euclidean', 'manhattan'],
        'shrink_threshold': [None, 0.1, 0.5, 1.0]}

# Linear Classifiers
lr={'penalty': ['l2'], 'C': [1]}
pa={'C':[1]}
perc={'penalty': ['l2','l1','elasticnet']}

# Neural Network
mlp={'mlpclassifier__random_state': [RANDOM_SEED],
    'mlpclassifier__activation': ['relu'],
     'mlpclassifier__solver': ['adam'],
     'mlpclassifier__learning_rate_init':[0.1, 0.01, 0.001, 0.0001],
     'mlpclassifier__max_iter':[3000],
     'mlpclassifier__early_stopping':[False],
     'mlpclassifier__hidden_layer_sizes':[[8,16,16,8],
                                          [8, 16, 16, 16, 16, 16, 16, 8],
                                          [16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16],
                                          [32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 32]
                                          ] }

# Naive Bayes
naive_Bayes_Bernoulli={'alpha': [1]}
naive_Bayes_Gauss={'priors': [None]}
multinomial_Naive_Bayes={'alpha': [1]}

# Ensemble method classifiers
randomForest={'random_state': [RANDOM_SEED],               
                'min_samples_split': [3, 5, 10],
              'n_estimators': [100, 300],
              'max_depth': [3, 5, 15, 25]}


adaBoost={'n_estimators': [50, 100], 'learning_rate': [0.01,0.05,0.1,0.3,1] }

extraTree={'random_state': [RANDOM_SEED],
            'n_estimators': range(50,126,25),
            'min_samples_leaf': range(20,50,5),
            'min_samples_split': range(15,36,5)}

gradientBoosting={'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01],
                  'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200]  }


# Quantum classifiers
ovr_ovo={'estimator__rescale':np.arange(0, 5, .2),
            'estimator__encoding':['amplit', 'stereo'],
            'estimator__class_weight':[None, 'balanced'],
            'estimator__n_copies':[2]}


pgm={'rescale':np.arange(0.5, 1.5, .5),
      'measure':['pgm'],
       # 'encoding':['amplit', 'stereo' 'proj'],
      'encoding':['proj'],

      #'class_weight':[None, 'balanced'],
      'n_copies':[2]

      }


kpgm={
      'measure':['pgm'],
      'encoding':['proj'],
      'n_copies':[6]
      }

hels={'rescale':np.arange(0.5, 1.5, .5),
      'measure':['hels'],
       # 'encoding':['amplit', 'stereo' 'proj'],
      'encoding':['proj'],

      #'class_weight':[None, 'balanced'],
      'n_copies':[2]

      }



# Function for choosing parameters
def param_grid(cl):

    if   'OneVs' in str(cl):                    return ovr_ovo
    elif 'PGMHQC_gpu_cpu_dtype' in str(cl) and cl.measure=='hels':     return hels
    elif 'PGMHQC_gpu_cpu_dtype' in str(cl):     return pgm
    elif 'KPGM' in str(cl):                     return kpgm
    elif 'LinearDiscriminant' in str(cl):       return lda
    elif 'QuadraticDiscriminant' in str(cl):    return qda
    elif 'Dummy' in str(cl):                    return dummy
    elif 'SVC' in str(cl):                      return svc
    elif 'KNeighborsClassifier' in str(cl):     return knn
    elif 'NearestCentroid' in str(cl):          return nc
    elif 'LogisticReg' in str(cl):              return lr
    elif 'PassiveAggressive' in str(cl):        return pa
    elif 'MLP' in str(cl):                      return mlp
    elif 'Bernoulli' in str(cl):                return naive_Bayes_Bernoulli
    elif 'Gauss' in str(cl):                    return naive_Bayes_Gauss
    elif 'Multinomial' in str(cl):              return multinomial_Naive_Bayes
    elif 'RandomForest' in str(cl):             return randomForest
    elif 'Extra' in str(cl):                    return extraTree

