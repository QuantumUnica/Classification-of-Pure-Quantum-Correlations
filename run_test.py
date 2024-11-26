import glob, os, joblib
import time
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import confusion_matrix

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Our custom modules 
from utils import get_dataset
import tuning_parameters
from quantum_classifiers.PGMHQC_gpu_cpu_dtype import PGMHQC_gpu_cpu_dtype

import torch


label_col = -1
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(dev))

for experiment in ['2_qubits_2_labels']:
    datasets_sizes = [10000] #range(10000, 35000, 5000)

    # Parameters for Crossvalidation and Grid search
    metric=metric='balanced_accuracy'
    n_folds=5
    n_grid_jobs = 1 #os.cpu_count() // 2

    # PGM parameter that splits the sum of quantum densities
    # done during centroid computation
    n_PGM_splits = 1

    classifiers = [
            ('PGM', PGMHQC_gpu_cpu_dtype(dtype=torch.float64, n_splits=n_PGM_splits, device=dev)),
            ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
        ]
    
    if experiment[0]=='2':
        classifiers = [classifiers[0]] + [('Hels', PGMHQC_gpu_cpu_dtype(n_splits=n_PGM_splits, measure='hels'))] + classifiers[1:]
    

    # Lists all datasets in the selected experiment
    data_path_list = glob.glob(os.path.join(os.getcwd(), experiment, 'data_split', '*.csv'), recursive=True) 
    splitPairs = []

    # Prepare pairs with train and test paths for a given dataset size
    for g in datasets_sizes:
        tr = [x for x in data_path_list if str(g) in x and 'Train' in x][0]
        te = [x for x in data_path_list if str(g) in x and 'Test' in x][0]
        splitPairs.append((tr, te))

    output_scores = []  # Will contain the rows of the output Dataframe

    for train_path, test_path in tqdm(splitPairs):

        dataset_name = os.path.splitext(os.path.split(train_path)[-1])[0] 
        dataset_name = '_'.join([ x for x in dataset_name.split('_') \
                                if x != '80' and x != '20' and x != 'perc' and x != 'Train' and x != 'Test' and x != ''])
        
        X_train, y_train = get_dataset(train_path, label_column_idx=label_col)
        X_test, y_test = get_dataset(test_path, label_column_idx=label_col)

        print('===========================================================================================')
        print('                     DATASET : ', dataset_name)
        print('                     INFO : ', X_train.shape, X_train.dtype)
        print('===========================================================================================')

        for clf_name, clf in classifiers:
            print()
            print("CLASSIFIER = ", clf_name)
            print()

            # It trains several instances of the same model with different parameter combinations. 
            # For each configuration it performs 5-Fold Cross Validation.
            cross_val_strategy = KFold(n_splits=n_folds, shuffle=True, random_state=tuning_parameters.RANDOM_SEED)
            
            start = time.time()

            models = GridSearchCV(clf,
                                    tuning_parameters.param_grid(clf),
                                    scoring=metric,
                                    cv=cross_val_strategy,
                                    n_jobs=n_grid_jobs,
                                    pre_dispatch=2*n_grid_jobs,
                                    verbose=0).fit(X_train, y_train)
            
            # Save the best model
            model_save_path = os.path.join('/'.join(train_path.split('/')[:-2]), 'models', f"{clf_name}_{dataset_name}_best_model.pkl")
            if not os.path.exists(os.path.dirname(model_save_path)):
                os.mkdir(os.path.dirname(model_save_path))

            joblib.dump(models.best_estimator_, model_save_path)
            print(f"Saved the best model for {clf_name} at {model_save_path}")
            loaded_model = joblib.load(model_save_path)
            
            # Make predictions
            y_pred_loaded_model = loaded_model.predict(X_test)
            y_pred = models.best_estimator_.predict(X_test)
            
            # Check prediction of the saved model
            assert np.array_equal(y_pred_loaded_model, y_pred), print("Test predicion with loaded model is different from the one in-memory")

            # save raw confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_save_path = os.path.join('/'.join(train_path.split('/')[:-2]), 'confusion_matrices', dataset_name)
            if not os.path.exists(os.path.dirname(cm_save_path)):
                os.mkdir(os.path.dirname(cm_save_path))

            labels = models.best_estimator_.classes_
            cm_with_labels = np.hstack((cm, np.array(labels)[...,np.newaxis]))
            np.save(os.path.splitext(cm_save_path)[0]+'_'+clf_name+'.npy', cm_with_labels)


            # Get best model test score
            best_mean_score = round( models.cv_results_['mean_test_score'][models.best_index_], 3)
            best_std_dev = round(models.cv_results_['std_test_score'][models.best_index_], 3)


            output_scores.append([dataset_name, clf_name, best_mean_score, models.best_params_, round(time.time() - start, 3)])
            

            print("THE BEST %s is" % metric, best_mean_score, '±', best_std_dev, '\n')
            
            print('with the following PARAMETER values:')
            print(models.best_params_)
            end = time.time()

            print('\n\nrunning time (secs) = ', round(end - start, 3))
            
            print("\n\n", "".join(['-']*100))

            score_save_path = os.path.join(experiment, 'score_files', metric + '.csv')
            if not os.path.exists(os.path.dirname(score_save_path)):
                os.mkdir(os.path.dirname(score_save_path))
                
            pd.DataFrame(output_scores).to_csv(score_save_path, index=False, header=None)



        