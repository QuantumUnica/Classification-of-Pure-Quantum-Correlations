import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
import torch
import scipy

class KPGM(BaseEstimator, ClassifierMixin):
    
    
    def __init__(self, n_copies = 1,  dtype = torch.float64, device='cpu'):

        self.n_copies = n_copies
        self.dtype = dtype
        self.device=device
        
        # Raise error if dtype is not torch.float32 or torch.float64
        if self.dtype not in [torch.float32, torch.float64]:
            raise ValueError('dtype should be torch.float32 or torch.float64 only')
    

    def fit(self, X, y):
        # Check data in X and y as required by scikit-learn v0.25
        X, y = self._validate_data(X, y, reset = True)
        
        # Ensure target array y is of non-regression type  
        # Added as required by sklearn check_estimator
        check_classification_targets(y)

        N = X.shape[0]  # number of training samples
        q = X.shape[1]  # number of features

        X = torch.tensor(X, dtype = self.dtype).to(self.device) 
        y = torch.tensor(y, dtype = self.dtype).to(self.device)

        # Store classes and encode y into class indexes
        self.classes_, y_class_index = np.unique(y.cpu().numpy(), return_inverse = True)
        
        # Number of classes, set as global variable
        global num_classes
        num_classes = len(self.classes_)

        self.training_row_vectors = X.view(N, 1, q)

        G = torch.matmul(X, X.t())    
        print("Gram shape: ", G.shape)

        if self.n_copies > 1:
            G = torch.pow(G, self.n_copies)

        self.G_sqrtPinv = torch.tensor( scipy.linalg.sqrtm( torch.pinverse(G).cpu().numpy() )
                                       , dtype=self.dtype).to(self.device)

        self.projectors_ = torch.zeros((num_classes, N, N), dtype=torch.float64).to(self.device)

        for k in range(num_classes):
            # Get indices of elements in y corresponding to class k
            k_class_idxs = torch.where(y.view(-1) == k)[0]
            
            # Use advanced indexing to update the diagonal elements
            self.projectors_[k, k_class_idxs, k_class_idxs] = 1
        
       
        return self

           

    def predict_proba(self, X):

        X = torch.tensor(X, dtype=self.dtype).to(self.device)
        check_is_fitted(self, ['projectors_'])

        class_z_probs = []

        for i in range(X.shape[0]): # Predict each test data sample

            z = X[i, ...]                  # get ith test vector
            z = z[np.newaxis, ... ,np.newaxis]  # test vector must be a column

            z_probs = []    # it will store class probs for current test sample

            # Iterate over classes projectors 
            for k in range(num_classes):
                w = self.training_row_vectors @ z   # it must use row vectors from training set
                w = w[..., -1].type_as(self.G_sqrtPinv)
                
                if self.n_copies > 1:
                    w = torch.pow(w, self.n_copies)

                p = (w.T @ self.G_sqrtPinv) @ self.projectors_[k,...] @ (self.G_sqrtPinv @ w) # probability of z to belong to class K
                p = p.cpu().numpy()[0,0]
                z_probs.append(p)

            class_z_probs.append(z_probs)

        return np.array(class_z_probs)

                
    

    def predict(self, X):
        predict_index = np.argmax(self.predict_proba(X), axis = 1)
        return self.classes_[predict_index]
    

    
