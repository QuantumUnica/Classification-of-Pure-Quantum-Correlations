# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:52:23 2021

@author: rober
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
import torch
from torch.nn.functional import normalize
from scipy import linalg

class PGMHQC_gpu_cpu_dtype(BaseEstimator, ClassifierMixin):
    """The Pretty Good Measurement (PGM) - Helstrom Quantum Centroid (HQC) classifier is a 
    quantum-inspired supervised classification approach for data with multiple classes.
                         
    Parameters
    ----------
    rescale : int or float, default = 1
        The dataset rescaling factor. A parameter used for rescaling the dataset. 
    encoding : str, default = 'amplit'
        The encoding method used to encode vectors into quantum densities. Possible values:
        'amplit', 'stereo'. 'amplit' means using the amplitude encoding method. 'stereo' means 
        using the inverse of the standard stereographic projection encoding method. Default set 
        to 'amplit'.
    n_copies : int, default = 1
        The number of copies to take for each quantum density. This is equivalent to taking 
        the n-fold Kronecker tensor product for each quantum density.
    measure : str, default = 'pgm'
        The measurement used to distinguish between quantum states. Possible values: 'pgm', 
        'hels'. The value 'pgm' stands for "Pretty Good Measurement", 'hels' stands for 
        "Helstrom measurement" (applicable only for binary classification). Default set to 
        'pgm'. 
    class_weight : str, default = None        
        Weights associated with classes. This is the class weights assigned to the quantum 
        centroids in the Pretty Good Measurement or Helstrom observable. Possible values: None,
        'balanced'. If None given, all classes are supposed to have weight one. The 'balanced' 
        mode uses the values of y to automatically adjust weights inversely proportional to class
        frequencies in the input data as n_samples / (n_classes * np.bincount(y)). Default set
        to None.       
    n_splits : int, default = 1
        The number of subset splits performed on the input dataset row-wise and on the number 
        of eigenvalues/eigenvectors of the Quantum Helstrom observable for optimal speed 
        performance. If 1 is given, no splits are performed. For optimal speed, recommend 
        using small values as close to 1 as possible. If memory blow-out occurs, increase 
        n_splits.
    dtype : torch.float32 or torch.float64, default = torch.float64
        The float datatype used for the elements in the Pytorch tensor dataset. Datatype has to
        be of float to ensure calculations are done in float rather than integer. To achieve
        higher n_copies without memory blow-out issues, reduce float precision, which may or may   
        not affect accuracy in a significant way.
    
    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Sorted classes.
    qcentroids_ : ndarray, shape (n_classes, (n_features + 1)**n_copies, (n_features + 1)**n_copies)
        Quantum Centroids for each class.
    pgms_ : list, shape (n_classes, (n_features + 1)**n_copies, (n_features + 1)**n_copies)
        Values for the Pretty Good Measurements. Only applicable when Pretty Good Measurement is 
        selected.
    pgm_bound_ : float
        Pretty Good Measurement bound is the upper bound on the probability that one can correctly
        discriminate whether a quantum density is of which of the (multiclass) N quantum density 
        patterns. Only applicable when Pretty Good Measurement is selected.
    proj_sums_ : list, shape (n_classes, (n_features + 1)**n_copies, (n_features + 1)**n_copies)
        Sum of the projectors of the Quantum Helstrom observable's unit eigenvectors, which has
        corresponding positive and negative eigenvalues respectively. Only applicable when Helstrom
        Measurement is selected.
    hels_bound_ : float
        Helstrom bound is the upper bound on the probability that one can correctly 
        discriminate whether a quantum density is of which of the two binary quantum density 
        pattern. Only applicable when Helstrom Measurement is selected.         
    """   
    
    # Initialize model hyperparameters
    def __init__(self, 
                 rescale = 1,
                 encoding = 'amplit',
                 n_copies = 1,  
                 measure = 'pgm',
                 class_weight = None, 
                 n_splits = 1,
                 dtype = torch.float64,
                device='cpu'):
        self.rescale = rescale
        self.encoding = encoding
        self.n_copies = n_copies
        self.measure = measure
        self.class_weight = class_weight
        self.n_splits = n_splits
        self.dtype = dtype
        self.device=device
        
        # Raise error if dtype is not torch.float32 or torch.float64
        if self.dtype not in [torch.float32, torch.float64, ]:
          raise ValueError('dtype should be torch.float32 or torch.float64 only')
    
    # Function for X_prime, set as global function
    global X_prime_func
    def X_prime_func(self, X, m):
        
        # Cast array X into a floating point tensor to ensure all following calculations below  
        # are done in float rather than integer, and send tensor X from CPU to GPU
        X = torch.tensor(X, dtype = self.dtype).to(self.device) 
        
        # Rescale X
        X = self.rescale*X
        
        # Calculate sum of squares of each row (sample) in X
        X_sq_sum = (X**2).sum(dim = 1)
        
        # Calculate X' using amplitude or inverse of the standard stereographic projection 
        # encoding method
        if self.encoding == 'amplit':
            X_prime = normalize(torch.cat([X, torch.ones(m, dtype = self.dtype) \
                                           .reshape(-1, 1).to(self.device) ], dim = 1), p = 2, dim = 1)
    
        elif self.encoding == 'stereo':
            X_prime = (1 / (X_sq_sum + 1)).reshape(-1, 1)*(torch.cat((2*X, (X_sq_sum - 1) \
                                                                      .reshape(-1, 1)), dim = 1))
        elif self.encoding == 'proj':
            X_prime=X
        else:
            raise ValueError('encoding should be "amplit", "stereo" or "proj"')
        
        return X_prime
        
        
    # Function for kronecker tensor product for PyTorch tensors, set as global function
    global kronecker
    def kronecker(A, B):
        return torch.einsum('nab,ncd->nacbd', A, B).view(A.size(0), 
                                                         A.size(1)*B.size(1), 
                                                         A.size(2)*B.size(2))
    

    # Set np.einsum subscripts (between unnested and nested objects) as a constant, set as global
    # variable
    global einsum_unnest, einsum_nest
    einsum_unnest = 'ij,ji->'
    einsum_nest = 'bij,ji->b'
    
    
    # Function for fit
    def fit(self, X, y):
        """Perform PGM-HQC classification with the amplitude and inverse of the standard 
        stereographic projection encoding methods, with the option to rescale the dataset prior 
        to encoding.
                
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples. An array of int or float.
        y : array-like, shape (n_samples,)
            The training input binary target values. An array of str, int or float.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Check data in X and y as required by scikit-learn v0.25
       # X, y = self._validate_data(X, y, reset = True)
        
        # Ensure target array y is of non-regression type  
        # Added as required by sklearn check_estimator
        #check_classification_targets(y)
            
        # Store classes and encode y into class indexes
        self.classes_, y_class_index = np.unique(y, return_inverse = True)
        
        # Number of classes, set as global variable
        global num_classes
        num_classes = len(self.classes_)
        
        # Raise error when there are more than 2 classes and Helstrom measurement is specified
        if num_classes > 2 and self.measure == 'hels':
            raise ValueError('Helstrom measurement can be applied for binary classification only')
        else:
            # Number of rows and columns in X
            m, n = X.shape[0], X.shape[1]
            
            # Calculate X_prime
            X_prime = X_prime_func(self, X, m)
                   
            # Number of columns in X', set as global variable
            global n_prime
            if self.encoding=='amplit' or self.encoding=='stereo':
                n_prime=n+1
            elif self.encoding=='proj':
                n_prime=n
        
            # Function to calculate number of rows (samples) and Quantum Centroids for each class 
            def qcentroids_terms_func(i):
                # Cast array y_class_index into a tensor and send from CPU to GPU
                # Determine rows (samples) in X' belonging to either class
                X_prime_class = X_prime[torch.CharTensor(y_class_index).to(self.device) == i]
                                    
                # Split X' belonging to either class into n_splits subsets, row-wise
                # Send tensors from GPU to CPU and cast tensors into arrays, use np.array_split()
                # because the equivalent torch.chunk() doesn't behave similarly to np.array_split()
                X_prime_class_split_arr = np.array_split(X_prime_class.to(self.device) .numpy(),
                                                         indices_or_sections = self.n_splits,
                                                         axis = 0)
            
                # Cast arrays back to tensors and send back from CPU to GPU
                X_prime_class_split = [torch.tensor(a, dtype = self.dtype).to(self.device)  
                                       for a in X_prime_class_split_arr]
            
                # Function to calculate sum of quantum densities belonging to each class, 
                # per subset split
                def X_prime_class_split_func(j):
                    # Counter for j-th split of X'
                    X_prime_class_split_jth = X_prime_class_split[j]
                
                    # Number of rows (samples) in j-th split of X'
                    m_class_split = X_prime_class_split_jth.shape[0]
                
                    # Encode vectors into quantum densities
                    density_chunk = torch.matmul(X_prime_class_split_jth.view(m_class_split, 
                                                                              n_prime, 1),
                                                 X_prime_class_split_jth.view(m_class_split, 
                                                                              1, n_prime))
                
                    # Calculate n-fold Kronecker tensor product
                    if self.n_copies == 1:
                        density_chunk = density_chunk
                    else:
                        density_chunk_copy = density_chunk
                        for _ in range(self.n_copies - 1):
                            density_chunk = kronecker(density_chunk, density_chunk_copy)
                    
                    # Calculate sum of quantum densities
                    density_chunk_sum = density_chunk.sum(dim = 0)
                    return density_chunk_sum

                # Number of rows/columns in density matrix, set as global variable
                global density_nrow_ncol
                density_nrow_ncol = n_prime**self.n_copies
            
                # Initialize tensor density_class_sum
                density_class_sum = torch.zeros([density_nrow_ncol, density_nrow_ncol], 
                                                dtype = self.dtype).to(self.device) 
                for j in range(self.n_splits):
                    # Calculate sum of quantum densities belonging to each class
                    density_class_sum = density_class_sum + X_prime_class_split_func(j)
            
                # Number of rows (samples) in X' belonging to each class
                m_class = X_prime_class.shape[0]
            
                # Function to calculate Quantum Centroid belonging to each class
                def qcentroid_func():
                    # Calculate Quantum Centroid belonging to each class
                    # Added ZeroDivisionError as required by sklearn check_estimator
                    try:
                        qcentroid = (1/m_class)*density_class_sum
                    except ZeroDivisionError:
                        qcentroid = 0 
                    return qcentroid
            
                # Calculate Quantum Centroid belonging to each class
                qcentroid_class = qcentroid_func()
                return m_class, qcentroid_class
            
            # Calculate number of rows (samples) and Quantum Centroids for each class 
            qcentroids_terms = [qcentroids_terms_func(i) for i in range(num_classes)]
           

            # Determine Quantum Centroids
            self.qcentroids_ = torch.stack([qcentroids_terms[z][1] for z in range(num_classes)], dim = 0)
            torch.set_printoptions(threshold=10_000)
          

            # Calculate class weight
            if self.class_weight == None:
                class_weight_terms = torch.tensor([qcentroids_terms[y][0] for y in range(num_classes)], \
                                                  dtype = self.dtype)/m
                
            elif self.class_weight == 'balanced':
                class_weight_terms = torch.tensor([1/num_classes for k in range(num_classes)], \
                                                  dtype = self.dtype)
            else:
                raise ValueError('class_weight should be None or "balanced"')
            
            # When Pretty Good Measurement is specified
            if self.measure == 'pgm':
                # Function to calculate R
                def R_func(a):
                    return class_weight_terms[a]*self.qcentroids_[a]

                # Calculate R
                R = torch.stack([R_func(a) for a in range(num_classes)], dim = 0).sum(dim = 0)
              
                # Calculate square root of pseudo inverse of R
                # Change datatype of R to float64 as the square root of a matrix calculation is highly 
                # senstive to numerical precision/rounding
                # Calculate pseudo inverse of R, send tensor from GPU to CPU and cast into an array
                # Use scipy.linalg.sqrtm() to calculate square root of the pseudo inverse of R because 
                # there is no equivalent function in PyTorch which behaves numerically similarly 
                # Remove complex part of the matrix created due to numerical precision/rounding issues
                # in machine language
                # Cast array back into a tensor and send back from CPU to GPU
                sqrt_pinv_R = torch.tensor(np.real(linalg.sqrtm(torch.pinverse(torch.as_tensor(R, dtype = \
                                           self.dtype)).to(self.device) .numpy())), dtype = self.dtype).to(self.device) 
                    
                # Calculate kernel of R
                # Change datatype of R to float64 as the kernel of a matrix calculation is highly
                # senstive to numerical precision/rounding
                # Send tensor from GPU to CPU and cast into an array, use scipy.linalg.null_space()
                # to calculate kernel because there is no equivalent function in PyTorch which
                # behaves numerically similarly
                # Cast array back into a tensor and send back from CPU to GPU
                ker_R = torch.tensor(linalg.null_space(torch.as_tensor(R, dtype = self.dtype).to(self.device)  \
                                     .numpy()), dtype = self.dtype).to(self.device) 
                    
                # Calculate projector of kernel of R
                proj_ker_R = torch.matmul(ker_R, ker_R.T)
                    
                # Function to calculate Pretty Good Measurement
                def pgm_func(b):
                    return torch.matmul(torch.matmul(sqrt_pinv_R, class_weight_terms[b]*self.qcentroids_[b]), 
                                        sqrt_pinv_R) + (1/num_classes)*proj_ker_R
                                               
                # Calculate Pretty Good Measurement
                self.pgms_ = torch.stack([pgm_func(b) for b in range(num_classes)], dim = 0)

                # Function to calculate PGM bound
                def pgm_bound_func(c):
                    return class_weight_terms[c]*torch.einsum(einsum_unnest, self.qcentroids_[c], self.pgms_[c])

                # Calculate PGM bound
                self.pgm_bound_ = torch.stack([pgm_bound_func(c) for c in range(num_classes)], dim = 0) \
                                             .sum(dim = 0).item()
            # When Helstrom measurement is specified
            elif self.measure == 'hels':
                # Calculate quantum Helstrom observable
                hels_obs = class_weight_terms[0]*self.qcentroids_[0] \
                           - class_weight_terms[1]*self.qcentroids_[1]
                
                # Number of rows/columns in density matrix, set as global variable
                global density_nrow_ncol
                density_nrow_ncol = hels_obs.shape[0]
                
                # Calculate eigenvalues w and unit eigenvectors v of the quantum Helstrom observable
                w, v = torch.linalg.eigh(hels_obs)
                
                # Length of w
                len_w = len(w)
                
                # Initialize tensor eigval_class
                eigval_class = torch.empty_like(w, dtype = self.dtype).to(self.device) 
                for d in range(len_w):
                    # Create a tensor of 0s and 1s to indicate positive and negative eigenvalues
                    # respectively
                    if w[d] > 0:
                        eigval_class[d] = 0
                    else:
                        eigval_class[d] = 1
                        
                # Transpose matrix v containing eigenvectors to row-wise
                eigvec = v.T
                
                # Function to calculate sum of the projectors corresponding to positive and negative
                # eigenvalues respectively
                def sum_proj_func(e):
                    # Split eigenvectors belonging to positive or negative eigenvalues into n_splits subsets
                    # Send tensors from GPU to CPU and cast tensors into arrays, use np.array_split()
                    # because the equivalent torch.chunk() doesn't behave similarly to np.array_split()
                    eigvec_class_split_arr_full = np.array_split(eigvec.to(self.device) .numpy()[eigval_class.to(self.device)  == e],
                                                                 indices_or_sections = self.n_splits,
                                                                 axis = 0)
                    
                    # Remove empty rows in eigvec_class_split_arr_full
                    eigvec_class_split_arr = [f for f in eigvec_class_split_arr_full if f.shape[0] > 0]
                    
                    # Cast arrays back to tensors and send back from CPU to GPU
                    eigvec_class_split = [torch.tensor(g, dtype = self.dtype).to(self.device) 
                                          for g in eigvec_class_split_arr]
                    
                    # Function to calculate sum of the projectors corresponding to positive and negative
                    # eigenvalues respectively, per subset split
                    def eigvec_class_split_func(h):
                        # Counter for h-th split of eigvec
                        eigvec_class_split_hth = eigvec_class_split[h]
                        
                        # Number of rows (samples) in h-th split of eigvec
                        m_eigvec_class_split = eigvec_class_split_hth.shape[0]
                        
                        # Calculate projectors corresponding to positive and negative eigenvalues
                        # respectively, per subset split
                        proj_split = torch.matmul(eigvec_class_split_hth.view(m_eigvec_class_split,
                                                                              density_nrow_ncol, 1),
                                                  eigvec_class_split_hth.view(m_eigvec_class_split,
                                                                              1, density_nrow_ncol))
                        
                        # Calculate sum of projectors
                        proj_split_sum = proj_split.sum(dim = 0)
                        return proj_split_sum
                    
                    # Determine length of eigvec_class_split_arr
                    eigvec_class_split_arr_len = len(eigvec_class_split_arr)
                    
                    # Initialize tensor proj_class_sum
                    proj_class_sum = torch.zeros([density_nrow_ncol, density_nrow_ncol],
                                                 dtype = self.dtype).to(self.device) 
                    for h in range(eigvec_class_split_arr_len):
                        # Calculate sum of the projectors corresponding to positive and negative eigenvalues
                        # respectively
                        proj_class_sum = proj_class_sum + eigvec_class_split_func(h)
                    return proj_class_sum
                
                # Calculate sum of the projectors corresponding to positive and negative eigenvalues
                # respectively
                self.proj_sums_ = torch.stack([sum_proj_func(0), sum_proj_func(1)], dim = 0)
                
                # Calculate Helstrom bound
                self.hels_bound_ = (class_weight_terms[0]*torch.einsum(einsum_unnest, self.qcentroids_[0],
                                                                      self.proj_sums_[0])).item() \
                                   + (class_weight_terms[1]*torch.einsum(einsum_unnest, self.qcentroids_[1],
                                                                        self.proj_sums_[1])).item()
            # When Pretty Good Measurement or Helstrom measurement is misspecified
            else:
                raise ValueError('measure should be "pgm" or "hels"')
        return self

           
    # Function for predict_proba
    def predict_proba(self, X):
        """Performs PMG-HQC classification on X and returns the trace of the dot product of the 
        densities and the POV (positive operator-valued) measure, i.e. the class probabilities.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples. An array of int or float.       
            
        Returns
        -------
        trace_matrix : array-like, shape (n_samples, n_classes)
            Each column corresponds to the trace of the dot product of the densities and the POV 
            (positive operator-valued) measure for each class, i.e. each column corresponds to the 
            class probabilities. An array of float.
        """
        # Send tensors self.pgms_ and self.proj_sums_ from GPU to CPU and cast into an array, and
        # check if fit had been called
        if self.measure == 'pgm':
            self.pgms_arr_ = self.pgms_.to(self.device) .numpy()
            check_is_fitted(self, ['pgms_arr_'])
        else:
            self.proj_sums_arr_ = self.proj_sums_.to(self.device) .numpy()
            check_is_fitted(self, ['proj_sums_arr_'])
               
        # Check data in X as required by scikit-learn v0.25
        X = self._validate_data(X, reset = False)
        
        # Number of rows in X
        m = X.shape[0]
        
        # Calculate X_prime
        X_prime = X_prime_func(self, X, m)
                       
        # Function to calculate trace values for each class
        def trace_func(i):
            # Split X' into n_splits subsets, row-wise
            # Send tensors from GPU to CPU and cast tensors into arrays, use np.array_split()
            # because the equivalent torch.chunk() doesn't behave similarly to np.array_split()
            X_prime_split_arr_full = np.array_split(X_prime.to(self.device) .numpy(),
                                                    indices_or_sections = self.n_splits,
                                                    axis = 0)
            
            # Remove empty rows in X_prime_split_arr_full
            X_prime_split_arr = [a for a in X_prime_split_arr_full if a.shape[0] > 0]

            # Cast arrays back to tensors and send back from CPU to GPU
            X_prime_split = [torch.tensor(q, dtype = self.dtype).to(self.device)  for q in X_prime_split_arr]
            
            # Function to calculate trace values for each class, per subset split
            def trace_split_func(j):
                # Counter for j-th split X'
                X_prime_split_jth = X_prime_split[j]
                
                # Number of rows (samples) in j-th split X'
                X_prime_split_m = X_prime_split_jth.shape[0]
                
                # Encode vectors into quantum densities
                density_chunk = torch.matmul(X_prime_split_jth.view(X_prime_split_m, n_prime, 1),
                                             X_prime_split_jth.view(X_prime_split_m, 1, n_prime))
                
                # Calculate n-fold Kronecker tensor product
                if self.n_copies == 1:
                    density_chunk = density_chunk
                else:
                    density_chunk_copy = density_chunk
                    for _ in range(self.n_copies - 1):
                        density_chunk = kronecker(density_chunk, density_chunk_copy)
                        
                # When Pretty Good Measurement is specified
                if self.measure == 'pgm':
                    # Calculate trace of the dot product of density of each row and Pretty Good
                    # Measurement
                    trace_class_split = torch.einsum(einsum_nest, density_chunk, self.pgms_[i])
                # When Helstrom measurement is specified
                else:
                    # Calculate trace of the dot product of density of each row and sum of 
                    # projectors with corresponding positive and negative eigenvalues respectively
                    trace_class_split = torch.einsum(einsum_nest, density_chunk, self.proj_sums_[i])
                return trace_class_split
            
            # Determine length of X_prime_split_arr
            X_prime_split_arr_len = len(X_prime_split_arr)

            # Initialize tensor trace_class
            trace_class = torch.empty([0], dtype = self.dtype).to(self.device) 
            for j in range(X_prime_split_arr_len):
                # Calculate trace values for each class, per subset split
                trace_class = torch.cat([trace_class, trace_split_func(j)], dim = 0)
            return trace_class
        
        # Calculate trace values for each class, send from GPU to CPU and cast into an array
        trace_matrix = torch.stack([trace_func(i) for i in range(num_classes)], dim = 1).to(self.device) .numpy()
        
        
        return trace_matrix
                
    
    # Function for predict
    def predict(self, X):
        """Performs PGM-HQC classification on X and returns the classes.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples. An array of int or float.
            
        Returns
        -------
        self.classes_[predict_trace_index] : array-like, shape (n_samples,)
            The predicted binary classes. An array of str, int or float.
        """
        # Determine column index with the higher trace value in trace_matrix
        # Cast predict_proba(X) from an array into a tensor and send from CPU to GPU
        # If both columns have the same trace value, returns column index 1, which is different 
        # to np.argmax() which returns column index 0
        predict_trace_index = torch.argmax(torch.tensor(self.predict_proba(X), 
                                                        dtype = self.dtype).to(self.device) , axis = 1)
  
        # Returns the predicted binary classes, send tensor from GPU to CPU and cast tensor
        # into an array
        return self.classes_[predict_trace_index.to(self.device) .numpy()]
    
    
    
    
    
    
    
    
    
    