import pandas as pd
from pandas.api.types import is_string_dtype
import numpy as np
import re

clfs_names_short_mapping = {"Bernoulli Naive Bayes":"B-NB",
    "Dummy Classifier":"Dummy",
    "Extra Tree":"ET",
    "Gaussian Naive Bayes":"G-NB",
    "Hels":"Hels",
    "Linear Discriminant Analysis": "LDA",
    "MLP": "MLP",
    "Nearest Centroid":"NC",
    "Nearest Neighbors":"NN",
    "PGM":"PGM",
    "Quadratic Discriminant Analysis": "QDA",
    "Random Forest": "RF",
    "Support Vector Machine": "SVM" }



def save_latex_table_from_winner_avg(file_path):

    """
    Produce and save the code to render a LaTex table with 
    average balanced accuracy per classifier among all datasets.

    The output file name will have a postfix more than the input file name. 

    Args:
        file_path (string): Path of txt file with average scores and winner classifier.

    """
   
    classifier_data = {} # Dictionary to store classifier data

    # Regular expression to extract classifier name, mean, and uncertainty
    pattern = r"(.+?) = ([0-9.]+) ± ([0-9.]+)"

    # Read the file and extract data
    with open(file_path, "r") as file:
        for line in file:
            if "BEST"  in line:
                continue

            match = re.match(pattern, line.strip())
            if match:
                classifier_name = match.group(1).strip()
                mean = float(match.group(2))
                uncertainty = float(match.group(3))
                classifier_data[classifier_name] = (mean, uncertainty)

    # Generate the LaTeX table
    latex_table = ""
    for classifier, (mean, uncertainty) in classifier_data.items():
        # Highlight the PGM classifier row
        if classifier == "PGM":
            latex_table += f"\\rowcolor{{lightblue3}}\n"

        short_name_substr = f" ({clfs_names_short_mapping[classifier]})"
        latex_table += f"{classifier} {'' if clfs_names_short_mapping[classifier] in ['Hels', 'PGM', 'MLP'] else short_name_substr} & {mean:.3f} $\\pm$ {uncertainty:.3f} \\\\\n"

    # Path to save the LaTeX table
    output_file_path = file_path[:-4]+'_latex_tab_code.txt'

    # Save the LaTeX table to a text file
    with open(output_file_path, "w") as output_file:
        output_file.write(latex_table)



def labeling_sort_key_function(s):
    """
    Sorting function for [Fact, Sep1, Sep2, ... SepN, Ent] criteria. 

    Args:
        s (string): String class label.

    Returns:
        int: Priority index of the intput string.

    """
    if s.startswith("Fact"):
        return (0, s)  # Fact strings have the highest priority
    elif s.startswith("Sep"):
        return (1, s)  # Sep strings are next, sorted lexicographically
    elif s == "Ent":
        return (2, s)  # Ent strings come last
    else:
        print("Wrong type string", s)
        #return (3, s)  # Any other string would go to the end
    


def get_str_labels(numeric_labels, clf_problem):   
    """
    Make string labels from numeric labels, for both EFS and Non-Locality classification problems.
    In EFS the class sorting criteria will be: [Fact, Sep1, Sep2, ... SepN, Ent].

    Args:
        num_labels (list of int): Sequence of numeric labels.
        clf_problem (string): Classification problem identifier: 'efs' or 'nonLoc'.

    Returns:
        list of string: String labels

    Examples:
        >>> get_str_labels([0,1,2,3,4,5,6])
        [Fact, Sep1, Sep2, Sep3, Sep4, Sep5, Ent]

    """
    if clf_problem == 'nonLoc':
        str_labels=['Not Violate', 'Violate'] 
    elif clf_problem == 'efs':
        labels = numeric_labels
        sep_numeric_label = labels[1:-1]

        d = {
        labels[0]: 'Fact',
        labels[-1]: 'Ent'
        }

        for i, nl in enumerate(sep_numeric_label):
            d[nl] = 'Sep' + str(i + 1)

        str_labels = [d[l] for l in labels]
        str_labels = sorted(str_labels, key=labeling_sort_key_function)

    return str_labels



def get_dataset(d_path, label_column_idx, addOnes=True):
    """
    Reads CSV file as a dataframe, replaces text labels with numeric labels starting from 0 

    Args:
        d_path (string): Path of CSV file.
        label_column_idx (int): Index of labels column.

    Returns:
        tuple of numpy.ndarray: Data and Labels

    """
    df = pd.read_csv(d_path, header=None)
    class_column = list(df.keys())[label_column_idx]

    if is_string_dtype(df[class_column]):    
       
        class_names = list(set(df[class_column]))
        class_names = sorted(class_names, key=labeling_sort_key_function)
        print(class_names)
        # convert it to numerical from 0 to nClasses-1)   
        for idx, key in enumerate( class_names ):
            df = df.replace(key,idx)

    X = df.to_numpy()[:,:label_column_idx]
    y = df.to_numpy()[:,label_column_idx].astype('uint8').flatten()

    non_zero_idxs = ~np.all(X == 0, axis=1)

    X = X[non_zero_idxs]  # rimuovo tutte le righe che contengono solo zeri
    y = y[non_zero_idxs]

    if X.dtype == 'object':
        X = X.astype(np.complex128)
    
    return X, y
 
