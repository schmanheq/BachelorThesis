import numpy as np
from sklearn.impute import SimpleImputer

def simpleImpute(x_input):
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    predictions = imp.transform(x_input)
    return predictions

