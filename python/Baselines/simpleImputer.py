import numpy as np
from sklearn.impute import SimpleImputer

def simpleImpute(x_input, batch_size):
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    x_input = x_input.reshape(-1,90)
    prediction = imp.fit_transform(x_input)
    return prediction.reshape(batch_size*10000,90)

