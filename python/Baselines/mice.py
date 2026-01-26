from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np

def mice(x_input):
    x_input = x_input.detach().cpu().numpy().astype(np.float64)
    x_input[x_input == 0] = np.nan
    imputer = IterativeImputer(max_iter=10, random_state=0)
    x_reconstructed = imputer.fit_transform(x_input)
    return x_reconstructed

