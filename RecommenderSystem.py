# Crystal A. Contreras  Spring CSC-480  Assignment 4
import numpy as np
import sys
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt

test_row = [1,3,5,2,3,4,5,2]

# Pass in dataframe of training data and target.
file_location = '/Users/crystalcontreras/Desktop/DePaul/2020Spring/CSC480/Assignment4/knn-csc480-a4.xls'
data = pd.read_excel(file_location, header=0).loc[0:19].drop('Unnamed: 0', axis=1)
test = pd.read_excel(file_location).loc[21:25].drop('Unnamed: 0', axis=1)

def predict_ratings(data, target, K):
    data_corr = data
    data_corr.replace(' ', np.nan, inplace=True)

    # For each row in dataframe, compute correlation of row with target row.
    corr = []
    for i in range(len(data_corr)):
        x = data_corr.iloc[i]
        y = target
        new_x = []
        new_y = []
        for j in range(len(target)):
            if not np.isnan(x[j]) and not np.isnan(y[j]):
                new_x.append(x[j])
                new_y.append(y[j])
        corr.append(stats.pearsonr(new_x, new_y)[0])
    data_corr['corr'] = corr

    # Sort by correlation
    data_corr.sort_values(by='corr', ascending=False, inplace=True)

    # If K contains non-positive correlations, reduce K to only include positive correlations.
    is_positive_corr = False
    new_K = K
    while not is_positive_corr:
        if data_corr.iloc[new_K - 1]['corr'] <= 0:
            new_K -= 1
        else:
            is_positive_corr = True

    # Take top K rows, and use weighted average to create a prediction for every column.
    
    print(data_corr)

predict_ratings(data, test.iloc[0], 11)
