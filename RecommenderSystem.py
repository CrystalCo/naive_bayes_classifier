# Crystal A. Contreras  Spring CSC-480  Assignment 4
import numpy as np
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt

test_row = [1,3,5,2,3,4,5,2]

# Pass in dataframe of training data and target.
file_location = '/Users/crystalcontreras/Desktop/DePaul/2020Spring/CSC480/Assignment4/knn-csc480-a4.xls'
data = pd.read_excel(file_location, header=0).loc[0:19].drop('Unnamed: 0', axis=1)
test = pd.read_excel(file_location).loc[21:25].drop('Unnamed: 0', axis=1)

def predict_ratings(data, target, K):
    """
        The predicted rating of user u_t on item i_t will be the weighted average of the ratings of the K 
        neighbors on item i_t (with the weight of the neighbor's being the similarity of that neighbor to u_t)
    """
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
    top_k_rows = data_corr.iloc[0:new_K]
    target_row = target
    movies = top_k_rows.columns
    predictions = pd.Series(target_row, copy=True)

    for movie in movies:
        numerator = 0
        denominator = 0
        # If the item has not been rated
        for index, row in top_k_rows.iterrows():
            if not np.isnan(row[movie]):
                # Add the weighted average to numerator and sum of correlations in denominator 
                numerator += row[movie] * row['corr']
                denominator += row['corr']
            
        average_rating = numerator / denominator
        predictions.update(pd.Series([average_rating], index=[movie]))

    return predictions



def predict_rating(data, target, target_item, K):
    """ 
        Compute the predicted rating of user `target` on item `target_item` 
        (assuming that `target` has not previously rated `target_item`). 
        Note also that if `target` is a test user who has an actual rating on item `target_item`, 
        the predicted rating for `target_item` can still be generated and compared to the actual 
        rating to measure prediction error rate.
    """
    predictions = predict_ratings(data, target, K)
    print(f'Predictions:\n{predictions}')
    movie_rating_prediction = predictions.loc[target_item]
    print(f'\nPrediction rating on movie {target_item}: {movie_rating_prediction}\n')
    return movie_rating_prediction
    

def recommend_top_n(data, target, N, K):
    """
        Given a user u_t and the number of desired recommendations N, generates the top N recommended items for u_t. 
    """
    target_recommendations = pd.Series(target, copy=True)
    rating_predictions = predict_ratings(data, target_recommendations, K)

    for index, value in target_recommendations.iteritems():
        if not np.isnan(value):
            rating_predictions[index] = np.nan

    rating_predictions.dropna(inplace=True)
    rating_predictions.sort_values(ascending=False)
    return rating_predictions[0:min(N, len(rating_predictions))]

    
def test_recommender(test_data):
    """
        Compute predicted ratings for the existing ratings of NU1-NU5. 
        Measure the Mean Absolute Error (MAE) on these predictions for the test users. 
        You can compute MAE by generating predictions for items already rated by NU1 through NU5 (e.g., for NU1 these are all items except "The DaVinci Code" and "Runny Babbit"). 
        In each case, you will use the remaining ratings of the target test user to generate prediction for the test item being considered. 
        Then, for each of these items you can compute the absolute value of the difference between the predicted and the actual ratings. 
        Finally, you can average these errors across all test cases to obtain the MAE.
    """
    pass


# Your program should allow you to specify a user in the data (e.g., a user's row number in the ratings matrix) and the value of N. 
target_user = test.iloc[0]
target_user.replace(' ', np.nan, inplace=True)
target_movie = 'THE DA VINCI CODE'
N = 2
K = 3 
# predict_rating(data, target_user, target_movie, 3)
print(recommend_top_n(data, target_user, N, K))










# The index is not a column; it is its own thing