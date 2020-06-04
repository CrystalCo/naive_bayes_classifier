# Crystal A. Contreras  Spring CSC-480  Assignment 4
import numpy as np
import pandas as pd 
from scipy import stats
import math


# Pass in dataframe of training data and target.
file_location = '/Users/crystalcontreras/Desktop/DePaul/2020Spring/CSC480/Assignment4/knn-csc480-a4.xls'
data = pd.read_excel(file_location, header=0).loc[0:19].drop('Unnamed: 0', axis=1)
data.replace(' ', np.nan, inplace=True)
test = pd.read_excel(file_location).loc[21:25].drop('Unnamed: 0', axis=1)
test.replace(' ', np.nan, inplace=True)

def average_ratings(data, target_item):
    """ Returns average rating for a given movie. """
    avg_rating = data[target_item].sum()
    if avg_rating:
        return avg_rating/len(data[target_item])
    else:
        return 0 

def predict_ratings(data, target, K):
    """
        data :: type DataFrame
        target :: type Series
        K :: type int; Represents k-neighbors from target item
        The predicted rating of user on target item using the weighted average of 
        the ratings of the K-nearest neighbors on target item.
    """
    data_corr = pd.DataFrame(data, copy=True)

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
    target_row = pd.Series(target, copy=True)
    movies = data_corr.columns
    movies = movies.drop(labels='corr')    # Drop 'corr' column
    predictions = pd.Series(target_row, copy=True)

    for movie in movies:
        numerator = 0
        denominator = 0
        # If the movie has not been rated by user, predict a rating
        for index, row in top_k_rows.iterrows():
            if not np.isnan(row[movie]):
                # Add the weighted average to numerator and sum of correlations in denominator 
                numerator += row[movie] * row['corr']
                denominator += row['corr']

        if not numerator:
            # If out of the k-nearest neighbors, no co-rated items exist, predict rating based on average rating
            predictions.update(pd.Series([average_ratings(data, movie)], index=[movie]))
        else:
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
    movie_rating_prediction = predictions.loc[target_item]
    return movie_rating_prediction
    
def recommend_top_n(data, target, N, K):
    """
        Given a user (target) and the number of desired recommendations N, 
        generates the top N recommended items for user. 
    """
    target_recommendations = pd.Series(target, copy=True)
    rating_predictions = predict_ratings(data, target_recommendations, K)

    for index, value in target_recommendations.iteritems():
        if not np.isnan(value):
            rating_predictions[index] = np.nan

    rating_predictions.dropna(inplace=True)
    rating_predictions.sort_values(ascending=False)
    return rating_predictions[0:min(N, len(rating_predictions))]

    
def test_recommender(data, test_data, K):
    """
        Compute predicted ratings for the existing ratings of NU1-NU5. 
        Measure the Mean Absolute Error (MAE) on these predictions for the test users. 
        You can compute MAE by generating predictions for items already rated by NU1 through NU5 (e.g., for NU1 these are all items except "The DaVinci Code" and "Runny Babbit"). 
        In each case, you will use the remaining ratings of the target test user to generate prediction for the test item being considered. 
        Then, for each of these items you can compute the absolute value of the difference between the predicted and the actual ratings. 
        Finally, you can average these errors across all test cases to obtain the MAE.
    """
    test_users = pd.DataFrame(test_data, copy=True)

    # Compute predicted ratings for the existing ratings of NU1-NU5. 
    numerator = 0
    denominator = len(test_users)

    for index, row in test_users.iterrows():
        # For each test user, use the remaining ratings of the target test user to generate prediction for the test item being considered. 
        predictions = predict_ratings(data, row, K)
        # Measure the Mean Absolute Error (MAE) on these predictions for the test users. 
        for i, prediction in predictions.iteritems():
            if not np.isnan(row.loc[i]):
                print(f'Prediction: {prediction}.   Actual: {row.loc[i]}')
                rating_difference = prediction - row.loc[i]
                numerator += abs(rating_difference)
        
        MAE = math.sqrt(numerator/denominator)
        print("MAE: ", MAE)




# Your program should allow you to specify a user in the data (e.g., a user's row number in the ratings matrix) and the value of N. 
target_user = test.iloc[0]
target_user.replace(' ', np.nan, inplace=True)
target_movie = 'THE DA VINCI CODE'
N = 2
K = 3 

# predict_rating(data, target_user, target_movie, 3)
# print(recommend_top_n(data, target_user, N, K))
test_recommender(data, test, K)




