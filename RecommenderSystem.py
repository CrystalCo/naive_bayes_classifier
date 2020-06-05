# Crystal A. Contreras  Spring CSC-480  Assignment 4
import numpy as np
import pandas as pd 
from scipy import stats
import math


# Pass in dataframe of training data and target.
file_location = '/Users/crystalcontreras/Desktop/DePaul/2020Spring/CSC480/Assignment4/knn-csc480-a4.xls'

data = pd.read_excel(file_location, header=0).loc[0:25].drop('Unnamed: 0', axis=1)
data.replace(' ', np.nan, inplace=True)

existing_user_data = pd.read_excel(file_location, header=0).loc[0:19].drop('Unnamed: 0', axis=1)
existing_user_data.replace(' ', np.nan, inplace=True)

new_user_data = pd.read_excel(file_location).loc[21:25].drop('Unnamed: 0', axis=1)
new_user_data.replace(' ', np.nan, inplace=True)

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
            # Append to lists only for movies that both users have rated
            if not np.isnan(x[j]) and not np.isnan(y[j]):
                new_x.append(x[j])
                new_y.append(y[j])
        # If the number of co-rated items is below 2, the correlation coefficient will be undefined. 
        # Check for this situation and not consider neighbors with too few overlapping items
        if len(new_x) < 2:
            corr.append(np.nan)
        else:
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
            if not np.isnan(row.loc[movie]):
                # Add the weighted average to numerator and sum of correlations in denominator 
                numerator += row.loc[movie] * row['corr']
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
        Compute the predicted rating of user `target` on `target_item` 
        (assuming that `target` has not previously rated `target_item`). 
        Note also that if `target` is a test user who has an actual rating on `target_item`, 
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

    
def test_recommender(data, test_data, K, print_diff=False):
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
    MAE_Avg = []

    for index, row in test_users.iterrows():
        if print_diff:
            print(f"User: NU{index}")
        # For each test user, use the remaining ratings of the target test user to generate prediction for the test item being considered. 
        predictions = predict_ratings(data, row, K)
        # Measure the Mean Absolute Error (MAE) on these predictions for the test users. 
        for i, prediction in predictions.iteritems():
            if not np.isnan(row.loc[i]):
                if print_diff:
                    print(f'Movie: {i}\tPrediction: {prediction}.\tActual: {row.loc[i]}')
                rating_difference = prediction - row.loc[i]
                numerator += abs(rating_difference)
        
        MAE = math.sqrt(numerator/denominator)
        MAE_Avg.append(MAE)

    if MAE_Avg != []:
        return np.sum(MAE_Avg)/len(MAE_Avg)
    else:
        return -1 




# implementations of KNN

print("Predict Rating...\n")

u_i = input("Enter the target user's row number in the ratings matrix: ")
u_i = int(u_i)
print("User: U{}\n".format(u_i+1))

target_user = data.iloc[u_i]
target_user.replace(' ', np.nan, inplace=True)
# print("Target user: {}\n".format(target_user))

target_movie = input("(Optional. Default is 'THE DA VINCI CODE') Enter the target movie you wish to predict: ")
if target_movie == '':
    target_movie = 'THE DA VINCI CODE'

K = input("Enter the number of nearest neighbors to check (K): ")
K = int(K)

predicted_rating = predict_rating(existing_user_data, target_user, target_movie, K)

print("\nPredicted Rating for movie {}: {}".format(target_movie, predicted_rating))



print("\n\nRecommendations...\n")

N = input("Enter the value of N for top N recommendations: ")
N = int(N)

print()
print(recommend_top_n(existing_user_data, target_user, N, K))


# Testing
print("\n\nTesting our Recommender...\n")
print("Mean Absolute Error for K = 3:")
print(test_recommender(existing_user_data, new_user_data, 3, True))

# Create a table of MAE values for values of K in {1, 2, ..., 20}.
mae_values_table = []
for k in range(1, 19):
    mae_values_table.append(test_recommender(existing_user_data, new_user_data, k))

# Next, using the best value of K from the previous part, compute the predicted ratings for NU1 and NU2 for all items that have not already been rated by these two users.
movies = data.columns
best_k = mae_values_table.index(min(mae_values_table)) + 1
print("\nPredicted rating for NU1:\n {}\n".format(recommend_top_n(existing_user_data, new_user_data.iloc[0], len(movies), best_k)))
print("\nPredicted rating for NU2:\n {}\n".format(recommend_top_n(existing_user_data, new_user_data.iloc[1], len(movies), best_k)))

# Finally, using your recommendation function (and K = 4) generate the top 3 recommendations for the following users: U2, U5, U13, and U20.
print("\nTop 3 Recommendations for U2:\n {}\n".format(recommend_top_n(existing_user_data, existing_user_data.iloc[1], 3, 4)))
print("\nTop 3 Recommendations for U5:\n {}\n".format(recommend_top_n(existing_user_data, existing_user_data.iloc[4], 3, 4)))
print("\nTop 3 Recommendations for U13:\n {}\n".format(recommend_top_n(existing_user_data, existing_user_data.iloc[12], 3, 4)))
print("\nTop 3 Recommendations for U20:\n {}\n".format(recommend_top_n(existing_user_data, existing_user_data.iloc[19], 3, 4)))

print("\n\nTesting complete.\n\n")



