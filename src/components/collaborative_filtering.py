# collaborative algorithm
# try to put the code in FUNCTIONS so that we will later call the functions in __init__.py


import numpy as np
import pandas as pd



from surprise import Dataset, Reader
from surprise import NMF, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.dataset import DatasetAutoFolds

from collections import defaultdict


def collaborative_filtering(input, algorithm, n_suggestion=10):
    """
    Using collaborative filtering to suggest n movies for users
    Input:
        input: csv file
        algorithm: like SVD(), NMF()
        n_suggestion: number of movies recommended for each user
     Return:
     A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
    """
    
    def load_data(path_csv):
        reader = Reader()
        ratings = pd.read_csv(path_csv)
        ratings.head()
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
        
        return data
        
    def fit_predict(algo, path_csv):
        """
        Train and predict in testset

        Returns:
        A tuple containing predictions and RMSE of the algorithm in testset
        """

        data = load_data(path_csv)
        trainset, testset = train_test_split(data, test_size=0.25)
        # fit
        algo.fit(trainset)
        predictions = algo.test(testset)
        
        return predictions, accuracy.rmse(predictions)
    
    def get_top_n(algo, path_csv, n=10):
        """Return the top-N recommendation for each user from a set of predictions.

        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        """
        
        predictions = fit_predict(algo, path_csv)[0]
        
        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n
    
    return get_top_n(algorithm, input)


## Example

top_n = collaborative_filtering(input='../input/ratings_small.csv', algorithm=NMF())

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])