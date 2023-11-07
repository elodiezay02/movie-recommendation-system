# hybrid approach
# try to put the code in FUNCTIONS so that we will later call the functions in __init__.py

# sử dụng lại các functions từ content-based, hoặc collaborative_filtering nếu có thể

# DL: thứ 7 4/11
import pandas as pd
import numpy as np

from surprise import SVD, BaselineOnly, SVDpp, NMF, SlopeOne, CoClustering, Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import accuracy
from sklearn.model_selection import train_test_split
from surprise import dump

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# chia train, test data
def train_test(rating_path, full_data_path):
    full_dataset = pd.read_csv(full_data_path)
    rating = pd.read_csv(rating_path)
    
    '''preprocessing'''
    full_dataset = full_dataset.drop('Unnamed: 0', axis = 1, inplace = True)
    full_dataset = full_dataset[['movieId', 'title', 'genres', 'description', 'popularity']]
    
    '''Merge full_dataset với rating, chia train: 80, test: 20'''
    data = rating.merge(movie, on = 'movieId')
    train_df, test_df = train_test_split(data, test_size = 0.2, random_state = 42, stratify=data['userId'])
    
def convert_traintest_dataframe_forsurprise(training_dataframe, testing_dataframe):
    '''Dùng để convert trainset, testset để dùng cho thư viện surprise'''
    reader = Reader(rating_scale=(0, 5))
    trainset = Dataset.load_from_df(training_dataframe[['userId', 'movieId', 'rating']], reader)
    testset = Dataset.load_from_df(testing_dataframe[['userId', 'movieId', 'rating']], reader)
    trainset = trainset.construct_trainset(trainset.raw_ratings)
    testset = testset.construct_testset(testset.raw_ratings)
    return trainset, testset

def knnbaseline(trainset, testset):
    sim_options = {'name': 'cosine', 'user_based': False} # compute  similarities between items
    knnbaseline_algo = KNNBaseline(sim_options=sim_options)

    knnbaseline_algo.fit(train_set)
    knnbaseline_predictions = knnbaseline_algo.test(test_set)

    file_name = 'KnnBaseline_model'
    dump.dump(file_name, algo=knnbaseline_predictions)

    accuracy.rmse(knnbaseline_predictions)
    accuracy.mae(knnbaseline_predictions)
    print("Done!")
    


def hybrid(input):
    pass
