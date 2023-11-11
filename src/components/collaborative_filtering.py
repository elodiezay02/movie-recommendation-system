
import numpy as np
import pandas as pd



from surprise import Dataset, Reader
from surprise import NMF, SVD
# from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.dataset import DatasetAutoFolds

from collections import defaultdict

class CollaborativeFiltering:
    def __init__(self, algorithm=None):
        self.trainset = None
        self.testset = None
        self.algorithm = algorithm
        self.predictions = None
        
    def load_data(self, trainset, testset):
        reader = Reader()
  
        train = Dataset.load_from_df(trainset[['userId', 'movieId', 'rating']], reader)
        test = Dataset.load_from_df(testset[['userId', 'movieId', 'rating']], reader)
        
        self.trainset = train.build_full_trainset()
        full_testset = test.build_full_trainset()
        self.testset = full_testset.build_testset()
        
    def fit_predict(self):
    
        self.algorithm.fit(self.trainset)
        predictions_test = self.algorithm.test(self.testset)
        self.predictions = predictions_test
        
        return accuracy.rmse(predictions_test), accuracy.mae(predictions_test)

