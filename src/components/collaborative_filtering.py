
import numpy as np
import pandas as pd
from surprise import dump


from surprise import Dataset, Reader
from surprise import NMF, SVD
# from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.dataset import DatasetAutoFolds

from collections import defaultdict

from evaluate_model import precision_recall_at_k, evaluate

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
        
    def fit(self):
    
        self.algorithm.fit(self.trainset)

    def predict(self):

        predictions_test = self.algorithm.test(self.testset)
        self.predictions = predictions_test
        rmse = accuracy.rmse(predictions_test)
        mae = accuracy.mae(predictions_test)

        file_name = str(self.algorithm).split('.')[-1].split()[0] + '_model'
        dump.dump(file_name, algo=self.algorithm)

        test_df = pd.DataFrame(self.predictions).drop(columns='details')
        test_df.columns = ['userId', 'movieId', 'rating', 'pred_rating']
        self.test_df = test_df
        pre, recall, f_measure = precision_recall_at_k(test_df, 3.5)
        
        return pre, recall, f_measure

