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

#read dataser
def read_data(data_path):
    '''read csv file, bỏ column Unnamed: 0'''
    df = pd.read_csv(data_path)
    col = 'Unnamed: 0'
    if col in df.columns:
        df.drop(col, axis = 1, inplace = True)
    return df

# chia train, test data
def train_test(rating_path, fulldata_path): #fulldata_path là file data cuối cùng của mình
    full_dataset = read_data(fulldata_path)
    rating = read_data(rating_path)
    
    '''preprocessing'''
    full_dataset = full_dataset.drop('Unnamed: 0', axis = 1, inplace = True)
    full_dataset = full_dataset[['movieId', 'title', 'genres', 'description', 'popularity']]
    
    '''Merge full_dataset với rating, chia train: 80, test: 20'''
    data = rating.merge(movie, on = 'movieId')
    train_df, test_df = train_test_split(data, test_size = 0.2, random_state = 42, stratify=data['userId'])
    return train_df, test_df '''train_df, test_df = train_test(rating_path, full_data_path)'''
    
def convert_traintest_dataframe_forsurprise(training_dataframe, testing_dataframe):
    '''Dùng để convert trainset, testset để dùng cho thư viện surprise'''
    reader = Reader(rating_scale=(0, 5))
    trainset = Dataset.load_from_df(training_dataframe[['userId', 'movieId', 'rating']], reader)
    testset = Dataset.load_from_df(testing_dataframe[['userId', 'movieId', 'rating']], reader)
    trainset = trainset.construct_trainset(trainset.raw_ratings)
    testset = testset.construct_testset(testset.raw_ratings)
    return trainset, testset '''train_set, test_set = convert_traintest_dataframe_forsurprise(training_dataframe, testing_dataframe)'''

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
    return knnbaseline_algo

def

def cosine_similarity(fulldata_path): #cái file data cuối cùng của mình
    '''tính cosine similarity dựa trên overview + tagline + 2*genres'''
    df = read_data(fulldata_path)
    df['description_genre'] = df['description']+ 2*df['genres']
    df['description_genre'] = df['description_genre'].fillna('')

    '''vẫn dùng TF-IDF matrix nhưng cộng với 2*genres để trở thành Count Vector'''

    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description_genre'])
    cosine_sim= linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def mapping_title_toIndex(fulldata_path):
    '''map title với index của table movie, index của title = index của bảng, value = title'''
    df =  read_data(fulldata_path):
    titles = df['title']
    indices = pd.Series(df.index, index=df['title'])
    return indices

def get_recommendation_new(title, full_dataset):#dùng file data cuối cùng, type(title) = string
    '''Model recommendation dựa trên Movie Similarity'''
    idx = mapping_title_toIndex[title] #lấy ra index của title
    if type(idx) != np.int64:
        if len(idx)>1:
            print("ALERT: Multiple values")
            idx = idx[0]
    sim_scores = list(enumerate(cosine_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return full_dataset['movieId'].iloc[movie_indices]

def genre_based_popularity(genres, full_dataset):#dùng file data cuối cùng, type(genre) = string
    '''Model recommendation dựa trên popularity'''
    mask = full_dataset.genres.apply(lambda x: genre in x) # trả về dạng bool, check xem genre có trong cái list genres đó k
    filtered_movie = full_dataset[mask]# trả về dataframe các film match với genre
    filtered_movie = filtered_movie.sort_values(by='popularity', ascending=False) #xếp theo độ phổ biến nhất
    return filtered_movie['movieId'].head(10).values.tolist() #trả về list top 10 movie similar
    
    
    
    
    
    


def hybrid(input):
    pass
