# hybrid approach
# try to put the code in FUNCTIONS so that we will later call the functions in __init__.py

# sử dụng lại các functions từ content-based, hoặc collaborative_filtering nếu có thể

# DL: thứ 7 4/11

'''Update vấn đề: hàm read_data có dùng dc converters cho genres hay k -> liên quan đến '''
import pandas as pd
import numpy as np

from surprise import SVD, BaselineOnly, SVDpp, NMF, SlopeOne, CoClustering, Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import accuracy
from sklearn.model_selection import train_test_split
from ast import literal_eval
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
def train_test_df(rating_data, full_data): #full_data là file data cuối cùng của mình tuwf get_movie_features (input la dataframe)
    '''chia tập train và tập test theo sklearn'''
    full_data = full_data[['movieId', 'title', 'genres', 'description', 'popularity']]
    
    '''Merge full_dataset với rating, chia train: 80, test: 20'''
    data = rating_data.merge(full_data, on = 'movieId')
    train_df, test_df = train_test_split(data, test_size = 0.2, random_state = 42, stratify=data['userId'])
    return train_df, test_df
    '''train_df, test_df = train_test(rating_path, full_data_path)'''
    
def convert_traintest_dataframe_forsurprise(train_df, test_df): #train, test_df lấy ở hàm train_test_df
    '''Dùng để convert trainset, testset để dùng cho thư viện surprise'''
    reader = Reader(rating_scale=(0, 5))
    train_convert = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    test_convert = Dataset.load_from_df(test_df[['userId', 'movieId', 'rating']], reader)
    train_convert = train_convert.construct_trainset(train_convert.raw_ratings)
    test_convert = test_convert.construct_testset(test_convert.raw_ratings)
    return train_convert, test_convert '''train_convert, test_convert = convert_traintest_dataframe_forsurprise(train_df, test_df)'''

def knnbaseline(train_convert, test_convert):
    sim_options = {'name': 'cosine', 'user_based': False} # compute  similarities between items
    knnbaseline_algo = KNNBaseline(sim_options=sim_options)

    knnbaseline_algo.fit(train_convert)
    knnbaseline_predictions = knnbaseline_algo.test(test_convert)

    file_name = 'KnnBaseline_model'
    dump.dump(file_name, algo=knnbaseline_predictions)

    accuracy.rmse(knnbaseline_predictions)
    accuracy.mae(knnbaseline_predictions)
    print("Done!")
    return knnbaseline_algo #phải đặt tên 1 biến là knnbaseline_algo = knnbaseline(trainset, testset)

def svd(train_convert, test_convert):
    svd_algo = SVD()

    svd_algo.fit(train_convert)
    svd_predictions = svd_algo.test(test_convert)

    file_name = 'svd_model'
    dump.dump(file_name, algo=svd_algo)

    accuracy.rmse(svd_predictions)
    accuracy.mae(svd_predictions)
    print("Done!")
    return svd_algo #phải đặt tên 1 biến là svd_algo = svd(train_convert, test_convert):

def svdpp(train_convert, test_convert):
    svdpp_algo = SVDpp()
    svdpp_algo.fit(train_convert)
    svdpp_predictions = svdpp_algo.test(test_convert)
    
    file_name = 'svdpp_model'
    dump.dump(file_name, algo=svdpp_algo)
    
    
    accuracy.rmse(svdpp_predictions)
    accuracy.mae(svdpp_predictions)
    print("Done!")
    return svdpp_algo #phải đặt tên 1 biến là svdpp_algo = svdpp(train_convert, test_convert):
    

def cosine_similarity(full_data): #cái dataframe từ file data cuối cùng của mình - get_movie_feature
    '''tính cosine similarity dựa trên overview + tagline + 2*genres'''
    full_data['description_genre'] = full_data['description']+ 2*full_data['genres']
    full_data['description_genre'] = full_data['description_genre'].fillna('')

    '''vẫn dùng TF-IDF matrix nhưng cộng với 2*genres để trở thành Count Vector'''

    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = tfidf.fit_transform(full_data['description_genre'])
    cosine_sim= linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def mapping_title_toIndex(full_data): #dataframe từ file data cuối cùng của mình - get_movie_feature
    '''map title với index của table movie, index của title = index của bảng, value = title'''
    titles = full_data['title']
    indices = pd.Series(full_data.index, index=full_data['title'])
    return indices

def get_recommendation_new(title, full_data):#dataframe từ file data cuối cùng của mình - get_movie_feature, type(title) = String
    '''Model recommendation dựa trên Movie Similarity'''
    idx = mapping_title_toIndex(full_data)[title] #lấy ra index của title
    if type(idx) != np.int64:
        if len(idx)>1:
            print("ALERT: Multiple values")
            idx = idx[0]
    sim_scores = list(enumerate(cosine_similarity(full_data)[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return full_data['movieId'].iloc[movie_indices]

def genre_based_popularity(genre, full_data):#dataframe từ file data cuối cùng của mình - get_movie_feature, type(genre) = String
    '''Model recommendation dựa trên popularity'''
    mask = full_data.genres.apply(lambda x: genre in x) # trả về dạng bool, check xem genre có trong cái list genres đó k
    filtered_movie = full_data[mask]# trả về dataframe các film match với genre
    filtered_movie = filtered_movie.sort_values(by='popularity', ascending=False) #xếp theo độ phổ biến nhất
    return filtered_movie['movieId'].head(10).values.tolist() #trả về list top 10 movie similar


def make_useinfo_df(full_data, train_df): #full_data - get_movie_feature; train_df: train của sklearn
    full_data['genres'] = full_data.genres.apply(lambda x: literal_eval(str(x)))
    train_df['genres'] = train_df.genres.apply(lambda x: literal_eval(str(x)))
    
    unique_genre = full_data['genres'].explode().unique() #unique genres của full_data
    genre_distribution = train_df['genres'].explode().value_counts() #unique genres của train_df
    
    # Make a dict assigning an index to a genre
    genre_dict = {k: v for v, k in enumerate(unique_genre)} #key-value: genre - encode

    user_ids = train_df['userId'].unique()
    user_df = pd.DataFrame(columns=['userId', 'user_vector', 'avg_rating', 'num_movies_rated'])
    for user_id in user_ids:
        user_rating_df = train_df[(train_df['userId'] == user_id)]
        user_vector = np.zeros(len(genre_dict))
        count_vector = np.zeros(len(genre_dict))

        user_avg_rating = 0
        movies_rated_count = 0
        for _, row in user_rating_df.iterrows():
            user_avg_rating += row.rating 
            movies_rated_count += 1
            genres = row.genres

            user_movie_vector = np.zeros(len(genre_dict))

            for g in genres:
                user_movie_vector[genre_dict[g]] = 1
                count_vector[genre_dict[g]] += 1
            
            user_vector += user_movie_vector*row.rating
        count_vector = np.where(count_vector==0, 1, count_vector)
        user_vector = np.divide(user_vector, count_vector)
        user_avg_rating /= movies_rated_count
        row_df = pd.DataFrame([[user_id, user_vector, user_avg_rating, movies_rated_count]], 
                          columns=['userId', 'user_vector', 'avg_rating', 'num_movies_rated'])
        user_df = pd.concat([user_df, row_df], ignore_index=True)
    return user_df
    '''đặt biến user_info = make_useinfo_df(full_data, train_df): trae về dataframe'''

def user_top_genre(userId, user_info): #user_info la dataframe dùng hàm make_useinfo_df(fulldata_path, trainset_path), type(userId) = int
    user_vec = user_info['user_vector'][user_info['userId'] == userId].values[0].copy()
    print("User Vector: ", user_vec)
    top_genre_indices = np.flip(np.argsort(user_vec))
    genre_list = []
    for i in top_genre_indices[:3]:
        genre_list.append(idx_to_genre[i])
    return genre_list '''đặt 1 biến là genre_list = user_top_genre(userId, user_info'''
    
def hybrid(userId, full_datas, train_df, test_df, train_convert, test_convert, knnbaseline_algo, svdpp_algo): #full_data là file data cuối cùng, train-test_df là qua sklearn
    user_movies = test_df[test_df['userId'] == userId]
    user_movies['est'] = user_movies['movieId'].apply(lambda x: 0.6*knnbaseline_algo.predict(userId,x).est + 0.4*svdpp_algo.predict(userId, x).est)    
    user_movies = user_movies.sort_values(by ='est', ascending=False).head(4)
    user_movies['Model'] = 'SVD + CF'
    
    recommend_list = user_movies[['movieId', 'est', 'Model']]

    movie_list = recommend_list['movieId'].values.tolist()
    sim_movies_list = []
    for movie_id in movie_list:
        # Call content based 
        movie_title = full_data['title'][full_data['movieId'] == movie_id].values[0]
        sim_movies = get_recommendations_new(movie_title, full_data) 
        sim_movies_list.extend(sim_movies)
    # Compute ratings for the popular movies
    for movie_id in sim_movies_list:
        pred_rating = 0.6*knnbaseline_algo.predict(userId, movie_id).est + 0.4*svdpp_algo.predict(userId, movie_id).est
        row_df = pd.DataFrame([[movie_id, pred_rating, 'Movie similarity']], columns=['movieId', 'est','Model'])
        recommend_list = pd.concat([recommend_list, row_df], ignore_index=True)

    # Popular based movies
    top_genre_list = user_top_genre(userId, user_info) #data frame user_info
    print("User top genre list: ", top_genre_list)

    popular_movies = []
    for top_genre in top_genre_list:
        popular_movies.extend(genre_based_popularity(top_genre, full_data))
    print("Final list: ", popular_movies)

    # Compute ratings for the popular movies
    for movie_id in popular_movies:
        pred_rating = 0.6*knnbaseline_algo.predict(userId, movie_id).est + 0.4*svdpp_algo.predict(userId, movie_id).est
        row_df = pd.DataFrame([[movie_id, pred_rating, 'Popularity']], columns=['movieId', 'est','Model'])
        recommend_list = pd.concat([recommend_list, row_df], ignore_index=True)
    recommend_list = recommend_list.drop_duplicates(subset=['movieId'])
    train_movie_list = train_df[train_df['userId']==userId]['movieId'].values.tolist()

    # Remove movies in training for this user
    mask = recommend_list.movieId.apply(lambda x: x not in train_movie_list)
    recommend_list = recommend_list[mask]
    
    return recommend_list

def get_title(x):
    '''lấy ra title của hàm hybrid'''
    mid = x['movieId']
    return full_data['title'][full_data['movieId'] == mid].values

def get_genre(x):
    '''get genre của hybrid'''
    mid = x['movieId']
    return full_data['genres'][full_data['movieId'] == mid].values
    
