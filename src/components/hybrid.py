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
def read_data(data_path, ):
    '''read csv file, bỏ column Unnamed: 0'''
    df = pd.read_csv(data_path)
    col = 'Unnamed: 0'
    if col in df.columns:
        df.drop(col, axis = 1, inplace = True)
    return df

# chia train, test data
def train_test_df(rating_path, fulldata_path): #fulldata_path là file data cuối cùng của mình
    ''''''
    full_dataset = read_data(fulldata_path)
    rating = read_data(rating_path)
    
    '''preprocessing'''
    full_dataset = full_dataset[['movieId', 'title', 'genres', 'description', 'popularity']]
    
    '''Merge full_dataset với rating, chia train: 80, test: 20'''
    data = rating.merge(movie, on = 'movieId')
    train_df, test_df = train_test_split(data, test_size = 0.2, random_state = 42, stratify=data['userId'])
    return train_df, test_df '''train_df, test_df = train_test(rating_path, full_data_path)'''
    
def convert_traintest_dataframe_forsurprise(train_df, test_df):
    '''Dùng để convert trainset, testset để dùng cho thư viện surprise'''
    reader = Reader(rating_scale=(0, 5))
    train_convert = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    test_convert = Dataset.load_from_df(test_df[['userId', 'movieId', 'rating']], reader)
    train_convert = train_convert.construct_trainset(train_convert.raw_ratings)
    test_convert = test_convert.construct_testset(test_convert.raw_ratings)
    return train_convert, test_convert '''train_convert, test_convert = convert_traintest_dataframe_forsurprise(train_df, test_df)'''

def knnbaseline(train_convert, test_convert): #traiset, testset ở đây là đã được convert qua hàm convert_traintest_dataframe_forsurprise(train_df, test_df)
    sim_options = {'name': 'cosine', 'user_based': False} # compute  similarities between items
    knnbaseline_algo = KNNBaseline(sim_options=sim_options)

    knnbaseline_algo.fit(train_convert)
    knnbaseline_predictions = knnbaseline_algo.test(test_convert)

    file_name = 'KnnBaseline_model'
    dump.dump(file_name, algo=knnbaseline_predictions)

    accuracy.rmse(knnbaseline_predictions)
    accuracy.mae(knnbaseline_predictions)
    print("Done!")
    return knnbaseline_algo #phải đặt tên 1 biến là knnbaseline_algo = knnbaseline(trainset, testset):

def svd(train_convert, test_convert):
    svd_algo = SVD()

    svd_algo.fit(train_convert)
    svd_predictions = svd_algo.test(test_convert)

    file_name = 'svd_model'
    dump.dump(file_name, algo=svd_algo)

    accuracy.rmse(svd_predictions)
    accuracy.mae(svd_predictions)
    print("Done!")
    return svd_algo #phải đặt tên 1 biến là svd_algo = svd(trainset, testset):

def svdpp(train_convert, test_convert):
    svdpp_algo = SVDpp()
    svdpp_algo.fit(train_convert)
    svdpp_predictions = svdpp_algo.test(test_convert)
    
    file_name = 'svdpp_model'
    dump.dump(file_name, algo=svdpp_algo)
    
    
    accuracy.rmse(svdpp_predictions)
    accuracy.mae(svdpp_predictions)
    print("Done!")
    return svdpp_algo #phải đặt tên 1 biến là svdpp_algo = svdpp(trainset, testset):
    

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
    df = read_data(full_dataset)
    mask = df.genres.apply(lambda x: genre in x) # trả về dạng bool, check xem genre có trong cái list genres đó k
    filtered_movie = df[mask]# trả về dataframe các film match với genre
    filtered_movie = filtered_movie.sort_values(by='popularity', ascending=False) #xếp theo độ phổ biến nhất
    return filtered_movie['movieId'].head(10).values.tolist() #trả về list top 10 movie similar

def user_top_genre(userId, user_info): #user_info la dataframe dùng hàm make_useinfo_df(fulldata_path, trainset_path)
    user_vec = user_info['user_vector'][user_info['userId'] == userId].values[0].copy()
    print("User Vector: ", user_vec)
    top_genre_indices = np.flip(np.argsort(user_vec))
    genre_list = []
    for i in top_genre_indices[:3]:
        genre_list.append(idx_to_genre[i])
    return genre_list

def make_useinfo_df(fulldata, train_df): #trainset_path ở đây là dùng train data được split bởi sklearn
    fulldata['genres'] = fulldata.genres.apply(lambda x: literal_eval(str(x)))
    train_df['genres'] = train_df.genres.apply(lambda x: literal_eval(str(x)))

    genre_distribution = train_df['genres'].explode().value_counts()
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
    return user_df = pd.concat([user_df, row_df], ignore_index=True)   
    
def hybrid(userId, full_dataset, train_df, test_df, train_convert, test_convert knnbaseline_algo, svdpp_algo): #full_dataset là file data cuối cùng, train-test_df là qua sklearn
    user_movies = test_df[test_df['userId'] == userId]
    user_movies['est'] = user_movies['movieId'].apply(lambda x: 0.6*knnbaseline_algo.predict(userId,x).est + 0.4*svdpp_algo.predict(userId, x).est)    
    user_movies = user_movies.sort_values(by ='est', ascending=False).head(4)
    user_movies['Model'] = 'SVD + CF'
    
    recommend_list = user_movies[['movieId', 'est', 'Model']]

    movie_list = recommend_list['movieId'].values.tolist()
    sim_movies_list = []
    for movie_id in movie_list:
        # Call content based 
        movie_title = full_dataset['title'][full_dataset['movieId'] == movie_id].values[0]
        sim_movies = get_recommendations_new(movie_title)
        sim_movies_list.extend(sim_movies)
    # Compute ratings for the popular movies
    for movie_id in sim_movies_list:
        pred_rating = 0.6*knnbaseline(train_convert, test_convert).predict(userId, movie_id).est + 0.4*svdpp_algo.predict(userId, movie_id).est
        row_df = pd.DataFrame([[movie_id, pred_rating, 'Movie similarity']], columns=['movieId', 'est','Model'])
        recommend_list = pd.concat([recommend_list, row_df], ignore_index=True)
    
