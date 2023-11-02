# chia tập train và test: groupby userId, sort by timestamp
# in each user, train = 70% first movies, test = 30% last movies

# content-based evaluate: Input bộ phim A, trong số bộ phim thuật toán 
# recommend (output) có bao nhiêu bộ phim được xem bởi 
# những người đã xem phim A

import pandas as pd
from get_movie_features import movie_features

credits_ = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\credits.csv'
keywords = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\keywords.csv'
links_small = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\links_small.csv'
movies_metadata = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\movies_metadata.csv'
ratings_small = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\ratings_small.csv'

rating_df = pd.read_csv(ratings_small)
rating_df_sorted = rating_df.sort_values(by=['userId', 'timestamp'], ignore_index=True)
movie_data = movie_features(movies_metadata, links_small, credits_, keywords)

# Train test split
def get_train_df(df, train_size = 0.7):
    train = df.head(round(train_size * len(df)))
    return train
    
def get_test_df(df, train_size = 0.7):
    test = df.tail(len(df) - round((train_size) * len(df)))
    return test

def train_test_split(sorted_df, train_size = 0.7):

    train_df = pd.DataFrame(sorted_df.groupby('userId').apply(lambda x: get_train_df(x, train_size)))
    test_df = pd.DataFrame(sorted_df.groupby('userId').apply(lambda x: get_test_df(x, train_size)))
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return train_df, test_df

def get_movie_id(movie_data_df, title):
    movieId = movie_data_df[movie_data_df.title == title]['movieId']
    return movieId

def metric1(movie_id, movie_rec_id, rating_df):

    # users who watch movie_id
    user_movie_id = rating_df[rating_df.movieId.isin(movie_id)]['userId']
    n_user_movie_id = user_movie_id.nunique()
    
    # user watch movie_id and movie_rec_id
    user_movie_rec_id = rating_df[(rating_df['userId'].isin(user_movie_id)) & \
                        rating_df['movieId'].isin(movie_rec_id)]['userId']
    n_user_movie_rec_id = len(user_movie_rec_id)

    return n_user_movie_rec_id / n_user_movie_id

train_df, test_df = train_test_split(rating_df_sorted, train_size=0.7)
print(train_df.shape, test_df.shape)

movie_title = 'Một bộ phim nào đó'
movie_data = movie_feature(movies_metadata, links_small, credits_, keywords)
movie_to_recommend = top_10_recommend(movie_data, movie_title)
evaluation_score = evaluate(movie_title, movie_to_recommend)


# Evaluate 1(metrics cho hướng content-based): Trong số những người xem bộ phim A, có bao nhiêu người xem 
# những bộ phim giống với phim A

# Evaluate 2: Trong số top N bộ phim recommend cho user X, có bao nhiêu bộ phim mà
# user X thực sự xem sau đó. (chia tập để đánh giá)

Content-based: Recommend top N bộ phim giống với bộ phim A
    Metrics: Trong số những người xem bộ phim A, có bao nhiêu người thực sự xem 
    top N những bộ phim giống với phim A

Collaborative: Recommend top N bộ phim cho user X thông qua những bộ phim mà
users giống X xem
    Với phương pháp này chia 2 tập: train và validate. Tập train đưa vào thuật toán
    để trích xuất phim sẽ recommend, tập validate dùng để đánh giá lại mô hình 
    recommend.
    Metrics: - Trong số top N bộ phim recommend cho user X, có bao nhiêu bộ phim mà
user X thực sự xem sau đó
             - Trong số top N bộ phim recommend cho user X, cosine similarity giữa
các bộ phim được recommend và các bộ phim user X thực sự đã xem là bao nhiêu?