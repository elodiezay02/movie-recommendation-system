# content-based: input 1 bộ phim => recommend những bộ phim cùng thể loại
# check theo timestamp, với mỗi người dùng, 
# chia tập train và test: groupby userId, train = 70% first movies, test = 30% last movies

import pandas as pd


rating_path = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\ratings_small.csv'
rating_df = pd.read_csv(rating_path)
rating_df_sorted = rating_df.sort_values(by=['userId', 'timestamp'], ignore_index=True)

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

train_df, test_df = train_test_split(rating_df_sorted, train_size=0.7)
print(train_df.shape, test_df.shape)