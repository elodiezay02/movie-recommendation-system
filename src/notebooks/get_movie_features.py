import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from ast import literal_eval
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import warnings; warnings.simplefilter('ignore')

def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def filter_keywords(x, s):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words

cols = ['id', 'movieId', 'title', 'genres', 'description', 'keywords', \
        'cast', 'director', 'spoken_languages', 'production_companies',\
        'production_countries', 'popularity', 'year', 'vote_average',\
         'vote_count', 'wr']

def read_dataset(metadata_path, links_small_path, credits_path,keywords_path):
    meta = pd.read_csv(metadata_path)
    links_small = pd.read_csv(links_small_path)
    credits = pd.read_csv(credits_path)
    keywords = pd.read_csv(keywords_path)
    return meta, links_small, credits, keywords

def cal_weighted_rating(meta, percentile=0.95):
    vote_counts = meta[meta['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = meta[meta['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)

    meta = meta[(meta['vote_count'].notnull()) \
                     & (meta['vote_average'].notnull())]
    meta['vote_count'] = meta['vote_count'].astype('int')
    meta['vote_average'] = meta['vote_average'].astype('int')
    meta['wr'] = meta.apply(lambda x: weighted_rating(x, m, C), axis=1)

    return meta
    
def movie_feature(metadata_path, links_small_path, credits_path,keywords_path, \
                  percentile=0.95, more_weight_on = None, \
                    stemmer = SnowballStemmer('english'), cols=cols):
    # read dataset
    meta, links_small, credits, keywords = read_dataset(metadata_path, \
        links_small_path, credits_path,keywords_path)

    # change type + drop
    links_small = links_small[links_small['tmdbId'].notnull()]
    links_small['tmdbId'] = links_small['tmdbId'].astype('int')

    meta = meta.drop([19730, 29503, 35587])
    meta['popularity'] = meta[meta['popularity'].notnull()]['popularity'].astype('float')
    meta['id'] = meta['id'].astype('int')
    meta['year'] = pd.to_datetime(meta['release_date'], errors='coerce').apply(\
        lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    meta = meta[meta.production_companies.notnull()]
    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')

    # calcualte weighted rating for movies
    meta = cal_weighted_rating(meta)

    # merge meta + link small => create a smaller dataset for recommend
    smd = meta[meta['id'].isin(links_small['tmdbId'])]

    # create description feature
    smd['tagline'] = smd['tagline'].fillna('')
    smd['description'] = smd['overview'] + smd['tagline']
    smd['description'] = smd['description'].fillna('')
    
    # merge credit + keywords + links_small
    smd = smd.merge(credits, on='id')
    smd = smd.merge(keywords, on='id')
    smd = smd.merge(links_small, left_on='id', right_on='tmdbId')

    # feature engineering
    literal_features = ['cast', 'spoken_languages', 'genres', 'keywords',\
                        'production_companies', 'production_countries']
    for fearture in literal_features:
        smd[fearture] = smd[fearture].apply(literal_eval)
        smd[fearture] = smd[fearture].apply(lambda x: [i['name'] \
                                            for i in x] if isinstance(x, list) else [])
    smd['crew'] = smd['crew'].apply(literal_eval)
    smd['director'] = smd['crew'].apply(get_director)

    # top 3 actors
    smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
   
    # Strip Spaces and Convert to Lowercase 
    smd['cast'] = smd['cast'].apply(lambda x: [str.lower(\
        i.replace(" ", "")) for i in x])
    smd['director'] = smd['director'].astype('str').apply(\
        lambda x: str.lower(x.replace(" ", "")))
    if more_weight_on:
        smd[more_weight_on] = smd[more_weight_on].apply(lambda x: [x,x,x])
    
    # choose keywords appear more than once + stemming
    s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
    s = s.value_counts()
    s = s[s > 1]
    smd['keywords'] = smd['keywords'].apply(lambda x: filter_keywords(x, s))
    smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    smd['keywords'] = smd['keywords'].apply(lambda x: \
                                            [str.lower(i.replace(" ", "")) for i in x])

    return smd[cols]
    
# credits_ = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\credits.csv'
# keywords = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\keywords.csv'
# links_small = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\links_small.csv'
# movies_metadata = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\movies_metadata.csv'
# smd = movie_feature(movies_metadata, links_small, credits_, keywords, more_weight_on='director')
# print(smd.columns)
