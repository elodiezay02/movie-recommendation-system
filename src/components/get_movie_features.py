import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from ast import literal_eval
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# Chuyển thêm keywords
def filter_keywords(x, s):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words

cols = ['id', 'imdb_id', 'title', 'genres', 'description', 'cast', \
        'cast_size', 'crew_size', 'director', 'keywords', 'popularity', 'vote_average',\
         'vote_count', 'year', 'wr', 'spoken_language']

def movie_feature(metadata_path, links_small_path, credits_path,keywords_path, \
                  percentile=0.95, more_weight_on = None, \
                    stemmer = SnowballStemmer('english'), cols=cols):
    # read dataset
    meta = pd.read_csv(metadata_path)
    links_small = pd.read_csv(links_small_path)
    credits = pd.read_csv(credits_path)
    keywords = pd.read_csv(keywords_path)

    # change type + drop
    links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
    meta = meta.drop([19730, 29503, 35587])
    meta['id'] = meta['id'].astype('int')
    meta['year'] = pd.to_datetime(meta['release_date'], errors='coerce').apply(\
        lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')

    vote_counts = meta[meta['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = meta[meta['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    # quantified movies: have more vote counts than ...(percentile)% others
    # qualified = meta[(meta['vote_count'] >= m) & (meta['vote_count'].notnull()) \
    #                  & (meta['vote_average'].notnull())]
    meta = meta[(meta['vote_count'].notnull()) \
                     & (meta['vote_average'].notnull())]
    meta['vote_count'] = meta['vote_count'].astype('int')
    meta['vote_average'] = meta['vote_average'].astype('int')
    meta['wr'] = meta.apply(lambda x: weighted_rating(x, m, C), axis=1)
    # meta = qualified.sort_values('wr', ascending=False)

    # merge meta + link small => create a smaller dataset for recommend
    smd = meta[meta['id'].isin(links_small)]

    # create description feature
    smd['tagline'] = smd['tagline'].fillna('')
    smd['description'] = smd['overview'] + smd['tagline']
    smd['description'] = smd['description'].fillna('')

    # transform text to vector
    # tf = word_to_vec(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    # tfidf_matrix = tf.fit_transform(smd['description'])
    
    # merge credit + keywords
    smd = smd.merge(credits, on='id')
    smd = smd.merge(keywords, on='id')

    # feature engineering
    smd['cast'] = smd['cast'].apply(literal_eval)
    smd['crew'] = smd['crew'].apply(literal_eval)
    smd['keywords'] = smd['keywords'].apply(literal_eval)
    smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
    smd['crew_size'] = smd['crew'].apply(lambda x: len(x))
    smd['director'] = smd['crew'].apply(get_director)
    smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in \
                                               x] if isinstance(x, list) else [])
    smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
    smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] \
                                            for i in x] if isinstance(x, list) else [])
    # Strip Spaces and Convert to Lowercase 
    smd['cast'] = smd['cast'].apply(lambda x: [str.lower(\
        i.replace(" ", "")) for i in x])
    smd['director'] = smd['director'].astype('str').apply(\
        lambda x: str.lower(x.replace(" ", "")))
    if more_weight_on:
        smd[more_weight_on] = smd[more_weight_on].apply(lambda x: [x,x,x])
    
    s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
    s = s.value_counts()
    s = s[s > 1]
    smd['keywords'] = smd['keywords'].apply(lambda x: filter_keywords(x, s))
    smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    smd['keywords'] = smd['keywords'].apply(lambda x: \
                                            [str.lower(i.replace(" ", "")) for i in x])

    smd[['spoken_languages', 'genres']] = smd[['spoken_languages', 'genres']].apply(\
        lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    
    
    return smd[cols]
    