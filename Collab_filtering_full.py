#!/usr/bin/env python
# coding: utf-8

# In[48]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate


import warnings; warnings.simplefilter('ignore')


# In[ ]:





# In[51]:


def top_10(metadata_path, links_small_path, credits_path,keywords_path, title):
    meta = pd.read_csv(metadata_path)
    credits = pd.read_csv(credits_path)
    keywords = pd.read_csv(keywords_path)
    links_small = pd.read_csv(links_small_path)
    meta['genres'] = meta['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    meta['year'] = pd.to_datetime(meta['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
    meta = meta.drop([19730, 29503, 35587])
    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')
    meta['id'] = meta['id'].astype('int')
    meta = meta.merge(credits, on='id')
    meta = meta.merge(keywords, on='id')
    smd = meta[meta['id'].isin(links_small)]
    smd.shape
    smd['cast'] = smd['cast'].apply(literal_eval)
    smd['crew'] = smd['crew'].apply(literal_eval)
    smd['keywords'] = smd['keywords'].apply(literal_eval)
    smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
    smd['crew_size'] = smd['crew'].apply(lambda x: len(x))
    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan
    smd['director'] = smd['crew'].apply(get_director)
    smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
    smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    smd['director'] = smd['director'].apply(lambda x: [x,x, x])
    
    keys = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
    keys.name = 'keyword'
    keys = keys.value_counts()
    keys[:10]
    keys = keys[keys > 1]
    stemmer = SnowballStemmer('english')
    stemmer.stem('dogs')
    def filter_keywords(x):
        words = []
        for i in x:
            if i in keys:
                words.append(i)
        return words
    smd['keywords'] = smd['keywords'].apply(filter_keywords)
    smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
    smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))
    count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    count_matrix = count.fit_transform(smd['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    smd = smd.reset_index()
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    meta = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = meta[meta['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = meta[meta['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)
    qualified = meta[(meta['vote_count'] >= m) & (meta['vote_count'].notnull()) & (meta['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified


# In[52]:


top_10('movies_metadata.csv', 'links_small.csv','credits.csv','keywords.csv', 'Forrest Gump')


# In[ ]:





# But there are still some problems here, most of the recommendations here still base on the vote of other users, it not personalize enough but we cant do much about it, although this can be fix by using hybrid method filtering but that is for other part of this project. This is the conclusion of content base filtering using cosine similarity
