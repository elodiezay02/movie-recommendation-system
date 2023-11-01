#!/usr/bin/env python
# coding: utf-8

# In[67]:


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


# In[101]:


movies = pd.read_csv('movies_metadata.csv')
print(movies.shape)
movies.head()


# In[69]:


movies['genres'] = movies['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[70]:


movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# In[71]:


links_small = pd.read_csv('links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')


# In[72]:


movies = movies.drop([19730, 29503, 35587])


# In[73]:


credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')


# In[74]:


keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
movies['id'] = movies['id'].astype('int')


# In[75]:


movies.shape


# In[76]:


movies = movies.merge(credits, on='id')
movies = movies.merge(keywords, on='id')


# In[77]:


movies.head()


# In[78]:


smd = movies[movies['id'].isin(links_small)]
smd.shape


# In[79]:


smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))


# In[80]:


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[81]:


smd['director'] = smd['crew'].apply(get_director)


# In[82]:


smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)


# In[83]:


smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# What we plan on doing is creating a metadata dump for every movie which consists of genres, director, main actors and keywords.
# 
# We then use a Count Vectorizer to create our count matrix as we did in the Description Recommender. 
# 
# The remaining steps are similar to what we did earlier: we calculate the cosine similarities and return movies that are most similar.

# In[84]:


smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


# In[85]:


smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x,x, x])


# In[86]:


keys = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
keys.name = 'keyword'


# In[87]:


keys = keys.value_counts()
keys[:10]


# Next, we remove all keywords that appear only once and then we will have to deal with the similars one by merge them into one like 'dog' and 'dogs'

# In[88]:


keys = keys[keys > 1]
stemmer = SnowballStemmer('english')
stemmer.stem('dogs')


# In[89]:


def filter_keywords(x):
    words = []
    for i in x:
        if i in keys:
            words.append(i)
    return words


# In[90]:


smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


# In[91]:


smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))


# In[92]:


count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])


# In[93]:


cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[94]:


cosine_sim[0]


# In[95]:


smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])


# In[96]:


get_recommendations('Forrest Gump').head(10)


# With this result, we can see that most of the recommendation are being done base on the crews and other factors of the film instead of just tagline, thats why the 'Frozen' has disappeared and being switched by other relative movies

# But we can notice that the recommender is not good yet, there are some problems with it and the most major one is that it recommend any movies that most suited the conditions without filtering, thats why some movies like "Death Become Her' and "What Lies Beneath" got recommend even with their low rating score
# 
# Thats why, we will continue to improve it by running through a filtering process of rating score

# In[99]:


def top_10_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    mean_vote = vote_averages.mean()
    cut_point = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= cut_point) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified


# In[100]:


top_10_recommendations('Forrest Gump')


# There we go, a better recommender using cosine similarity on crews, taglines, genres, etc,... with the improvement of vote filtering

# But there are still some problems here, most of the recommendations here still base on the vote of other users, it not personalize enough but we cant do much about it, although this can be fix by using hybrid method filtering but that is for other part of this project. This is the conclusion of content base filtering using cosine similarity
