#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


movies = pd.read_csv('movies_metadata.csv')
print(movies.shape)
print(movies.head)


# In[4]:


movies['genres'] = movies['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# The first step is to determine an appropriate value for m, the minimum votes required to be listed in the chart. We will use 95th percentile as our cutoff. In other words, for a movie to feature in the charts, it must have more votes than at least 95% of the movies in the list.

# In[5]:


vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
mean_vote = vote_averages.mean()
mean_vote


# In[6]:


cut_point = vote_counts.quantile(0.95)
cut_point


# In[7]:


movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# In[8]:


qualified = movies[(movies['vote_count'] >= cut_point) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified.shape


# Therefore, to qualify to be considered for the chart, a movie has to have at least 434 votes on TMDB. We also see that the average rating for a movie on TMDB is 5.244 on a scale of 10. 2274 Movies qualify to be on our chart.

# In[9]:


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+cut_point) * R) + (cut_point/(cut_point+v) * mean_vote)


# In[10]:


qualified['wr'] = qualified.apply(weighted_rating, axis=1)


# In[11]:


qualified = qualified.sort_values('wr', ascending=False).head(250)


# In[12]:


qualified.head(15)


# Next, we will build charts for specific genre of the movies

# In[13]:


s = movies.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_movies = movies.drop('genres', axis=1).join(s)


# In[14]:


def build_chart(genre, percentile=0.85):
    df = gen_movies[gen_movies['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    mean_vote = vote_averages.mean()
    cut_point = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= cut_point) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+cut_point) * x['vote_average']) + (cut_point/(cut_point+x['vote_count']) * mean_vote), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified


# In[15]:


build_chart('Comedy').head(15)


# RECOMMENDER_CONTENT_BASED

# Because of how big the data is, we decide to use sub-categories to do the recommender 

# In[16]:


links_small = pd.read_csv('links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')


# In[17]:


movies = movies.drop([19730, 29503, 35587])


# In[18]:


movies['id'] = movies['id'].astype('int')


# In[19]:


smd = movies[movies['id'].isin(links_small)]
smd.shape


# We have 9099 movies avaiable in our small movies metadata dataset

# # Movie Description Based Recommender

# In[20]:


smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')


# In[21]:


tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])


# In[22]:


tfidf_matrix.shape


# ## Cosine simiarity

# In[23]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[24]:


cosine_sim[0]


# returns the 30 most similar movies based on the cosine similarity score.

# In[25]:


smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])


# In[34]:


def top_10_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    #return top 10 recommendation movie
    return titles.iloc[movie_indices].head(10)


# In[35]:


top_10_recommendations('Modern Times')


# In[36]:


top_10_recommendations('Forrest Gump')


# As you can see, here the system only recommend the user base on their tagline, thats why in the recommendation of 'Forrest Gump', we can see 'Frozen' as one of the top. within the next part, we will try to improve this model
