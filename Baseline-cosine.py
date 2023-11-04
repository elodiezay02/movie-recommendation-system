#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from get_movie_features import movie_feature

import warnings; warnings.simplefilter('ignore')

# credits_ = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\credits.csv'
# keywords = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\keywords.csv'
# links_small = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\links_small.csv'
# movies_metadata = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\movies_metadata.csv'

# smd = movie_feature(movies_metadata, links_small, credits_, keywords, more_weight_on='director')
# vec_to_word = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0.0, stop_words='english')

# In[8]:


def baseline_model(smd, title, vec_to_word):
    smd['tagline'] = smd['tagline'].fillna('')
    smd['description'] = smd['overview'] + smd['tagline']
    smd['description'] = smd['description'].fillna('')
    tfidf_matrix = vec_to_word.fit_transform(smd['description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    smd = smd.reset_index()
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    #return top 10 recommendation movie
    return titles.iloc[movie_indices].head(10)


# In[9]:


# print(top_10_content_based(smd, 'Forrest Gump', vec_to_word))


# In[ ]:




