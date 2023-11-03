# content-based algorithm
# try to put the code in FUNCTIONS so that we will later call the functions in __init__.py

# DL: thứ 7 4/11
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd

from get_movie_features import movie_feature

# credits_ = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\credits.csv'
# keywords = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\keywords.csv'
# links_small = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\links_small.csv'
# movies_metadata = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\movies_metadata.csv'

# smd = movie_feature(movies_metadata, links_small, credits_, keywords, more_weight_on='director')
# vec_to_word = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0.0, stop_words='english')

def top_10_content_based(smd, title, vec_to_word):
    smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
    smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))
    count_matrix = vec_to_word.fit_transform(smd['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    smd = smd.reset_index()
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:30]
    movie_indices = [i[0] for i in sim_scores]
    qualified = smd.iloc[movie_indices]
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified['title']

# print(top_10_content_based(smd, 'Forrest Gump', vec_to_word))