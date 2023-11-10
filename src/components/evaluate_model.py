import pandas as pd
from get_movie_features import movie_feature
from sklearn.model_selection import train_test_split

def split_data(rating_df):
    train_df, test_df = train_test_split(rating_df, \
                                         test_size=0.2, stratify=rating_df['userId'], \
                                        random_state=42)
    return train_df, test_df

def precision_recall_at_k(test_df, threshold):
    """Return recall and precision

    Args:
        test_df: prediction dataframe, with 4 columns: userId, movieId, true_rating, pred_rating
        threshold: if rating > threshold, movie is believed to be relevant

    Returns:
    Recall@K: Proportion of relevant items that are recommended, dict-like
    Precision@K: Proportion of recommended items that are relevant, dict-like
        Movie is relevant if true_rating > threshold
        Movie is recommend when pred_rating > threshold
    """
    recalls = dict()
    precisions = dict()

    for userId, group in test_df.groupby('userId'):

        filter_rel = group[group['rating'] > threshold]
        filter_rec = group[group['pred_rating'] > threshold]
        filter_rel_rec = group[(group['pred_rating'] > threshold) & \
                               (group['rating'] > threshold)]

        # Number of relevant items
        n_rel = len(filter_rel)

        # Number of recommended items in top k
        n_rec = len(filter_rec)

        # Number of relevant and recommended items in top k
        n_rel_rec = len(filter_rel_rec)

        recalls[userId] = n_rel_rec/n_rel if n_rel != 0 else 1
        precisions[userId] = n_rel_rec/n_rec if n_rec != 0 else 1

    precision = sum(prec for prec in precisions.values())/len(precisions)
    recall = sum(rec for rec in recalls.values())/len(recalls)
    fmeasure = (2*precision*recall)/(precision + recall)

    return recall, precision, fmeasure
    