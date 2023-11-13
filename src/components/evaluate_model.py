import pandas as pd
from get_movie_features import movie_feature
from sklearn.model_selection import train_test_split

def split_data(rating_df):
    train_df, test_df = train_test_split(rating_df, \
                                         test_size=0.2, stratify=rating_df['userId'], \
                                        random_state=42)
    return train_df, test_df

def check_movieId(pred_df, val_df):
    result = pred_df['movieId'].isin(val_df[val_df['userId'] == \
                                   pred_df['userId'].iloc[0]]['movieId'])
    return result.reset_index(drop=True)

def evaluate(pred_df, val_df):
    """ Proportion of movies recommended that were actually watched by users

    Args:
    pred_df: dataframe, 2 columns: userId, movieId
    val_df: dataframe, 2 columns: userId, movieId

    Returns:
    
    """
    result = pred_df.groupby('userId').apply(lambda x: check_movieId(x, val_df))
    result = result.groupby('userId').sum() / result.groupby('userId').count()

    return result.mean().mean()

def precision_recall_at_k(prediction_df, threshold):
    """Return recall and precision, F-1 scrore for collaborative + hybrid

    Args:
        prediction_df: prediction dataframe, with 4 columns: userId, movieId, true_rating, pred_rating
        threshold: if rating > threshold, movie is believed to be relevant

    Returns:
    Recall: Proportion of relevant items that are recommended, dict-like
    Precision: Proportion of recommended items that are relevant, dict-like
        Movie is relevant if true_rating > threshold
        Movie is recommend when pred_rating > threshold
    """
    recalls = dict()
    precisions = dict()

    for userId, group in prediction_df.groupby('userId'):

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
    