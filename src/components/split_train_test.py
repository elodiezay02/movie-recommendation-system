from sklearn.model_selection import train_test_split
import pandas as pd

# thay rating_path
rating_path = r'E:\School\DE_AN\Movie-Recommendation-System\src\data\ratings_small.csv'
rating_df = pd.read_csv(rating_path)

train_df, test_df = train_test_split(rating_df, random_state=42, \
                                     stratify=rating_df['userId'], test_size=0.25)

train_df.to_csv('train_set.csv', index=False)
test_df.to_csv('test_set.csv', index=False)