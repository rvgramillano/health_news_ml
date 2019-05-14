from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# TEXT EMBEDDINGS
WORK_DIR = '/Users/rvg/Documents/other_projects/health_news_twitter/'

df = pd.read_pickle(WORK_DIR + 'data/cleaned_df.pkl')

tfidfvectorizer = TfidfVectorizer(min_df=2)
tfidf=tfidfvectorizer.fit_transform(df.tweet_text)
sources = df.source