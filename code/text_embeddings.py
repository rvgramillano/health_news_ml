from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

# TEXT EMBEDDINGS
WORK_DIR = '/Users/rvg/Documents/other_projects/health_news_twitter/'

df = pd.read_pickle(WORK_DIR + 'data/cleaned_df.pkl')

min_df_arr = range(1,10)

tfidf_vectorizer = TfidfVectorizer(min_df=4)
tfidf = tfidf_vectorizer.fit_transform(df.tweet_text)
sources = df.source

true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(tfidf)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names()

for i in range(true_k):
    print('Cluster %d:' % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
