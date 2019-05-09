import pandas as pd
import numpy as np
import os


WORK_DIR = '/Users/rvg/Documents/other_projects/health_news_twitter/'
NEWS_SOURCES = WORK_DIR + 'data/Health-Tweets/'

total_df = pd.DataFrame(columns=['tweet_ID', 'tweet_time', 'tweet_text'])

for news_source in os.listdir(NEWS_SOURCES):
    df = pd.read_csv(WORK_DIR + 'data/Health-Tweets/' + news_source, sep='|', header=None, names=['tweet_ID', 'tweet_time', 'tweet_text'])
    total_df = pd.concat([total_df, df])

# let's check that all tweets occur in the same year with +0000
zones = total_df.tweet_time.str.extract(r'(\+.{4})')
print(np.unique(zones.values))

years = total_df.tweet_time.str[-4:]
print(np.unique(years))

