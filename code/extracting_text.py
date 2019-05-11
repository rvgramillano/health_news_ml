import pandas as pd
import numpy as np
import os

WORK_DIR = '/Users/rvg/Documents/other_projects/health_news_twitter/'
NEWS_SOURCES = WORK_DIR + 'data/Health-Tweets/'

df = pd.DataFrame(columns=['tweet_ID', 'tweet_time', 'tweet_text', 'source'])

news_source_dict = dict()
i = 0
for news_source in os.listdir(NEWS_SOURCES):
    news_source_dict[news_source[:-4]] = i
    df_i = pd.read_csv(WORK_DIR + 'data/Health-Tweets/' + news_source, sep='|', header=None, names=['tweet_ID', 'tweet_time', 'tweet_text'])
    df_i['source'] = np.ones(len(df_i), dtype=int) * i
    df = pd.concat([df, df_i])
    i += 1

# let's check that all tweets occur in the same year with +0000 UTC offset
zones = df.tweet_time.str.extract(r'(\+.{4})')
print(np.unique(zones.values))
# all have IDENTICALLY +0000 offset

years = df.tweet_time.str[-4:]
print(np.unique(years))

# first let's remove the tweet_id as it contains no relevant info
df.drop(['tweet_ID'], axis=1, inplace=True)

# so years range from 2011-2015 and are ALL preceded by +0000.
# let's clean up the tweet_time column
# first remove offset because strptime has a hard time formatting and they're ALL +0000
df.tweet_time.replace(regex=True, inplace=True, to_replace=r'(\+.{4})', value=r'')

# then convert to datetime
df.tweet_time = pd.to_datetime(df.tweet_time, format="%a %b %d %H:%M:%S  %Y")

# now we clean up the tweet_text column

# remove hyperlinks
df.tweet_text = df.tweet_text.str.replace(r'http\S+', r'')

# remove unnecessary punctuation, tags
df.tweet_text = df.tweet_text.str.replace(r'[^\w\s]', r'')

# convert all text to lowercase
df.tweet_text = df.tweet_text.str.lower()

# remove all stop words
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
df['tweet_text'] = df['tweet_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))




