

"""### Configure and run Twint (twitter scrapper)"""
# Commented out IPython magic to ensure Python compatibility.
#core setup.





##importing things here!
import streamlit as st
### pandas helps us visualize and sort data, we'll need it if you want to see results.
import pandas as pd

### matplot helps us visualize data too.
import matplotlib.pyplot as plt

### ntlk is a word processing library, we can use it to parse our tweets for our goal.
import nltk
from wordcloud import WordCloud,STOPWORDS
nltk.download('punkt')   
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

### tweepy is the official twitter python library/api; this is how we'll be able to source our tweets.
### textblob is a text processing library.
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob

### display is so we can display a dataframe from pandas or something else.
from IPython.display import display

st.set_page_config(layout="wide")
st.title("Tsunamifi's Twitter Sentiment Analysis Bot")
st.write("This WebAPP will allow you to plug in a topic from twitter and determine if their tweets are Positive, Negative or Neutral.")

# defining twitter auth setup here!
# from config import bearer_token


bearer_token = "AAAAAAAAAAAAAAAAAAAAAHbwcAEAAAAAA3Heaep4TEp0EOTiUpPnTPLy9A4%3DspSy2AYLMulUovSsOjv9AT99T2GBnyJDmaPV7YzOXrvbHtJPWH"
client = tweepy.Client(bearer_token=bearer_token)

# Get user information..
def get_username(user_id):
    result = client.get_user(id=user_id, user_fields=['username'])
    return result.data.username


def get_tweets(search_query, max_results):
    # The tweet.fields query parameters [attachments,author_id,context_annotations,conversation_id,created_at,entities,
    # geo,id,in_reply_to_user_id,lang,non_public_metrics,organic_metrics,possibly_sensitive,promoted_metrics,public_metrics,
    # referenced_tweets,reply_settings,source,text,withheld]
    result = client.search_recent_tweets(query=search_query,
                                         tweet_fields=['conversation_id', 'context_annotations', 'created_at',
                                                       'author_id',
                                                       'public_metrics', 'lang', 'source'],
                                         max_results=max_results)
    return result



search_query = 'from:CyBearNanook -is:retweet (#JohnsonOut109)'
result = get_tweets(search_query=search_query, max_results=11)


for data in result.data:
    # print(data)
    tweet_id = data.id
    conversation_id = data.conversation_id
    author_id = data.author_id
    username = get_username(author_id)
    created_at = data.created_at
    lang = data.lang
    tweet = data.text
    source = data.source
    public_metrics = data.public_metrics
    # print("\n" + "-" * 80)
   st.write(f"Tweet ID: {tweet_id}\n"
          f"Conversation ID: {conversation_id}\n"
          f"Author ID: {author_id}\n"
          f"Username: {username}\n"
          f"Created at: {created_at}\n"
          f"Language: {lang}\n"
          f"Tweet: {tweet}\n"
          f"Public metrics: {public_metrics}\n",
          f"Source: {source}")
    # print("-" * 80)     

st.title("Choose Topic on twitter to analyze")
#st.write("You're welcome to use both a user and topic but both at the same time are not required, you can use one or the other too if you'd like.")


# Take off...


with st.form(key='vars'):
        texti = st.text_input(label='Choose topic')
        numberi = st.number_input(label= 'How many tweets should we source?')
        submit = st.form_submit_button(label='Submit')
        
if submit:
    get_tweets()
else:
    pass
