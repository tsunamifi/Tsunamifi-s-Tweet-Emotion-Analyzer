# -*- coding: utf-8 -*-
"""tsunamifis_twittr_sentiment_analysis_botv1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1c3F8klCW9uaJllbkFLyeQipfOu2rDRPr

# Tsunamifi's Twitter Sentiment Analysis Bot
This notebook will allow you to plug in a user OR topic from twitter and determine if their tweets are Positive, Negative or Neutral.

Why? for fun, probably.

# Setup
"""

# Commented out IPython magic to ensure Python compatibility.
#core setup.

## installing things here (for non colab use, uncomment if you need to)

#pip install tweepy
#pip install nltk
#pip install pandas
#pip install textblob

##importing things here!
import streamlit as st
### pandas helps us visualize and sort data, we'll need it if you want to see results.
import pandas as pd

### ntlk is a word processing library, we can use it to parse our tweets for our goal.
import nltk
nltk.download('all-corpora')

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
def auth():

    ## keys and token from twitter'
    consumer_key = "KLhloeEIOr2de1Cnz7ddcxcmT"
    consumer_secret = "WPQeRE5skCsCfBK8inJSPFTOEFMCUrPGMUyKsi1kjo8xKDaoxQ"
    access_token = "1450493640132923397-NQ1fupgKuJZKZbPsi3gYw9EnngxapO"
    access_token_secret = "eH3IvjkUatJuU7EdYtoSMZckBbIWtPpvYNFUW078VvMK4"

    ## attempting auth...
    try:
      auth = OAuthHandler(consumer_key, consumer_secret)
      auth.set_access_token(access_token, access_token_secret)
      api = tweepy.API(auth)
      return api
    except:
      print("Error: Authentification Failed, try again?")
      exit(1)

# this will clean unnecessary and maybe complicated things out of a tweet
# like links or #'s 
def cleanup(text):

     ## replaces all letters and numbers associated with chars like "\/:"
     ## (which are chars used in links) with spaces which removes them.
      text = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', text)

      text = ' '.join(text)
      return text

# lets find out the cleaned tweets' emotion!
def get_tweet_score(analysis):

    ##then scores the tweet
    if analysis.sentiment.polarity > 0:
      return 'positive'
    elif analysis.sentiment.polarity == 0:
      return 'neutral'
    elif analysis.sentiment.polarity < 0:
      return 'negative'



# inner workings
## we're gonna grab this x amount of tweets to parse
def fetch_tweets(query, count = 50):

    api = auth()
    ### empty list to hold tweets
    tweets = []  

    try:
      collected_tweets = api.search(q = query, count = count)

      for tweet in collected_tweets:
        parsed_tweet = tweet.text
        clean_tweet = cleanup(parsed_tweet)
        stem_tweet = TextBlob(clean_tweet)
        scored_tweet = get_tweet_score(stem_tweet)
        tweets.append((parsed_tweet, clean_tweet, scored_tweet))
        return tweets
    except tweepy.TweepError as e:
      print("Error : " + str(e))
      exit(1)

st.title("Choose Topic or User to analyze")
st.write("You're welcome to use both a user and topic but both at the same time are not required, you can use one or the other too if you'd like.")


# Take off...


with st.form(key='topic):
	text_input = st.text_input(label='Choose topic')
    number_input = st.number_input(label= 'How many tweets should we source?')
	submit_button = st.form_submit_button(label='Submit')

tweets = fetch_tweets(query = text_input, count = number_input)

## sort and grab percentages between each type
## of tweet with pandas..
df = pd.DataFrame(tweets, columns= ['tweets', 'clean_tweets', 'sentiment'])


### dropping duplicate tweets too..
df = df.drop_duplicates(subset='clean_tweets')
df.to_csv('tweetbank.csv', index= False)

ptweets = df[df['sentiment'] == 'positive']
print("Percentage of positive tweets from: " + Topic + " {} %".format(100*len(ptweets)/len(tweets)))
  
ntweets = df[df['sentiment'] == 'negative']
print("Percentage of negative tweets from: " + Topic + " {} %".format(100*len(ntweets)/len(tweets)))
print("Neutral tweets percentage from: " + Topic + " {} %  ".format(100*(len(tweets) -(len( ntweets )+len( ptweets)))/len(tweets)))

display(df)



