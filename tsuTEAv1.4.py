# -*- coding: utf-8 -*-
"""tsunamifis_twittr_sentiment_analysis_botv1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1c3F8klCW9uaJllbkFLyeQipfOu2rDRPr

# Tsunamifi's Twitter Sentiment Analysis Bot
This notebook will allow you to plug in a user OR topic from twitter and determine if their tweets are Positive, Negative or Neutral.

Why? for fun, probably."""

#core setup.


##importing things here!


### streamlit is a python web app framework, we're using it for its widgets and cute gui.
import streamlit as st

### pandas helps us sort data, we'll need it if you want to see results.
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
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob

### MISC
import re
import time

# streamlit formatting
st.set_page_config(layout="wide")
st.title("Twitter Sentiment Analysis Bot")
st.write("This WebAPP will allow you to plug in a topic from twitter and determine if the general discussion is positive, negative or neutral.")
st.title("Choose Topic on twitter to analyze")

with st.form(key='vars'):
        texti = st.text_input(label='Choose topic')
        numberi = st.number_input(label= 'How many tweets should we source?', step=1, value=15)
        submit = st.form_submit_button(label='Submit')


    



# inner workings

## defining twitter auth setup here!
def auth():

    ### keys and token from twitter'
    consumer_key = "KLhloeEIOr2de1Cnz7ddcxcmT"
    consumer_secret = "WPQeRE5skCsCfBK8inJSPFTOEFMCUrPGMUyKsi1kjo8xKDaoxQ"
    access_token = "1450493640132923397-NQ1fupgKuJZKZbPsi3gYw9EnngxapO"
    access_token_secret = "eH3IvjkUatJuU7EdYtoSMZckBbIWtPpvYNFUW078VvMK4"

    ### attempting auth...
    try:
      auth = OAuthHandler(consumer_key, consumer_secret)
      auth.set_access_token(access_token, access_token_secret)
      api = tweepy.API(auth)
      return api
    except:
      st.error("Error: Authentification Failed, try again?")
      exit(1)

## this will clean unnecessary and maybe complicated things out of a tweet
## like links or #'s 
def cleanup(text):

     ### replaces all letters and numbers associated with chars like "\/:"
     ### (which are chars used in links) with spaces which removes them.
     ### we're also tokenizing each word here
      
      text = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', text)
  
      text_tokens = word_tokenize(text)
      text = [word for word in text_tokens if not word in stopwords.words()]

      text = ' '.join(text)
      return text

## here we're getting rid of parts of words that dont mean anything
## in sentiment analysis so we'll end up with just scoring rootwords
def root(text):

  porter = PorterStemmer()
  token_words = word_tokenize(text)
  root_sentence = []
  for word in token_words:
    root_sentence.append(porter.stem(word))
  return " ".join(root_sentence)    
    
    
## lets find out the cleaned tweets' general emotion!
def get_tweet_score(analysis):

    ###"scoring" tweet
    if analysis.sentiment.polarity > 0:
      return 'positive'
    elif analysis.sentiment.polarity < 0:
      return 'negative'
    else:
      return 'neutral'


## we're gonna grab this x amount of tweets to parse
def fetch_tweets(query, count = 50):

    api = auth()
    ### empty list to hold tweets
    tweets = []  

    collected_tweets = api.search(q = query + ' -filter:retweets', count = count)
    
    ### this is a pipeline to do all our our tweet processing
    for Tweet in collected_tweets:
        parsed_tweet = Tweet.text
        clean_tweet = cleanup(parsed_tweet)
        stem_tweet = TextBlob(root(clean_tweet))
        scored_tweet = get_tweet_score(stem_tweet)
        tweets.append((parsed_tweet, clean_tweet, scored_tweet))
        
    return tweets
    



# Take off...

## this is how everything comes into play
def run():
 tweets = fetch_tweets(query = texti, count = numberi)

 ### stuff our collected tweets in to a pandas dataframe
 ### and also specifying which columns.
 df = pd.DataFrame(tweets, columns= ['Tweets', 'Scrubbed Tweets', 'Result'])

 ### dropping duplicate tweets too..
 df = df.drop_duplicates(subset='Scrubbed Tweets')
 df.to_csv('tweetbank.csv', index= False)

 ### calculate and display total percentages
 ptweets = df[df['Result'] == 'positive']
 posper = (100*len(ptweets)/len(tweets))
  
 ntweets = df[df['Result'] == 'negative']
 negper = (100*len(ntweets)/len(tweets))
       
 nuper = (100 - posper - negper)
    
 st.write("Here's the overall climate concerning " + texti)
 col1, col2, col3 = st.columns(3)
 col1.metric('Positive Tweets', f'{posper}%')
 col2.metric('Negative Tweets', f'{negper}%')
 col3.metric('Neutral Tweets', f'{nuper}%') 
    
 ### generate wordcloud
 twt = " ".join(df['Scrubbed Tweets'])
 wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=2000, height=1500).generate(twt)
 plt.show()
 fig = plt.figure(1,figsize=(8, 8))
 plt.axis('off')
 plt.imshow(wordcloud)   
    
 col4, col5 = st.columns(2)   
 with col4:
        st.caption('Here is our data for reference')
        st.dataframe(df)
 with col5:
    st.caption(f' Here are the words most commonly association with {texti}')
    st.pyplot(fig)
 st.success('Done!')

## loading spinner, why because its cute.    
def spin():
  with st.spinner('Collecting tweets...'):
    time.sleep(3)
    
if submit:
  spin()
  try:  
    run()
  except tweepy.TweepError as e:
    st.error("There's something afoot... looks like " f'{e}')
else:
    pass
