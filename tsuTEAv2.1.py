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


##importing things here!
import streamlit as st
### pandas helps us chart and sort data, we'll need it if you want to see results.
import pandas as pd

import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

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



### twint is a twitter python library/api to scrape tweets; this is how we'll be able to source our tweets.
### textblob is a text processing library.
import re
import twint
from textblob import TextBlob

# streamlit formatting
st.set_page_config(layout="wide")
st.title("Tsunamifi's Twitter Sentiment Analysis Bot")
st.write("This WebAPP will allow you to plug in a topic from twitter and determine if their tweets are Positive, Negative or Neutral")
st.title("Choose Topic on twitter to analyze")
#st.write("You're welcome to use both a user and topic but both at the same time are not required, you can use one or the other too if you'd like.")

with st.form(key='vars'):
        texti = st.text_input(label='Choose topic')
        numberi = st.number_input(label= 'How many tweets should we source?')
        submit = st.form_submit_button(label='Submit')
        
c = twint.Config()
c.Search = texti
c.Limit = numberi
c.Pandas = True
c.Lang = "en"
twint.run.Search(c)
   
tweetsdf = twint.storage.panda.Tweets_df
  


# this will clean unnecessary and maybe complicated things out of a tweet
# like links or #'s 
def cleanup(text):

     ## replaces all letters and numbers associated with chars like "\/:"
     ## (which are chars used in links) with spaces which removes them.
     ## we're also tokenizing each word here
        
      parsed_tweet = text
      
      text = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', text)
  
      text_tokens = word_tokenize(text)
      text = [word for word in text_tokens if not word in stopwords.words()]

      text = ' '.join(text)
      return text

# here we're getting rid of parts of words that dont mean anything
# in sentiment analysis so we'll end up with just scoring rootwords
def root(text):

   porter = PorterStemmer()
   token_words = word_tokenize(text)
   root_sentence = []
   for word in token_words:
      root_sentence.append(porter.stem(word))
   return " ".join(root_sentence)

# dropping duplicate tweets & then cleaning them too.. 
tweetsdf = tweetsdf.drop_duplicates(subset=['date', 'tweet'])
tweetsdf.reset_index(inplace=True)
tweetsdf.drop("index",axis =1,inplace=True  
              
tweetsdf["tweetsC"] = tweetsdf["tweet"].apply(cleanup)] 
tweetsdf["tweetsC"] = tweetsdf["tweet"].apply(root)] 

      
# lets find out the cleaned tweets' emotion!
tweetsdf["polarity"] = tweetsdf["tweetsC"].apply(lambda x: TextBlob(x).sentiment[0]) 
tweetsdf["result"] = tweetsdf["polarity"].apply(lambda x: 'Positive' if x > 0 else('negative' if x<0 else 'neutral'))
              


# Take off...        
def run():
                                          
 ptweets = tweetsdf[tweetdf['result'] == 'Positive']
 posper = (100*len(ptweets)/len(tweets))
 st.write(f'Positive tweets {posper} %')
  
 ntweets = tweetsdf[tweetdf['result'] == 'Negative']
 negper = (100*len(ntweets)/len(tweets))
 st.write(f'Negative tweets {negper} %')
       
 nuper = (100 - posper - negper)
 st.write(f'Neutral tweets {nuper} %')
   
 st.dataframe(tweetsdf[['date','username','tweet','result]])
 
 wcloud = st.checkbox(label='Generate word cloud')

 twt = " ".join(tweetsdf['tweets(cleaned)'])
 wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', width=2500, height=2000).generate(twt)

 if wcloud:
   plt.figure(1,figsize=(8, 8))
   plt.axis('off')
   plt.imshow(wordcloud)
   plt.show()
   st.pyplot(wordcloud)
 else:
    pass



if submit:
    run()
else:
    pass
