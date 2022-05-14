

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

st.title("Choose Topic on twitter to analyze")
#st.write("You're welcome to use both a user and topic but both at the same time are not required, you can use one or the other too if you'd like.")
with st.form(key='vars'):
        texti = st.text_input(label='Choose topic')
        numberi = st.number_input(label= 'How many tweets should we source?', step=1)
        submit = st.form_submit_button(label='Submit')
# defining twitter auth setup here!
# from config import bearer_token


bearer_token = "AAAAAAAAAAAAAAAAAAAAAHbwcAEAAAAAA3Heaep4TEp0EOTiUpPnTPLy9A4%3DspSy2AYLMulUovSsOjv9AT99T2GBnyJDmaPV7YzOXrvbHtJPWH"
client = tweepy.Client(bearer_token=bearer_token)

# this will clean unnecessary and maybe complicated things out of a tweet
# like links or #'s 
def cleanup(text):

     ## replaces all letters and numbers associated with chars like "\/:"
     ## (which are chars used in links) with spaces which removes them.
     ## we're also tokenizing each word here
      text = text.lower()
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
    
    
# lets find out the cleaned tweets' emotion!
def get_tweet_score(analysis):

    ##then scores the tweet
    if analysis.sentiment.polarity > 0:
      return 'positive'
    elif analysis.sentiment.polarity < 0:
      return 'negative'
    else:
      return 'neutral'

# Get user information..
def get_username(user_id):
    result = client.get_user(id=user_id, user_fields=['username'])
    return result.data.username


def get_tweets(search_query, max_results):
    tweets = []


    # The tweet.fields query parameters [attachments,author_id,context_annotations,conversation_id,created_at,entities,
    # geo,id,in_reply_to_user_id,lang,non_public_metrics,organic_metrics,possibly_sensitive,promoted_metrics,public_metrics,
    # referenced_tweets,reply_settings,source,text,withheld]
    result = tweepy.Cursor(client.search_recent_tweets, query=search_query, tweet_fields=['conversation_id', 'created_at','author_id'], max_results=max_results)
    tweets = list(result.items(20))
    for tweet in tweets:
      tweets.append((tweet.id, tweet.created_at, tweet.author.screen_name, tweet.text))
    return tweets

 




result = get_tweets(search_query=texti, max_results=numberi)   

def print():
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
    #st.write(f"Tweet ID: {tweet_id}\n"
     #     f"Conversation ID: {conversation_id}\n"
      #    f"Author ID: {author_id}\n"
       #   f"Username: {username}\n"
        #  f"Created at: {created_at}\n"
         # f"Language: {lang}\n"
          #f"Tweet: {tweet}\n"
          #f"Public metrics: {public_metrics}\n",
          #f"Source: {source}")
    # print("-" * 80)   
   # df = pd.json_normalize(tweets_data, columns= ['created_at', 'username', 'tweets', 'results'])
    df = pd.DataFrame([tweet for tweet in tweets], columns=["tweet_id", "tweet_date", "tweet_username", "tweet_text"])

  
 ### dropping duplicate tweets too..
 
    df["tweetsC"] = df["tweets"].apply(cleanup) 
    df["tweetsC"] = df["tweets"].apply(root)
    df = df.drop_duplicates(subset='tweetsC')
    df.to_csv('tweetbank.csv', index= False)
      
  # lets find out the cleaned tweets' emotion!
    df["polarity"] = df["tweetsC"].apply(lambda x: TextBlob(x).sentiment[0]) 
    df["result"] = df["polarity"].apply(lambda x: 'Positive' if x > 0 else('negative' if x<0 else 'neutral'))
    

    ptweets = df[df['result'] == 'positive']
    posper = (100*len(ptweets)/len(tweets))
    st.write(f'Positive tweets {posper} %')
  
    ntweets = df[df['result'] == 'negative']
    negper = (100*len(ntweets)/len(tweets))
    st.write(f'Negative tweets {negper} %')
       
    nuper = (100 - posper - negper)
    st.write(f'Neutral tweets {nuper} %')
   
    st.dataframe(df)

    wcloud = st.checkbox(label='Generate word cloud')

    twt = " ".join(df['tweetsc'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', width=2500, height=2000).generate(twt)

    if wcloud:
      plt.show()
      plt.figure(1,figsize=(8, 8))
      plt.axis('off')
      plt.imshow(wordcloud)
      st.pyplot(wordcloud)
    else:
      pass

 


# Take off...

        
if submit:
    print()
else:
    pass
