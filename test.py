

"""### Configure and run Twint (twitter scrapper)"""
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

### display is so we can display a dataframe from pandas or something else.
from IPython.display import display


st.set_page_config(layout="wide")
st.title("Tsunamifi's Twitter Sentiment Analysis Bot")
st.write("This WebAPP will allow you to plug in a topic from twitter and determine if their tweets are Positive, Negative or Neutral")


with st.form(key='vars'):
        texti = st.text_input(label='Choose topic')
        numberi = st.number_input(label= 'How many tweets should we source?')
        submit = st.form_submit_button(label='Submit')
        
# inner workings
## we're gonna grab this x amount of tweets to parse

st.title("Choose Topic on twitter to analyze")
#st.write("You're welcome to use both a user and topic but both at the same time are not required, you can use one or the other too if you'd like.")


# Take off...



        
def run():

 ## sort and grab percentages between each type
 ## of tweet with pandas..

 ### dropping duplicate tweets too..
 
                                           
 ptweets = tweetsdf[tweetdf['result'] == 'Positive']
 posper = (100*len(ptweets)/len(tweets))
 st.write(f'Positive tweets {posper} %')
  
 ntweets = tweetsdf[tweetdf['result'] == 'Negative']
 negper = (100*len(ntweets)/len(tweets))
 st.write(f'Negative tweets {negper} %')
       
 nuper = (100 - posper - negper)
 st.write(f'Neutral tweets {nuper} %')
   
 st.dataframe(tweetsdf)
 tweetsdf[['tweets','tweets(cleaned)','result']].st.dataframe()
 wcloud = st.checkbox(label='Generate word cloud')

 twt = " ".join(df['clean_tweets'])
 wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', width=2500, height=2000).generate(twt)

 if wcloud:
   plt.show()
   st.pyplot(wordcloud)
 else:
    pass

 plt.figure(1,figsize=(8, 8))
 plt.axis('off')
 plt.imshow(wordcloud)

if submit:
    twintConfig()
    st.dataframe(tweets_df)
else:
    pass
#for compatibility issues with twint

def twintConfig():
    c = twint.Config()
    c.Search = texti
    c.Limit = numberi
    c.Pandas = True
    c.Lang = "en"
    twint.run.Search(c)

"""# precleaning"""

tweets_df.shape

tweets_df = tweets_df.drop(["Unnamed: 0", 'created_at', 'user_id_str', 'link', 'urls', 'photos', 'video',
       'thumbnail', 'retweet','nreplies', 'nretweets', 'quote_url', 'near', 'geo', 'source', 'user_rt_id', 'user_rt',
       'retweet_id', 'reply_to', 'retweet_date', 'translate', 'trans_src',
       'trans_dest'],axis = 1)

tweets_df.head(2)

"""#### Language analysis 

Although the language tag doesnt seem to get it right 100% of the time, we will drop these rows that arent english but keep undefined:
* und = undefined --- this will also include tweets with only hashtags so we will keep this
* en = english 
"""

tweets_df["language"].unique()

# remove all rows where language is not english or undefined
tweets_df = tweets_df[tweets_df["language"].isin([ 'und', 'en'])]

"""#### Remove unnecessary rows 
* Remove tweets from Bank owned accounts i.e. FNBSA
* Remove duplicates where tweet, bank and date are the same 
* Reindex dataframe
"""

# remove rows where username is in bank_search
tweets_df = tweets_df[ ~tweets_df["username"].str.lower().str.contains('fnb|standardbank|nedbank|absa|capitec',regex = True)]

#Drop duplicated tweets 
tweets_df = tweets_df.drop_duplicates(subset=['date',"tweet","Bank"],keep="first")

# reset the index for visualisation later
tweets_df.reset_index(inplace=True)
tweets_df.drop("index",axis =1,inplace=True)

len(tweets_df)

"""### Cleaning tweet data 
* Remove punctuation, hashtags, symbols etc

***Note:*** 
Cleaning will take a very long time depeding on your size of data and processing speeds \
In order to do parallel processing download the file (https://github.com/Slyth3/Sentiment-analysis-on-South-African-Banks/blob/main/multi_clean.py) \
Then import this file and run using the below:
* import multi_clean
* import multiprocessing as mp
*from multiprocessing import  Pool
* p = mp.Pool(mp.cpu_count())
* cleaned_list = p.map(multi_clean.clean_text,base_tweets["tweet"])
* p.close()

#### Tweet cleaning
"""

def clean_text(text):  
    pat1 = r'@[^ ]+'                   #@signs
    pat2 = r'https?://[A-Za-z0-9./]+'  #links
    pat3 = r'\'s'                      #floating s's
    pat4 = r'\#\w+'                     # hashtags
    pat5 = r'&amp '
    pat6 = r'[^A-Za-z\s]'         #remove non-alphabet
    combined_pat = r'|'.join((pat1, pat2,pat3,pat4,pat5, pat6))
    text = re.sub(combined_pat,"",text).lower()
    return text.strip()

# Commented out IPython magic to ensure Python compatibility.
tweets_df["cleaned_tweet"] = tweets_df["tweet"].apply(clean_text)

#drop empty rows
tweets_df = tweets_df [ ~(tweets_df["tweet"] =="")]

tweets_df["cleaned_tweet"].head()

"""## Sentiment analysis (TextBlob)"""

# Commented out IPython magic to ensure Python compatibility.
print("Running sentiment process")
for row in tweets_df.itertuples():
    tweet = tweets_df.at[row[0], 'cleaned_tweet']
 
     #run sentiment using TextBlob
     analysis = TextBlob(tweet)
 
     #set value to dataframe
     tweets_df.at[row[0], 'polarity'] = analysis.sentiment[0]
     tweets_df.at[row[0], 'subjectivity'] = analysis.sentiment[1]
 
     #Create Positive / negative column depending on polarity
     if analysis.sentiment[0]>0:
         tweets_df.at[row[0], 'Sentiment'] = "Positive"
     elif analysis.sentiment[0]<0:
         tweets_df.at[row[0], 'Sentiment'] = "Negative"
     else:
         tweets_df.at[row[0], 'Sentiment'] = "Neutral"

tweets_df[["cleaned_tweet","polarity","Sentiment"]].head(5)
st.dataframe(tweets_df)
