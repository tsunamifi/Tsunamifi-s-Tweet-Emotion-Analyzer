import streamlit as st
import twint
import pandas as pd
import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)             
import matplotlib.pyplot as plt

#optional: for reading and concatenation of previous files
import glob                     
import os

import numpy as np
import datetime as dt

#cleaning
import re
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords             

# Sentiment Analysis
from textblob import TextBlob

#word cloud
from wordcloud import WordCloud

"""### Configure and run Twint (twitter scrapper)"""

#for compatibility issues with twint
runit = st.button('run')
bank_search = {"FNB":"FNBSA", "StandardBank":"StandardBankZA OR \"Standard Bank\" OR \"standard bank\"","Nedbank":"Nedbank OR nedbank","ABSA": "Absa OR ABSA OR absa OR AbsaSouthAfrica","Capitec":"CapitecBankSA OR Capitec or capitec"}
if runit
   st.dataframe(tweets_df[["cleaned_tweet","polarity","Sentiment"]])
else
 pass
def twintConfig(date_from,date_to, search_string):
    c = twint.Config()
    c.Search = search_string[1]
    c.Since = date_from
    c.Until = date_to
    c.Pandas = True
    c. Pandas_au = True          
    c.Pandas_clean=True
    #c.Hide_output = True
    #c.Resume = "./ResumeID/resume_id_"+search_string[0]+".txt"
    twint.run.Search(c)

"""### Run twint"""

since = input("Input a start date eg 2021-09-17: ")
until = input("Input an end date eg 2021-09-18: ")

def Run_Twint(search_vals):
    
    #set empty dataframe for join
    out_df= pd.DataFrame()
    
    for bank in search_vals.items():
        print ("running for search item: "+bank[0]+"\n")
        print ("Search string: "+bank[1]+"\n")
        
        #run twint
        twintConfig(since,until, bank)
        
        #get dataframe
        tweets_df = twint.storage.panda.Tweets_df
        
        #join Dataframes and create 'Bank' column
        tweets_df["Bank"]= bank[0]
        out_df = pd.concat([out_df,tweets_df])
        
    return out_df

tweets_df = Run_Twint(bank_search)

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
# %%time
# tweets_df["cleaned_tweet"] = tweets_df["tweet"].apply(clean_text)

#drop empty rows
tweets_df = tweets_df [ ~(tweets_df["tweet"] =="")]

tweets_df["cleaned_tweet"].head()

"""## Sentiment analysis (TextBlob)"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# print("Running sentiment process")
# for row in tweets_df.itertuples():
#     tweet = tweets_df.at[row[0], 'cleaned_tweet']
# 
#     #run sentiment using TextBlob
#     analysis = TextBlob(tweet)
# 
#     #set value to dataframe
#     tweets_df.at[row[0], 'polarity'] = analysis.sentiment[0]
#     tweets_df.at[row[0], 'subjectivity'] = analysis.sentiment[1]
# 
#     #Create Positive / negative column depending on polarity
#     if analysis.sentiment[0]>0:
#         tweets_df.at[row[0], 'Sentiment'] = "Positive"
#     elif analysis.sentiment[0]<0:
#         tweets_df.at[row[0], 'Sentiment'] = "Negative"
#     else:
#         tweets_df.at[row[0], 'Sentiment'] = "Neutral"

tweets_df[["cleaned_tweet","polarity","Sentiment"]].head(5)
