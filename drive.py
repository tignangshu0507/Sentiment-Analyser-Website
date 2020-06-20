# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:15:17 2020

@author: KIIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from sklearn import preprocessing 
from sklearn import datasets, linear_model
#reading the dataset
df = pd.read_csv("train.csv",encoding='latin-1')

from tweepy import API 
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
  
from flask import Flask,render_template,request,jsonify

from textblob import TextBlob
 
#import twitter_credentials

# Variables that contains the user credentials to access Twitter API 
ACCESS_TOKEN = "1113114787473371136-UIFWlK19yFNw7XEQhSft1jfiHbzhce"
ACCESS_TOKEN_SECRET = "rVson5wuySWh2qejUDxtdeZQJJGmeAJdHcOCyInd3CZao"
CONSUMER_KEY = "kcECETl2cHMHFvcmZg00JJ7ox"
CONSUMER_SECRET = "bIgz1L54uQGYFAHIZynzj4GY33cAL3spPSO6v4n6dMbOauk0RH"
#please do change the value of the variables after you have created a twitter app copy your own credentials to the variables 


import numpy as np
import pandas as pd
import re


# # # # TWITTER CLIENT # # # #
class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets


# # # # TWITTER AUTHENTICATER # # # #
class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        return auth

# # # # TWITTER STREAMER # # # #
class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """
    def __init__(self):
        self.twitter_autenticator = TwitterAuthenticator()    

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = TwitterListener(fetched_tweets_filename)
        auth = self.twitter_autenticator.authenticate_twitter_app() 
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords: 
        stream.filter(track=hash_tag_list)


# # # # TWITTER STREAM LISTENER # # # #
class TwitterListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """
    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True
          
    def on_error(self, status):
        if status == 420:
            # Returning False on_data method in case rate limit occurs.
            return False
        print(status)
        
class TweetAnalyzer():
    """
    Functionality for analyzing and categorizing content from tweets.
    """

    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['tweets'])

        df['id'] = np.array([tweet.id for tweet in tweets])
        df['len'] = np.array([len(tweet.text) for tweet in tweets])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        df['source'] = np.array([tweet.source for tweet in tweets])
        df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])

        return df


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('TEST.html')

@app.route("/search",methods=["POST"])
def search():
    user = request.form.get("tweets")
    
    if __name__ == '__main__':
        twitter_client = TwitterClient()
        tweet_analyzer = TweetAnalyzer()

        api = twitter_client.get_twitter_client_api()

        tweets = api.user_timeline(screen_name=user, count=200)

        df_test = tweet_analyzer.tweets_to_data_frame(tweets)

    df_test_predicted=df_test.copy()

# # Visualizations

# Layered Time Series:
    time_likes = pd.Series(data=df_test_predicted['likes'].values, index=df_test_predicted['date'])
    time_likes.plot(figsize=(16, 4), label="likes", legend=True)


    time_retweets = pd.Series(data=df_test_predicted['retweets'].values, index=df_test_predicted['date'])
    time_retweets.plot(figsize=(16, 4), label="retweets", legend=True)
    plt.show()
  
# # Analysing Data
    df.head()
    df.tail()
    df_test.head()
    df.info()
    df.isnull().sum()
    df_test.isnull().sum()
    df.Sentiment.value_counts()
    df['length'] = [len(t) for t in df.SentimentText]

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.boxplot(df.length)
    plt.show()

#since Twitter Maximum Character Length is 280 so we remove the tweets beyondthat limit
    df.drop(df[df['length'] > 280].index, inplace = True) 

#Converting html entities
    from html.parser import HTMLParser
    html_parser = HTMLParser()
# Created a new columns i.e. clean_tweet contains the same tweets but cleaned version
    df['clean_tweet'] = df['SentimentText'].apply(lambda x: html_parser.unescape(x))
    df_test['clean_tweet'] = df_test['tweets'].apply(lambda x: html_parser.unescape(x))

    import re
#Removing "@user" from all the test dataset tweets
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
            return input_txt
# remove twitter handles (@user)
    df_test['clean_tweet'] = np.vectorize(remove_pattern)(df_test['clean_tweet'], "@[\w]*")
    df_test.head(10)
#Changing all the tweets into lowercase
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: x.lower())
    df.head(10)
    df_test['clean_tweet'] = df_test['clean_tweet'].apply(lambda x: x.lower())
    df_test.head(10)
# Apostrophe Dictionary
    apostrophe_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not","couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is",
    "i'd": "I would",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have","mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had","that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will","we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have","who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
    }

    def lookup_dict(text, dictionary):
        for word in text.split():
            if word.lower() in dictionary:
                if word.lower() in text.split():
                    text = text.replace(word, dictionary[word.lower()])
    return text

    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: lookup_dict(x,apostrophe_dict))
    df_test['clean_tweet'] = df_test['clean_tweet'].apply(lambda x: lookup_dict(x,apostrophe_dict))

    short_word_dict = {
"121": "one to one",
"a/s/l": "age, sex, location",
"adn": "any day now",
"afaik": "as far as I know",
"afk": "away from keyboard",
"aight": "alright",
"alol": "actually laughing out loud",
"b4": "before",
"b4n": "bye for now",
"bak": "back at the keyboard",
"bf": "boyfriend",
"bff": "best friends forever",
"bfn": "bye for now",
"bg": "big grin",
"bta": "but then again",
"btw": "by the way",
"cid": "crying in disgrace","cnp": "continued in my next post",
"cp": "chat post",
"cu": "see you",
"cul": "see you later",
"cul8r": "see you later",
"cya": "bye",
"cyo": "see you online",
"dbau": "doing business as usual",
"fud": "fear, uncertainty, and doubt",
"fwiw": "for what it's worth",
"fyi": "for your information",
"g": "grin",
"g2g": "got to go",
"ga": "go ahead",
"gal": "get a life",
"gf": "girlfriend","gfn": "gone for now",
"gmbo": "giggling my butt off",
"gmta": "great minds think alike",
"h8": "hate",
"hagn": "have a good night",
"hdop": "help delete online predators",
"hhis": "hanging head in shame",
"iac": "in any case",
"ianal": "I am not a lawyer",
"ic": "I see",
"idk": "I don't know",
"imao": "in my arrogant opinion",
"imnsho": "in my not so humble opinion",
"imo": "in my opinion",
"iow": "in other words",
"ipn": "I’m posting naked","irl": "in real life",
"jk": "just kidding",
"l8r": "later",
"ld": "later, dude",
"ldr": "long distance relationship",
"llta": "lots and lots of thunderous applause",
"lmao": "laugh my ass off",
"lmirl": "let's meet in real life",
"lol": "laugh out loud",
"ltr": "longterm relationship",
"lulab": "love you like a brother",
"lulas": "love you like a sister",
"luv": "love",
"m/f": "male or female","m8": "mate",
"milf": "mother I would like to fuck",
"oll": "online love",
"omg": "oh my god",
"otoh": "on the other hand",
"pir": "parent in room",
"ppl": "people",
"r": "are",
"rofl": "roll on the floor laughing",
"rpg": "role playing games",
"ru": "are you",
"shid": "slaps head in disgust",
"somy": "sick of me yet",
"sot": "short of time",
"thanx": "thanks",
"thx": "thanks",
"ttyl": "talk to you later",
"u": "you",
"ur": "you are",
"uw": "you’re welcome",
"wb": "welcome back",
"wfm": "works for me",
"wibni": "wouldn't it be nice if","wtf": "what the fuck",
"wtg": "way to go",
"wtgp": "want to go private",
"ym": "young man",
"gr8": "great"
    }

    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: lookup_dict(x,short_word_dict))
    df_test['clean_tweet'] = df_test['clean_tweet'].apply(lambda x: lookup_dict(x,short_word_dict))

    emoticon_dict = {
":)": "happy",
":‑)": "happy",
":-]": "happy",
":-3": "happy",
":->": "happy",
"8-)": "happy",
":-}": "happy",
":o)": "happy",
":c)": "happy",
":^)": "happy",
"=]": "happy",
"=)": "happy",
"<3": "happy",
":-(": "sad",
":(": "sad",
":c": "sad",
":<": "sad",
":[": "sad",
">:[": "sad",
":{": "sad",
">:(": "sad",
":-c": "sad",
":-< ": "sad",
":-[": "sad",
":-||": "sad"
    }

    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: lookup_dict(x,emoticon_dict))
    df_test['clean_tweet'] = df_test['clean_tweet'].apply(lambda x: lookup_dict(x,emoticon_dict))

    import re
#replacing special characters with space
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x))
    df_test['clean_tweet'] = df_test['clean_tweet'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x))

#tokenizing a collection of text documents and building a vocabulary of known words,and also encoding new documents using that vocabulary.
#count the number of words (term frequency)
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(df['clean_tweet'])
    X_train_counts.shape

    X_test_counts = count_vect.transform(df_test['clean_tweet'])
    X_test_counts.shape

#computing the IDF values
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf.shape

    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    X_test_tfidf.shape
    y = df['Sentiment']

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, y, test_size=0.30, random_state=42)
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.1, class_prior=None, fit_prior=True)
    model.fit(X_train_tfidf,y)
    y_predict1 = model.predict(X_test)
    print(f1_score(y_test, y_predict1, average="macro"))

    Y_test_pred=model.predict(X_test_tfidf)
    Y_test_pred
    df_test_predicted['Sentiment']=Y_test_pred
    df_test_predicted.head(10)

    df_test_predicted.tail()
    return jsonify({"success":True,"tweets":df_test_predicted})
app.run