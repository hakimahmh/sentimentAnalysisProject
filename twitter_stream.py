import tweepy
from config import TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
from models.sentiment_model import get_sentiment

auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
api = tweepy.API(auth)

def get_twitter_sentiment(keyword, count=10):
    tweets = api.search_tweets(q=keyword, count=count, lang='en')
    results = []
    for tweet in tweets:
        sentiment, score = get_sentiment(tweet.text)
        results.append({"tweet": tweet.text, "sentiment": sentiment, "score": score})
    return results
