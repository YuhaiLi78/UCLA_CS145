import tweepy
import json

consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

def process_or_store(tweet):
    print(json.dumps(tweet))
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

for friend in tweepy.Cursor(api.friends).items():
    process_or_store(friend)