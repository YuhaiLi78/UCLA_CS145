from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
# Go to http://apps.twitter.com and create an app.
# The consumer key and secret will be generated for you after
consumer_key="qLK1LPJfsLSeqSqybpXmEVo68"
consumer_secret="ofjUokR46EkuP5lNMqqgtatjeaofvmixSX5lTgjgPmAbMQ8OSl"
access_token="925228398363918336-btpd6lJcKkdBtmd2bjQr51cMYT4aNAG"
access_token_secret="T1QPY6edOPi9GA7sPjyRR6gJsNot65eGCO6HGAuWX2jq2"

class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """
    def on_data(self, data):
        print(json.dumps(data))
        return True

    def on_error(self, status):
        print(status)

if __name__ == '__main__':
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l)
    stream.filter(locations=[-122.75,36.8,-121.75,37.8])
