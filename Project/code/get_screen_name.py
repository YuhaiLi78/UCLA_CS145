# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json

# We use the file saved from last step as example
tweets_filename = 'stream_output.json'
tweets_file = open(tweets_filename, "r")
screenname_file = open("screen_name.txt", "w")
for line in tweets_file:
    try:
        # Read in one line of the file, convert it into a json object 
        tweet = json.loads(line.strip())
        if 'text' in tweet: # only messages contains 'text' field is a tweet
            screenname_file.write(tweet['user']['screen_name']) # screen name of the user
            screenname_file.write("\n")
    except:
        # read in a line is not in JSON format (sometimes error occured)
        continue
tweets_file.close()
screenname_file.close()
