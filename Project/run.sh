#!/bin/bash

echo "----------------------------------------------------------------------------------------"
echo "We will be using a Python library called Python Twitter Tools to connect to Twitter API and downloading the data from Twitter."
echo "Download the Python Twitter tools at https://pypi.python.org/pypi/twitter."
echo "Install the Python Twitter Tools package by typing in commands:"
echo "$ python setup.py --help"
echo "$ python setup.py build"
echo "$ python setup.py install"
echo "----------------------------------------------------------------------------------------"
echo "stream_output.json contains the streaming tweets in Json format"
echo "screan_name.txt contains the users names extracted from stream_output"
echo "user_tweets.json contain the tweets of each user"
# run the twitter_streaming.py which stores the data into file stream_output.json
python3 code/twitter_streaming.py

# get screen name of users
python3 code/get_screen_name.py

# grab timeline of each user
python3 code/twitter_user.py > user_tweets.json

# preprocess the data
python3 code/process_json.py