from tweepy.streaming import StreamListener
from tweepy import Stream
from tweepy import OAuthHandler
import re
import json
import csv
import traceback

# PLEASE INPUT PERSONAL API KEY INFORMATION
consumer_key = "INSERT CONSUMER KEY HERE"
consumer_secret_key = "INSERT CONSUMER SECRET KEY HERE"
access_token = "INSERT ACCESS TOKEN HERE"
access_token_secret = "INSERT ACCESS SECRET TOKEN HERE"

# List of words to search for
search_list = ['COVID Canada', 'coronavirus Canada', 'covid Trudeau', 'coronavirus Trudeau' 'Ontario covid', 'coronavirus Ontario']
# search_list = ['COVID USA', 'coronavirus USA', 'trump coronavirus', 'trump COVID', 'trump covid', 'New York covid', 'New York coronavirus', 'New York COVID', 'COVID united states', 'coronavirus united states' ]
# search_list = ['COVID China', 'coronavirus China', 'chinese virus', 'wuhan virus', 'covid wuhan']

tweet_count = 0

# Number of tweets to download
num_tweets = 1000


# file = open("China.txt", "a")
# file = open("USA.txt", "a")
file = open("CANADA.txt", "a")
file.close()

class StdOutListener(StreamListener):

	def on_data(self,data):
		global tweet_count
		global num_tweets
		global stream
		if tweet_count < num_tweets:
			try: 
				tweet_data = json.loads(data)
				pattern1 = re.compile(r'\n')
				tweet_txt = pattern1.sub(r'', tweet_data['text'])
				pattern2 = re.compile(r'RT')
				tweet = pattern2.sub(r'', tweet_txt)
				
				# PLEASE CHANGE THE PATH TO YOUR DESIRED FILE FOLDER.
				# This is where the extracted tweets will go.
				# f = open("/home/Desktop/Covid19_Sentiment_Analysis/Data/China.txt", "a+")
				# f = open("/home/Desktop/Covid19_Sentiment_Analysis/Data/USA.txt", "a+")
				f = open("/home/Desktop/Covid19_Sentiment_Analysis/Data/CANADA.txt", "a+")
				
				if not tweet_data['text'].startswith('RT'):
					f.write(str(tweet_data['id']) + "\t" + tweet + "\n")
					tweet_count+=1
			except BaseException as error:
				traceback.print_exc()
				print(error)
			return True
		else:
			stream.disconnect()

	def on_error(self, status):
		print(status)

l = StdOutListener()
auth = OAuthHandler(consumer_key, consumer_secret_key)
auth.set_access_token(access_token, access_token_secret)
stream = Stream(auth, l)

#search filter
stream.filter(track = search_list, languages = ['en'])
