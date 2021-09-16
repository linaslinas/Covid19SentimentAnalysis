# Reference: https://www.youtube.com/watch?v=WSTob0j3I3E&feature=youtu.be

from tweepy.streaming import StreamListener
from tweepy import Stream
from tweepy import OAuthHandler
import re
import json
import csv
import traceback

consumer_key = "UGGE1vG4ZomvH976bJfFPOYWS"
consumer_secret_key = "TtaSzLKWdh4eoY2o7EZSw0iGHycHzo8XulvQfoeXrCleXoSPwR"
access_token = "1263719683607285762-3LBpajGEjhjzO7gjaNNcCRjUL8SAk1"
access_token_secret = "K1Px9ib8OUjxR4fy25MKkxOkSiNYFJStidU2pBtebw84Q"

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
				# f = open("/home/lina/Desktop/Covid19_Sentiment_Analysis/Data/China.txt", "a+")
				# f = open("/home/lina/Desktop/Covid19_Sentiment_Analysis/Data/USA.txt", "a+")
				f = open("/home/lina/Desktop/Covid19_Sentiment_Analysis/Data/CANADA.txt", "a+")
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