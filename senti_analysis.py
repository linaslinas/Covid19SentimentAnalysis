import re
import pandas as pd
import matplotlib.pyplot as plt 
import collections
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import spacy 
from langdetect import detect, DetectorFactory
import numpy as np
import random
import csv
from nltk.corpus import twitter_samples
from textblob import TextBlob
from itertools import chain
from functools import reduce
import emoji
import matplotlib.pyplot as plt; plt.rcdefaults()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import sys


def remove_url(x):
	url_pattern = re.compile(r'https?://\S+|www\.\S+')
	replace_url = url_pattern.sub(r'', str(x))
	return replace_url


def remove_usernames(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt  


# Remove retweets, links, hashtags, tags
def remove_phrases(x):
	x = re.sub(r'^RT[\s]+', '', x)
	x = re.sub(r'https?:\/\/.*[\r\n]*', '', x)
	x = re.sub(r'#', '', x)
	x = re.sub(r'@[A-Za-z0–9]+', '', x) 
	return x


def sentiment_scores(sentence):  

    # Create a SentimentIntensityAnalyzer object. 
    senti_obj = SentimentIntensityAnalyzer() 
  
    # Polarity_scores method of SentimentIntensityAnalyzer 
    # Oject gives a sentiment dictionary which contains pos, neg, neu, and compound scores. 
    sentiment_dict = senti_obj.polarity_scores(sentence) 
  
    if sentiment_dict['compound'] > 0.00 : 
        return "Positive"
  
    elif sentiment_dict['compound'] < 0.00 : 
        return "Negative"
  
    else : 
        return "Neutral"


# String matching to obtain the country of the data file: (Canada.txt, China.txt, USA.txt)
def get_input_file_name(filename):
	country_name = re.search('(?P<name>.*)\.txt$', filename).group('name')
	return country_name


filename = sys.argv[1]

data = pd.read_csv(filename, names=['id', 'text'], sep ="\t", quoting=csv.QUOTE_NONE, error_bad_lines=False)

# Remove URLs 
data['text'] = data['text'].apply(remove_url)

# Remove Twitter usernames
data['text'] = np.vectorize(remove_usernames)(data['text'], "@[\w]*:") 

# Remove hashtags, twitter usernames
data['text'] = data['text'].apply(remove_phrases)

tweet_text = []
for i in range(1000):
	tweet_text.append([])

# Split each tweet into sentences
for i in range(len(data['text'])):
	text_tuples = TextBlob(data['text'][i])
	text_sentences = text_tuples.sentences
	tweet_text[i].append(text_sentences)


positive_sentiments = []
negative_sentiments = []
neutral_sentiments = []

# Determine whether the sentences are positive attitudes, negative attitudes, or neutral attitudes
for i in range(len(tweet_text)):
	for j in range(len(tweet_text[i][0])):
		if (sentiment_scores(tweet_text[i][0][j])) == "Negative":
			negative_sentiments.append([tweet_text[i][0][j], "Negative"])

		if (sentiment_scores(tweet_text[i][0][j])) == "Positive":
			positive_sentiments.append([tweet_text[i][0][j], "Positive"])	

		if (sentiment_scores(tweet_text[i][0][j])) == "Neutral":
			neutral_sentiments.append([tweet_text[i][0][j], "Neutral"])


# Total number of sentences in the entire data set
total_sentences = len(positive_sentiments) + len(negative_sentiments) + len(neutral_sentiments)

print("Number of positive sentiments: ", len(positive_sentiments))
print("Number of negative sentiments: ", len(negative_sentiments))
print("Number of neutral sentiments: ", len(neutral_sentiments))
print("Total Number of sentiments: ", total_sentences)


# Calculate the percentage of positive, negative and neutral comments over the total number of sentences
positive_sent_percent = len(positive_sentiments)/total_sentences
print("Percentage of positive sentiments:  ", positive_sent_percent)

negative_sent_percent = len(negative_sentiments)/total_sentences
print("Percentage of negative sentiments: ", negative_sent_percent)

neutral_sent_percent = len(neutral_sentiments)/total_sentences
print("Percentage of neutral sentiments: ", neutral_sent_percent)


# Bar graph showing the amount of positive, negative, and neutral sentiments
label = ['Positive', 'Neutral', 'Negative']
sentiment_percentages = [positive_sent_percent*100, negative_sent_percent*100, negative_sent_percent*100]
index = np.arange(len(label))
plt.bar(index, sentiment_percentages)
plt.xlabel('Sentiment Groups', fontsize=10)
plt.ylabel('Sentiment Percentages (%)', fontsize=10)
plt.xticks(index, label, fontsize=10, rotation=30)
country_name = get_input_file_name(filename)
plt.title(country_name + ' Sentiments')
plt.show()