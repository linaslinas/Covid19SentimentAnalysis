# Citation:
# The following code is based off of the TA's code
# https://www.youtube.com/watch?v=Cz2hSCEDJqs&feature=youtube
import re
import pandas as pandas
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


def detect_language(x):
	DetectorFactory.seed = 0
	try:
		lang = detect(x)
		return(lang)
	except:
		return("other")

def remove_url(x):
	url_pattern = re.compile(r'https?://\S+|www\.\S+')
	replace_url = url_pattern.sub(r'', str(x))
	return replace_url

def remove_username(x):
	url_pattern = re.compile(r'@\S+|www\.\S+')
	replace_url = url_pattern.sub(r'', str(x))
	return replace_url

#Reference: https://stackoverflow.com/questions/50830214/remove-usernames-from-twitter-data-using-python
def remove_usernames(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt  

#Reference: https://medium.com/analytics-vidhya/sentiment-analysis-on-ellens-degeneres-tweets-using-textblob-ff525ea7c30f
def clean(x):
	x = re.sub(r'^RT[\s]+', '', x)
	x = re.sub(r'https?:\/\/.*[\r\n]*', '', x)
	x = re.sub(r'#', '', x)
	x = re.sub(r'@[A-Za-z0–9]+', '', x) 
	return x

#Reference: https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/
#sentimental analysis
def sentiment_scores(sentence): 
  
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 
  
    # polarity_scores method of SentimentIntensityAnalyzer 
    # oject gives a sentiment dictionary. 
    # which contains pos, neg, neu, and compound scores. 
    sentiment_dict = sid_obj.polarity_scores(sentence) 
      
    # print("Overall sentiment dictionary is : ", sentiment_dict) 
    # print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative") 
    # print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 
    # print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive") 
  
    # print("Sentence Overall Rated As", end = " ") 
  
    # decide sentiment as positive, negative and neutral 
    if sentiment_dict['compound'] > 0.00 : 
        # print("Positive") 
        return "Positive"
  
    elif sentiment_dict['compound'] < 0.00 : 
        # print("Negative") 
        return "Negative"
  
    else : 
        # print("Neutral") 
        return "Neutral"

#Reference: https://stackoverflow.com/questions/33161769/not-reading-all-rows-while-importing-csv-into-pandas-dataframe?fbclid=IwAR3non0phADUq4bPpR1-mUg3-H2USZfVe77CJ5thqQmt3j2EveAdiZGKK44
columns = ['id', 'text']

#read in the file that I want to extract/parse
file = pandas.read_csv('/home/lina/Desktop/cmpt459/cmpt459Project/Canada.txt', names = columns, sep = "\t", quoting=csv.QUOTE_NONE, error_bad_lines=False)
# file = pandas.read_csv('/home/lina/Desktop/cmpt459/cmpt459Project/USA.txt', names = columns, sep = "\t", quoting=csv.QUOTE_NONE, error_bad_lines=False)
#file = pandas.read_csv('/home/lina/Desktop/cmpt459/cmpt459Project/China.txt', names = columns, sep = "\t", quoting=csv.QUOTE_NONE, error_bad_lines=False)


#initialize a list containing 1000 empty lists
#the 1000 empty lists will be used for storing each tweet text after it's broken up into several sentences
tweet_text = []
for i in range(1000):
	tweet_text.append([])

file.head()

#remove urls
file['cleaned_urls'] = file['text'].apply(remove_url)

#Reference: https://stackoverflow.com/questions/50830214/remove-usernames-from-twitter-data-using-python
#remove twitter usernames
file['cleaned_usernames'] = np.vectorize(remove_usernames)(file['cleaned_urls'], "@[\w]*:") 

#remove hashtags, twitter usernames
file['cleaned'] = file['cleaned_usernames'].apply(clean)

vader = SentimentIntensityAnalyzer()

#split each tweet into sentences
for i in range(len(file['cleaned'])):
	text_tuples = TextBlob(file['cleaned'][i])
	text_sentences = text_tuples.sentences
	tweet_text[i].append(text_sentences)

print(tweet_text)

positive_sentiments = []
negative_sentiments = []
neutral_sentiments = []
count = 0

#determine the if the sentences are have positive attitudes, negative attitudes, or neutral attitudes
for i in range(len(tweet_text)):
	for j in range(len(tweet_text[i][0])):
		if (sentiment_scores(tweet_text[i][0][j])) == "Negative":
			negative_sentiments.append([tweet_text[i][0][j], "Negative"])

		if (sentiment_scores(tweet_text[i][0][j])) == "Positive":
			positive_sentiments.append([tweet_text[i][0][j], "Positive"])	

		if (sentiment_scores(tweet_text[i][0][j])) == "Neutral":
			neutral_sentiments.append([tweet_text[i][0][j], "Neutral"])

		#count the total number of sentences in the entire dataset
		count = count + 1


#-------------------------------------------------------------------------------------------------------
#print the total number of positive sentiments, negative sentiments, and neutral sentiments
#print the total number of sentences in the dataset
print("Number of positive sentiments")
print(len(positive_sentiments))
print("Number of negative sentiments")
print(len(negative_sentiments))
print("Number of neutral sentiments")
print(len(neutral_sentiments))
print("Total Number of sentiments")
print(count)
#-------------------------------------------------------------------------------------------------------

#------------------calculate the percentage of positive, negative and neutral comments over the total number of sentences/phrases-----------------
print("************************************************")
print("Percentage of positive sentiments")
positive_sent_percent = len(positive_sentiments)/count
print(positive_sent_percent)
print("************************************************")

print("************************************************")
print("Percentage of negative sentiments")
negative_sent_percent = len(negative_sentiments)/count
print(negative_sent_percent)
print("************************************************")

print("************************************************")
print("Percentage of neutral sentiments")
neutral_sent_percent = len(neutral_sentiments)/count
print(neutral_sent_percent)
print("************************************************")
#-------------------------------------------------------------------------------------------------------------------------------------

#plot the results
# label = ['Positive', 'Neutral', 'Negative']
# sentiment_percentages = [positive_sent_percent*100, negative_sent_percent*100, negative_sent_percent*100]
# index = np.arange(len(label))
# plt.bar(index, sentiment_percentages)
# plt.xlabel('Sentiment Groups', fontsize=10)
# plt.ylabel('Sentiment Percentages', fontsize=10)
# plt.xticks(index, label, fontsize=10, rotation=30)
# plt.title('Sentiment Groups and Their Corresponding Percentages')
# plt.show()

for i in range(10):
	print(positive_sentiments[i])

# #Reference: https://www.programiz.com/python-programming/writing-csv-files
# with open('sentiment_results.csv', 'w', newline='') as results:
# 	writer = csv.writer(results)
# 	for i in range(len(positive_sentiments)):
# 		writer.writerow(positive_sentiments[i])