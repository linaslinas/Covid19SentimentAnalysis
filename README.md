# Covid19 Sentiment Analysis

### Project Goal: 
<p> The goal of this project is to explore how citizens feel towards countries of high, moderate, and low Covid risks. <p>

### Motivation:
<p> During a pandemic, government health officials would be interested in knowing how people generally react, feel, or respond towards countries of high, low, and moderate risks of Covid. Understanding how people feel towards countries of high, moderate, and low Covid risk is important for government health officials to determine if there's a need to provide additional mental health services such as mental health help lines, community based mental health clinics, or free guidance counselling services to citizens during the pandemic. <p>

### Methodology:
1. Categorize all countries into 3 different risk groups (high, moderate, low) based on the number of confirmed COVID19 cases and case fatality rate of each country.
    <p> Run the code with the following command: <p>
```
python3 cluster_countries.py
```
2. Choose 1 country from each risk group. 
     - Canada was chosen to represent the low risk group
     - China was chosen to represent the moderate risk group
     - USA was chosen to represent the high risk group
   <p> Collect Twitter Tweets related to COVID19 for each country: Canda, China, USA. <p>
   <p> Run the code with the following command: <p>
```
python3 extract_tweets.py
```
3. For each country, use sentiment analysis tools to analyze and determine whether each Twitter Tweet is positive, negative, or neutral.
     <p> Run the code with the following commands: <p>
  
```
python3 senti_analysis.py Canada.txt
```
```
python3 senti_analysis.py China.txt
```
```
python3 senti_analysis.py USA.txt
```

#### Note:
<p> To install python packages, run the following command: <p>

```
pip3 install [package name]
```
### Results:
<p> Through clustering the countries into high, moderate, low risk levels, it seems that USA is the only country that is categorized in the high risk group. China is one of the countries of moderate risk for COVID19. Canada is one of the many countries that have a low risk of COVID19. <p>
  
![ScreenShot](https://github.com/linaslinas/Covid19SentimentAnalysis/blob/main/Graphs/risks_levels_worldwide.png)
  
<p> After collecting and analyzing Twitter Tweets for each country, it seems that there is no significant difference in sentiments for countries of high, moderate, and low risks. <p>
  
![ScreenShot](https://github.com/linaslinas/Covid19SentimentAnalysis/blob/main/Graphs/USA_sentiments.png)

![ScreenShot](https://github.com/linaslinas/Covid19SentimentAnalysis/blob/main/Graphs/china_sentiments.png)

![ScreenShot](https://github.com/linaslinas/Covid19SentimentAnalysis/blob/main/Graphs/canada_sentiments.png)

### Conclusion:
<p> People generally don't feel significantly different towards countries of high, moderate, or low COVID risks.<p>
