# Import necessary libraries
from flask import Flask, render_template, request, jsonify
import requests
import csv
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd


app = Flask(__name__)


BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAGKbsgEAAAAA4%2FnKz62jBf76XMy8H4Ltg3fGwxo%3DkuqX40PUdvnn4yHbZpLNCd5tNfScoNaMZmA4WPtPcI5JaI4ZJU'


def get_user_tweets(username):
    # API endpoint URL
    url = f"https://api.twitter.com/2/tweets/search/recent?query=from:{username}&max_results=50&tweet.fields=created_at,entities"
    # Authorization header
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

    
    response = requests.get(url, headers=headers)
    # Check if request was successful
    if response.status_code != 200:
        print("Error:", response.text)
        return []

   
    tweet_data = response.json()
    return tweet_data.get("data", [])


def contains_keywords(username):
    
    keywords = [
        "crypto", "money", "stocks", "wealth", "invest", "investing", "portfolio"
    ]
   
    tweets = get_user_tweets(username)

    
    for tweet in tweets:
        for keyword in keywords:
            if keyword in tweet.get("text", "").lower():
                return True

    return False


def get_average_tweet_time(username):
    # Get user's tweets
    tweets = get_user_tweets(username)
    # Check if enough tweets are available for calculation
    if not tweets or len(tweets) < 2:
        return None  # Not enough tweets for meaningful calculation

    
    tweet_times = []
    for tweet in tweets:
        created_at = tweet.get('created_at')
        if created_at:
            try:
                tweet_times.append(
                    datetime.fromisoformat(created_at.replace('Z', '+00:00')))
            except ValueError:
                continue

    
    if len(tweet_times) < 2:
        return None
    time_diffs = [(tweet_times[i] - tweet_times[i - 1]).total_seconds()
                  for i in range(1, len(tweet_times))]

    return sum(time_diffs) / len(time_diffs)

def get_user_data(username):
  
    url = f"https://api.twitter.com/2/users/by/username/{username}?user.fields=public_metrics"
   
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

 
    response = requests.get(url, headers=headers)
   
    if response.status_code != 200:
        print("Error:", response.text)
        return {}

    # Parse response JSON
    user_data = response.json().get('data', {})
    return user_data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
   
    username = request.form['username']
   
    user_data = get_user_data(username)
   
    contains_keywords_result = contains_keywords(username)
   
    average_tweet_time = get_average_tweet_time(username)
    if average_tweet_time is None:
        average_tweet_time = 0

   
    hashtags_count = 0
    tweets = get_user_tweets(username)
    for tweet in tweets:
        hashtags_count += len(tweet.get("entities", {}).get("hashtags", []))

   
    follower_count = user_data.get('public_metrics', {}).get('followers_count', 0)
    following_count = user_data.get('public_metrics', {}).get('following_count', 0)
    percent_difference = ((following_count - follower_count) / follower_count) * 100 if follower_count != 0 else 0

   
    predict1, predict2, predict3 = predict(average_tweet_time, contains_keywords_result, percent_difference)

   
    return render_template('result.html', username=username, follower_count=follower_count,
                           following_count=following_count, contains_keywords_result=contains_keywords_result,
                           average_tweet_time=average_tweet_time, hashtags_count=hashtags_count,
                           percent_difference=percent_difference, predict1=predict1, predict2=predict2,
                           predict3=predict3)


def predict(tweetTime, containKey, diff):
   
    data = pd.read_csv('dataset.csv') 
  
    X = data[['AverageTweetTime', 'ContainsKeywords', 'PercentDifference']]  # Features
    y = data['Bot']  # Labels
  
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
  
    feature_vector = [[tweetTime, containKey, diff]]
    
    prediction = model.predict(feature_vector)
    
   
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    prediction2 = model.predict(feature_vector)
    
    
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    prediction3 = model.predict(feature_vector)
    
    return prediction, prediction2, prediction3


if __name__ == '__main__':
    app.run(debug=True)
