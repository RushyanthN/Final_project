from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vectorizer = TfidfVectorizer()


sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

def train_model(df):
    try:
        X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['sentiment'], test_size=0.2, random_state=42)
        
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)
        model = MultinomialNB()
        model.fit(X_train_vectorized, y_train)
        return model, vectorizer
    except (ValueError, IOError) as e:
        print(f"Error occurred while training the model: {e}")
        return None, None

def predict_sentiment(text, model, vectorizer):
    try:
        X_new = vectorizer.transform([text])
     
        sentiment_label = model.predict(X_new)
        
        sentiment_count_label = get_sentiment_label(sentiment_label)

        global sentiment_counts
        sentiment_counts[sentiment_count_label.capitalize()] += 1

        return  get_sentiment_label(sentiment_label)
    

    except (ValueError, AttributeError) as e:
        print(f"Error occurred during sentiment analysis: {e}")
        return None

def get_sentiment_label(sentiment):
    if sentiment == '1':
        return 'negative'
    elif sentiment == '3':
        return 'neutral'
    elif sentiment == '5':
        return 'positive'
    else:
        return 'unknown'
    
def plot_sentiment_pie_chart():
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=labels, autopct='%1.1f%%')
    ax.axis('equal')
    ax.set_title("Sentiment Analysis Distribution")

    return fig