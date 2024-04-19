from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def train_model(df):
    try:
        X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['sentiment'], test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer()
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
        predicted_sentiment = model.predict(X_new)[0]
        return get_sentiment_label(predicted_sentiment)
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