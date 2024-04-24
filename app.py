import os
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from utils.b2 import B2
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from model import train_model, predict_sentiment, get_sentiment_label, plot_sentiment_pie_chart




# Load environment variables
load_dotenv()

def clean_text(text):
    text = ' '.join(word for word in text.split() if not word.startswith('@'))
    text = re.sub(r'@\[A-Za-z0-9]+', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9\s]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

# Caching function
@st.cache_data
def get_data():
    try:
        REMOTE_DATA = 'Apple-Twitter-Sentiment-DFE_encoded11.csv'
        b2 = B2(endpoint=os.environ['B2_ENDPOINT'], key_id=os.environ['B2_KEYID'], secret_key=os.environ['B2_APPKEY'])
        b2.set_bucket(os.environ['B2_BUCKETNAME'])
        df_apple = b2.get_df(REMOTE_DATA)
        sentiment_1 = df_apple[df_apple['sentiment'] == '1'].sample(n=423, random_state=42)
        sentiment_3 = df_apple[df_apple['sentiment'] == '3'].sample(n=423, random_state=42)
        sentiment_5 = df_apple[df_apple['sentiment'] == '5'].sample(n=423, random_state=42)
        df_apple = pd.concat([sentiment_1, sentiment_3, sentiment_5])
        df_apple['clean_text'] = df_apple['text'].apply(clean_text)
        return df_apple[["clean_text", "sentiment"]]
    except (KeyError, ValueError, IOError) as e:
        st.error(f"Error occurred while loading data: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Streamlit app initialization
    st.title("Apple Product Sentiment Analysis")

    # Get the data
    df_apple = get_data()

    # User input
    user_input = st.text_area("Enter text related to Apple products:", key="unique_user_input")

    # Sentiment analysis
    if st.button("Analyze Sentiment"):
        if user_input:
            try:
                cleaned_text = clean_text(user_input)
                #print(cleaned_text)
                model, vectorizer = train_model(df_apple)
                #print(vectorizer.transform([cleaned_text]))
                predicted_sentiment = predict_sentiment(cleaned_text, model, vectorizer)
                st.write(f"The sentiment is **{predicted_sentiment.capitalize()}**.")  
                                  
            except (ValueError, AttributeError, ImportError) as e:
                st.error(f"Error occurred during sentiment analysis: {e}")
                st.write("Please check the implementation of the sentiment analysis model.")
        else:
            st.warning("Please enter some text to analyze.")
    st.subheader("Sentiment Analysis Distribution")
    if st.button("Plot Sentiment Pie Chart"):
        fig = plot_sentiment_pie_chart()
        st.pyplot(fig)