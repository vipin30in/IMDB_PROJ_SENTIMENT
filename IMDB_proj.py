import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# Title and description
st.title("Movie Review Sentiment Analysis")
st.markdown("""
This app predicts whether a movie review is positive or negative using machine learning.
Simply enter your review in the text box below!
""")

# Text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Function to load and train the model
@st.cache_resource
def load_model():
    # Load your dataset
    file_path = 'IMDB Dataset.csv'  # Make sure to upload this file
    try:
        data = pd.read_csv(file_path, names=['Review', 'Sentiment'], header=0)
        
        # Preprocess the data
        data['Cleaned_Review'] = data['Review'].apply(preprocess_text)
        data['Sentiment'] = data['Sentiment'].map({'positive': 1, 'negative': 0})
        
        # Vectorize the text
        vectorizer = CountVectorizer(max_features=5000)
        X = vectorizer.fit_transform(data['Cleaned_Review']).toarray()
        y = data['Sentiment']
        
        # Train the model
        model = LogisticRegression(max_iter=200, solver='liblinear')
        model.fit(X, y)
        
        return vectorizer, model
    except FileNotFoundError:
        st.error("Please make sure the IMDB Dataset.csv file is in the same directory as this script.")
        return None, None

# Load the model and vectorizer
vectorizer, model = load_model()

# Create the main input area
st.subheader("Enter Your Movie Review")
user_input = st.text_area("", height=150)

# Add a predict button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        try:
            # Preprocess the input
            processed_input = preprocess_text(user_input)
            
            # Vectorize the input
            input_vector = vectorizer.transform([processed_input]).toarray()
            
            # Make prediction
            prediction = model.predict(input_vector)[0]
            probability = model.predict_proba(input_vector)[0]
            
            # Display results
            st.subheader("Analysis Results")
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.success("This review appears to be POSITIVE! 😊")
                else:
                    st.error("This review appears to be NEGATIVE! 😔")
            
            with col2:
                # Display confidence scores
                st.write("Confidence Scores:")
                st.progress(probability[1])
                st.write(f"Positive: {probability[1]:.2%}")
                st.progress(probability[0])
                st.write(f"Negative: {probability[0]:.2%}")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add instructions and about section in the sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses machine learning to analyze movie reviews and predict whether they are positive or negative.
    
    The model is trained on the IMDB dataset and uses:
    - Text preprocessing
    - CountVectorizer for feature extraction
    - Logistic Regression for classification
    """)
    
    st.header("Instructions")
    st.markdown("""
    1. Enter your movie review in the text box
    2. Click the "Predict Sentiment" button
    3. View the prediction and confidence scores
    
    The longer and more detailed the review, the better the prediction!
    """)

