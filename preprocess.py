import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.initializers import GlorotUniform
import gender_edit


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the dataset into a pandas DataFrame
data = pd.read_csv('rotten_tomatoes_critic_reviews.csv')
#data = data.sample(frac=0.25, random_state=42)
print("dropping empty rows...")
data = data.dropna(subset=['review_type', 'review_content'])
print("dropped data empty rows...")
# Preprocessing function
def preprocess(review):
    # Tokenization
    tokens = word_tokenize(review.lower())
    
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    return ' '.join(filtered_tokens)  # Join the tokens back into a single string
print("creating new labels for pred...")
# Apply preprocessing to the text data
data['preprocessed_review'] = data['review_content'].apply(preprocess)
data = data[data['review_type'].str.lower().isin(['fresh', 'rotten'])]
print("creating gender labels...")
# Replace 'fresh' and 'rotten' with 1 and 0
data['review_type'] = data['review_type'].str.lower().replace({'fresh': 1, 'rotten': 0})
# Split the dataset into training and testing sets

data.to_csv('prep_data.csv', index=False)