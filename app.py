import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.models import load_model
import nltk

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load the pre-trained model
model = load_model('SarcasmDetection_model.h5')
tokenizer_obj = Tokenizer()

# Function to clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

# Function to tokenize and clean the text data
def CleanTokenize(df):
    head_lines = []
    lines = df["headline"].values.tolist()
    for line in lines:
        line = clean_text(line)
        tokens = word_tokenize(line)
        words = [word for word in tokens if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        words = [w for w in words if not w in stop_words]
        head_lines.append(words)
    return head_lines

# Function to predict sarcasm
def predict_sarcasm(text, max_length=100):
    x_final = pd.DataFrame({"headline": [text]})
    test_lines = CleanTokenize(x_final)
    test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
    test_review_pad = pad_sequences(test_sequences, maxlen=max_length, padding='post')
    pred = model.predict(test_review_pad)
    confidence = pred[0][0] * 100  # Convert to percentage
    return confidence, "It's a sarcasm!" if confidence >= 50 else "It's not a sarcasm."

# Streamlit app
st.set_page_config(page_title="Sarcasm Detection App", layout="wide")

st.title("🤖 Sarcasm Detection")
st.write("This app detects whether a given text is sarcastic or not.")

# Sidebar options
st.sidebar.header("Input Options")
input_type = st.sidebar.radio("Choose input type:", ["Text Input", "File Upload"])

# Text input
if input_type == "Text Input":
    input_text = st.text_area("Enter Text:", placeholder="Type something sarcastic...")

    if st.button("Predict"):
        if input_text:
            confidence, result = predict_sarcasm(input_text)
            st.success(f"**Result:** {result}")
            st.info(f"**Confidence:** {confidence:.2f}%")
        else:
            st.error("Please enter some text to predict.")

# File upload option
elif input_type == "File Upload":
    uploaded_file = st.file_uploader("Choose a file", type="txt")
    if uploaded_file is not None:
        text = str(uploaded_file.read(), "utf-8")
        st.write("File content:")
        st.write(text)

        if st.button("Predict from File"):
            confidence, result = predict_sarcasm(text)
            st.success(f"**Result:** {result}")
            st.info(f"**Confidence:** {confidence:.2f}%")

# Additional info section
st.sidebar.header("About")
st.sidebar.info(
    """
    This Sarcasm Detection app uses a pre-trained neural network model 
    built with TensorFlow/Keras. It can detect sarcasm in text with a 
    good level of confidence. 
    - Input: Single text or text file.
    - Output: Sarcastic or Not Sarcastic, with confidence level.
    """
)

st.sidebar.markdown(
    """
    Developed by: [Your Name](https://www.linkedin.com)
    """
)
