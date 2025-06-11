import streamlit as st
import pickle
import string
import nltk
import ssl

# Handle SSL certificate issues that sometimes occur on Streamlit Cloud
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Force download NLTK data
@st.cache_resource
def setup_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    return True


setup_nltk()

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load the saved models
@st.cache_resource
def load_models():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model


tfidf, model = load_models()

st.title("📱 Email/SMS Spam Classifier")
st.write("Enter a message below to check if it's spam or not.")

input_sms = st.text_area("Enter the message", height=100, placeholder="Type your message here...")

if st.button("🔍 Predict", type="primary"):
    if input_sms.strip():
        with st.spinner('Analyzing message...'):
            # 1- pre-process
            transformed_sms = transform_text(input_sms)
            # 2- Vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3- Predict
            result = model.predict(vector_input)[0]

            # 4- Display
            if result == 1:
                st.error("🚨 **SPAM DETECTED**")
                st.write("This message appears to be spam.")
            else:
                st.success("✅ **NOT SPAM**")
                st.write("This message appears to be legitimate.")
    else:
        st.warning("⚠️ Please enter a message to classify.")