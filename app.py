import streamlit as st
import pickle
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer


tokenizer = TreebankWordTokenizer()
port_stemmer = PorterStemmer()


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def clean_text(text):
    
    tokens = tokenizer.tokenize(text)

    cleaned = []
    for word in tokens:
        word = word.lower()
        if word not in string.punctuation and not re.match(r'\d+', word):
            if word not in stopwords.words('english'):
                stemmed = port_stemmer.stem(word)
                cleaned.append(stemmed)

    return ' '.join(cleaned)


st.title('📩 SMS Spam Classifier')

input_sms = st.text_input("Enter your message below:")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning('⚠️ Please enter a message before predicting.')
    else:
       
        transformed_text = clean_text(input_sms)

       
        vector_input = tfidf.transform([transformed_text])

      
        result = model.predict(vector_input)[0]

      
        if result == 1:
            st.error("🚨 This message is classified as **Spam**.")
        else:
            st.success("✅ This message is **Not Spam**.")
