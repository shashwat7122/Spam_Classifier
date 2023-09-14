import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
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
    for  i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tf = pickle.load(open('vectorize.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")
if st.button('Predict'):
        #preprocess
        transformed_sms = transform_text(input_sms)

        #vectorize
        v_input = tf.transform([transformed_sms])

        #predict
        result = model.predict(v_input)

        #display
        if result == 1:
            st.header("This message is a Spam")
        else:
            st.header("This message is Not a Spam")