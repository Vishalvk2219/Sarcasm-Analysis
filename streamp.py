import streamlit as st
import mainrun


st.title("Sentiment-Analysis")

que = st.text_input("Enter you sentence or review for Sentiment-Analysis")

if(st.button("Check")):
	st.text(mainrun.predict_sarcasm(que , max_length=25))

