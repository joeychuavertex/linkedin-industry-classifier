import streamlit as st
import pandas as pd
from transformers import pipeline

linkedin_industry_csv = pd.read_csv("industry_codes_linkedin_v1.csv")
linkedin_industry = linkedin_industry_csv['description']

st.title("LinkedIn Industry Classifier")

query = st.text_input("Search industries ")
# query = st.text_area("Search industries by name or description", height=200)
st.caption("It will take around 30 seconds.")

if query:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    output = classifier(query, linkedin_industry, multi_label=True)
    output = pd.DataFrame(data=output)
    output = output.sort_values("scores", ascending=False)
    output = output[['labels', 'scores']]

else:
    output = linkedin_industry


st.table(output)
