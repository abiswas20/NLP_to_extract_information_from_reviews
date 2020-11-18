import streamlit as st
import pandas as pd
import numpy as np

st.title('review_analyzer')

path = st.text_input('path to dataset (csv)')
nrows=int(st.text_input('number of rows'))

def load_data(nrows):
    data=pd.read_csv(path,nrows=nrows)
    lowercase=lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data.head(10)

show_df=st.checkbox('Do you want to see the data?')

if show_df:
    X=load_data(nrows)
    X



