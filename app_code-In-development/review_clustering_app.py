import streamlit as st
import pandas as pd
import numpy as np

st.title('review_analyzer')

path = st.text_input('path to dataset (csv)', key='path')
asin=st.text_input('asin:',key='asin')
nrows=int(st.text_input('number of rows to load',key='nrows'))

get_data=st.checkbox('Do you want to load data?',key='get_data')

if get_data:
    product_reviews_ratings=pd.read_csv(path,nrows=nrows,error_bad_lines=False)

show_df=st.checkbox('Do you want to see the data?',key='show_df')
row_views=int(st.text_input('number of rows to view',key='row_views'))


if show_df:
    row_views=int(row_views)
    X=product_reviews_ratings.head(row_views)
    st.dataframe(X)


reviews_1=product_reviews_ratings[product_reviews_ratings['asin']==asin]
reviews_1.dropna(inplace=True)

product_list=product_reviews_ratings['asin'].value_counts().index[:5]

show_most_reviewed=st.checkbox('Show 5 most reviewed products')
if show_most_reviewed:
    st.dataframe(product_list)

reviews_with_ratings=st.multiselect(
        "which reviews do you want to include?",
        [1,2,3,4,5]
        )

selected_reviews=pd.DataFrame()
for i in reviews_with_ratings:
    selected_reviews=pd.concat([selected_reviews,reviews_1[reviews_1['overall']==i]])

#Creating a dataframe with only the text of selected reviews
data=selected_reviews[['reviewText','overall']]

#reset index
#data.reset_index(inplace=True)

#And let's drop all nulls
#data.dropna(inplace=True)

#data.drop('index',axis=1,inplace=True)
#st.dataframe(data)

show_selected_data=st.checkbox('Do you want to see reviews with selected ratings?',key='show_reviews_with_selected_data')

if show_selected_data:
    st.dataframe(selected_reviews)
