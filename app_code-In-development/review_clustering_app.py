import gc
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import hdbscan as hdbscan
from sentence_transformers import SentenceTransformer

gc.set_threshold(3,3,3)

st.title('review_analyzer')

path = st.text_input('path to dataset (csv)', key='path')
nrows=st.number_input('number of rows to load',format='%f',key='nrows')

get_data=st.checkbox('Do you want to load data?',key='get_data')

if get_data:
    product_reviews_ratings=pd.read_csv(path,nrows=nrows,error_bad_lines=False)

show_df=st.checkbox('Do you want to see the data?',key='show_df')
row_views=st.number_input('number of rows to view',format='%f',key='row views')

if show_df:
    row_views=int(row_views)
    X=product_reviews_ratings.head(row_views)
    st.dataframe(X)

gc.collect()

product_list=product_reviews_ratings['asin'].value_counts().index[:5]
show_most_reviewed=st.checkbox('Show 5 most reviewed products')
if show_most_reviewed:
    st.dataframe(product_list)

gc.collect()

asin=st.text_input('asin for specific product:',key='asin')
reviews_1=product_reviews_ratings[product_reviews_ratings['asin']==asin]
reviews_1.dropna(inplace=True)

reviews_with_ratings=st.multiselect(
        "which reviews do you want to include?",
        [1,2,3,4,5]
        )

gc.collect()

selected_reviews=pd.DataFrame()
for i in reviews_with_ratings:
    selected_reviews=pd.concat([selected_reviews,reviews_1[reviews_1['overall']==i]])

show_selected_data=st.checkbox('Do you want to see reviews with selected ratings?',key='show_reviews_with_selected_data')

if show_selected_data:
    st.dataframe(selected_reviews)

#Creating a dataframe with only the text of selected reviews
data=selected_reviews[['reviewText','overall']]

#reset index
data.reset_index(inplace=True)
data.drop('index',axis=1,inplace=True)

#And let's drop all nulls
data.dropna(inplace=True)

ratings_and_reviews_choice=st.checkbox('Do you want to look at a view with only review text and ratings?')

if ratings_and_reviews_choice:
    st.dataframe(data)

gc.collect()

#sentence embeddings with bert
bert_embed=st.checkbox('Do you want to embed using bert?',key='bert_embed')
if bert_embed:
    model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
    embeddings = model.encode(data['reviewText'], show_progress_bar=True)

#reduce dimensionality to 6, keeping neighbors at 100 
umap_embeddings = umap.UMAP(n_neighbors=100,n_components=6,metric='cosine').fit_transform(embeddings)

#clustering using HDBSCAN
clusters=hdbscan.HDBSCAN(min_cluster_size=16,min_samples=10,cluster_selection_epsilon=0.5,metric='euclidean',cluster_selection_method='eom').fit(umap_embeddings)


#Let's explore what clusters and the docs in each cluster
labels=np.unique(clusters.labels_)

#we can use the UMAP method we used previously to create embedding in 2d and visualize the findings.
umap_embeddings_2D=umap.UMAP(n_neighbors=100,n_components=2,metric='cosine').fit_transform(embeddings)
df_umap_embeddings_2D=pd.DataFrame(umap_embeddings_2D,columns=['x','y'])
df_umap_embeddings_2D['label']=clusters.labels_

cluster_contributions=df_umap_embeddings_2D['label'].value_counts(normalize=True)  #contribution (in pct) of each cluster

labels_and_contributions=st.checkbox('Show cluster labels and relative contributions')
if labels_and_contributions:
    st.dataframe(cluster_contributions)

#Now let's plot df_umap_embeddings_2D
show_2d_clusters=st.checkbox('Show clusters on a 2D plot')
if show_2d_clusters:
    fig,ax=plt.subplots()
    fig=plt.figure(figsize=(16,13))
    ax = fig.add_subplot(111)
    ax.scatter(x=df_umap_embeddings_2D['x'],y=df_umap_embeddings_2D['y'],c=df_umap_embeddings_2D['label'],cmap='cividis')
    st.pyplot(fig,clear_figure=False)

#Let's create a column with labels in the original dataframe
data['label']=df_umap_embeddings_2D['label']

selected_clusters=[i for i in range(len(df_umap_embeddings_2D['label'].value_counts(normalize=True))-2) if df_umap_embeddings_2D['label'].value_counts(normalize=True)[i]>0.02]

df_umap_embeddings_2D_selected_clusters=df_umap_embeddings_2D[df_umap_embeddings_2D['label'].isin(selected_clusters)]
df_umap_embeddings_2D_selected_clusters.plot(x='x',y='y',kind='scatter',c='label',cmap='cividis',figsize=(16,12))


#We can go through the same exercise with 3D.

show_3d_clusters=st.checkbox('Show clusters in a 3D plot')
if show_3d_clusters:
    umap_embeddings_3D=umap.UMAP(n_neighbors=100,n_components=3,metric='cosine').fit_transform(embeddings)
    df_umap_embeddings_3D=pd.DataFrame(umap_embeddings_3D,columns=['x','y','z'])
    df_umap_embeddings_3D['label']=clusters.labels_

    df_umap_embeddings_3D_selected_clusters=df_umap_embeddings_3D[df_umap_embeddings_3D['label'].isin(selected_clusters)]

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    for cluster in selected_clusters:
        ax.scatter(df_umap_embeddings_3D_selected_clusters.loc[df_umap_embeddings_3D_selected_clusters['label']==cluster]['x'],\
                   df_umap_embeddings_3D_selected_clusters.loc[df_umap_embeddings_3D_selected_clusters['label']==cluster]['y'],\
                   df_umap_embeddings_3D_selected_clusters.loc[df_umap_embeddings_3D_selected_clusters['label']==cluster]['z'],\
                   cmap='cividis',label=cluster)
    ax.legend()

    st.pyplot(fig,clear_figure=False)
