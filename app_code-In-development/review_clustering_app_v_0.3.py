import gc
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import hdbscan as hdbscan
from embedding import bert_sent_embed
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

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
    embeddings=bert_sent_embed(data['reviewText'])

#reduce dimensionality to 6, keeping neighbors at 100 
umap_embeddings = umap.UMAP(n_neighbors=100,n_components=6,metric='cosine').fit_transform(embeddings)

#clustering using HDBSCAN
clusters=hdbscan.HDBSCAN(min_cluster_size=16,min_samples=10,cluster_selection_epsilon=0.5,metric='euclidean',cluster_selection_method='eom').fit(umap_embeddings)


#Let's explore what clusters and the docs in each cluster
labels=np.unique(clusters.labels_)

cluster_labels=pd.DataFrame(clusters.labels_)
cluster_contributions=cluster_labels.value_counts(normalize=True)  #contribution (in pct) of each cluster is the same

print(cluster_labels)

labels_and_contributions=st.checkbox('Show cluster labels and relative contributions')
if labels_and_contributions:
    st.dataframe(pd.DataFrame(cluster_contributions))


#Let's create a column with labels in the original dataframe
data['label']=clusters.labels_       #changing from df_umap_embeddings_2D['label']

#selected_clusters=[i for i in range(-1,(len(labels)-1)) if (cluster_contributions[i])>0.02]
selected_clusters=[]
for i in cluster_contributions.index:
    if cluster_contributions[i]>=0.02:
        selected_clusters.append(i[0])


print('cluster_contributions',cluster_contributions)
print('selected_clusters',selected_clusters)

gc.collect()


#We can go through the same exercise with 3D.
show_3d_clusters=st.checkbox('Show major clusters in 3D plot',key='3D_major_clusters')
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

gc.collect()

#We can label the original dataframe and extract reviews corresponding to each label.
data_selected_clusters=data[data['label'].isin(selected_clusters)].copy(deep=True)


#create docs from data in each cluster, ready to be fit to TF-IDF vectorizer
def create_doc(input_dataframe,clusters_to_select):
  docs=[]
  for label in clusters_to_select:
    doc=input_dataframe[input_dataframe['label']==label]
    docs.append(doc)
  return docs

docs=create_doc(data_selected_clusters,selected_clusters)

############## Function Starts ################

def docs_TFIDF_vectorizer(docs):
  from sklearn.feature_extraction.text import TfidfVectorizer

  stop_words = text.ENGLISH_STOP_WORDS.union(['00', '10', '100', '12', '15', '16', '20', '200', '24', '25',\
       '2nd', '30', '40', '45', '50', '60', '75', '80', '90','!',"''","'m","'s",',','.','...','He','I','It','My','Of','``',\
        '!',"''","'m","'re","'s",',','-','.','...','9','An','book','Ca','Do','I','It','S.','``','!',"''","'s",'(',')',',','-','.',\
        'b', 'c', 'd', 'e', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y','&',"'ll",'-D','5',':','?',\
        "'", '0', '1', '2', '3', '4', '6', '7', '8', 'A', 'C', 'D', 'H', 'M', 'O', 'S', '`','#',"'ve",'*','--','..','....','10/10','4/5',';',\
        'As','At','HE','IS','IT','If','In','MY','No','ON','On','PR','SO','So','St','To','US','We','/', 'E', 'N', 'P', 'R', 'T', 'U', 'W', 'Y',\
        '$','%',"'S","'d",'.....','1/2','1/3','105','12-lead','125','14','150','198','1\\23\\18','22','221','27','3-3.5','35','AM','Be','By',\
        'CK','DJ','De','Dr','HM','JE','K.','L','MB','Mr','Ms','R.','TO','W.','YA','B', 'J', 'K', '\\','@','Im','Me','Is','000','100ish','11',\
        '178','191','1945','1st','2015','2016','260','36','360','3rds','3star','55','70','78','80s','86','87','99','_____'])
  
  #initialize TFIDF vectorizer
  vectorizer = TfidfVectorizer(stop_words=stop_words,ngram_range=(1,4))

  #create an empty list
  tfidf_vectorized_docs=[]

  #loop over the docs
  for doc in docs:
    X=vectorizer.fit_transform(doc['reviewText'])
    tfidf_vectorized_docs.append((vectorizer.get_feature_names(),X))
  
  return tfidf_vectorized_docs

############## Function Ends ################

tfidf_data=docs_TFIDF_vectorizer(docs)

#The corresponding dataframes with tfidf data maybe called df_unclustered, df_0, df_1, df_2. We can easily assign names, as follows:
X=[pd.DataFrame(tfidf_data[i][1].todense(),columns=tfidf_data[i][0]) for i in range(len(tfidf_data))]
print('X:\n',X)

print('df_list_comprehension:\n',[df for df in X])

#List comprehension to create separate list for each cluster
n_prominent_words=[df.sum().nlargest(15).index for df in X]
print('n_prominent_words:', n_prominent_words)

df_n_prominent_words=pd.DataFrame(n_prominent_words).T
#df_n_prominent_words=pd.DataFrame(n_prominent_words)
print('df_n_prominent_words:',df_n_prominent_words)


#make a cluster name list equal to selected_clusters
cluster_names_list=['cluster_'+ str(i) for i in selected_clusters]
print('cluster_names_list before assignment:',cluster_names_list)
df_n_prominent_words.columns=cluster_names_list


prominent_words_by_cluster=st.checkbox('Show prominent words by cluster',key='prominent_words_by_cluster')
if prominent_words_by_cluster:
    st.dataframe(df_n_prominent_words)


