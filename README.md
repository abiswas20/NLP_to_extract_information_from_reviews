## Natural Language Processing (NLP) to Extract Information from Reviews

### Apratim Biswas, November 2020

---

### Problem Statement  
Popular businesses offering mutiple products and services receive large volume of customer feedback in the form of text reviews. While listening to such feedback is essential for a business to succeed, attempting to do so would involve large amount of human capital. The aim of this project is to use Natural Language Processing (NLP) to extract and aggregate information from large volume of customer reviews? If yes, explore methods to do so.

### Goal  
Our goal is two-fold:  
i) Cluster reviews based on their content. This helps us learn about customers.  
ii) Extract high-level information at topic level from the feedback.  

### Data Source and Preprocessing  
Original dataset was obtained from repository maintained by ianmo Ni, Jiacheng Li, Julian McAuley at UCSD [1](https://nijianmo.github.io/amazon/index.html). It contained reviews posted on Amazon from May 1996 till October 2018. Only the data on book reviews was used was used for the project. The dataset with book reviews was over 35 GB  by itself,and included more than 51 million reviews. It was processed in chunks, randomly selecting 25% of reviews each time. 

The processed dataset included only reviews from the five most reviewed books that were posted after 2015-07-01. 

### Executive Summary

Listening to customer feedback is an integral part of the business improvement process. In today's world a large part of this feedback comes in the form of review text. Digesting such large volume and extracting information is a very resource intensive process. This project explores a new approach using NLP to extract information. In the first step, the reviews are clustered. Each cluster of reviews is studied further to learn about factors they emphasize. The first step gives us a look at the various clusters of customers. In the second step we learn which attribute of our product is important to each cluster of customer.

The work in this project was done with book reviews. But that's just incidental. The big idea is to develop a general approach that can be applied to all types of products. What I present here is just the beginning and I plan to build a more general application that retrieve high level information from reviews of any type of product.

The original dataset used in this study included book reviews on Amazon from 1996 to 2018.Preprocessing involved filtering them to include only the top 5 most reviewed books since 7/1/15. From this smaller subset, only reviews with 3 star and 4 star ratings, for product with asin # 0312577222 and  # 038568231X were selected for the study. While this report focuses entirely on # 0312577222, results were comparable for # 038568231X. 

There are two reasons for chosing reviews with only 3 and 4 stars:  
i) they tend to be balanced with constructive information; and,  
ii) 3 and 4 stars are on the positive side of the spectrum. It is likely they are open to our product/service. And likewise, investments made in targeting them by addressing their needs is expected to yield a higher return. 

Sentence-transformers was used to encode review texts at the sentence level using BERT. UMAP was used to reduce dimensionality to 6 before clustering reviews with HDBSCAN. To create easily interpretable visualizations, I reduced original encoded data to 2 and 3 dimensions. It’s important to note that reducing dimensionality comes at a cost: information loss. An unfortunately side-effect of this is points that form obvious clusters in higher dimensions may appear 'not-so-clusterable' at lower dimension.

High TF-IDF terms were identified for each cluster. High-level information evident from such terms were summarized. In short, our approach involved clustering reviews and then studying each cluster as list of topics. This helps us achieve both of our goals: learn about our customers, and learn about our own products/services.

### Presentation Slides  

Presentation slides can be found at this [link](https://docs.google.com/presentation/d/1MDVXKrzkVPxmOz1KgxXRsLLzTfZhwPlyfoVQfiSez9g/edit?usp=sharing).

### Summary  

1. Book reviews with 3 and 4 star reviews of the 2 most reviewed books on Amazon were selected for this study. 
2. Sentence-transformer was used that utilized BERT to tokenize sentences. 
3. UMAP (Uniform Manifold Approximation and Projection) was used to reduce dimensionality and clusters were identified using HDBSCAN.  
4. Clusters were visualized in 2D and 3D.  
5. Terms with high TF-IDF scores were listed for each cluster. An attempt was made to summarize the content of reviews in each cluster at a very high level.  

Put in another way, the reviews were clustered and studied as list of topics. This helps us with both: learn about our customers, and learn about our own products/services.


## References

1. https://nijianmo.github.io/amazon/index.html
