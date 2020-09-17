import pandas as pd
import numpy as np
import nltk

# Below libraries are for text processing using NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Below libraries are for feature representation using sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Below libraries are for similarity matrices using sklearn
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
nltk.download('wordnet')

def read_and_preprocess():
    news_articles = pd.read_json("News_Category_Dataset_v2.json", lines = True)
    news_articles = news_articles[news_articles['date'] >= pd.Timestamp(2018,1,1)]
    news_articles = news_articles[news_articles['headline'].apply(lambda x: len(x.split())>5)]
    news_articles.sort_values('headline',inplace=True, ascending=False)
    duplicated_articles_series = news_articles.duplicated('headline', keep = False)
    news_articles = news_articles[~duplicated_articles_series]
    news_articles.index = range(news_articles.shape[0])

    return news_articles, duplicated_articles_series

def list_category():
    news_articles, duplicated_articles_series = read_and_preprocess()
    categories = news_articles.category.unique().tolist()
    idxs2cat = {k:v for k, v in enumerate(categories)}
    cat2idxs = {k:v for v, k in idxs2cat.items()}
    return categories, idxs2cat, cat2idxs

def recommend_articles(category):
    categories, idxs2cat, cat2idxs = list_category()
    news_articles, duplicated_articles_series = read_and_preprocess()
    cat_df = news_articles[news_articles.category == category]

    cat_healine = cat_df.headline.tolist()

    return cat_healine

def tfidf_based_model(row_index, num_similar_items):

    news_articles, duplicated_articles_series = read_and_preprocess()

    # Adding a new column containing both day of the week and month, it will be required later while recommending based on day of the week and month
    news_articles["day and month"] = news_articles["date"].dt.strftime("%a") + "_" + news_articles["date"].dt.strftime("%b")

    news_articles_temp = news_articles.copy()

    for i in range(len(news_articles_temp["headline"])):
        string = ""
        for word in news_articles_temp["headline"][i].split():
            word = ("".join(e for e in word if e.isalnum()))
            word = word.lower()
            if not word in stop_words:
                string += word + " "  

        news_articles_temp.at[i,"headline"] = string.strip()

    tfidf_headline_vectorizer = TfidfVectorizer(min_df = 0)
    tfidf_headline_features = tfidf_headline_vectorizer.fit_transform(news_articles_temp['headline'])

    couple_dist = pairwise_distances(tfidf_headline_features,tfidf_headline_features[row_index])
    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
    df = pd.DataFrame({'publish_date': news_articles['date'][indices].values,
               'headline':news_articles['headline'][indices].values,
                'Euclidean similarity with the queried article': couple_dist[indices].ravel()})
    # print("="*30,"Queried article details","="*30)
    # print('headline : ',news_articles['headline'][indices[0]])
    # print("\n","="*25,"Recommended articles : ","="*23)
    
    #return df.iloc[1:,1]
    return df.iloc[1:,]



# category = 'SCIENCE'
# print('RECOMMENDATION FOR CATEGORY:', category)
# cat_df = news_articles[news_articles.category == category]
# cat_idx = np.random.choice(cat_df.index)
# # Adding a new column containing both day of the week and month, it will be required later while recommending based on day of the week and month
# news_articles["day and month"] = news_articles["date"].dt.strftime("%a") + "_" + news_articles["date"].dt.strftime("%b")

# news_articles_temp = news_articles.copy()

# for i in range(len(news_articles_temp["headline"])):
#     string = ""
#     for word in news_articles_temp["headline"][i].split():
#         word = ("".join(e for e in word if e.isalnum()))
#         word = word.lower()
#         if not word in stop_words:
#           string += word + " "  
#     # if(i%1000==0):
#     #   print(i)           # To track number of records processed
#     news_articles_temp.at[i,"headline"] = string.strip()

# lemmatizer = WordNetLemmatizer()

# tfidf_headline_vectorizer = TfidfVectorizer(min_df = 0)
# tfidf_headline_features = tfidf_headline_vectorizer.fit_transform(news_articles_temp['headline'])

# def tfidf_based_model(row_index, num_similar_items):
#     couple_dist = pairwise_distances(tfidf_headline_features,tfidf_headline_features[row_index])
#     indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
#     df = pd.DataFrame({'publish_date': news_articles['date'][indices].values,
#                'headline':news_articles['headline'][indices].values,
#                 'Euclidean similarity with the queried article': couple_dist[indices].ravel()})
#     print("="*30,"Queried article details","="*30)
#     print('headline : ',news_articles['headline'][indices[0]])
#     print("\n","="*25,"Recommended articles : ","="*23)
    
#     #return df.iloc[1:,1]
#     return df.iloc[1:,]
# tfidf_based_model(cat_idx, 11)