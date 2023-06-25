#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


credits.head()


# In[5]:


movies = movies.merge(credits,on='title')


# In[6]:


movies.head(1)


# In[7]:


# genere
# id
# keywords
# title
# overview
# cast
# crew

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[8]:


movies.head()


# In[9]:


movies.isnull().sum()


# In[10]:


movies.dropna(inplace=True)


# In[11]:


movies.duplicated().sum()


# In[12]:


movies.iloc[0].genres


# In[13]:


# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# ['Action','Adventure','Fantasy','Sifi']


# In[14]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[15]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L    


# In[16]:


movies['genres'] = movies['genres'].apply(convert)


# In[17]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[18]:


movies.head()


# In[19]:


def convert3(obj):
    L = []
    counter = 0 
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L   


# In[20]:


movies['cast'] = movies['cast'].apply(convert3)


# In[21]:


movies.head()


# In[22]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L        


# In[23]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[24]:


movies.head()


# In[25]:


movies['overview'][0]


# In[26]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[27]:


movies.head()


# In[28]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[29]:


movies.head()


# In[30]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[31]:


movies.head()


# In[32]:


new_df = movies[['movie_id','title','tags']]


# In[33]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[34]:


new_df.head()


# In[35]:


import nltk


# In[36]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[37]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)   


# In[38]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[39]:


new_df['tags'][0]


# In[40]:


new_df['tags']= new_df['tags'].apply(lambda x:x.lower())


# In[41]:


new_df.head()


# In[42]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[43]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[44]:


vectors


# In[45]:


vectors[0]


# In[46]:


from sklearn.metrics.pairwise import cosine_similarity


# In[47]:


similarity = cosine_similarity(vectors)


# In[48]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[49]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    
    


# In[50]:


recommend('Batman Begins')


# In[51]:


import pickle


# In[52]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[53]:


new_df['title'].values


# In[54]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[55]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




