#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd


# In[63]:


df=pd.read_csv("spotify_millsongdata.csv")


# In[64]:


df.head(5)


# In[65]:


df.tail(5)


# In[66]:


df.shape


# In[67]:


df.isnull().sum()


# In[68]:


df=df.sample(5000).drop('link',axis=1).reset_index(drop=True)


# In[69]:


df.head(10)


# In[70]:


df['text'][0]


# In[71]:


df.shape


# TEXT CLEANING /TEXT PROCESSING

# In[72]:


df['text']=df['text'].str.lower().replace(r'^\w\s',' ').replace(r'\n',' ',regex=True)


# In[73]:


df.tail(5)


# In[74]:


import nltk
from nltk.stem.porter import PorterStemmer


# In[75]:


stemmer=PorterStemmer()


# In[76]:


def token(txt):
    token = nltk.word_tokenize(txt)
    a=[stemmer.stem(w) for w in token]
    return " ".join(a)


# In[77]:


token("you are beautiful")


# In[53]:


df['text'].apply(lambda x: token(x))


# In[54]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[55]:


tfid=TfidfVectorizer(analyzer='word',stop_words='english')


# In[56]:


matrix=tfid.fit_transform(df['text'])


# In[57]:


similer=cosine_similarity(matrix)


# In[58]:


similer[1]


# In[59]:


df[df['song']=='Six Inch Gold Blade'].index[0]


# # RECOMMENDER FUNCTION
# 

# In[60]:


def recommender(song_name):
    idx=df[df['song']==song_name].index[0]
    distance=sorted(list(enumerate(similer[idx])),reverse=True,key=lambda x:x[1])
    song=[]
    for s_id in distance[1:5]:
        song.append(df.iloc[s_id[0]].song)
        return song


# In[61]:


recommender("Come With Me")


# In[ ]:





# In[ ]:




