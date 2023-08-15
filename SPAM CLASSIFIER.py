#!/usr/bin/env python
# coding: utf-8

# In[227]:


import os
import nltk
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm
from numpy.linalg import eig
import math
import random
import sys
import seaborn as sns
from random import sample
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sortedcontainers import SortedList, SortedSet, SortedDict


# # MAKING THE SYSTEM PATH FOR READING THE SPAM AND HAM

# In[228]:


dataset_spam="C:/Users/bisha/ASSIGNMENT 3/dataset/dataset/spam"


# In[229]:


dataset_ham="C:/Users/bisha/ASSIGNMENT 3/dataset/dataset/ham"


# In[289]:


test="C:/Users/bisha/ASSIGNMENT 3/dataset/dataset/test"


# # COLLECTIONS OF MAILS

# In[267]:


spam = []
ham = []
document=[]


# In[ ]:





# # CHANGE THE DIRECTORY FOR READING THE SPAM DATA AND ITERATE THROUGH ALL THE FILE OF SPAM

# In[231]:


def read_text_file(file_path, storage):
    f = open(file_path, errors="ignore")
    storage.append(f.read())


# In[232]:


def read_spam_data():
    
    os.chdir(dataset_spam)
    for file in os.listdir():
            read_text_file(file, spam)


# # CHANGE THE DIRECTORY FOR READING THE HAM DATA AND  ITERATE THROUGH ALL THE FILE OF HAM

# In[233]:


def read_ham_data():
          
    os.chdir(dataset_ham)
    for file in os.listdir():
            read_text_file(file, ham)


# In[ ]:





# In[ ]:





# # TEXT CLEANING FOR REMOVING THINGS LIKE PUNCTUATION ETC.

# In[234]:


#text cleaning   
def punctuation_and_stopwords_remove(email):
    
    email_no_punctuation = [ch for ch in email if ch not in string.punctuation]
    email_no_punctuation_no_stopwords = [word.lower() for word in email_no_punctuation if word.lower() not in stopwords.words("english")]
    email_no_punctuation_no_stopwords_isalpha =  [word.lower() for word in email_no_punctuation_no_stopwords if (word.isalpha() == True)]
    return email_no_punctuation_no_stopwords_isalpha
    


# # CLEANING THE  DATA

# In[235]:


def clean_data():
    for i in range(len(spam)):
        nltk_tokens =  nltk.word_tokenize(spam[i])
        spam[i] = punctuation_and_stopwords_remove(nltk_tokens)
    
    for i in range(len(ham)):
        nltk_tokens =  nltk.word_tokenize(ham[i])
        ham[i] = punctuation_and_stopwords_remove(nltk_tokens)


# # CREATING A COMMON COLLECTION OF ALL SPAM AND HAM DATA IN A COMMON DOCUMENTS NAMED VECTORIZER

# In[236]:


def create_vectorizer():
    
    for i in range(len(spam)):
        document.append(" ".join(spam[i]))
            
    for i in range(len(ham)):
        document.append(" ".join(ham[i]))


# In[268]:


read_spam_data()
read_ham_data()
clean_data()
create_vectorizer()


# In[269]:


len(document)


# # ITS BASICALLY COUNTING THE WORDS

# In[270]:


vectorizer = CountVectorizer()
vectorizer.fit(document)
#print("Vocabulary: ", vectorizer.vocabulary_)
header = vectorizer.vocabulary_.keys()
header = list(header)
vector = vectorizer.transform(document)

X = vector.toarray()


# In[271]:


X.shape


# In[272]:


data_spam = 0
data_ham = 0
for i in range(len(spam)):
    data_spam += X[i]


# In[273]:


data_nonspam = 0
for i in range(len(ham)):
    data_nonspam += X[i + len(spam)]


# In[274]:


data_spam = data_spam + np.ones(X.shape[1])
data_nonspam = data_nonspam + np.ones(X.shape[1])
data_spam = data_spam / (len(spam) + 1)
data_nonspam = data_nonspam / (len(ham) + 1)
data_spam = data_spam / sum(data_spam)
data_nonspam = data_nonspam/sum(data_nonspam)


# In[275]:


p_ml = len(ham)/(len(spam)+len(ham))
q_ml = 1 - p_ml


# # HERE WE ARE BUILDING THE SEPERATOR

# In[276]:


s = np.zeros(X.shape[1])


# In[277]:


for i in range(X.shape[1]):
    nom = data_spam[i]*(1-data_nonspam[i])
    denom = data_nonspam[i] * (1 - data_spam[i])
    s[i] = math.log(nom/denom)


# # HERE WE ARE BUILDING THE BIAS TERM

# In[278]:


bias_term = math.log(p_ml/q_ml)
for i in range(X.shape[1]):
    bias_term += math.log((1 - data_spam[i])/(1 - data_nonspam[i]))  


# In[279]:


decision = []


# In[280]:


len(decision)


# In[281]:


y = []
for i in range(len(spam)):
    y.append(1)
for i in range(len(ham)):
    y.append(0)


# In[282]:


for i in range(X.shape[0]):
    d = np.matmul(s.T, (X[i]/sum(X[i]))) + bias
    if(d > 0):
        decision.append(1)
    else:
        decision.append(0)


# In[283]:


len(X.shape)


# In[284]:


len(spam)
len(ham)


# In[285]:


count = 0
for i in range(len(decision)):
    if(decision[i] != y[i]):
        count += 1


# # FINDING AND PRINTING THE ACCURACY OF TRAINING DATA

# In[287]:


accuracy = (X.shape[0]-count)/X.shape[0]
print('training accuracy', accuracy * 100, "%")  


# In[290]:


os.chdir(test) 
test_mail_data = []
test_mail_filename = []


# # READING THE TEST MAIL

# In[291]:


for file in os.listdir():
    test_mail_filename.append(file)
    read_text_file(file, test_mail_data)


# # CLEANING THE TEST MAIL

# In[293]:


for i in range(len(test_mail_data)):
        nltk_tokens =  nltk.word_tokenize(test_mail_data[i])
        test_mail_data[i] = punctuation_and_stopwords_remove(nltk_tokens)
        test_mail_data[i] = " ".join(test_mail_data[i])


# # CONVERTING EMAIL INTO A VECTOR USING THE SAME VECTORIZING INSTANCE

# In[294]:


test_vector = vectorizer.transform(test_mail_data).toarray()
test_predict = []


# In[297]:


for i in range(len(test_vector)):
    d = np.matmul(s.T, (test_vector[i]/sum(test_vector[i]))) + bias
    if(d > 0):
        test_predict.append(1)
    else:
        test_predict.append(0)


# In[298]:


for i in range(len(test_predict)):
    res = ''
    if(test_predict[i] == 1):
        res = 'spam'
    else: 
        res = 'non-spam'
    print(test_mail_filename[i], 'is predicted as', res)





