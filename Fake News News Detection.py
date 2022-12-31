#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# In[3]:


df=pd.read_csv("C:\\Users\\grsam\\Desktop\\train.csv")


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


plt.figure(figsize=(16,9))
sns.countplot(df.label)


# In[7]:


df['message']=df['author']+' '+df['title']


# In[8]:


print(df['message'])


# In[9]:


X = df.drop(columns='label', axis=1)
Y = df['label']


# In[10]:


print(X)
print(Y)


# In[11]:


pd=PorterStemmer()


# In[12]:


def stemming(message):
    review = re.sub('[^a-zA-Z]',' ',str(message))
    review = review.lower()
    review = review.split()
    review = [pd.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review


# In[ ]:


df['message'] = df['message'].apply(stemming)


# In[13]:


print(df['message'])


# In[14]:


X=df['message'].values
Y=df['label'].values


# In[15]:


print(X)


# In[16]:


print(Y)


# In[17]:


Y.shape


# In[18]:


vectorizer=TfidfVectorizer()
vectorizer.fit(X)

X=vectorizer.transform(X)


# In[19]:


print(X)


# In[20]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=3)


# In[21]:


from sklearn import metrics
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[1]:


#support vector machine
clf=svm.SVC(kernel='linear')
clf.fit(X_train,Y_train)
svm_ac=clf.predict(X_test)


# In[23]:


print("Accuracy Score using Support Vector MAchine",accuracy_score(Y_test,svm_ac))


# In[24]:


cm = metrics.confusion_matrix(Y_test, svm_ac)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])


# In[25]:


print(classification_report(Y_test,svm_ac))


# In[26]:


#decision tree
DT=DecisionTreeClassifier(max_depth=3,random_state=2)
DT.fit(X_train,Y_train)
dlt_pre=DT.predict(X_test)


# In[27]:


acc_dt=accuracy_score(Y_test,dlt_pre)


# In[28]:


print("Accuracy score of Decision Tree Model",acc_dt)


# In[29]:


cm = metrics.confusion_matrix(Y_test, dlt_pre)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])


# In[30]:


dlt_pre= DT.predict(X_test)


# In[31]:


print(classification_report(Y_test,dlt_pre))


# In[32]:


#multinomial algorithm
classifier=MultinomialNB()


# In[40]:


classifier.fit(X_train, Y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(Y_test, pred)
print("accuracy:   %0.4f" % score)


# In[34]:


cm = metrics.confusion_matrix(Y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])


# In[35]:


print(classification_report(Y_test,pred))


# In[36]:


#PASSIVE AGGRESSIVE CLASSIFIER ALGORITHM
linear_clf = PassiveAggressiveClassifier(max_iter=50)
linear_clf.fit(X_train, Y_train)
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(Y_test, pred)
print("accuracy:   %0.3f" % score)


# In[37]:


cm = metrics.confusion_matrix(Y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])


# In[38]:


print(classification_report(Y_test,pred))


# In[ ]:





# In[ ]:




