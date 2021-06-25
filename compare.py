import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import string
import hypeparameter_tune as hpt 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

max = dict()

def stemmer (text):
    text = text.split()
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words

def text_process(text):   
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]   
    return " ".join(text)

def train_classifier(clf, feature_train, labels_train):    
    clf.fit(feature_train, labels_train)
    
def predict_labels(clf, features):
    return (clf.predict(features))

def dataswo(files):
    fpath = files['path']
    df1 = pd.read_csv(fpath)    
    df1 = df1.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
    text_feat = df1['text'].copy()
    text_feat = text_feat.apply(text_process)
    vectorizer = TfidfVectorizer("english")
    features = vectorizer.fit_transform(text_feat)
    features_train, features_test, labels_train, labels_test = train_test_split(features, df1['spam'], test_size=0.3, random_state=111)
    svc = SVC()
    mnb = MultinomialNB()
    dtc = DecisionTreeClassifier(random_state=111)

    clfs = {'SVC' : svc,'NB': mnb, 'DT': dtc}

    pred_scores = []
    itm = 0
    for k,v in clfs.items():
        train_classifier(v, features_train, labels_train)
        pred = predict_labels(v,features_test)
        itm = itm + 1
        pred_scores.append((itm, [k, accuracy_score(labels_test,pred)]))

    df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Classifier','Score1'])
    print(df)

    return df

def datasw(files):
    global max
    fpath = files['path']
    df1 = pd.read_csv(fpath)    
    df1 = df1.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
    text_feat = df1['text'].copy()
    text_feat = text_feat.apply(text_process)
    vectorizer = TfidfVectorizer("english")
    features = vectorizer.fit_transform(text_feat)
    features_train, features_test, labels_train, labels_test = train_test_split(features, df1['spam'], test_size=0.3, random_state=111)

    svc = SVC(kernel='sigmoid', gamma=1.0)
    mnb = MultinomialNB(alpha=0.2)
    dtc = DecisionTreeClassifier(min_samples_split=9, random_state=111)

    clfs = {'SVC' : svc, 'NB': mnb, 'DT': dtc}

    pred_scores = []
    itm = 0
    for k,v in clfs.items():
        train_classifier(v, features_train, labels_train)
        pred = predict_labels(v,features_test)
        itm = itm + 1
        pred_scores.append((itm, [accuracy_score(labels_test,pred)]))

    df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score2'])
    print(df)

    return df

def datastem(files):
    fpath = files['path']
    df1 = pd.read_csv(fpath)    
    df1 = df1.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
    text_feat = df1['text'].copy()
    text_feat = text_feat.apply(stemmer)
    vectorizer = TfidfVectorizer("english")
    features = vectorizer.fit_transform(text_feat)
    features_train, features_test, labels_train, labels_test = train_test_split(features, df1['spam'], test_size=0.3, random_state=111)

    svc = SVC(kernel='sigmoid', gamma=1.0)
    mnb = MultinomialNB(alpha=0.2)
    dtc = DecisionTreeClassifier(min_samples_split=9, random_state=111)

    clfs = {'SVC' : svc,'NB': mnb, 'DT': dtc}

    pred_scores = []
    itm = 0
    for k,v in clfs.items():
        train_classifier(v, features_train, labels_train)
        pred = predict_labels(v,features_test)
        itm = itm + 1
        pred_scores.append((itm, [accuracy_score(labels_test,pred)]))

    df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score3'])
    print(df)

    return df

def datalen(files):
    fpath = files['path']
    df1 = pd.read_csv(fpath)    
    df1 = df1.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
    text_feat = df1['text'].copy()
    text_feat = text_feat.apply(text_process)
    vectorizer = TfidfVectorizer("english")
    features = vectorizer.fit_transform(text_feat)
    features_train, features_test, labels_train, labels_test = train_test_split(features, df1['spam'], test_size=0.3, random_state=111)
        
    svc = SVC(kernel='sigmoid', gamma=1.0)
    mnb = MultinomialNB(alpha=0.2)
    dtc = DecisionTreeClassifier(min_samples_split=9, random_state=111)

    clfs = {'SVC' : svc, 'NB': mnb, 'DT': dtc}

    pred_scores = []
    itm = 0
    for k,v in clfs.items():
        train_classifier(v, features_train, labels_train)
        pred = predict_labels(v,features_test)
        itm = itm + 1
        pred_scores.append((itm, [accuracy_score(labels_test,pred)]))

    df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score4'])
    print(df)

    return df

def cll_all(files):
    #get_values(files)
    df1 = dataswo(files)
    df2 = datasw(files)
    df3 = datastem(files)
    df4 = datalen(files)
    df = pd.concat([df1, df2, df3, df4], axis=1, join='inner')
    
    return df

def get_values(files):
    global max
    max = hpt.menu(files)
