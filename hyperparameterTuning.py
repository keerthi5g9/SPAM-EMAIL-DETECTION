import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

max_dict = dict()

def process_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    words = ""
    for i in text:
        stemmer = SnowballStemmer("english")
        words += (stemmer.stem(i)) + " "
    return words

def SVC_class(files):
    global max_dict
    fpath = files['path']
    datasets = pd.read_csv(fpath)
    text_feat = datasets['text'].copy()
    text_feat = text_feat.apply(process_text)
    vectorizer = TfidfVectorizer("english")
    features = vectorizer.fit_transform(text_feat)
    features_train, features_test, labels_train, labels_test = train_test_split(features, datasets['spam'], test_size=0.3, random_state=111)

    best_ser = dict()
    pred_scores = []
    krnl = {'rbf': 'rbf', 'polynominal': 'poly', 'sigmoid': 'sigmoid'}
    for k, v in krnl.items():
        itm = 0
        for i in np.linspace(0.05, 1, num=20):
            svc = SVC(kernel=v, gamma=i)
            svc.fit(features_train, labels_train)
            pred = svc.predict(features_test)
            itm = itm + 1
            pred_scores.append((itm, [k, i, accuracy_score(labels_test, pred)]))

    df = pd.DataFrame.from_items(pred_scores, orient='index', columns=['Kernel', 'Gamma', 'Score'])
    df['Score'].plot(kind='line', figsize=(11, 6), ylim=(0.8, 1.0))
    print("SVC : \n", df[df['Score'] == df['Score'].max()])
    sbst = df[df['Score'] == df['Score'].max()].head(1)
    sbst_id = sbst.first_valid_index()
    ker = sbst.get_value(sbst_id, 'Kernel')
    gam = sbst.get_value(sbst_id, 'Gamma')
    score = sbst.get_value(sbst_id, 'Score')
    best_ser.update({
        'kernel': ker,
        'gamma': gam,
        'score': score
    })
    max_dict['Support Vector Classifier'] = best_ser
    return best_ser


def MNB_class(files):
    global max_dict
    fpath = files['path']
    datasets = pd.read_csv(fpath)
    text_feat = datasets['text'].copy()
    text_feat = text_feat.apply(process_text)
    vectorizer = TfidfVectorizer("english")
    features = vectorizer.fit_transform(text_feat)
    features_train, features_test, labels_train, labels_test = train_test_split(features, datasets['spam'], test_size=0.3, random_state=111)

    best_ser = dict()
    pred_scores = []
    itm = 0
    for i in np.linspace(0.05, 1, num=20):
        mnb = MultinomialNB(alpha=i)
        mnb.fit(features_train, labels_train)
        pred = mnb.predict(features_test)
        itm = itm + 1
        pred_scores.append((itm, [i, accuracy_score(labels_test, pred)]))

    df = pd.DataFrame.from_items(pred_scores, orient='index', columns=['Alpha', 'Score'])
    df.plot(figsize=(11, 6))
    print("MNB : \n", df[df['Score'] == df['Score'].max()])
    sbst = df[df['Score'] == df['Score'].max()].head(1)
    sbst_id = sbst.first_valid_index()
    alp = sbst.get_value(sbst_id, 'Alpha')
    score = sbst.get_value(sbst_id, 'Score')
    best_ser.update({
        'alpha': alp,
        'score': score
    })
    max_dict['Multinomial Naive Bayes Classifier'] = best_ser
    return best_ser


def DT_class(files):
    global max_dict
    fpath = files['path']
    datasets = pd.read_csv(fpath)
    text_feat = datasets['text'].copy()
    text_feat = text_feat.apply(process_text)
    vectorizer = TfidfVectorizer("english")
    features = vectorizer.fit_transform(text_feat)
    features_train, features_test, labels_train, labels_test = train_test_split(features, datasets['spam'], test_size=0.3, random_state=111)

    best_ser = dict()
    pred_scores = []
    itm = 0
    for i in range(2, 21):
        dtc = DecisionTreeClassifier(min_samples_split=i, random_state=111)
        dtc.fit(features_train, labels_train)
        pred = dtc.predict(features_test)
        itm = itm + 1
        pred_scores.append((itm, [i, accuracy_score(labels_test, pred)]))

    df = pd.DataFrame.from_items(pred_scores, orient='index', columns=['Min_Samples', 'Score'])
    df.plot(figsize=(11, 6))
    print("DT : \n", df[df['Score'] == df['Score'].max()])
    sbst = df[df['Score'] == df['Score'].max()].head(1)
    sbst_id = sbst.first_valid_index()
    nt = sbst.get_value(sbst_id, 'Min_Samples')
    score = sbst.get_value(sbst_id, 'Score')
    best_ser.update({
        'min_samples': nt,
        'score': score
    })
    max_dict['Decision Tree Classifier'] = best_ser
    return best_ser


def KNN_class(files):
    global max_dict
    fpath = files['path']
    datasets = pd.read_csv(fpath)
    text_feat = datasets['text'].copy()
    text_feat = text_feat.apply(process_text)
    vectorizer = TfidfVectorizer("english")
    features = vectorizer.fit_transform(text_feat)
    features_train, features_test, labels_train, labels_test = train_test_split(features, datasets['spam'],
                                                                                test_size=0.3, random_state=111)

    best_ser = dict()
    pred_scores = []
    itm = 0
    for i in range(3, 61):
        knc = KNeighborsClassifier(n_neighbors=i)
        knc.fit(features_train, labels_train)
        pred = knc.predict(features_test)
        itm = itm + 1
        pred_scores.append((itm, [i, accuracy_score(labels_test, pred)]))

    df = pd.DataFrame.from_items(pred_scores, orient='index', columns=['N_Neighours', 'Score'])
    df.plot(figsize=(11, 6))
    print("DT : \n", df[df['Score'] == df['Score'].max()])
    sbst = df[df['Score'] == df['Score'].max()].head(1)
    sbst_id = sbst.first_valid_index()
    nt = sbst.get_value(sbst_id, 'N_Neighours')
    score = sbst.get_value(sbst_id, 'Score')
    best_ser.update({
        'n_neighbors': nt,
        'score': score
    })
    max_dict['KNN Classifier'] = best_ser
    return best_ser

def menu(files):
    print("\n\nBasic Classifiers Hyperparameter tuning......")
    print("\nSupport Vector Classifier Tuning.....")
    SVC_class(files)
    print("\nMultinomial Naive Bayes Classifier Tuning.....")
    MNB_class(files)
    print("\nDecision Tree Classifier Tuning.....")
    DT_class(files)
    print("\nKNN Classifier Tuning.....")
    KNN_class(files)
    return max_dict