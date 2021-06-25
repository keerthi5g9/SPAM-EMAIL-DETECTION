import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk 
import re
import string
from nltk.stem import SnowballStemmer
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score,  precision_score, recall_score
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.tree import DecisionTreeClassifier 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.utils import shuffle
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import hypeparameter_tune as hp
from pickle import dump, load
import warnings
warnings.filterwarnings("ignore")


def dataset_prop(files):
    c_dir = dict()
    fpath = files['path']
    ds = pd.read_csv(fpath)
    shape = ds.shape
    colmns = ds.columns
    print("Shape : ", shape)
    print("Columns : ", colmns)
    print("Head (5) : ", ds.head(5))
    print("Tail (5) : ", ds.tail(5))

    count_Class=pd.value_counts(ds["spam"], sort= True)

    count1 = Counter(" ".join(ds[ds['spam']==0]["text"]).split()).most_common(20)
    df1 = pd.DataFrame.from_dict(count1)
    df1 = df1.rename(columns={0: "words in non-spam", 1 : "count"})
    count2 = Counter(" ".join(ds[ds['spam']==1]["text"]).split()).most_common(20)
    df2 = pd.DataFrame.from_dict(count2)
    df2 = df2.rename(columns={0: "words in spam", 1 : "count_"})
    c_dir.update({
        'Ham': count1,
        'Spam': count2,
        'Shape': shape,
        'Columns': colmns,
        'CClass': count_Class
    })
    return c_dir
    

def model_assessment(u_classify, y_data, predicted_class):   
    mod_ass = dict()
    conf_mtrx = confusion_matrix(y_data,predicted_class)
    acc_score = accuracy_score(y_data,predicted_class)
    prec_score = precision_score(y_data,predicted_class)
    rec_score = recall_score(y_data,predicted_class)
    f1_scr = f1_score(y_data,predicted_class)
    print('confusion matrix : ',conf_mtrx)
    print('accuracy : ',acc_score)
    print('precision : ',prec_score)
    print('recall : ',rec_score)
    print('f-Score : ',f1_scr)
    mod_ass.update({
        "confusion" : conf_mtrx,
        "accuracy" : acc_score,
        "precision" : prec_score,
        "recall" : rec_score,
        "f1" : f1_scr
    })
    return mod_ass

def tra_tt_split(files):
    fpath = files['path']
    datasets = pd.read_csv(fpath)
    dsets = shuffle(datasets)
    d_train, d_test, l_train, l_test = model_selection.train_test_split(datasets['text'],datasets['spam'],test_size=0.33, random_state=42)
    s_dict = dict()
    s_dict.update({
        "d_train": d_train,
        "d_test": d_test,
        "l_train": l_train,
        "l_test": l_test
    })    
    return s_dict

def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

def features_transform(mail, dtrain, var1):
    bow = CountVectorizer(analyzer=process_text)
    bow_transformer = bow.fit(dtrain)
    messages_bow = bow_transformer.transform(mail)
    print('\nsparse matrix shape:', messages_bow.shape)
    print('number of non-zeros:', messages_bow.nnz) 
    print('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])), '\n')
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)
    if var1 in ["MNB", "SVM", "DecT", "KNC"]:
        sav_fit(vr1=bow.vocabulary_, var=var1)
    return messages_tfidf

def sav_fit(vr1, var):
    fname = 'vect'+str(var)+'.pkl'
    dump(vr1,open(fname, 'wb'))

def train_classifier(clf, f_train, l_train, typ):
    model = clf.fit(f_train, l_train)
    file = str(typ)+'.pkl'
    print("name : ",file)
    dump(model, open(file, 'wb'))


def Multi_NB(files, var):
    res_df = list()
    fpath = files['path']
    datasets = pd.read_csv(fpath)
    datasets = datasets.dropna()
    datasets.drop_duplicates(inplace=True)
    dsets = shuffle(datasets)
    if int(var) == 1:
        d_train, d_test, l_train, l_test = model_selection.train_test_split(datasets['text'],datasets['spam'],test_size=0.33, random_state=42)
        dtrain_msg = features_transform(mail=d_train, dtrain=d_train, var1='MNB')
        alp_dict = hp.MNB_class(files)
        modelMNB = naive_bayes.MultinomialNB(alpha=alp_dict['alpha'])    
        train_classifier(modelMNB, dtrain_msg, l_train, typ="MNB")
        modelMNB.fit(dtrain_msg, l_train)
        pred_train = modelMNB.predict(dtrain_msg)
        mnb_dict = model_assessment(u_classify='MultiNB', y_data=l_train, predicted_class=pred_train)
        return mnb_dict
    elif int(var) == 2:
        print("Inside testing phase : ")
        d_test = dsets['text']    
        modelMNB = load(open('MNB.pkl', 'rb'))
        vect = load(open('vectMNB.pkl', 'rb'))
        tf = TfidfTransformer()
        load_vect = CountVectorizer(vocabulary=vect)
        last = len(d_test)
        for i in range(0, last):
            if d_test.get(i) != None:
                tup = [d_test[i],]
                dtest_msg = tf.fit_transform(load_vect.fit_transform(tup))
                pred = modelMNB.predict_proba(dtest_msg)
                pred_test = modelMNB.predict(dtest_msg)
                res_df.append((i+1, [pred[0][0], pred[0][1], pred_test[0]]))
        df = pd.DataFrame.from_items(res_df, orient='index', columns=['Class O', 'Class 1', 'Result'])
        print(df.head(20))
        return df
    else:
        d_train, d_test, l_train, l_test = model_selection.train_test_split(datasets['text'],datasets['spam'],test_size=0.33, random_state=42)
        modelMNB = load(open('MNB.pkl', 'rb'))
        vect = load(open('vectMNB.pkl', 'rb'))
        tf = TfidfTransformer()
        load_vect = CountVectorizer(vocabulary=vect)
        dtest_msg = tf.fit_transform(load_vect.fit_transform(d_test))
        pred_test = modelMNB.predict(dtest_msg)
        mnb_dict = model_assessment(u_classify='MultiNB', y_data=l_test, predicted_class=pred_test)
        return mnb_dict
        
def SVM(files, var):
    res_df = list()
    fpath = files['path']
    datasets = pd.read_csv(fpath)
    datasets = datasets.dropna()
    datasets.drop_duplicates(inplace=True)
    dsets = shuffle(datasets)
    if int(var) == 1:
        d_train, d_test, l_train, l_test = model_selection.train_test_split(datasets['text'],datasets['spam'],test_size=0.33, random_state=42)
        dtrain_msg = features_transform(mail=d_train, dtrain=d_train, var1='SVM')
        krnl = hp.SVC_class(files)
        model_svm=svm.SVC(kernel=krnl['kernel'], gamma=krnl['gamma'], probability=True)
        train_classifier(model_svm, dtrain_msg, l_train, typ="SVM")
        model_svm.fit(dtrain_msg, l_train)
        pred_train = model_svm.predict(dtrain_msg)
        mnb_dict = model_assessment(u_classify='SVM', y_data=l_train, predicted_class=pred_train)
        return mnb_dict
    elif int(var) == 2:
        print("Inside testing phase : ")
        d_test = dsets['text']   
        model_svm = load(open('SVM.pkl', 'rb'))
        vect = load(open('vectSVM.pkl', 'rb'))
        tf = TfidfTransformer()
        load_vect = CountVectorizer(vocabulary=vect)
        last = len(d_test)
        for i in range(0, last):
            if d_test.get(i) != None:
                tup = [d_test[i],]
                dtest_msg = tf.fit_transform(load_vect.fit_transform(d_test))
                pred_test = model_svm.predict(dtest_msg)        
                pred = model_svm.predict_proba(dtest_msg)
                res_df.append((i+1, [pred[0][0], pred[0][1], pred_test[0]]))
        df = pd.DataFrame.from_items(res_df, orient='index', columns=['Class O', 'Class 1', 'Result'])
        print(df.head(15))
        return df
    else:
        d_train, d_test, l_train, l_test = model_selection.train_test_split(datasets['text'],datasets['spam'],test_size=0.33, random_state=42)
        model_svm = load(open('SVM.pkl', 'rb'))
        vect = load(open('vectSVM.pkl', 'rb'))
        tf = TfidfTransformer()
        load_vect = CountVectorizer(vocabulary=vect)
        dtest_msg = tf.fit_transform(load_vect.fit_transform(d_test))
        pred_test = model_svm.predict(dtest_msg)
        mnb_dict = model_assessment(u_classify='SVM', y_data=l_test, predicted_class=pred_test)
        return mnb_dict

def D_Tree(files, var):
    res_df = list()
    fpath = files['path']
    datasets = pd.read_csv(fpath)
    datasets = datasets.dropna()
    datasets.drop_duplicates(inplace=True)
    dsets = shuffle(datasets)
    if int(var) == 1:
        d_train, d_test, l_train, l_test = model_selection.train_test_split(datasets['text'],datasets['spam'],test_size=0.33, random_state=42)
        dtrain_msg = features_transform(mail=d_train, dtrain=d_train, var1='DecT')
        mss = hp.DT_class(files)
        model_dtree=DecisionTreeClassifier(min_samples_split=mss['min_samples'], random_state=111)
        train_classifier(model_dtree, dtrain_msg, l_train, typ="DecT")
        model_dtree.fit(dtrain_msg, l_train)
        pred_train = model_dtree.predict(dtrain_msg)
        mnb_dict = model_assessment(u_classify='Decision Tree', y_data=l_train, predicted_class=pred_train)
        return mnb_dict    
    elif int(var) == 2:    
        print("Inside testing phase : ")
        d_test = dsets['text']
        model_dtree = load(open('DecT.pkl', 'rb'))
        vect = load(open('vectDecT.pkl', 'rb'))
        tf = TfidfTransformer()
        load_vect = CountVectorizer(vocabulary=vect)
        last = len(d_test)
        for i in range(0, last):
            if d_test.get(i) != None:
                tup = [d_test[i],]
                dtest_msg = tf.fit_transform(load_vect.fit_transform(tup))
                pred_test = model_dtree.predict(dtest_msg)
                pred =model_dtree.predict_proba(dtest_msg)
                res_df.append((i+1, [pred[0][0], pred[0][1], pred_test[0]]))
        df = pd.DataFrame.from_items(res_df, orient='index', columns=['Class O', 'Class 1', 'Result'])
        print(df.head(15))
        return df
    else:
        d_train, d_test, l_train, l_test = model_selection.train_test_split(datasets['text'],datasets['spam'],test_size=0.33, random_state=42)
        model_dtree = load(open('DecT.pkl', 'rb'))
        vect = load(open('vectDecT.pkl', 'rb'))
        tf = TfidfTransformer()
        load_vect = CountVectorizer(vocabulary=vect)
        dtest_msg = tf.fit_transform(load_vect.fit_transform(d_test))
        pred_test = model_dtree.predict(dtest_msg)
        mnb_dict = model_assessment(u_classify='Decision Tree', y_data=l_test, predicted_class=pred_test)
        return mnb_dict

def menu(files):
    menu = '''
                Basic Classifiers : 
            ---------------------------
            1. MultiNomial Naive Bayes
            2. Support Vector Machine
            3. Decision Tree (criteria : 'gini'/'entropy')
           '''
    print(menu)
    print("\n\Basic Classifiers training......")
    print("\Multinomial Naive Bayes Classifier Training.....")
    Multi_NB(files, 1)
    print("\Support Vector Classifier Training.....")
    SVM(files, 1)
    print("\Decision Tree Classifier Training.....")
    D_Tree(files, 1)
