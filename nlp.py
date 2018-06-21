#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 21:16:49 2018

@author: princesegzy01
"""

import os
import csv

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

#import numpy as np
import pandas as pd



#Cleaning the text
import re

from nltk.corpus import stopwords
#we need to download stopword list to filter out 
#nltk.download('stopwords')
    


#ibrary for stemming
#from nltk.stem.porter import PorterStemmer
#ps = PorterStemmer()


#from nltk.stem import SnowballStemmer

#import lematizer too as against stemming
from nltk.stem import WordNetLemmatizer

#download wordnet dictionary
#nltk.download('wordnet')
wordnet_Lemmatizer = WordNetLemmatizer()


#import pos tagger
#because for our lemmatization to function perfectly,
# we need to supply the part of speech we are lemmatizing for 
#using the pos_tag class under nltk
#from nltk import word_tokenize, pos_tag


#this is for tokenizing sentence.. will be good for text summarization
#from nltk.tokenize import sent_tokenize 


#download the universet tag words like noun, pronoun as against usin NN, VBD
#nltk.download('brown')
#nltk.download('universal_tagset')
nltk.corpus.brown.tagged_words(tagset='universal')


#we are using the averaged_perceptron_tagger algorithm to tag our tokens
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')
#print(pos_tag(word_tokenize("Dive into NLTK: Part-of-speech tagging and POS Tagger")))





def CustomStopWord():
    
    # remove all word in the stopword list and leave only 4 letter words leaving oout a list of exception below
    stopWordList = stopwords.words('english')
    #rem_stopwords = ['Not', 'No', 'None']
    for word in list(stopWordList):  # iterating on a copy since removing will mess things up
        #if word in rem_stopwords:
            #stopWordList.remove(word)
            
        exception_list = ['no', 'not', 'none']
        if len(word) > 4 or word in exception_list:
            stopWordList.remove(word)
    return stopWordList
    
def WsDisAmbiguation(sent):
    
    #sentence = "man took course subject"
    #sent = sent.split()
    #sent = word_tokenize(sentence)    
    #sent = word_tokenize("We are going out. Just you and me.")
    total_count_word = len(sent) -1
    
    finalWord = []
    
    for num in range(total_count_word):
    
        for word in sent:
            
            word1 = sent[num]
            word2 = word 
            
            #
            if word1 == word2 :
                continue
            
            #pos TAG for the 2 words to compare
            #pos_word_tagged_1 = nltk.pos_tag(word_tokenize(word1))[0][1]
            #pos_word_tagged_2 = nltk.pos_tag(word_tokenize(word2))[0][1]
            
            
            if len(wordnet.synsets(word1, pos='n')) == 0 or len(wordnet.synsets(word2, pos='n')) == 0  :
                finalWord.append(word1)
                continue
            
            
            #Dont compare similarity if they are not of the same type
            #if pos_word_tagged_1 == pos_word_tagged_2 :
            
            word1_syn = wordnet.synsets(word1, pos = 'n')[0]
            word2_syn = wordnet.synsets(word2, pos='n')[0]
             
            result = word1_syn.lch_similarity(word2_syn)

            if result < 2.5 :
                finalWord.append(word1)
                continue
                
            
                
    response = " ".join(sorted(set(finalWord), key=finalWord.index))
    #cprint(response)
    return response

def WriteDictToCSV(csv_file,csv_columns,dict_data):
    #try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in dict_data:
            writer.writerow(data)
    #except IOError as (errno, strerror):
            #print("I/O error({0}): {1}".format(errno, strerror))    
    #return            


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None # for easy if-statement 


#get all the content in the datasets
df =  pd.read_csv('nigeria_security.csv', encoding = 'latin-1')

train_dataset_filter = pd.DataFrame()
train_dataset_filter['text'] = df.Tweet

#train_dataset.drop['id','date','query_string','user']


#dictionary array to hold individual dictionary result
#dict_data_csv = []



#corpus = []
corpus_lemma = []

#lets use Vader Sentimental Analysis to analyse our tweets
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

#Vader Classifier
vader_dataset = train_dataset_filter
vader_dataset['vader'] = ""
vader_dataset['result'] = ""

#loop through all that values in the csv
#to classify the vader tweets

#0 - Negative
#1 - Positive
#2 - Neutral
for i  in range(0,len(train_dataset_filter)):
    #Remove all other characters excepts for a-z 
    
    original_tweet = train_dataset_filter.iloc[i][0]
    
    ss = sid.polarity_scores(original_tweet)
    
    
    if ss["compound"] == 0.0: 
        vader_dataset['vader'][i] = 2
    elif ss["compound"] > 0.0:
        vader_dataset['vader'][i] = 1
    else:
        vader_dataset['vader'][i] = 0
  
  
# plot the vader bar chart
df_chart = vader_dataset 
f = pd.DataFrame()
f['class'] = df_chart.vader.unique()
f['total'] = df_chart.groupby(['vader'])['result'].agg('count')

ax = f.plot.bar(x='class', y='total', rot=0, legend = False)
ax.set(xlabel='Sentiment Review', ylabel='Result')
ax.set_xticklabels(['Negative','Positive', 'Neutral'])

        
#loop through all that values in the csv
for i  in range(0,len(vader_dataset)):
    
    #Remove all other characters excepts for a-z 
    #tweet = re.sub('[^a-zA-Z]', ' ' , dataset['Tweet'][i])
    tweet = re.sub('[^a-zA-Z]', ' ' , vader_dataset.iloc[i][0])
    tweet = re.sub(r'http\S+', '', tweet)



    #Change all tweets to lower case
    tweet = tweet.lower()
    
    #Remove Text that dosent help our machine learning algorithm
    #this is also useful to reduce our sparse matrix
    
    
    #loop through each tweets by splitting each tweets and check if stop words is available for us to filter out
    #tweet = tweet.split()
    
    tweet = nltk.word_tokenize(tweet)
    
    #tweet = [word for word in tweet if not word in set(stopwords.words('english'))]
    
    tweet = [word for word in tweet if not word in set(CustomStopWord())]
    
    
    tweet = WsDisAmbiguation(tweet)
    
    tweet = nltk.word_tokenize(tweet)

    
    tweet_lemma = tweet
    
    #next stage is stemming which take roots of all the words
    #stemming algorithm sometimes do too much or too little
    #tweet = [ps.stem(word) for word in tweet]
    
    tweet_lemma = [wordnet_Lemmatizer.lemmatize(word) for word in tweet_lemma]
    
    
    #remove tokens not available in wordnet DICTIONARY
    tweet_lemma = [word for word in tweet_lemma if wordnet.synsets(word)]
    
    #We are supposed to use lematization, but we are facing some isuues below
    #stemming has limits. For example, Porter stems both happiness and happy to happi,
    #while WordNet lemmatizes the two words to themselves.
    #The WordNet lemmatizer also requires specifying the word’s part of speech — otherwise, it assumes the word is a noun.
    #Finally, lemmatization cannot handle unknown words: for example, Porter stems both iphone and iphones to iphon,
    #while WordNet lemmatizes both words to themselves.
    
    #convert our tweets from list back to string
    tweet = ' '.join(tweet)
    tweet_lemma = ' '.join(tweet_lemma)
    
    #add each cleaned tweets to our corpus
    corpus_lemma.append(tweet_lemma)


#station_idx = map(lambda x: corpus_lemma.index(x), corpus_lemma)
#bag_of_words = TfidfVectorizer()
#X = bag_of_words.fit_transform(corpus_lemma)
#chi2score = chi2(X, station_idx)[0]



bag_of_words = TfidfVectorizer()
X = bag_of_words.fit_transform(corpus_lemma)
feature_names = bag_of_words.get_feature_names()

doc = 0
feature_index = X[doc,:].nonzero()[1]
tfidf_scores = zip(feature_index, [X[doc, x] for x in feature_index])



score = []
word = []
for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
    print(w, s)
    x = score.append(s)
    y = word.append(w)

scores = pd.DataFrame({'word' : word, 'score' : score})

scores.plot.bar(x='word', y='score', legend = False, rot=90)
scores.xlabel(x, rotation=90)
scores.set(xlabel='Tokens', ylabel='Weighted Scores')



with open('NlP_Result.csv', 'w') as fp :
    writer = csv.writer(fp, delimiter=',')
    writer.writerow(["Kbest","Feature Extraction","Classifier","Accuracy","stopword removal", "wsd"])  # write header
    
    
    #doc2vec vs Bag of word
    #======================== BAG OF WORDS MODEL : COUNT VECTORIZER ===================================
    #Count Vectorizer  with different classifeir algorithm    
    #Create our bag of word Model for count vectorizer
    
    #total_bag_of_words_model = ["CountVectorizer","TfidfTransformer","TfidfVectorizer","HashingVectorizer","chi2"]
    total_bag_of_words_model = ["CountVectorizer","TfidfVectorizer"]
    
    for bag_of_word_model in total_bag_of_words_model :
        
            
        if bag_of_word_model == "CountVectorizer" :
            
            bag_of_words = CountVectorizer()
        
        if bag_of_word_model == "TfidfTransformer" :
            bag_of_words = TfidfTransformer()
            
        if bag_of_word_model == "TfidfVectorizer" :
            
            bag_of_words = TfidfVectorizer()
            
        if bag_of_word_model == "HashingVectorizer" :
            
            bag_of_words = HashingVectorizer()
        
        #print("")
        #print("")
        #print("#*******   BAG OF WORDS : ", bag_of_word_model , "*****************")
       
        #Bag of word is matrix
        X = bag_of_words.fit_transform(corpus_lemma).toarray()
        y = vader_dataset.iloc[:, 1].values
        y= y.astype('int') 
        
        #Dimension Reductionality
        #Using CH2 as parameter for selecting KBest   
        
        #list = [10,50,100,500,1000,1500,2000,2500,3000,3500,4000,4500,5283]
        list = [5000]
        #list = [1000,3500,5283]
        
        #result = {'SVC': 0, 'RandomForestClassifier' : 0,'GaussianNB' : 0, 'SDGClassifier' : 0 , 'LogisticRegression' : 0,  'KNeighborsClassifier' : 0, 'DecisionTreeClassifier': 0, 'Kbest' : 0, 'Feature Extraction' : '', 'stopword removal' : '', 'wsd' : ''}
        
        #result['wsd'] = "Yes"
        #result['stopword removal'] = "Yes"
        #result['Feature Extraction'] = bag_of_word_model
        
        
    
        for knum in list:
            
            
            
        
            X_new = SelectKBest(chi2, k=knum).fit_transform(X, y)
    
            #split data into test and train
            from sklearn.cross_validation import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.20, random_state = 0)
            
            
            total_classifier = ["GaussianNB","SDGClassifier","LogisticRegression","KNeighborsClassifier","DecisionTreeClassifier","RandomForestClassifier","SVC"]
            
            for classifier_name in total_classifier:
                
                
                
                print("")
                #print("")
                #print("#******* Using  Bag of Word : ", bag_of_word_model , ", & Classifier :  "  , classifier_name , "*****************")
            
                if classifier_name == "GaussianNB" :
                    from sklearn.naive_bayes import GaussianNB
                    classifier = GaussianNB()
                    
                if classifier_name == "SDGClassifier" :
                    from sklearn.linear_model import SGDClassifier
                    classifier = SGDClassifier()
                    
                if classifier_name == "LogisticRegression" :
                    from sklearn.linear_model import LogisticRegression
                    classifier = LogisticRegression()
                    
                if classifier_name == "KNeighborsClassifier" :
                    from sklearn.neighbors import KNeighborsClassifier
                    classifier = KNeighborsClassifier()
                    
                if classifier_name == "DecisionTreeClassifier" :
                    from sklearn.tree import DecisionTreeClassifier
                    classifier = DecisionTreeClassifier()
                    
                if classifier_name == "RandomForestClassifier" :
                    from sklearn.ensemble import RandomForestClassifier
                    classifier = RandomForestClassifier()
                    
                if classifier_name == "SVC" :
                    from sklearn.svm import SVC
                    classifier = SVC()
                    
                
                
                
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                
                #Making the confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                
                #print("Confusion Matrix Output for ",  classifier)
                #print(cm)
                
                accuracy = (cm[0][0] + cm[1][1]) / len(y_test)
                
                #result[classifier_name] = accuracy
                #result['Kbest']  =  knum
                #print(">>> Knum : ", knum, " classifier : ", classifier_name, " bag_of_word_model : ", bag_of_word_model)
                
                #print(">>>>> Knum : ", knum)
                #print("Accuracy : Bag of words : ", bag_of_word_model, ", Classifier " , classifier_name , " :- " ,  accuracy)
                #print(result)
                #print(classification_report(y_test, y_pred))
                #print("#***************   End : Classifier " , classifier_name , " ***********************")
                #print("")
                #print("")
                
            #print(result)
                #dict_data_csv.append(result)
                writer.writerow([knum,bag_of_word_model, classifier_name,accuracy,"yes", "yes"])  # write header
    
                
            #print("#***************   End BAG OF WORDS " , bag_of_word_model , " ***********************")
            #print("")
            print("done")
        
#csv_columns = ['Kbest','SVC','RandomForestClassifier','GaussianNB', 'SDGClassifier', 'LogisticRegression',  'KNeighborsClassifier' , 'DecisionTreeClassifier', 'Feature Extraction', 'stopword removal', 'wsd']
#WriteDictToCSV("Nlp_result_output.csv",csv_columns,dict_data_csv)
        
print("Sentiment Analysis Done")



#tfv=TfidfVectorizer(min_df=0, max_features=3000, strip_accents='unicode',lowercase =True, analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1,1),  use_idf=True,smooth_idf=True, sublinear_tf=True, stop_words = "english")   
#data=tfv.fit_transform(text)

#HashingVectorizer



    
    
