from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

class NaiveBayes:
    """
    Fits a Naive Bayes model using the data in a given dataset for training and validation
    using K-Folds cross validation
    """
    def __init__(self, dataset_name):
        
        gnb = GaussianNB()
        skf = StratifiedKFold(n_splits=3) # 3 folds calculated, resulting in 66/33 train/test split for each

        dataset = pd.read_csv(dataset_name)
        self.symptom_set = dataset.columns[:-1].values.tolist() #Exclude last col, prognosis col

        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        self.num_rows = len(X)
        self.num_cols = len(X[0])

        self.score = 0
        for train_index, test_index in skf.split(X,y): #Iterate through each fold
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = gnb.fit(X, y)
            score = gnb.score(X_test, y_test)
            if (score > self.score): #Keep most accurate model
                self.model = model
                self.score = score
                best_train_index = train_index
                best_test_index = test_index


    """
    Return a dictionary of disease to likelihoods given a list of symptoms
    """
    def get_predictions(self, symptoms):
        X = self.format_symptoms(symptoms) #Uses helper to properly form symptoms
        classes = self.model.classes_ #Possible diseases
        probabilities = self.model.predict_proba([X])
        return dict(zip(classes, probabilities[0])) #gets vector of probabilities for each class (disease)

    """
    Returns one prediction for a disease given a list of symptoms
    """
    def get_prediction(self, symptoms):
        X = self.format_symptoms(symptoms) #Uses helper to properly form symptoms
        return self.model.predict([X])

    """
    Helper function to convert a list of symptoms to properly formed input vector X (containing one hot encoding of symptoms)
    """
    def format_symptoms(self, symptoms):
        symptom_index = dict(zip(self.symptom_set, range(len(self.symptom_set)))) #Convert list to (word -> index) dict

        X = np.zeros(len(self.symptom_set)) 
        for symptom in symptoms:
            if symptom in symptom_index:
                X[symptom_index[symptom]] = 1
        return X

    """
    Get the accuarcy of the model for a given test set between 0 and 1
    The accuarcy will vary depending on the test set given
    (ie. If set is same as training set, accuarcy will be 100%)
    """
    def get_test_score(self):
        return self.score

#Testing
"""
nb = NaiveBayes('Training.csv')
symptoms = ['continuous_sneezing', 'chills']
print('Prediction: ' + str(nb.get_prediction(symptoms)))
print('Test accuarcy: ' + str(nb.get_test_score()))
"""