from backend.api.ml.naivebayes_kfolds import NaiveBayes
from backend.api.ml.randomforest_kfolds import RandomForest
from backend.api.ml.svm_kfolds import SVM
from backend.api.ml.logisticreg_kfolds import LR
from backend.api.ml.neuralnet_kfolds import NN

from sklearn.decomposition import PCA
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

class ModelSelector:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        nb = NaiveBayes(dataset_name)
        rf = RandomForest(dataset_name)
        svm = SVM(dataset_name)
        lr = LR(dataset_name)
        nn = NN(dataset_name)
        self.models = {'Naive Bayes': nb, 'Random Forest': rf, 'Support Vector Machine': svm, 'Logistic Regression': lr, 'Neural Net': nn}
        self.disease_list = nb.model.classes_
        self.num_rows = nb.num_rows
        self.num_cols = nb.num_cols
        self.num_classes = len(self.disease_list)


    """
    Automatically selects the name of the best model to use given the properties of the initial dataset
    and given threshold parameters
    Use in conjunction with get_models to select the model to use
    """
    def suggest_model(self, avg_classes_per_example_threshold=10):
        avg_examples_per_class = self.num_rows/self.num_classes

        nb_score = self.models['Naive Bayes'].get_test_score()
        rf_score = self.models['Random Forest'].get_test_score()
        svm_score = self.models['Support Vector Machine'].get_test_score()
        lr_score = self.models['Logistic Regression'].get_test_score()
        nn_score = self.models['Neural Net'].get_test_score()

        if avg_examples_per_class < avg_classes_per_example_threshold: # Small dataset for number of possible diseases
            model_array = [('Naive Bayes', nb_score), ('Random Forest', rf_score)]
        else:
            model_array = [('Naive Bayes', nb_score), ('Random Forest', rf_score), ('Support Vector Machine', svm_score), ('Logistic Regression', lr_score), ('Neural Net', nn_score)]

        # Higher priority models for tie breakers
        model_priority = {'Neural Net': 5, 'Support Vector Machine': 4, 'Logistic Regression': 3, 'Random Forest': 2, 'Naive Bayes': 1}

        sorted_model_array = sorted(model_array, key=lambda x: (x[1], model_priority[x[0]]), reverse=True)
        return sorted_model_array[0][0]


    """
    Return all models in a map
    """
    def get_models(self):
        return self.models


    """
    Plots the correct/incorrect predictions of each model after a PCA transformation
    """
    def plot_PCA(self, num_points=300):
        matplotlib_axes_logger.setLevel('ERROR') #Supress warnings for color arg

        dataset = pd.read_csv(self.dataset_name)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        pca = PCA(n_components=2) #2d plot
        pca.fit(X)

        # Sample random points in dataset to plot (up to a max of num_points)
        len_X = len(X)
        if num_points > len_X:
            indices_to_plot = range(0, len_X)
        else:
            indices_to_plot = random.sample(range(0, len(X)), num_points)

        pca_X = np.hstack((pca.transform(X[indices_to_plot]), np.c_[y[indices_to_plot]])) #Contains PCA points and REAL label
        
        #Get model predictions for each point to plot
        model_predictions = {'Naive Bayes': [], 'Random Forest': [], 'Support Vector Machine': [], 'Logistic Regression': [], 'Neural Net': []}
        seen_predictions = set()
        for i in indices_to_plot:
            for model in model_predictions:
                prediction = self.models[model].model.predict([X[i]])[0]
                model_predictions[model].append(prediction)
                seen_predictions.add(prediction)

        #Setup colour map
        label_colour_map = dict() 
        num_unique_predictions = len(seen_predictions)
        c_iter = iter(cm.rainbow(np.linspace(0,1,num_unique_predictions)))
        for label in seen_predictions:
            label_colour_map[label] = next(c_iter)

        #Setup subplots for each model
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12,12))
        fig.suptitle('2 component PCA\nCorrect prediction = o   Incorrect prediction = x', fontsize = 20)
        model_plots = {'Naive Bayes': axs[0][0], 'Random Forest': axs[0][1], 'Support Vector Machine': axs[1][0], 'Logistic Regression': axs[1][1], 'Neural Net': axs[2][0]}
        fig.delaxes(axs[2][1]) #Delete extra axis

        #Plot each model's predictions
        for model in model_plots:
            model_plots[model].set_xlabel('Principal Component 1', fontsize = 8)
            model_plots[model].set_ylabel('Principal Component 2', fontsize = 8)
            model_plots[model].set_title(model, fontsize = 12)

            i = 0
            for row in pca_X:
                x = row[0]
                y = row[1]
                real_label = row[2] 
                prediction_label = model_predictions[model][i]
                colour = label_colour_map[prediction_label]

                if (real_label != prediction_label): #Incorrect labels get x
                    model_plots[model].scatter(x,y,c=colour,label=prediction_label,marker='x', s=5)
                else:
                    model_plots[model].scatter(x,y,c=colour,label=prediction_label,marker='o', s=5)
                i += 1

            model_plots[model].grid() #Turn on grid

            # Turn off tick labels
            model_plots[model].set_yticklabels([])
            model_plots[model].set_xticklabels([])

        #Construct legend for all plots
        legend_elements = []
        for label, colour in label_colour_map.items():
            element = Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=colour, markersize=8)
            legend_elements.append(element)

        fig.legend(handles=legend_elements, prop={'size': 8}, loc='lower right', bbox_to_anchor=(0.5, 0., 0.5, 0.5), ncol=2)

        plt.show()
   

    """
    Gets the predictions for all the models in a map
    """
    def get_all_predictions(self, symptoms):
        ret = {}
        ret['Naive bayes'] = self.models['Naive bayes'].get_prediction(symptoms)
        ret['Random Forest'] = self.models['Random Forest'].get_prediction(symptoms)
        ret['Support Vector Machine'] = self.models['Support Vector Machine'].svm.get_prediction(symptoms)
        ret['Logistic Regression'] = self.models['Logistic Regression'].lr.get_prediction(symptoms)
        ret['Neural Net'] = self.models['Neural Net'].nn.get_prediction(symptoms)
        return ret


    """
    Gets the test accuarcies for all the models in a map
    """
    def get_all_test_accuarcies(self):
        ret = {}
        ret['Naive bayes'] = self.models['Naive bayes'].get_test_score()
        ret['Random Forest'] = self.models['Random Forest'].get_test_score()
        ret['Support Vector Machine'] = self.models['Support Vector Machine'].get_test_score()
        ret['Logistic Regression'] = self.models['Logistic Regression'].get_test_score()
        ret['Neural Net'] = self.models['Neural Net'].get_test_score()
        return ret


#Testing (example usage)
#selector = ModelSelector('Training.csv')
#symptoms = ['runny_nose','skin_rash','nodal_skin_eruptions']

"""
prediction_map = selector.get_all_predictions(symptoms)
print('Predictions:')
for key, value in prediction_map.items():
    print (key, value)
"""

"""
test_map = selector.get_all_test_accuarcies()
print('Test scores:')
for key, value in test_map.items():
    print (key, value)
"""

#selector.plot_PCA(num_points=500)

"""
model = selector.suggest_model()
print(model)
"""