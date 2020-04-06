import os
import ast
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from backend.api.ml.nncf import NNCF
from backend.api.ml.svm_kfolds import SVM
from backend.api.ml.ModelSelector import ModelSelector
from backend.api.ml.word2vec import WordToVec

model_selector = ModelSelector(os.path.join(settings.BASE_DIR, "Training.csv"))
wordtovec = WordToVec(os.path.join(settings.BASE_DIR, "Training.csv"), os.path.join(settings.BASE_DIR, "Word2vec.bin"))

class Diagnosis(APIView):

    def post(self, request, format=None):
        symptoms = ast.literal_eval(request.data['symptoms'])
        return Response(model_selector.get_models()[model_selector.suggest_model()].get_prediction(symptoms)[0])

class SimilarSymptoms(APIView):
    
    def post(self, request, format=None):
        input_symptoms = ast.literal_eval(request.data['symptoms'])
        nncf = NNCF(os.path.join(settings.BASE_DIR, "Training.csv"), 10)
        prediction = nncf.get_nearest_symptoms(input_symptoms, 5, 0.00000001)
        return Response(prediction)

class CheckSymptom(APIView):
    def post(self, request, format=None):
        symptom = request.data['symptom'].strip()
        return Response(wordtovec.check_symptom(symptom))