from gensim.models import KeyedVectors
import csv

class WordToVec:

    def __init__(self, training_csv, vector_bin):
        self.symptoms_dataset = None
        with open(training_csv, newline='') as f:
            csv_reader = csv.reader(f)
            self.symptoms_dataset = next(csv_reader)
        self.model = KeyedVectors.load_word2vec_format(vector_bin, binary=True, limit=300000)

    def check_symptom(self, symptom):
        res = {
            "symptom_in_dataset": None,
            "symptom_similar": None,
            "symptom_not_in_dataset": None
        }

        if symptom in self.symptoms_dataset:
            res["symptom_in_dataset"] = symptom 
            return res

        symptoms_in_dataset = []
        symptoms_not_in_dataset = []
    
        # Get list of words that are similar to the symptom entered 
        try:
            similar_words = self.model.most_similar(positive=[symptom], negative=None, topn = 100)
        except KeyError:
            similar_words = []



        # Check if similar words appear in ML symptom set
        for symptom_data in similar_words:
            if symptom_data[0] in self.symptoms_dataset:
                res["symptom_similar"] = symptom_data[0]
                return res

        # no matches to dataset
        res["symptom_not_in_dataset"] = symptom 
        return res