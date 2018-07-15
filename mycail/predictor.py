import json
import thulac
import pickle
import sys
from tfcode.model_v1.load_predict import predict_result

sys.path.append("..")

class Predictor:
    def __init__(self):
        self.batch_size = 128
        self.cut = thulac.thulac(seg_only = True)
        self.load_pre = predict_result()

    def predict(self, content):
        result = []
        test_data = []
        for a in range(0, len(content)):
            fact = self.cut.cut(content[a], text = True).split(' ')
            test_data.append(fact)
        accu_result = self.load_pre.predict_to_result(test_data)
        for ac in accu_result:
            result.append({
                "accusation": ac,
                "imprisonment": 6,
                "articles":[47]
            })
        return result

