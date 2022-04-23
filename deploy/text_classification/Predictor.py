import tensorflow as tf
import joblib
import numpy as np
import json
import traceback
import sys
import os


class Predictor(object):
    
    def __init__(self):
        self.loaded = False

    def load(self):
        print("Loading model",os.getpid())
        self.model = tf.keras.models.load_model('model.h5', compile=False)
        self.labelencoder = joblib.load('labelencoder.pkl')
        self.loaded = True
        print("Loaded model")



    def predict(self, X,features_names):
        # data = request.get("data", {}).get("ndarray")
        # mult_types_array = np.array(data, dtype=object)
        print ('step1......')
        print(X)
        X = tf.constant(X)
        print ('step2......')
        print(X)
        if not self.loaded:
            self.load()
#         result = self.model.predict(X)
        try:
            result = self.model.predict(X)
        except Exception as e:
            print(traceback.format_exception(*sys.exc_info()))
            raise # reraises the exception
                
        print ('step3......')
        result = tf.sigmoid(result)
        print ('step4......')
        print(result)
        result = tf.math.argmax(result,axis=1)
        print ('step5......')
        print(result)
        print(result.shape)
        
        print(self.labelencoder.inverse_transform(result))
        print ('step6......')
        

        json_results = {"Predicted Class": str(self.labelencoder.inverse_transform(result)),"Predicted Class Label": json.dumps(result.numpy(), cls=JsonSerializer)}
        
        
        return json_results
        # return json.dumps(result.numpy(), cls=JsonSerializer)

class JsonSerializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (
        np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)