import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

class Transformer(object):
    def __init__(self):
        self.tokenizer = joblib.load('tokenizer.pkl')
        
    def transform_input(self, X, feature_names, meta):
        print ('step1......')
        print(X)
        print(feature_names)
# #         X = X[0]
#         print(X[0])
        output = self.tokenizer.texts_to_sequences(X)
        print ('step2......')
        print(output)
        
        output = pad_sequences(output, maxlen=348,padding='post')
        print ('step3......')
        print(output)
#         output = tf.constant(output)
#         print ('step4......')
#         print(output)
        return output