import sys
import os
import tensorflow as tf
import joblib
import numpy as np
import json
import traceback
import mlflow
from minio import Minio
import openshift as oc
from jinja2 import Template
from tensorflow.keras.preprocessing.sequence import pad_sequences


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
        return json.dumps(result.numpy(), cls=JsonSerializer)

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




class DownloadArtifact():
    '''
    Build Lstm model for tensorflow
    ----------

    Returns
    -------
    self.model:
        Deep learning based Model  
    
    '''
    def __init__(self, MLFLOW, MODEL_NAME='lstmt1', MODEL_VRSION='1',HOST = "http://mlflow:5500" ,MLFLOW_S3_ENDPOINT_URL = 'http://minio-ml-workshop:9000',AWS_ACCESS_KEY_ID='minio',AWS_SECRET_ACCESS_KEY='minio123',AWS_REGION='us-east-1',AWS_BUCKET_NAME ='mlflow'):
        
        self.mlflow = MLFLOW
        self.model_name = MODEL_NAME
        self.model_version = MODEL_VRSION
        self.host = HOST
        self.mlflow_s3_endpoint_url = MLFLOW_S3_ENDPOINT_URL
        self.aws_access_key_id = AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = AWS_SECRET_ACCESS_KEY
        self.aws_region = AWS_REGION
        self.aws_bucket_name = AWS_BUCKET_NAME
        self.SetUpOS()
        
    def SetUpOS(self):
        

        os.environ['MLFLOW_S3_ENDPOINT_URL'] = self.mlflow_s3_endpoint_url 
        os.environ['AWS_ACCESS_KEY_ID']= self.aws_access_key_id 
        os.environ['AWS_SECRET_ACCESS_KEY'] = self.aws_secret_access_key
        os.environ['AWS_REGION'] = self.aws_region
        os.environ['AWS_BUCKET_NAME']= self.aws_bucket_name
        # os.environ['MODEL_NAME'] = 'rossdemo'
        # os.environ['MODEL_VERSION'] = '1'
        # os.environ['OPENSHIFT_CLIENT_PYTHON_DEFAULT_OC_PATH'] = '/tmp/oc'



    def get_s3_server(self):
        minioClient = Minio('minio-ml-workshop:9000',
                        access_key='minio',
                        secret_key='minio123',
                        secure=False)

        return minioClient


    def Init_Mlfow(self):
        self.mlflow.set_tracking_uri(self.host)
        print(self.host)
        # Set the experiment name...
        #mlflow_client = mlflow.tracking.MlflowClient(HOST)


    def download_artifacts(self):
        print("retrieving model metadata from mlflow...")
        model = self.mlflow.pyfunc.load_model(
            model_uri=f"models:/{self.model_name}/{self.model_version}"
        )
        print(model)

        run_id = model.metadata.run_id
        experiment_id = self.mlflow.get_run(run_id).info.experiment_id

        print("initializing connection to s3 server...")
        minioClient = self.get_s3_server()

    #     artifact_location = mlflow.get_experiment_by_name('rossdemo').artifact_location
    #     print("downloading artifacts from s3 bucket " + artifact_location)

        data_file_model = minioClient.fget_object("mlflow", f"/{experiment_id}/{run_id}/artifacts/model/model.h5", "model.h5")
        data_file_model2 = minioClient.fget_object("mlflow", f"/{experiment_id}/{run_id}/artifacts/model/tokenizer.pkl", "tokenizer.pkl")
        data_file_model3 = minioClient.fget_object("mlflow", f"/{experiment_id}/{run_id}/artifacts/model/labelencoder.pkl", "labelencoder.pkl")


        #Using boto3 Download the files from mlflow, the file path is in the model meta
        #write the files to the file system
        print("download successful")

        return run_id