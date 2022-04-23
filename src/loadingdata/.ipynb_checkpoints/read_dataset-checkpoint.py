import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from minio import Minio
class ReadData():
    '''
    Turn raw data into features for modeling
    ----------

    Returns
    -------
    self.final_set:
        Features for modeling purpose
    self.labels:
        Output labels of the features
    enc: 
        Ordinal Encoder definition file
    ohe:
        One hot  Encoder definition file
    '''
    def __init__(self, S3BucketName = "raw-data-saeed",FILE_NAME="data.csv", SPLIT_RATE=.2, INPUT_FEATURE_NAME ='consumer_complaint_narrative' , OUTPUT_FEATURE_NAME='product',MLFLOW_S3_ENDPOINT_URL = "minio-ml-workshop:9000",AWS_ACCESS_KEY_ID='minio',AWS_SECRET_ACCESS_KEY = 'minio123',SECURE = False):
#     def __init__(self, *args, **kwargs):
        self.file_name =  FILE_NAME 
        self.client = []
        self.S3BucketName = S3BucketName
        self.split_rate = SPLIT_RATE
        self.aws_access_key_id = AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = AWS_SECRET_ACCESS_KEY
        self.mlflow_s3_endpoint_url = MLFLOW_S3_ENDPOINT_URL
        self.secure = SECURE
        
        self.in_fe_name = 'consumer_complaint_narrative'
        self.out_fe_name = 'product'
        self.df = []
        self.train_data = []
        self.test_data = []

        self.train_labels = []
        self.test_labels = []
        self.enc = []
        
    
        
#         self.final_set,self.labels = self.build_data()
    def ReadS3Bucket(self):
        '''
        Read Data from S3. bucket
        ----------
        
        Returns
        -------
        Dataframe representation of the csv file
        '''
        
        csv_file = self.client.get_object(self.S3BucketName, self.file_name)
        self.df = pd.read_csv(csv_file)

        
#         return self.df
    ## read the data from the source file
    def SplitData(self):
        '''
        Reading the csv file
        ----------
        
        Returns
        -------
        Dataframe representation of the csv file
        '''

        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(self.df[self.in_fe_name], self.df[self.out_fe_name],stratify=self.df[self.out_fe_name], 
                                                    test_size=self.split_rate)
        
#         return self.train_data, self.test_data, self.train_labels, self.test_labels
    def LabelEncoding(self):
        '''
        GetRequired Info
        ----------
        
        Returns
        -------
        Dataframe representation of the csv file
        '''
        ##label encoding target variable
        self.enc = preprocessing.LabelEncoder()
        self.train_labels = self.enc.fit_transform(self.train_labels)
        self.test_labels = self.enc.fit_transform(self.test_labels)

        print(self.enc.classes_)
        print(np.unique(self.train_labels, return_counts=True))
        print(np.unique(self.test_labels, return_counts=True))


        
    def GetS3Server(self):
        
        self.client = Minio(self.mlflow_s3_endpoint_url,
                        access_key=self.aws_access_key_id,
                        secret_key=self.aws_secret_access_key,
                        secure=self.secure)

        return self.client
    ## address the missing information
    def ReadDataFrameData(self):
        '''
        Replace the missing value with the zero.
        ----------
        
        Returns
        -------
        Dataframe with replaced missing value.
        '''
        self.client = self.GetS3Server()
        self.ReadS3Bucket()
        self.SplitData()
        self.LabelEncoding()
        
        
        
        return self.train_data, self.test_data, self.train_labels, self.test_labels,self.enc