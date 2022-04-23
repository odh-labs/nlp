import os
import tensorflow as tf
import subprocess
import joblib
import mlflow
class MLflow():
    '''
    Define a class for MLflow configuration
    ----------

    Returns
    -------
    self.model:
        Deep learning based Model  
    
    '''
    def __init__(self, MLFLOW, HOST,EXPERIMENT_NAME):
        self.mlflow = MLFLOW
        self.host = HOST
        self.experiment_name = EXPERIMENT_NAME
        

    def SetUp_Mlflow(self):
        '''
        Setup MLflow
        ----------
        
        Returns
        -------
        
        '''       

        # Connect to local MLflow tracking server
        self.mlflow.set_tracking_uri(self.host)

        # Set the experiment name...
        self.mlflow.set_experiment(self.experiment_name)

        self.mlflow.tensorflow.autolog()
        return self.mlflow

    


    def mlflow_grid_search(methodtoexecute, methodarguments):
        with mlflow.start_run(tags= {
            "mlflow.source.git.commit" : get_git_revision_hash() ,
            "mlflow.user": get_git_user(),
            "mlflow.source.git.repoURL": get_git_remote(),
            "git_remote": get_git_remote(),
            "mlflow.source.git.branch": get_git_branch(),
            "mlflow.docker.image.name": os.getenv("JUPYTER_IMAGE", "LOCAL"),
            "mlflow.source.type": "NOTEBOOK",
    #         "mlflow.source.name": ipynbname.name()
        }) as run:
            methodtoexecute(**methodarguments)
            record_details(mlflow)

        return run
    
    
    
    

import tensorflow as tf


    

class TrainModel():
    '''
    Build Lstm model for tensorflow
    ----------

    Returns
    -------
    self.model:
        Deep learning based Model
    
    '''
    
    def __init__(self, MODEL, TOKENIZER, ENC,TRAIN_DATA, TRAIN_LABELS,TEST_DATA, TEST_LABELS,HOST, EXPERIMENT_NAME, BATCH_SIZE=64,EPOCHS=10):
        self.model_checkpoint_callback = []
        self.enc = ENC
        self.tokenizer = TOKENIZER
        self.mlflow = mlflow
        self.model = MODEL
        self.train_data = TRAIN_DATA
        self.train_labels = TRAIN_LABELS
        self.test_data  = TEST_DATA
        self.test_labels = TEST_LABELS
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.host = HOST
        self.experiment_name = EXPERIMENT_NAME
        self.history = []
    def get_git_revision_hash(self):
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'])

    def get_git_revision_short_hash(self):
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])

    def get_git_remote(self):
        return subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'])

    def get_git_user(self):
        return subprocess.check_output(['git', 'config', 'user.name'])

    def get_git_branch(self):
        return subprocess.check_output(['git', 'branch', '--show-current'])

    def get_pip_freeze(self):
        return subprocess.check_output(['pip', 'freeze']).splitlines()


    def record_details(self):
        """
        This method is the anchor poijt and more activiteis will go in it
        :param mlflow:
        :return:
        """
        with open("pip_freeze.txt", "wb") as file:
            for line in self.get_pip_freeze():
                file.write(line)
                file.write(bytes("\n", "UTF-8"))
        self.mlflow.log_artifact("pip_freeze.txt")
        file.close()
        self.mlflow.log_artifact("model.h5", artifact_path="model")
        self.mlflow.log_artifact("tokenizer.pkl", artifact_path="model")
        self.mlflow.log_artifact("labelencoder.pkl", artifact_path="model")

        # os.remove("pip_freeze.txt")
        # os.remove("model.h5")
        # os.remove("tokenizer.pkl")
        # os.remove("labelencoder.pkl")

    def DefineCheckPoint(self):
        '''
        Define the model
        ----------
        
        Returns
        -------
        
        '''
        #Bidirectional LSTM
        checkpoint_filepath = 'model.h5'
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_acc',
            mode='max',
            save_best_only=True)
        
        
    
    def SavePKL(self):
        '''
        Define the model
        ----------
        
        Returns
        -------
        
        '''
        joblib.dump(self.enc, 'labelencoder.pkl')  
        joblib.dump(self.tokenizer, 'tokenizer.pkl')  

        
    
        
    def ModelTraining(self):
        '''
        Define the model
        ----------
        
        Returns
        -------
        
        '''
        # self.SavePKL()
        self.DefineCheckPoint()
        self.mlflow = MLflow(self.mlflow, self.host,self.experiment_name).SetUp_Mlflow()
        with self.mlflow.start_run(tags= {
                "mlflow.source.git.commit" : self.get_git_revision_hash() ,
                "mlflow.user": self.get_git_user(),
                "mlflow.source.git.repoURL": self.get_git_remote(),
                "git_remote": self.get_git_remote(),
                "mlflow.source.git.branch": self.get_git_branch(),
                "mlflow.docker.image.name": os.getenv("JUPYTER_IMAGE", "LOCAL"),
                "mlflow.source.type": "NOTEBOOK",
        #         "mlflow.source.name": ipynbname.name()
            }) as run:
                self.history = self.model.fit(self.train_data, self.train_labels,
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         validation_data=(self.test_data, self.test_labels),callbacks=[self.model_checkpoint_callback])
                self.record_details()
        return self.model,self.history
        # return self.final_set,self.labels, self.enc, self.ohe,self.encoding_flag
    