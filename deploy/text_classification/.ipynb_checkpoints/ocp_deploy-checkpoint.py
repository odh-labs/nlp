import sys
import os
import mlflow
from minio import Minio
import openshift as oc
from jinja2 import Template
import tensorflow as tf
import time

os.environ['MLFLOW_S3_ENDPOINT_URL']='http://minio-ml-workshop:9000'
os.environ['AWS_ACCESS_KEY_ID']='minio'
os.environ['AWS_SECRET_ACCESS_KEY']='minio123'
os.environ['AWS_REGION']='us-east-1'
os.environ['AWS_BUCKET_NAME']='mlflow'
# os.environ['MODEL_NAME'] = 'lstmv7'
# os.environ['MODEL_VERSION'] = '1'
os.environ['OPENSHIFT_CLIENT_PYTHON_DEFAULT_OC_PATH'] = '/tmp/oc'

HOST = "http://mlflow:5500"

model_name = os.environ["MODEL_NAME"]
model_version = os.environ["MODEL_VERSION"]
build_name = f"seldon-model-{model_name}-v{model_version}"

def get_s3_server():
    minioClient = Minio('minio-ml-workshop:9000',
                    access_key='minio',
                    secret_key='minio123',
                    secure=False)

    return minioClient


def init():
    mlflow.set_tracking_uri(HOST)
    print(HOST)
    # Set the experiment name...
    #mlflow_client = mlflow.tracking.MlflowClient(HOST)

    
def download_artifacts():
    print("retrieving model metadata from mlflow...")
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )
    print(model)
    
    run_id = model.metadata.run_id
    experiment_id = mlflow.get_run(run_id).info.experiment_id
    
    print("initializing connection to s3 server...")
    minioClient = get_s3_server()

#     artifact_location = mlflow.get_experiment_by_name('rossdemo').artifact_location
#     print("downloading artifacts from s3 bucket " + artifact_location)

    data_file_model = minioClient.fget_object("mlflow", f"/{experiment_id}/{run_id}/artifacts/model/model.h5", "model.h5")
    # TODO  # REPLACE IT WITH FOR LOOP TO AUTOMATICA
    data_file_tokenizer = minioClient.fget_object("mlflow", f"/{experiment_id}/{run_id}/artifacts/model/tokenizer.pkl", "tokenizer.pkl")
    data_file_labelencoder = minioClient.fget_object("mlflow", f"/{experiment_id}/{run_id}/artifacts/model/labelencoder.pkl", "labelencoder.pkl")
    data_file_tokenizer = minioClient.fget_object("mlflow", f"/{experiment_id}/{run_id}/artifacts/model/requirements.txt", "requirements.txt")
    #Using boto3 Download the files from mlflow, the file path is in the model meta
    #write the files to the file system
    print("download successful")
    
    return run_id
    
        
init()
try:
    run_id = download_artifacts()
except:
    print('Something happened in download_artifacts')
    print(model_name)
    print (model_version)
    sys.exit(-1)

print("Start OCP things...")

server = "https://" + os.environ["KUBERNETES_SERVICE_HOST"] + ":" + os.environ["KUBERNETES_SERVICE_PORT"] 
print(server)

#build from source Docker file
with open('/var/run/secrets/kubernetes.io/serviceaccount/token', 'r') as file:
    token = file.read()
print(f"Openshift Token{token}")

#/var/run/secrets/kubernetes.io/serviceaccount
with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace', 'r') as namespace:
    project = namespace.read()
print(f"Project name: {project}")

#build from source Docker file
with oc.api_server(server):
    with oc.token(token):
        with oc.project(project), oc.timeout(10*60):
            print('OpenShift client version: {}'.format(oc.get_client_version()))
            #print('OpenShift server version: {}'.format(oc.get_server_version()))

            build_config = oc.selector(f"bc/{build_name}").count_existing() #.object
            print(oc.get_project_name())
            print(build_config)
            if build_config == 0:
                oc.new_build(["--strategy", "docker", "--binary", "--docker-image", "registry.access.redhat.com/ubi8/python-38:1-71", "--name", build_name ])
            else:
                build_details = oc.selector(f"bc/{build_name}").object()
                print(build_details.as_json())

            print("Starting Build and Wiating.....")
            build_exec = oc.start_build([build_name, "--from-dir", ".", "--follow", "--build-loglevel", "10"])
            print("Build Finished")
            status = build_exec.status()
            print(status)
            for k, v in oc.selector([f"bc/{build_name}"]).logs(tail=500).items():
                print('Build Log: {}\n{}\n\n'.format(k, v))

            #seldon_deploy = oc.selector(f"SeldonDeployment/{build_name}").count_existing()
            #experiment_id = mlflow.get_run(run_id).info.experiment_id

            template_data = {"experiment_id": run_id, "model_name": model_name, "image_name": build_name, "project": project}
            applied_template = Template(open("SeldonDeploy.yaml").read())
            print(applied_template.render(template_data))
            oc.apply(applied_template.render(template_data))

            route_count = oc.selector(f"route/{build_name}").count_existing()
            print(route_count)
            if route_count == 0:
                service_name = "model-" + run_id + "-" + model_name
                while True:
                    service_count = oc.selector(f"service/{service_name}").count_existing()
                    if service_count > 0:
                        service = oc.selector(f"service/{service_name}").object()
                        print(service.name())
                        oc.oc_action(oc.cur_context(), "expose", cmd_args=["service", service.name(), "--name", service.name()])
                        break
                    else:
                        print(f"Service name does not exist {service_name}")
                        time.sleep(10)
            else:
                print(f"Route already exists {service_name}")


# #build from source Docker file
# with oc.api_server(server):
#     with oc.token(token):
#         with oc.project(project), oc.timeout(10*60):
#             print('OpenShift client version: {}'.format(oc.get_client_version()))
#             #print('OpenShift server version: {}'.format(oc.get_server_version()))

#             build_config = oc.selector(f"bc/{build_name}").count_existing() #.object
#             print(oc.get_project_name())
#             print(build_config)
#             if build_config == 0:
#                 oc.new_build(["--strategy", "docker", "--binary", "--docker-image", "registry.access.redhat.com/ubi8/python-38:1-71", "--name", build_name ])
#             else:
#                 build_details = oc.selector(f"bc/{build_name}").object()
#                 print(build_details.as_json())

#             print("Starting Build and Wiating.....")
#             build_exec = oc.start_build([build_name, "--from-dir", ".", "--follow", "--build-loglevel", "10"])# docker build and push
#             print("Build Finished")
#             status = build_exec.status()
#             print(status)
#             for k, v in oc.selector([f"bc/{build_name}"]).logs(tail=500).items():
#                 print('Build Log: {}\n{}\n\n'.format(k, v))

#             seldon_deploy = oc.selector(f"SeldonDeployment/{build_name}").count_existing()

#             experiment_id = mlflow.get_run(run_id).info.experiment_id

#             if seldon_deploy == 0:
#                 template_data = {"experiment_id": run_id, "model_name": model_name, "image_name": build_name, "project": project}
#                 applied_template = Template(open("SeldonDeploy.yaml").read())
#                 print(applied_template.render(template_data))
#                 oc.create(applied_template.render(template_data))

                
                
                
                