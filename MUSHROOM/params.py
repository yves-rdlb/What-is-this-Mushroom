import os

##################  VARIABLES  ##################
BUCKET_NAME = os.environ.get("BUCKET_NAME")
GCP_REGION = os.environ.get("GCP_REGION")
GCP_PROJECT = os.environ.get("GCP_PROJECT")

##################  CONSTANTS  #####################
LOCAL_MODEL_PATH = os.path.join(os.path.expanduser('~'), "code", "yves-rdlb", "What-is-this-Mushroom")
