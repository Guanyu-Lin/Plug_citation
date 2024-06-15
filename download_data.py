from huggingface_hub import hf_hub_download
import os
from huggingface_hub import HfApi
from huggingface_hub import login
login()
READ_WRITE_TOKEN = os.environ['READ_WRITE']
api = HfApi(token = READ_WRITE_TOKEN)
DATA_REPO_ID = "cmulgy/rag_data"
hf_hub_download(repo_id=DATA_REPO_ID, filename="rag_train_with_psg.tar.gz", local_dir = ".", repo_type="dataset")
