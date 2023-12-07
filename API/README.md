# NLP API

API for heavy loading processes like LLM > Spacy. Load once, avoid waiting hereafter

Current features:

- Spacy:
  - morphological analysis
- Llama 2 (requires huggingface key and model sign up):
  - translation function
  
  

API routes at http://0.0.0.0:6003/docs

### 

## Setup

For enable translation via llama 2:

1. Sign-up/log-in at https://huggingface.co

2. Get a hugging face key (`<your huggingface key>`) at https://huggingface.co 

3. Sign access to lightweight Llama 2 model at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

   > HARDWARE REQUIREMENTS:
   >
   > - a GPU with at least 9GB available space
   >
   > - CPU option is not implemented

### Option 1. Create virtual environment and activate it (recommended)

```bash
python -m venv venv
source ./venv/bin/activate
```

And then:

```bash
pip install -r requirements.txt
```

start API:

```bash
uvicorn API:app --host 0.0.0.0 --port 6003
```

Test available API routes at http://0.0.0.0:6003/docs

### Option 2. Create a docker image based on `Dockerfile` file and run a container

1. Create volume and image

```bash
# Path to this folder
# create volume for storing .cache large downloads 
docker volume create cache
# create image
docker build --tag nlp .
```

2. Start a container, setup it internally and test it:

```bash
# run interactive sesion
docker run -it --gpus all -v $PWD:/app -v cache:/root/.cache -p 6003:6003 --name nlp_api nlp bash
# set huggingface key environment variable
export hf_auth=<your huggingface key here>
# run 
uvicorn API:app --host 0.0.0.0 --port 6003
```

3. Test available API routes at http://0.0.0.0:6003/docs
4. Everything in order? Type <key>Ctrl+p</key> then <key>Ctrl+q</key> to turn interactive mode to daemon mode

