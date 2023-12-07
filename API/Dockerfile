# CUDA: docker run -it --gpus all -v $(pwd):/app --name nombre_contenedor nombre_imagen

# IMAGES BASE
# https://stackoverflow.com/questions/56405159/what-is-the-difference-between-devel-and-runtime-tag-for-a-docker-container
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# update fetches the latest version of the package list from your distro's software repository, and any third-party repositories you may have configured. In other words, it'll figure out what the latest version of each package and dependency is, but will not actually download or install any of those updates.
RUN apt update
# packages can be upgraded, run apt list --upgradable
# sudo apt upgrade command downloads and installs the updates for each outdated package and dependency on your system
RUN apt -y upgrade

# ································
# · INSTALACIÓN
# ································
RUN apt install -y python3 python3-pip

# COPY requirements.txt /app/requirements.txt
# RUN pip install -r /app/requirements.txt
# optimizadores NN

###########################
# NN
###########################
# Collecting torch>=1.10.0 (from accelerate)
# Collecting huggingface-hub (from accelerate)
# Collecting nvidia-cuda-nvrtc-cu12==12.1.105 (from torch>=1.10.0->accelerate)

# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# ERROR
#   Failed to import transformers.integrations.bitsandbytes because of the following error (look up to see its traceback):
#   CUDA Setup failed despite GPU being available. Please run the following command to get more information:
# SOLUTION: don't install torch, let this packages install the dependency in a a compatible version 
RUN pip install bitsandbytes accelerate peft xformers einops
RUN pip install transformers 
# RUN pip install openai
RUN pip install --force-reinstall -v "openai==0.27.2"
###########################
# EXTRAS
###########################
# API
RUN pip install uvicorn fastapi pydantic
# Collecting numpy<1.28.0,>=1.21.6 (from scipy)
RUN pip install pandas scipy numpy openpyxl
# DEBUG
RUN pip install loguru fire tmuxp 
###########################
# Spacy
###########################
RUN pip install spacy
# EN english
# RUN python -m spacy download en_core_web_lg # Vocabulary, syntax, entities, vectors
RUN python3 -m spacy download en_core_web_trf # no vector info
# ES spanish
# RUN python -m spacy download es_core_news_lg # Vocabulary, syntax, entities, vectors
RUN python3 -m spacy download es_dep_news_trf  # no vector info, no entities


# COPY *.py /app
# COPY . /app
WORKDIR /app

CMD uvicorn API:app --host 0.0.0.0 --port 6003 --reload
# ENTRYPONT ["python3", "tensor-code.py"]