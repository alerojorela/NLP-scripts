import os
import json
import spacy
import time

from loguru import logger
from pydantic import BaseModel
from fastapi import FastAPI

# <Copy&paste your User Access Token from Hugging Face>
import torch
from torch import cuda, bfloat16
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

#####################################
# FROM THIS PROJECT
#####################################
# from utils import explacy
import explacy


default_text= "Hola, ¿cómo te encuentras?"


class LlamaInference:
    def __init__(self,
                 model_id: str = 'meta-llama/Llama-2-7b-hf',
                 # device: str='gpu',
                 bits: int = 8):

        # https://huggingface.co/meta-llama
        # model_id = 'meta-llama/Llama-2-7b-hf'
        # NOOOO model_id = 'meta-llama/Llama-2-7b-chat-hf'  # GPU: 6600 MB (55%)
        # model_id = 'meta-llama/Llama-2-13b-hf'  # 26 Gb GPU: 10800 MB (86%)
        # model_id = 'meta-llama/Llama-2-70b-hf'  # > 100Gb
        self.model_id = model_id
        """
        /opt/conda/lib/python3.10/site-packages/transformers/utils/hub.py:374: FutureWarning: The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.
        """

        device = f'cuda:{cuda.current_device()}' if cuda.is_available(
        ) else 'cpu'
        # device = 'cpu'  # NUEVO
        if device != 'cpu':
            device_name = torch.cuda.get_device_name()
            logger.info(f"Using device: {device} ({device_name})")

        logger.info(
            f"Loading {self.model_id} model with {bits} bits {'quantization' if bits < 16 else ''}...")

        if bits == 4:
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=bfloat16,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                # Optimization
                device_map='auto',
                quantization_config=bnb_config,
                # load_in_4bit=True,  # nuevo, alternativo a lo anterior
                use_auth_token=hf_auth
            )
        elif bits == 8:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                # Optimization
                device_map='auto',
                load_in_8bit=True,  # nuevo, alternativo a lo anterior
                use_auth_token=hf_auth
            )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                # config=model_config,

                # Optimization
                device_map='auto',
                # quantization_config=bnb_config,
                # load_in_4bit=True,  # nuevo, alternativo a lo anterior
                # load_in_8bit=True,  # nuevo, alternativo a lo anterior
                # max_memory=f'{max_memory}GB',  # nuevo
                # offload_folder="/mnt/datos/temp/save_folder",  # nuevo

                use_auth_token=hf_auth
            )

        # model.eval() # <<<<<<<<<<<<<
        logger.info(f"Model loaded on {device}")

        # TOKENIZER
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            use_auth_token=hf_auth
        )
        # End of string signifier is used by llama 2 - {EOS} or </s>
        # tokenizer.pad_token = tokenizer.eos_token

        self._generate_text = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            task='text-generation',
            # ValueError: `temperature` (=0.0) has to be a strictly positive float, otherwise your next token scores will be invalid. If you're looking for greedy decoding strategies, set `do_sample=False
            # temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            temperature=0.1,  # ERROR SI offload_folder y offload_folder si CPU

            max_new_tokens=512,  # mex number of tokens to generate in the output
            repetition_penalty=1.1,  # without this output begins repeating
            # device=device
        )

    # @print_processing_time
    def predict(self, prompt: str):
        res = self._generate_text(prompt)
        # logger.error(res)
        output = res[0]["generated_text"]
        return output

    def predict_from_dict(self, sample: dict):
        if sample.get('input'):
            sample['input'] = "```" + str(sample['input']) + "```"
        # print('expected: ', sample.get('output'))

        prompt = self.prompts.generate_input_prompt(sample)
        return self.predict(prompt)


llama_models = ["meta-llama/Llama-2-7b-chat-hf",
                "meta-llama/Llama-2-13b-chat-hf",
                "meta-llama/Llama-2-70b-chat-hf"]
llama_model = llama_models[0]

# region Loading
hf_auth = None
try:
    # read environment variable
    hf_auth = os.environ['hf_auth']
except Exception as error:
    logger.error("""hf_auth environment variable not set. Translator won't be available
                 Solution:
                 1. signup/login at https://huggingface.co
                 2. get hugging face key
                 3. sign access to model at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
                 3. at your CLI execute: `export hf_auth=<my hugging face key>""") 
else:
    try:
        inferencer = LlamaInference(model_id=llama_model, bits=8)
        pass
    except Exception as error:
        logger.error(f"{llama_model} model loading failed. Translator won't be available") 


logger.info('Loading Spacy model...')
start_time = time.time()
nlp_model = {}
# nlp_model["es"] = spacy.load("es_core_news_lg")  # has vector info
nlp_model["es"] = spacy.load("es_dep_news_trf")  # no vector info, no entities
execution_time = (time.time() - start_time)
print(f'Spacy loading took {str(execution_time)} seconds\n')

# endregion



def serialize_token(token):
    properties = {'text': token.text,
                'lemma': token.lemma_,
                'POS': token.pos_,
                'morph': token.morph.to_dict(),
                # dep
                # parent
                'index': token.i
                }
    return properties


app = FastAPI()
"""
class text_data(BaseModel):
    text: str
    language: str
"""

@app.get("/")
async def root():
    return {'message': "consult API in /docs"}

@app.get("/parse")
def parse(text:str=default_text, language: str = "es"):
    doc = nlp_model[language](text)
    explacy.print_parse_info(nlp_model[language], text)
    result = [serialize_token(_) for _ in doc]
    return result

def ask_llama(prompt: str):
    result = inferencer.predict(prompt)
    result = result[len(prompt):].strip()
    return result

@app.get("/query")
def parse(task:str):
    sys_msg = "You are a helpful research assistant. You are brief and get straight to the point"
    prompt = f"""[INST] <<SYS>>{sys_msg}<</SYS>> {task} [/INST]"""
    result = ask_llama(prompt)
    return result


@app.get("/translate")
def parse(text:str=default_text, language: str = "english"):
    sys_msg = "You are a helpful research assistant. You are brief and get straight to the point"
    # task = f"Traduce al inglés el siguiente texto delimitado por triples comillas '''{text}'''"
    task = f"""Translate to {language} the following text delimited by triple quotes. Don't provide any information but the translation.
text: '''{text}'''
translation: """
    prompt = f"""[INST] <<SYS>>{sys_msg}<</SYS>> {task} [/INST]"""
    result = ask_llama(prompt)
    if result.startswith('"'):
        result = result[1:]
    if result.endswith('"'):
        result = result[:-1]
    return result
