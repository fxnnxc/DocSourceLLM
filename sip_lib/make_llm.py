import os
# from accelerate import infer_auto_device_map
# from accelerate import dispatch_model
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import OPTForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM

PYTHIA_MODEL_SIZES_LAYERS = {
    '70m':6, 
    '160m':12, 
    '410m':24, 
    '1b':16, 
    '1.4b':24, 
    '2.8b':32, 
    '6.9b':32, 
    '12b':36
}

LLAMA_MODEL_SIZES_LAYERS = {
    '7b':32, 
    '13b':40, 
}

OPT_MODEL_SIZES_LAYERS = {
    '350m' : 24,
    '2.7b' : 32,
    '6.7b':32, 
    '13b':40, 
}

def infer_llama_device_map(model_size, split:int):
    model_size = LLAMA_MODEL_SIZES_LAYERS[model_size]
    device_map = {'model.embed_tokens':0, 
                  'model.norm':split-1,
                  'lm_head':split-1,
                  }
    gpu = 0
    per_block = model_size//split
    for i in range(1, model_size+1):
        device_map[f'model.layers.{i-1}'] = gpu
        if i % per_block ==0 and  gpu < split-1:
            gpu += 1
    return device_map   

def infer_opt_device_map(model_size, split:int):
    size = model_size
    model_size = OPT_MODEL_SIZES_LAYERS[model_size]
    device_map = {
                'decoder.embed_tokens':0, 
                'decoder.embed_positions':0,
                'decoder.final_layer_norm':split-1,
                  }
    if size  == "350m":
        device_map['decoder.project_in'] = 0
        device_map['decoder.project_out'] = split-1
    gpu = 0
    per_block = model_size//split
    for i in range(1, model_size+1):
        device_map[f'decoder.layers.{i-1}'] = gpu
        if i % per_block ==0 and  gpu < split-1:
            gpu += 1
    return device_map   


def infer_pythia_device_map(model_size, split:int):
    model_size = PYTHIA_MODEL_SIZES_LAYERS[model_size]
    device_map = {'gpt_neox.embed_in':0, 
                  'gpt_neox.final_layer':split-1,
                  'gpt_neox.final_layer_norm':split-1,
                  'embed_out': split-1,
                  }
    gpu = 0
    per_block = model_size//split
    for i in range(1, model_size+1):
        device_map[f'gpt_neox.layers.{i-1}'] = gpu
        if i % per_block ==0 and  gpu < split-1:
            gpu += 1
    return device_map        

def make_language_model_and_tokenizer(lm_model, lm_size, lm_cache_dir, num_gpus, **kwargs):    
    if lm_model == "pythia":
        if 'EleutherAI' not in lm_cache_dir:
            lm_cache_dir = os.path.join(lm_cache_dir, 'EleutherAI')
        if num_gpus == 0:
            my_device_map = "cpu"
        else:
            my_device_map = infer_pythia_device_map(lm_size, num_gpus)
        if lm_size in PYTHIA_MODEL_SIZES_LAYERS.keys():
            model = GPTNeoXForCausalLM.from_pretrained(
                os.path.join('EleutherAI', f"pythia-{lm_size}-deduped"),
                revision='step143000',
                cache_dir=lm_cache_dir,
                device_map = my_device_map,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join('EleutherAI', f"pythia-{lm_size}-deduped"),
                revision='step143000',
                cache_dir=lm_cache_dir,
            )
            tokenizer.pad_token_id = tokenizer.eos_token_id

    elif lm_model in ["llama2", 'llama2_chat'] :
        # https://huggingface.co/meta-llama/Llama-2-7b
        if 'llama' not in lm_cache_dir:
            lm_cache_dir = os.path.join(lm_cache_dir, 'llama')
        if lm_size in LLAMA_MODEL_SIZES_LAYERS.keys():
            my_device_map = infer_llama_device_map(lm_size, num_gpus)
            if 'chat' in lm_model:
                model_name = f"meta-llama/Llama-2-{lm_size}-chat-hf"
            else:
                model_name = f"meta-llama/Llama-2-{lm_size}-hf"
                
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                cache_dir=lm_cache_dir,
                resume_download=True,
                device_map=my_device_map
            )
            tokenizer = LlamaTokenizer.from_pretrained(
                model_name, 
                cache_dir=lm_cache_dir
            )
            tokenizer.sep_token_id =  tokenizer.eos_token_id 
            tokenizer.pad_token_id =  tokenizer.eos_token_id 
    elif lm_model in ['opt']:
        if 'opt' not in lm_cache_dir:
            lm_cache_dir = os.path.join(lm_cache_dir, 'opt')
        
        if lm_size in OPT_MODEL_SIZES_LAYERS.keys():
            my_device_map = infer_opt_device_map(lm_size, num_gpus)

        model_name = f"facebook/opt-{lm_size}"
        model = OPTForCausalLM.from_pretrained(model_name, 
                                        cache_dir=lm_cache_dir, 
                                        device_map='auto'
                                        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=lm_cache_dir)
        tokenizer.sep_token_id =  tokenizer.eos_token_id 
        tokenizer.pad_token_id =  tokenizer.eos_token_id 

    else:
        raise ValueError()
    # dispatch_model(model, device_map)
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = make_language_model_and_tokenizer('pythia', '70m', 'outputs', 0)
    print(model)