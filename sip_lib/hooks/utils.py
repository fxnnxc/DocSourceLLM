

from sip_lib.make_llm import (LLAMA_MODEL_SIZES_LAYERS, 
                              PYTHIA_MODEL_SIZES_LAYERS, 
                              OPT_MODEL_SIZES_LAYERS)

def get_lm_layer_names(lm_model, lm_size, indices=-1):
    module_names = [] 
    if  "pythia" in lm_model:
        name = "gpt_neox.layers"
        num_layers = PYTHIA_MODEL_SIZES_LAYERS[lm_size]
    elif "llama2" in lm_model:
        name = "model.layers"
        num_layers = LLAMA_MODEL_SIZES_LAYERS[lm_size]
    elif "opt" in lm_model:
        name = "decoder.layers"
        num_layers = OPT_MODEL_SIZES_LAYERS[lm_size]
        
    if indices == -1 :
        indices = [i for i in range(num_layers)] + [num_layers]
    else:
        assert max(indices) <= num_layers, f"{indices} | MAX: {num_layers}"
    module_names = module_names + [f'{name}.{i-1}' for i in indices if i>0]
    return module_names, indices


def get_module_by_name(lm_model, name):
    from functools import reduce
    return reduce(getattr, name.split("."), lm_model)