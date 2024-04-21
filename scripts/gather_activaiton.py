import os 
import torch 
import pickle 
from tqdm import tqdm 
from omegaconf import OmegaConf

def main(args):
    """ Activation gather-pipeline
    1. create LM model
    2. create LM hook wrapper 
    3. create raw dataset 
    4. create tokenized dataset 
    5. gather activations 
    6. save activations
    """
    # --------------------------------------------
    #                  AH-pipeline
    # --------------------------------------------
    flags  = OmegaConf.create({})
    for k, v in vars(args).items():
        print(">>>", k, ":" , v)
        setattr(flags, k, v)

    # --------------------------------------------
    # 1. make lm model 
    from sip_lib.make_llm import make_language_model_and_tokenizer
    lm_model, tokenizer = make_language_model_and_tokenizer(**flags)

    # --------------------------------------------
    # 2. create hook for LM
    from sip_lib.hooks.hooks import register_hooks
    from sip_lib.hooks.utils import get_lm_layer_names, get_module_by_name
    flags.hook_layers = eval(flags.hook_layers)
    hook_module_names, indices = get_lm_layer_names(flags.lm_model, flags.lm_size, flags.hook_layers)
    if flags.hook_layers == -1:
        flags.hook_layers = indices

    # --------------------------------------------
    # 3. create raw dataset 
    from sip_lib.data.get_data import make_data
    dataset = make_data(flags.data, 'sip', flags.data_cache_dir)
    n = len(dataset['source_label'])
    labels = dataset['source_label']
    indices = [i for i in range(n) if labels[i] < flags.max_labels]
    dataset = dataset.select(indices)

    # --------------------------------------------
    # 4. create tokenized datset 
    from sip_lib.data.tokenize import make_raw_dataset_and_dataloader
    dataset, dataloader = make_raw_dataset_and_dataloader(dataset, 
                                                                tokenizer, 
                                                                'text',  
                                                                flags.batch_size, 
                                                                flags.batch_size,  
                                                                num_proc=1,
                                                                max_token_length=flags.max_token_length)
    
    # --------------------------------------------
    # 5. gather activations 
    print("🚀 Start gathering activations...")
    print(f"gather activation of {flags.lm_model}-{flags.lm_size}, {flags.data},{flags.hook_layers}")
    n_layers = len(flags.hook_layers)
    hiddens = None 
    losses = [] 
    idx = 0
    with tqdm(enumerate(dataloader), total=len(dataloader)) as pbar:
        pbar.set_description("Gathering activation")
        for step, batch in pbar:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to("cuda:0")

            with torch.no_grad():
                outputs = lm_model(batch['input_ids'], labels=batch['input_ids'], attention_mask=batch['attention_mask'],  output_hidden_states=True)
                logits = outputs.logits
                if hiddens is None:
                    # allocate memory first 
                    size = (n_layers, len(dataloader.dataset),  *outputs.hidden_states[-1].shape[1:])
                    print('Pre allocated memory of size... ', size) 
                    hiddens = torch.zeros(*size).half()
                    flags.hidden_size = size
                    print("done")
                losses.append(outputs.loss.item())
                for i in range(logits.shape[0]):
                    for j, layer in enumerate(flags.hook_layers):
                        hiddens[j, idx, ...] = outputs.hidden_states[layer][i].detach().cpu().half()
                    idx+=1

    # --------------------------------------------
    # 6. save the hiddens 
    if not os.path.exists(flags.save_dir):
        os.makedirs(flags.save_dir)
    with open(os.path.join(flags.save_dir, 'hiddens.pkl'), 'wb') as f:
            pickle.dump(hiddens, f, pickle.HIGHEST_PROTOCOL)
    OmegaConf.save(flags, os.path.join(flags.save_dir, 'config.yaml'))
    print(f"activations are saved at :{os.path.join(flags.save_dir, 'hiddens.pkl')}")
    print("done\n\n")


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lm_model")
parser.add_argument("--lm_size")
parser.add_argument("--lm_cache_dir")
parser.add_argument("--num_gpus", type=int)
parser.add_argument("--data")
parser.add_argument("--data_cache_dir")
parser.add_argument("--hook_layers", type=str)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--max_labels", type=int)
parser.add_argument("--max_token_length", type=int, default=-1)
parser.add_argument("--save_dir")

args = parser.parse_args()
main(args)

