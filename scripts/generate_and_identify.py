from sip_lib.make_llm import make_language_model_and_tokenizer
from sip_lib.utils.colorize import colorize
from IPython.display import display, HTML
from omegaconf import OmegaConf
import numpy as np 
import matplotlib
import re 
import os 
import datetime 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path")
parser.add_argument("--prompt", default="ChatGPT is ")
parser.add_argument("--logit_filter", default=0.5, type=float)

args = parser.parse_args()

current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
save_dir = os.path.join(args.path, "generated", current_time)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



flags = OmegaConf.load(os.path.join(args.path, 'config.yaml'))
lm_model, tokenizer = make_language_model_and_tokenizer(lm_cache_dir=flags.activation_gather_config.lm_cache_dir, 
                                                        num_gpus=flags.activation_gather_config.num_gpus, **flags)
import torch 
from sip_lib.identifiers.get_identifier import make_identifier
identifier_model = make_identifier(flags.identifier_model,  **flags)
identifier_model.load_state_dict(torch.load(os.path.join(args.path, "model.pt")))
identifier_model.to("cuda:0")
identifier_model.eval()

print(flags)

# 1. generate
input_ids = tokenizer([args.prompt], return_tensors="pt" )
for k, v in input_ids.items():
    input_ids[k] = v.to("cuda:0")
output = lm_model.generate(**input_ids, max_length=50, do_sample=False, pad_token_id=tokenizer.eos_token_id,)
text = tokenizer.batch_decode(output)[0]

# 2. gather hiddens
input_ids = tokenizer([text], return_tensors="pt" )
for k, v in input_ids.items():
    input_ids[k] = v.to("cuda:0")
hiddens = lm_model.forward(**input_ids, output_hidden_states=True).hidden_states[flags.hook_layer]
print("----------------------------------")
print("> Hidden shape:", hiddens.shape)
print(" Generated: ")
print(text)


if flags.source_label_type == "unigram":
    new_x = hiddens
elif flags.source_label_type == "bigram":
    first_gram = hiddens[:,:-1, ...] # drop the last 
    second_gram = hiddens[:,1:,...] # drop the first
    new_x = torch.cat([first_gram, second_gram], dim=-1)
elif flags.source_label_type == "trigram":
    first_gram = hiddens[:,:-2, ...] # drop the last 
    second_gram = hiddens[:,1:-1,...] # drop the first
    third_gram = hiddens[:,2:, ...]
    new_x = torch.cat([first_gram, second_gram, third_gram], dim=-1)

output = identifier_model(new_x.to("cuda:0"))
id_to_vocab = {v:k for k,v in tokenizer.get_vocab().items()}
tokenized = tokenizer(text)['input_ids']
decoded_text = tokenizer.decode(tokenized) 
decoded_tokens = [] 
for idx in tokenized:
    decoded_tokens.append(id_to_vocab[idx])
    


labels = output.argmax(dim=-1).squeeze(0)
logits = torch.softmax(output, dim=-1).max(dim=-1)[0].squeeze(0)
labels[logits<args.logit_filter] = -1

decoded_tokens_for_print = [re.sub("<0x0A>", "<br>", d) for d in decoded_tokens]
decoded_tokens_for_print = [re.sub("<s>", "", d) for d in decoded_tokens_for_print]

words = decoded_tokens_for_print
color_array = labels.detach().cpu().numpy()
color_array_set = sorted(list(set(color_array)))
color_array = [color_array_set.index(k) for k in color_array]

# visualize labels
def cmap(x):
    cmap = matplotlib.colormaps.get_cmap('tab20')
    if x==0:
        return (1,1,1)
    else:
        return cmap(x)

s = colorize(words, color_array, color_map_version=1, custom_mapping=cmap)
# display(HTML(s))
with open(os.path.join(save_dir, f'source_labels.html'), 'w') as f:
    f.write(s)


# visualize argmax logits
color_array = logits.detach().cpu().numpy()
s = colorize(words, color_array, color_map_version=1)
# display(HTML(s))
with open(os.path.join(save_dir, f'source_logits.html' ), 'w') as f:
    f.write(s)
    
# save info
with open(os.path.join(save_dir, f'source_labels.txt'), 'w') as f:
    f.write("> Labels:\n")
    f.write(str(labels)+"\n\n")
    f.write("> text:\n")
    f.write(text+"\n\n")
    
    f.write("> Predicted Labels \n")
    for label in color_array_set:
        if label != -1:
            f.write(f"label:{label}\n")

print("---------------------------------------------------")
print("Predicted Argmax Labels:")
print(labels)
print(f"🚀 done. see:{save_dir}\n")
