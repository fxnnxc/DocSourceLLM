
# 2024.04.06
# author : Bumjin Park
data=wikitext-103-v1
lm_cache_dir=' ~/.cache/huggingface/hub'
data_cache_dir='~/.cache/huggingface/datasets'

num_gpus=1
batch_size=4
max_token_length=256
max_labels=100

# model / size / layers
# please see sip_lib/make_llm.py to find the number of layers
llama1=('llama2' '7b' '[26]')
llama2=('llama2' '13b' '[32]')
llama3=('llama2_chat' '7b' '[26]')
llama4=('llama2_chat' '13b' '[32]')
pythia1=('pythia' '70m' '[4,5]')
pythia2=('pythia' '160m' '[8]')
pythia3=('pythia' '410m' '[18]')
pythia4=('pythia' '1b' '[12]')
pythia5=('pythia' '1.4b' '[20]')
pythia6=('pythia' '2.8b' '[26]')
pythia7=('pythia' '6.9b' '[26]')
pythia8=('pythia' '12b' '[28]')


# 🥕 put other models for gathering activations 
models=(
    pythia1
    # pythia4
)

for p in ${models[@]}
do 
    declare -n pair=$p 
    pair=("${pair[@]}")
    lm_model=${pair[0]}
    lm_size=${pair[1]}
    hook_layers=${pair[2]}

    save_dir='outputs/gather_activaiton/'$lm_model'_'$lm_size/$hook_layers
    python scripts/gather_activaiton.py \
        --lm_model $lm_model \
        --lm_size $lm_size \
        --lm_cache_dir $lm_cache_dir \
        --num_gpus $num_gpus \
        --data $data \
        --data_cache_dir $data_cache_dir \
        --max_labels $max_labels \
        --hook_layers $hook_layers \
        --batch_size $batch_size \
        --save_dir $save_dir \
        --max_token_length $max_token_length
done 


