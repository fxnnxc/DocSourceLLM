
# 2024.04.06
# author : Bumjin Park
data=wikitext-103-v1
lm_cache_dir='~/.cache/huggingface/hub'
data_cache_dir='~/.cache/huggingface/datasets'

num_gpus=1
batch_size=4
max_token_length=256
max_labels=100

# model / size / layers
# please see sip_lib/make_llm.py to find the number of layers
llama1=('llama2' '7b' '[26]')

# 🥕 put other models for gathering activations 
models=(
    llama1
)

for p in ${models[@]}
do 
    declare -n pair=$p 
    pair=("${pair[@]}")
    lm_model=${pair[0]}
    lm_size=${pair[1]}
    hook_layers=${pair[2]}

    save_dir='outputs/gather_activaiton/'$lm_model'_'$lm_size/$hook_layers
    # python scripts/gather_activaiton.py \
    #     --lm_model $lm_model \
    #     --lm_size $lm_size \
    #     --lm_cache_dir $lm_cache_dir \
    #     --num_gpus $num_gpus \
    #     --data $data \
    #     --data_cache_dir $data_cache_dir \
    #     --max_labels $max_labels \
    #     --hook_layers $hook_layers \
    #     --batch_size $batch_size \
    #     --save_dir $save_dir \
    #     --max_token_length $max_token_length
done 

tiny=('[128]')
mlp_types=(
    tiny 
)

# ----------------------------------------
# Parameters 
data=wikitext-103-v1
data_cache_dir='~/.cache/huggingface/datasets'
batch_size=64
seed_data=0
seed=0
device=cuda:0
lr=1e-3
optim=adam
split=0.7
num_epochs=10
cut_labels=100

# ----------------------------------------
# LM Model Loop
# ----------------------------------------
models=(
    llama1
)

for p in ${models[@]}
do 
declare -n pair=$p 
pair=("${pair[@]}")
lm_model=${pair[0]}
lm_size=${pair[1]}
hook_layers=${pair[2]}
hidden_dir='outputs/gather_activaiton/'$lm_model'_'$lm_size/$hook_layers

# ----------------------------------------
# Representation Loop
# ----------------------------------------
for source_label_type in bigram
do 
# ----------------------------------------
# MLP Model Loop
# ----------------------------------------
for mlp_name in ${mlp_types[@]}
do 
declare -n mlp=$mlp_name 
mlp_pair=("${mlp[@]}")
identifier_model=mlp
linear_hidden_size=${mlp_pair[0]}
linear_activation=relu

save_dir='outputs/train_identifier/cut_labels_'$cut_labels'/'$lm_model'_'$lm_size'/'$mlp_name'/'$source_label_type'/seed_'$seed
# python scripts/train_identifier.py \
#     --identifier_model $identifier_model \
#     --source_label_type $source_label_type \
#     --lm_model $lm_model \
#     --lm_size $lm_size \
#     --hidden_dir $hidden_dir \
#     --data $data \
#     --data_cache_dir $data_cache_dir \
#     --save_dir $save_dir \
#     --batch_size $batch_size \
#     --lr $lr \
#     --optim $optim \
#     --seed $seed \
#     --seed_data $seed_data \
#     --split $split \
#     --num_epochs $num_epochs \
#     --device $device \
#     --cut_labels $cut_labels \
#     --linear_hidden_size $linear_hidden_size \
#     --linear_activation $linear_activation
done 
done 
done 

python scripts/generate_and_identify.py --path $save_dir/layer_26  --prompt "The ship"