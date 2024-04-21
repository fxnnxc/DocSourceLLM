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

models=(
    pythia1
    # pythia4
)

linear=('[]')
tiny=('[128]')
small=('[256,128]')
medium=('[512,256,128]')
large=('[1024,512,256,128]')
mlp_types=(
    linear 
    tiny 
    small 
    medium 
    large
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
for source_label_type in unigram bigram trigram 
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
python scripts/train_identifier.py \
    --identifier_model $identifier_model \
    --source_label_type $source_label_type \
    --lm_model $lm_model \
    --lm_size $lm_size \
    --hidden_dir $hidden_dir \
    --data $data \
    --data_cache_dir $data_cache_dir \
    --save_dir $save_dir \
    --batch_size $batch_size \
    --lr $lr \
    --optim $optim \
    --seed $seed \
    --seed_data $seed_data \
    --split $split \
    --num_epochs $num_epochs \
    --device $device \
    --cut_labels $cut_labels \
    --linear_hidden_size $linear_hidden_size \
    --linear_activation $linear_activation
done 
done 
done 