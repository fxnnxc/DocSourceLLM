# Source Identification of LLM 

🌻Official Repo for ICPRAI 2024 - Identifying the Source of Generation for Large Language Models

<p align="center" >
<img src="/assets/paper_visualize_new.png" width="80%">
</p> 


```bash 
pip install -e . 
```

## Example Code 
Please run the following code first which trains identifier for **Llama2 model**. 

```bash 
bash shells/example.sh
```

### Option 1 
Check the save file in 

```bash 
outputs/train_identifier/cut_labels_100/llama2_7b/tiny/bigram/seed_0/layer_26/generated
```

### Option 2 

Check [the jupyter notebook:visualize](visualize.ipynb) for the interactive codes. 


## Code Structure 

```
--scripts
    -- gather_activation.py     # gather activation of LLM
    -- train_identifier.py      # trains an identifier 
    -- generate_and_identify.py # generated texts from a prompt and identify labels
--sip_lib
    --data          # processing docuemnts.
    --hooks         # for gathering activations 
    --identifiers   # torch modules for the FFNs
    --utils         # store utility functions
    --make_llm.py   # LLM loading and info
```

## Run all 

To run all LLMs, MLP types, n_grams run the following shell scripts 
```bash
bash shells/gather_activaiton.sh
bash shells/train_identifiers.sh
```

## Citation 

```
TBD
```