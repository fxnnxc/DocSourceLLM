

import torch 
from torch.utils.data import DataLoader
from transformers import default_data_collator

def get_process_fn(tokenizer, process_type, max_token_length, **kwargs ):
    def text_preprocess_function(examples):        
        batch_size = len(examples['text']) 
        inputs = [str(text) for text in examples['text']]
        inputs = [" ".join([t for t in text.split()][:max_token_length]) for text in inputs]
        model_inputs = tokenizer(inputs)
        for i in range(batch_size):
            # cut the text and source length
            model_inputs["input_ids"][i] = model_inputs["input_ids"][i][:max_token_length-1]
            input_length = len(model_inputs["input_ids"][i])
            padding_length = max_token_length - (input_length) 
            model_inputs["input_ids"][i] = torch.tensor(
                                            [tokenizer.pad_token_id] * padding_length 
                                            + model_inputs["input_ids"][i][:max_token_length-1] 
                                            )
            model_inputs['attention_mask'][i] = torch.tensor([0] * (padding_length)+ [1] * (max_token_length - padding_length))
            
            assert model_inputs['input_ids'][i].shape == (max_token_length,), model_inputs['input_ids'][i].shape
            assert model_inputs['attention_mask'][i].shape == (max_token_length,), model_inputs['attention_mask'][i].shape
        return model_inputs
    
    def text_and_index_preprocess_function(examples):        
        batch_size = len(examples['text']) 
        inputs = [str(text) for text in examples['text']]
        inputs = [" ".join([t for t in text.split()][:max_token_length]) for text in inputs]
        model_inputs = tokenizer(inputs)
        model_inputs['label_index'] = [None for i in range(batch_size)]
        
        for i in range(batch_size):
            # cut the text and source length
            model_inputs["input_ids"][i] = model_inputs["input_ids"][i][:max_token_length-1]
            input_length = len(model_inputs["input_ids"][i])
            padding_length = max_token_length - (input_length)
            model_inputs["input_ids"][i] = torch.tensor(
                                            [tokenizer.pad_token_id] * padding_length 
                                            + model_inputs["input_ids"][i][:max_token_length-1] 
                                            )
            model_inputs['attention_mask'][i] = torch.tensor([0] * (padding_length)+ [1] * (max_token_length - padding_length))
            # -----------------
            model_inputs['label_index'][i] = torch.tensor(examples['source_index'][i])
            assert model_inputs['input_ids'][i].shape == (max_token_length,), model_inputs['input_ids'][i].shape
            assert model_inputs['attention_mask'][i].shape == (max_token_length,), model_inputs['attention_mask'][i].shape
        return model_inputs
    
    if process_type == "text":
        return text_preprocess_function
    elif process_type == "text_index":
        return text_and_index_preprocess_function
    else:
        raise ValueError("not implemented {0}".format(process_type))
        

def make_raw_dataset_and_dataloader(dataset, tokenizer, process_type,  proc_batch_size, batch_size,  num_proc, max_token_length,  **kwargs):
    col_names = dataset.column_names
    if max_token_length <=0:
        max_token_length = max([ len(string.split(" ")) for string in  dataset['text']]) + 2 
        print(f"max_token_length is set to the maximum: {max_token_length}")
    dataset = dataset.map(
                get_process_fn(tokenizer, process_type, max_token_length, **kwargs),
                batched=True,
                num_proc=num_proc,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset...",
                batch_size=proc_batch_size,
        )
    
    dataset_for_loader = dataset.remove_columns(col_names)
    train_dataloader = DataLoader(dataset_for_loader, shuffle=False, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
    return  dataset, train_dataloader