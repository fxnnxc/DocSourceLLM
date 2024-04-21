import os 
import torch 
import pickle 
import datetime 
import numpy as np 
import gc 
import copy
import random 
from tqdm import tqdm 
from omegaconf import OmegaConf
from transformers import get_scheduler
from sip_lib.identifiers.get_identifier import make_identifier
from sip_lib.utils.train_helper import TrainHelper
from sip_lib.utils.seed_data import split_indices
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

def train(identifier, num_outputs, x, y, flags, **kwargs):
    # --------------------------------------------
    # 3. make identifier model 
    class TorchDataset(Dataset):
        def __init__(self, x, y, train, seed_data, split, **kwargs):
            train_indices, test_indices = train_indices,  test_indices = split_indices(len(x), seed_data, split)
            if train :
                self.x = x[train_indices]
                self.y = y[train_indices]
            else:
                self.x = x[test_indices]
                self.y = y[test_indices]
            print("size:", self.x.size(), self.y.size())
        def __len__(self):
            return len(self.x)
        def __getitem__(self, idx):
            x = self.x[idx].float()
            y = self.y[idx]
            return x, y

    if flags.source_label_type == "unigram":
        # replace the token dim to the data dimension
        new_x = x.reshape(-1, x.size(-1))
        new_y = y.repeat_interleave(x.size(1))
        
    elif flags.source_label_type == "bigram":
        first_gram = x
        second_gram = x.clone()
        
        first_gram = first_gram[:,:-1, ...] # drop the last 
        second_gram = second_gram[:,1:,...] # drop the first
        new_x = torch.cat([first_gram, second_gram], dim=-1)
        new_x = new_x.reshape(-1, new_x.size(-1))
        new_y = y.repeat_interleave(x.size(1)-1)
         
    elif flags.source_label_type == "trigram":
        first_gram = x
        second_gram = x.clone()
        third_gram = x.clone()
        
        first_gram = first_gram[:,:-2, ...] # drop the last 
        second_gram = second_gram[:,1:-1,...] # drop the first
        third_gram = third_gram[:,2:, ...]
        new_x = torch.cat([first_gram, second_gram, third_gram], dim=-1)
        new_x = new_x.reshape(-1, new_x.size(-1))
        new_y = y.repeat_interleave(x.size(1)-1)

    else:
        raise ValueError(f"not implemented source_label_type {flags.source_label_type}")
    
    print("label data:", new_x.shape, new_y.shape)
    
    gpt_hidden_size = x.shape[-1]
    flags.gpt_hidden_size = gpt_hidden_size
    flags.num_outputs = num_outputs
    identifier_model = make_identifier(identifier,  **flags)

        
    torch.save(identifier_model.state_dict(), os.path.join(flags.save_dir,  'model.pt'))
    
    # --------------------------------------------
    # 4. train identifier model 
    train_dataset = TorchDataset(new_x, new_y, True, flags.seed_data, flags.split, )
    test_dataset  = TorchDataset(new_x, new_y, False, flags.seed_data, flags.split,)
    train_dataloader =  DataLoader(train_dataset, batch_size=flags.batch_size, shuffle=True)
    eval_dataloader =  DataLoader(test_dataset, batch_size=flags.batch_size, shuffle=False)
    train_dataloader_for_eval = DataLoader(train_dataset, batch_size=flags.batch_size, shuffle=False)
    
    th = TrainHelper(num_steps_per_epoch=len(train_dataloader), num_eval=5, num_save=1, num_epochs=flags.num_epochs)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    if flags.optim == "adam" :
        optimizer = torch.optim.Adam(identifier_model.parameters(), lr=flags.lr)
    elif flags.optim == "sgd" : 
        optimizer = torch.optim.SGD(identifier_model.parameters(), lr=flags.lr)
    elif flags.optim == "rmsprop" : 
        optimizer = torch.optim.RMSprop(identifier_model.parameters(), lr=flags.lr)
        
    lr_scheduler = get_scheduler(
                    name="linear",
                    optimizer=optimizer,
                    num_warmup_steps=1,
                    num_training_steps=th.num_train_steps,
                )
        
    summary_writer = SummaryWriter(log_dir=os.path.join(flags.save_dir, ))
    
    device = flags.device
    identifier_model.to(device)
    target_holder = torch.zeros(flags.batch_size, num_outputs).to(flags.device)
    
    OmegaConf.save(flags, os.path.join(flags.save_dir, "config.yaml"))
    with tqdm(range(th.num_train_steps)) as pbar: 
        for epoch in range(th.num_epochs):
            for batch in train_dataloader:
                pbar.update(1)
                th.update_global_step()
                
                x = batch[0]
                y = batch[1]
            
                x = x.to(device)            
                y = y.to(device)            
                
                target = target_holder[:len(y), ...]
                target.fill_(0)
                for i, label in enumerate(y.int()):
                    target[i, label] = 1.0      
                
                y_hat = identifier_model(x)
                loss = loss_fn(y_hat, target)      
                          
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(identifier_model.parameters(), 5.0)
                optimizer.step()
                lr_scheduler.step()
                pbar.set_postfix({"loss": f"{loss.item():.3f}", })
                                
                if th.global_step % 100 == 0:
                    summary_writer.add_scalar("in_training/loss", loss.item(), th.global_step)
                                
                if th.is_eval_step():
                    identifier_model.eval()
                    for name, dataloader in zip([ 'train', 'eval',], [ train_dataloader_for_eval, eval_dataloader]):
                        losses = [] 
                        count = 0
                        eq = 0 
                        for step, batch in enumerate(dataloader):
                            with torch.no_grad():
                                x = batch[0]
                                y = batch[1]
                                x = x.to(device)            
                                y = y.to(device)            
                                y_hat = identifier_model(x)
                                
                                target = target_holder[:len(y), ...]
                                target.fill_(0)
                                for i, label in enumerate(y.int()):
                                    target[i, label] = 1.0      
                                
                                y_hat = identifier_model(x)
                                loss = loss_fn(y_hat, target)   
                                
                                eq += (y_hat.argmax(dim=-1) == target.argmax(dim=-1)).sum().item()       
                                losses.append(loss) # squared error summed by all minibatch samples
                                count += y.size(0)
                        eq = eq/count
                        loss = (torch.sum(torch.tensor(losses)).item() / count) 
                        print(f"[EVAL-{name}] {flags.lm_model} {flags.lm_size} {flags.data}  step:{th.global_step} |  acc:{eq:.3f} | loss:{loss} |")
                        summary_writer.add_scalar(f"{name}/loss", loss, th.global_step)
                        summary_writer.add_scalar(f"{name}/acc", eq, th.global_step)
                    identifier_model.train()

                                        
    torch.save(identifier_model.state_dict(), os.path.join(flags.save_dir, 'model.pt'))
    flags.done = True
    OmegaConf.save(flags, os.path.join(flags.save_dir, "config.yaml"))
    

def main(args):
    """ P-pipeline
    1. create sip dataset 
    2. load hiddens 
    3. create identifier model
    4. train the model 
    """
    # --------------------------------------------
    #                  A-pipeline
    # --------------------------------------------
    flags  = OmegaConf.create({})
    flags.done = False
    flags.datetime = datetime.datetime.now().strftime("%m%d_%H%M%S")
    
    for k, v in vars(args).items():
        print(">>>", k, ":" , v)
        setattr(flags, k, v)

    random.seed(flags.seed)
    np.random.seed(flags.seed)
    torch.manual_seed(flags.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)

    # --------------------------------------------
    # 2. create sip dataset 
    from sip_lib.data.get_data import make_data
    dataset = make_data(flags.data, 'sip', flags.data_cache_dir)
    if flags.cut_labels != -1:
        n = len(dataset['source_label'])
        labels = dataset['source_label']
        indices = [i for i in range(n) if labels[i] < flags.cut_labels]
        dataset = dataset.select(indices)
        
    min_label = min(dataset['source_label'])
    max_label = max(dataset['source_label'])
    num_outputs = max_label - min_label + 1

    # --------------------------------------------
    # 3. load hiddens 
    a_flags  = OmegaConf.load(os.path.join(flags.hidden_dir, 'config.yaml'))
    hiddens  = pickle.load(open(os.path.join(flags.hidden_dir, 'hiddens.pkl'), 'rb'))
    if flags.cut_labels != -1:
        hiddens_temp = hiddens[:,:len(dataset['source_label']), ...]
        del hiddens
        gc.collect()
        hiddens = hiddens_temp
    
    # --------------------------------------------
    # 4. train
    print("🚀Start training identification...")
    print(f"train of {flags.lm_model}-{flags.lm_size}, {a_flags.hook_layers}, ")
    flags.activation_gather_config = a_flags
    identifier = flags.identifier_model
    base_dir = copy.deepcopy(flags.save_dir)
    for hidden_index, hidden_layer in enumerate(a_flags.hook_layers):
        flags.save_dir = os.path.join(base_dir, f"layer_{hidden_layer}")
        flags.hook_layer = hidden_layer
        if not os.path.exists(flags.save_dir):
            os.makedirs(flags.save_dir)
        
        x = hiddens[hidden_index]
        y = torch.tensor(dataset['source_label']).long()
        assert x.size(0) == y.size(0)
        print("📌 X (Samples, Sequences, Hiddens):", x.size())
        print("📌 Y (Samples):", y.size())
        print("📌 #documents:", len(y.unique()))
        train(identifier, num_outputs, x, y, flags)        
    # --------------------------------------------
    # 6. save the hiddens 


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--identifier_model")
parser.add_argument("--lm_model")
parser.add_argument("--lm_size")
parser.add_argument("--seed_data", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--data")
parser.add_argument("--data_cache_dir")
parser.add_argument("--hidden_dir")
parser.add_argument("--cut_labels", type=int, default=-1)

parser.add_argument("--source_label_type", type=str)
parser.add_argument("--save_dir")
parser.add_argument("--test", action='store_true')

parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--lr",  type=float),
parser.add_argument("--split",  type=float),
parser.add_argument("--optim", type=str),
parser.add_argument("--device", type=str),
parser.add_argument("--linear_hidden_size"),
parser.add_argument("--linear_activation", default='relu', type=str),
# parser.add_argument("--linear-n-layers", default=2, type=int),

args = parser.parse_args()
main(args)

