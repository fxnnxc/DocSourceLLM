import re 
import os 
import pandas as pd 
from tqdm import tqdm 
from datasets import Dataset, DatasetDict

def load_pg19(data_cache_dir, cut_labels=100, max_lines=256, **kwargs):
    if 'pg19' not in data_cache_dir:
        data_cache_dir = os.path.join(data_cache_dir, 'pg19')
    metadata = pd.read_csv(os.path.join(data_cache_dir, 'metadata.csv'), header=None)

    train_dir = sorted(os.listdir(os.path.join(data_cache_dir, 'train')))[:cut_labels]
    texts = [] 
    labels = []
    book_names = []
    current_label = 0 
    pattern = re.compile("[\w.,]+")
        
    print("pg19 data is loading....")
    for idx in tqdm(range(len(train_dir))):
        book = train_dir[idx]
        book_idx = int(book.split(".")[0])
        meta = metadata[metadata[0]==book_idx]
        label = meta[0].values[0]
        string_label = meta[1].values[0]
        text = []
        count =0
        # ----- read file ----- 
        with open(os.path.join(data_cache_dir, 'train', book), 'r') as f :
            while True:
                line = f.readline()
                if not line or count > max_lines:                
                    break
                line = re.findall(pattern, line, )
                line = " ".join(line)
                line = re.sub("\d+:\d+", "",  line)
                line = re.sub("\d+ \d+", "",  line)
                line = re.sub("\n", "",  line)
                text.append(line)
                count += 1
                
        if len(text) > max_lines:
            text = " ".join(text)
            if text is not None:
                texts.append(text)
                labels.append(current_label)
                book_names.append(string_label)
                current_label += 1 
    
    train_dataset = Dataset.from_dict({'text': texts, 'source_label':labels, 'book_names': book_names})
    return train_dataset