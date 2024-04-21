import re 
from tqdm import tqdm 



def process_sip(name, dataset, **kwargs):
    if name =="wikitext-103-v1":
        dataset = make_eleuther_wikitext_source_dataset(dataset)
    return dataset

def make_eleuther_wikitext_source_dataset(dataset):    
    pattern = re.compile("=.*=")
    HOLDER = {}
    def get_source(example):
        page = example['page']
        full = re.findall(pattern, page)[0]
        source = full.strip().split("=")[1]
        if source not in HOLDER:
            HOLDER[source] = len(HOLDER)
        return {'source_text': source, 'source_label':HOLDER[source]}
    dataset = dataset.map(get_source)        
    return dataset



