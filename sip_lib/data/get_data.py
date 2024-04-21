
from .wikitext import load_wikitext
from .sip_dataset import process_sip
from .pg19 import load_pg19

DATASET_NAMES = ['wikitext-103-v1', 
                 'fever-v1.0', 
                 'fever-v2.0',
                 'fever-wiki_pages',
                 'true_false',
                 'pg19']

def make_data(name, data_type, data_cache_dir, **kwargs):
    assert name in DATASET_NAMES
    assert data_type in ['raw', 'sip']
    if 'wikitext' in name:
        if data_type == "raw":
            dataset = load_wikitext(name, data_cache_dir)
            dataset = dataset.rename_columns({'page': 'text'})
            return dataset
        else:
            dataset = load_wikitext(name, data_cache_dir)
            dataset = process_sip(name, dataset, **kwargs)
            dataset = dataset.rename_columns({'page': 'text'})
            return dataset
    elif 'pg19' in name:
        if data_type == "raw":
            dataset = load_pg19(data_cache_dir=data_cache_dir, **kwargs)
            return dataset 
        else:
            dataset = load_pg19( data_cache_dir=data_cache_dir, **kwargs)
            return dataset 
            