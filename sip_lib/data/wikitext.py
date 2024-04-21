
from datasets import load_dataset
def load_wikitext(name, cache_dir):
    assert name in ['wikitext-103-v1', 'wikitext-2-v1'] 
    raw_datasets = load_dataset('EleutherAI/wikitext_document_level',   
                            name=name,
                            cache_dir=cache_dir  # '/data/EleutherAI_wikitext'
                    )
    train = raw_datasets['train'] 
    return train
