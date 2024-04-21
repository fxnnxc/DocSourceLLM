

from sip_lib.identifiers.mlp_identifier import MLPIdentifier
# from sip_lib.probes.rnn_prober import RNNProbe 

def make_identifier(identifier, gpt_hidden_size, num_outputs, source_label_type, **kwargs):    
    if identifier == "mlp":
        if source_label_type == "unigram":
            hidden_size=gpt_hidden_size
        elif source_label_type == "bigram":
            hidden_size=gpt_hidden_size*2
        elif source_label_type == "trigram":
            hidden_size=gpt_hidden_size*3
            
        identifier = MLPIdentifier(hidden_size=hidden_size, num_outputs=num_outputs, **kwargs)
    else:
        raise ValueError(f"not implemented identifier {identifier}")
    return identifier