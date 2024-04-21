
import torch 
import torch.nn.functional as F 

def get_forward_hook(hook_type):
    def forward_hook_tokens(module, input, output):
        module.output =  output
        
    def forward_hook_sentence(module, input, output):
        with torch.no_grad():
            output = output.clone().detach()
            output = output.mean(dim=1)
        module.output = output 
        
    if hook_type == "tokens":
        return forward_hook_tokens
    
    if hook_type == "sentence":
        return forward_hook_sentence
    else:
        raise ValueError(f"{hook_type} is not implemented")



def register_hooks(modules, hook_type):
    hooks = [] 
    # register all the current hooks 
    for m in modules:
        hook_fn = get_forward_hook(hook_type)
        hook = m.register_forward_hook(hook_fn)
        hooks.append(hook)
        print(f"{m}:{hook}")
    print("[INFO] hooks are registered.")
    return hooks 

def remove_hooks(hooks):
    while len(hooks) > 0 :
        hook = hooks.pop()
        hook.remove()
    print("hooks are removed")
