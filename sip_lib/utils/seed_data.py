import numpy as np 

def split_indices(length, seed_data, split):
    np.random.seed(seed_data)
    full_indices = [i for i in range(length)]
    train_indices = np.random.choice(full_indices, size= int(split *length), replace=False).tolist()
    test_indices = list(set(full_indices)-set(train_indices))
    return train_indices, test_indices