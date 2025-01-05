import numpy as np

input_ids = np.load('/media/sien/media/data/train_data/tensor_1.npy',allow_pickle=True).item()

print(input_ids['input_ids'])
tmp = input_ids['input_ids']
len(tmp)