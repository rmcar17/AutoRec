from abc import ABC, abstractmethod

import random

class Batcher:

    def __init__(self, data, batch_size):
        self.data = data
        self.n_rows = data.shape[0]
        self.batch_size = batch_size

        self.cur_indices = list(range(self.n_rows))

    def __len__(self):
        return self.n_rows
    
    def __iter__(self):
        self.cur_indices = list(range(self.n_rows))
        random.shuffle(self.cur_indices)

        return self
    
    def __call__(self, seed=None):
        random.seed(seed)
        
        return self

    @abstractmethod
    def __next__(self):
        pass


class MaskBatcher(Batcher):
    
    def __init__(self, data, data_mask, batch_size):
        super().__init__(data, batch_size)
        self.mask = data_mask
    
    def __next__(self):
        if len(self.cur_indices) < 1:
            raise StopIteration
            
        n = 0
        batch_indices = []
        while self.cur_indices and n < self.batch_size:
            idx = self.cur_indices.pop()
            batch_indices.append(idx)
            n += 1

        data_batch = self.data[batch_indices, :]
        mask_batch = self.mask[batch_indices, :]

        return data_batch, mask_batch