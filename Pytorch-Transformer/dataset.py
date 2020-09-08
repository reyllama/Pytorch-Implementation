from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class IMDB(Dataset):
    def __init__(self, review, sentiment):
        self.review = review
        self.sentiment = sentiment

    def __getitem(self, index):
        review = self.review[index]
        sentiment = self.sentiment[index]
        return {'review': torch.LongTensor(review), 'sentiment': torch.LongTensor([sentiment])}

    def __len__(self):
        return len(self.sentiment)
