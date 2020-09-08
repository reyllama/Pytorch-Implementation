from model import *
from dataset import *
import torch
import numpy
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import random
import os
import time

class MetaCritic(nn.Module):
    def __init__(self, n_layers, d_model, n_head, d_head, n_seq, n_vocab, n_out):
        super().__init__()
        self.transformer = Transformer(n_layers, d_model, n_head, d_head, n_seq, n_vocab)
        self.out = nn.Linear(d_model, n_out)

    def forward(self, enc_in, dec_in):
        dec_out, enc_self_attn_prob, dec_self_attn_prob, dec_enc_attn_prob = self.transformer(enc_in, dec_in)
        dec_out, _ = torch.max(dec_out, dim=1)
        out = self.out(dec_out)

        return out, enc_self_attn_prob, dec_self_attn_prob, dec_enc_attn_prob

def accuracy(yhat, y):
    return torch.sum((yhat>0.5)==y).item() / len(y)

def train(model, df, epochs=5, lr=0.002, batch_size=128):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    cuda = torch.cuda.is_available()
    base = np.Inf

    for fold in range(1, 6):
        train, valid = df[df['kfold']!=fold].reset_index(drop=True), df[df['kfold']==fold].reset_index(drop=True)
        xtrain, xvalid = train.review.values, valid.review.values
        xtrain, xvalid = torch.utils.rnn.pad_sequence(xtrain, batch_first=True, padding=0), torch.utils.rnn.pad_sequence(xvalid, batch_first=True, padding=0)

        traindset = IMDB(xtrain, train.sentiment.values)
        validdset = IMDB(xvalid, valid.sentiment.values)
        trainloader = DataLoader(traindset, batch_size=batch_size, shuffle=True)
        validloader = DataLoader(validdset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            loss_t, loss_v = 0,0
            acc_t, acc_v = 0,0

            for batch in trainloader:
                optimizer.zero_grad()
                x, y = batch['review'], batch['sentiment'].float()
                if cuda:
                    x, y = x.cuda(), y.cuda()
                out = model(x)
                acc_t += accuracy(out.detach().cpu(), y.detach().cpu())
                loss = criterion(out, y)
                loss_t += loss.item()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                for batch in validloader:
                    x, y = batch['review'], batch['sentiment'].float()
                    if cuda:
                        x, y = x.cuda(), y.cuda()
                    out = model(x)
                    acc_v += accuracy(out.detach().cpu(), y.detach().cpu())
                    loss = criterion(out, y)
                    loss_v += loss.item()

            print("Fold: {}/{}, Epoch: {}/{}".format(fold, 5, epoch, epochs))
            print("Train Acc: {:.2f}".format(100*acc_t/len(trainloader)))
            print("Valid Acc: {:.2f}".format(100*acc_v/len(validloader)))

            if loss_v < base:
                torch.save(model.state_dict(), 'best_model.pt')
                print("Validation Loss decreased from {:.3f} ---> {:.3f}, saving model...".format(base, loss_v))
                base = loss_v
