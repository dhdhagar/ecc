# Preprocess mini_newsgroups
#
# Usage:
#   python preprocess.py
#

import collections
import glob
import os
import pickle
import random 

from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from IPython import embed


nlp = English()
tokenizer = Tokenizer(nlp.vocab)


class SparseToDenseEncoder(pl.LightningModule):

    def __init__(self, in_dim, out_dim):
        super().__init()
        self.encoder = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        embedding = self.encoder(x)
        return F.normalize(embedding)

    def training_step(self, batch, batch_idx):
        x, y = batch
        embeds = self(x)

        # TODO: figure this out

        loss = F.mse_loss(x_hat, x)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def get_doc_text(doc_path):
    with open(doc_path, 'r', encoding='utf8', errors='ignore') as f:
        full_doc = f.read()
        header_end = full_doc.find('\n\n')
        body = full_doc[header_end:].strip()
    return body


if __name__ == '__main__':

    random.seed(42)

    train_size = 50
    dev_size = 25
    test_size = 25

    files_by_split = collections.defaultdict(list)
    labels_by_split = collections.defaultdict(list)

    for group in glob.glob('raw/mini_newsgroups/*'):
        group_name = group.split('/')[-1]
        group_files = glob.glob(group + '/*')

        assert len(group_files) == train_size + dev_size + test_size
        random.shuffle(group_files)
        
        files_by_split['train'].extend(group_files[:train_size])
        files_by_split['dev'].extend(
                group_files[train_size:train_size+dev_size])
        files_by_split['test'].extend(group_files[-test_size:])

        labels_by_split['train'].extend([group_name] * train_size)
        labels_by_split['dev'].extend([group_name] * dev_size)
        labels_by_split['test'].extend([group_name] * test_size)

    train_docs = [get_doc_text(doc_path) 
            for doc_path in files_by_split['train']]
    dev_docs = [get_doc_text(doc_path) 
            for doc_path in files_by_split['dev']]
    test_docs = [get_doc_text(doc_path) 
            for doc_path in files_by_split['test']]

    corpus = train_docs + dev_docs + test_docs
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(corpus)
    
    train_embeds = vectorizer.transform(train_docs)
    dev_embeds = vectorizer.transform(dev_docs)
    test_embeds = vectorizer.transform(test_docs)

    embed()
    exit()
