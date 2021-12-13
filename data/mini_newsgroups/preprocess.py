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

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import AUROC
import pytorch_lightning as pl

from IPython import embed


nlp = English()
tokenizer = Tokenizer(nlp.vocab)


class SparseToDenseEncoder(pl.LightningModule):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = nn.Linear(in_dim, out_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.auroc = AUROC(pos_label=1)

    def forward(self, x):
        embedding = self.encoder(x)
        return F.normalize(embedding)

    def _compute_all_pairs(self, batch):
        x, y = batch
        batch_size = x.shape[0]
        embeds = self(x)
        sims = embeds @ embeds.T
        lbls = (y[:, None] == y[None, :]).float()
        idx = torch.triu_indices(batch_size, batch_size, 1)
        return (sims[idx[0], idx[1]], lbls[idx[0], idx[1]])

    def training_step(self, batch, batch_idx):
        pred, target = self._compute_all_pairs(batch)
        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, target = self._compute_all_pairs(batch)
        val_loss = self.loss_fn(pred, target)
        val_auroc = self.auroc(pred, target.long())
        metrics = {'val_loss': val_loss, 'val_auroc': val_auroc}
        self.log_dict(metrics)
        return metrics

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_auroc = torch.stack([x['val_auroc'] for x in outputs]).mean()
        print('val_loss', avg_loss.item())
        print('val_auroc', avg_auroc.item())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def get_doc_text(doc_path):
    with open(doc_path, 'r', encoding='utf8', errors='ignore') as f:
        full_doc = f.read()
        header_end = full_doc.find('\n\n')
        body = full_doc[header_end:].strip()
    return body


def sparse_matrix_to_sparse_tensor(matrix):
    mx = matrix.tocoo()
    values = mx.data
    indices = np.vstack((mx.row, mx.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = mx.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


if __name__ == '__main__':
    seed = 42
    pl.utilities.seed.seed_everything(seed)

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
    vectorizer = TfidfVectorizer(stop_words=['english'], max_df=250, min_df=3)
    vectorizer.fit_transform(corpus)
    
    train_sparse_embeds = vectorizer.transform(train_docs)
    dev_sparse_embeds = vectorizer.transform(dev_docs)
    test_sparse_embeds = vectorizer.transform(test_docs)

    train_feats_tensor = sparse_matrix_to_sparse_tensor(train_sparse_embeds)
    dev_feats_tensor = sparse_matrix_to_sparse_tensor(dev_sparse_embeds)
    test_feats_tensor = sparse_matrix_to_sparse_tensor(test_sparse_embeds)

    # remap split label names to ids
    all_labels = set([label for label in labels_by_split['test']])
    label_id_map = {v: i for i, v in enumerate(all_labels)}
    train_labels = [label_id_map[label] for label in labels_by_split['train']]
    dev_labels = [label_id_map[label] for label in labels_by_split['dev']]
    test_labels = [label_id_map[label] for label in labels_by_split['test']]

    train_ds = TensorDataset(train_feats_tensor, torch.LongTensor(train_labels))
    dev_ds = TensorDataset(dev_feats_tensor, torch.LongTensor(dev_labels))

    train_loader = DataLoader(train_ds, batch_size=100, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=100)

    dense_dim = 64
    model = SparseToDenseEncoder(train_sparse_embeds.shape[1], dense_dim)
    trainer = pl.Trainer(max_epochs=1000, check_val_every_n_epoch=100)
    trainer.fit(model, train_loader, dev_loader)
    trainer.validate(model, dev_loader)
    
    embed()
    exit()
