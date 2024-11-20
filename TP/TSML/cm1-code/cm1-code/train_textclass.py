#!/usr/bin/env python3
import sys, torch, collections, tqdm, pdb
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

################################################################################

class GRUClassifier(nn.Module):
  
  def __init__(self, d_embed, d_hidden, d_in, d_out):
    super().__init__() 
    self.embed = nn.Embedding(d_in, d_embed, padding_idx=0)
    self.gru = nn.GRU(d_embed, d_hidden, batch_first=True, bias=False)    
    self.dropout = nn.Dropout(0.1)
    self.decision = nn.Linear(d_hidden, d_out)      
    
  def forward(self, idx_words):
    embedded = self.embed(idx_words)    
    hidden = self.gru(embedded)[1].squeeze(dim=0)    
    return self.decision(self.dropout(hidden))    

################################################################################

class BOWClassifier(nn.Module):
  
  def __init__(self, d_embed, d_in, d_out):
    super().__init__() 
    self.embed = nn.Embedding(d_in, d_embed, padding_idx=0)
    self.dropout = nn.Dropout(0.3)
    self.decision = nn.Linear(d_embed, d_out)      
    
  def forward(self, idx_words):
    embedded = self.embed(idx_words)
    averaged = torch.mean(embedded, dim=1) # dim 0 is batch    
    return self.decision(self.dropout(averaged))

################################################################################

def perf(model, dev_loader, criterion):
  model.eval()
  total_loss = correct = 0
  for (X, y) in dev_loader:
    with torch.no_grad():
      y_scores = model(X) 
      total_loss += criterion(y_scores, y)
      y_pred = torch.max(y_scores, dim=1)[1] # argmax
      correct += torch.sum(y_pred.data == y)
  total = len(dev_loader.dataset)
  return total_loss / total, correct / total

################################################################################
    
def fit(model, epochs, train_loader, dev_loader):
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters()) 
  for epoch in range(epochs):
    model.train()
    total_loss = 0
    for (X, y) in tqdm.tqdm(train_loader) :      
      optimizer.zero_grad()
      y_scores = model(X)    
      loss = criterion(y_scores, y)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()  
    print("train_loss = {:.4f}".format(total_loss / len(train_loader.dataset)))
    print("dev_loss = {:.4f} dev_acc = {:.4f}".format(*perf(model, dev_loader, criterion)))

################################################################################

def pad_tensor(X, max_len):
  res = torch.full((len(X), max_len), 0)
  for (i, row) in enumerate(X) :
    x_len = min(max_len, len(X[i]))
    res[i,:x_len] = torch.LongTensor(X[i][:x_len])
  return res

################################################################################

def read_corpus(filename, wordvocab, tagvocab, train_mode=True, batch_mode=True):
  if train_mode :
    wordvocab = collections.defaultdict(lambda : len(wordvocab))
    wordvocab["<PAD>"]; wordvocab["<UNK>"] # Create special token IDs      
    tagvocab = collections.defaultdict(lambda : len(tagvocab))
  words, tags = [], []
  with open(filename, 'r', encoding="utf-8") as corpus:
    for line in corpus:
      fields = line.strip().split()
      tags.append(tagvocab[fields[0]])
      if train_mode :
        words.append([wordvocab[w] for w in fields[1:]])
      else :
        words.append([wordvocab.get(w, wordvocab["<UNK>"]) for w in fields[1:]])
  if batch_mode :
    dataset = TensorDataset(pad_tensor(words, 40), torch.LongTensor(tags))
    return DataLoader(dataset, batch_size=32, shuffle=train_mode), wordvocab, tagvocab 
  else :
    return words, tags, wordvocab, tagvocab
    
################################################################################

if __name__ == "__main__" : 
  if len(sys.argv) != 4 : # Prefer using argparse, more flexible
    print("Usage: {} trainfile.txt devfile.txt bow|gru".format(sys.argv[0]), file=sys.stderr) 
    sys.exit(-1)   
  hp = {"model_type": sys.argv[3], "d_embed": 250, "d_hidden": 200}
  train_loader, wordvocab, tagvocab = read_corpus(sys.argv[1], None, None)
  dev_loader, _, _ = read_corpus(sys.argv[2], wordvocab, tagvocab, train_mode=False)
  if hp["model_type"] == "bow" :
    model = BOWClassifier(hp["d_embed"], len(wordvocab), len(tagvocab))
  else :
    model = GRUClassifier(hp["d_embed"], hp["d_hidden"], len(wordvocab), len(tagvocab))
  fit(model, 30, train_loader, dev_loader)
  torch.save({"wordvocab": dict(wordvocab), 
              "tagvocab": dict(tagvocab), 
              "model_params": model.state_dict(),
              "hyperparams": hp}, "model.pt")

# ./train_textclass.py dumas_train.txt dumas_dev.txt bow
# mv model.pt model-bow.pt              
# ./train_textclass.py dumas_train.txt dumas_dev.txt gru
# mv model.pt model-gru.pt
