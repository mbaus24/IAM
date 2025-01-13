import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from conllu import parse_incr
import argparse
import json
from torch.utils.data import DataLoader, Dataset

# Extraction de sequoia
sys.path.append("../TP1/")

# Charger les phrases depuis le corpus
phrases = []
for sent in parse_incr(open("../TP1/sequoia-ud.parseme.frsemcor.simple.small", encoding='UTF-8')):
    mots = [tok["form"] for tok in sent if tok["upos"] in ["NOUN", "PROPN", "NUM"]]
    if mots:
        phrases.append(mots)

print("Exemples de phrases extraites :", phrases[:10])

# Initialisation du tokenizer et du modèle
model_name = "almanach/camembert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Préparation des embeddings
embeddings = []
for phrase in phrases:
    tok_sents = tokenizer(phrase, is_split_into_words=True, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        emb_sent = model(**tok_sents)['last_hidden_state'][0]
        embeddings.append(emb_sent)

# Alignement des sub-tokens et moyennage
final_embeddings = []
for phrase, emb_sent in zip(phrases, embeddings):
    word_embeddings = []
    for i, word_id in enumerate(tok_sents.word_ids()):
        if word_id is not None and word_id < len(phrase):
            word_embeddings.append(emb_sent[i].detach().numpy())

    # Moyennage des embeddings des sub-tokens pour chaque mot
    phrase_embeddings = []
    for word_idx in range(len(phrase)):
        word_embs = [emb for i, emb in enumerate(word_embeddings) if tok_sents.word_ids()[i] == word_idx]
        if word_embs:
            phrase_embeddings.append(np.mean(word_embs, axis=0))

    final_embeddings.append(phrase_embeddings)

print("Exemples d' embeddings finaux :", final_embeddings[:1])

# Dataset class
class SuperSenseDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Define the classifier
class SuperSenseClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(SuperSenseClassifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels)
        )

    def forward(self, x):
        return self.mlp(x)

# Training script
def train_model(args):
    with open(args.train_labels, 'r') as f:
        labels = json.load(f)

    dataset = SuperSenseDataset(final_embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    num_labels = len(set(labels))
    classifier = SuperSenseClassifier(input_dim=len(final_embeddings[0][0]), num_labels=num_labels)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        classifier.train()
        for embeddings, labels in dataloader:
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item()}")

    torch.save(classifier.state_dict(), args.output_model)

# Prediction script
def predict(args):
    with open(args.label_map, 'r') as f:
        label_map = json.load(f)
    rev_label_map = {v: k for k, v in label_map.items()}

    classifier = SuperSenseClassifier(input_dim=len(final_embeddings[0][0]), num_labels=len(label_map))
    classifier.load_state_dict(torch.load(args.model))

    predictions = []
    for embeddings in final_embeddings:
        logits = classifier(torch.tensor(embeddings, dtype=torch.float32))
        pred_labels = torch.argmax(logits, dim=-1).tolist()
        predictions.append([rev_label_map[label] for label in pred_labels])

    with open(args.output_file, 'w') as f:
        json.dump(predictions, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', choices=['train', 'predict'], required=True)
    parser.add_argument('--train_labels', type=str, help='Path to training labels')
    parser.add_argument('--label_map', type=str, help='Path to label map JSON')
    parser.add_argument('--output_model', type=str, help='Path to save the trained model')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--model', type=str, help='Path to trained model for prediction')
    parser.add_argument('--output_file', type=str, help='Path to save predictions')

    args = parser.parse_args()

    if args.script == 'train':
        train_model(args)
    elif args.script == 'predict':
        predict(args)
