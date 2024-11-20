import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from Vocab import Vocab

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embeddings(x).view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][:-1]), torch.tensor(self.data[idx][-1])

def prepare_data(file_path, vocab, k=3):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().split()
    
    data = []
    for i in range(len(text) - k):
        sequence = [vocab.get_index(word) for word in text[i:i+k+1]]
        data.append(sequence)
    return data

def train(model, train_loader, num_epochs, learning_rate):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

def main():
    vocab = Vocab("embeddings-word2vecofficial.train.unk5.txt")
    data = prepare_data("Le_comte_de_Monte_Cristo.train.unk5.tok", vocab)
    random.shuffle(data)

    dataset = TextDataset(data)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LanguageModel(len(vocab), embedding_dim=100, hidden_dim=128)
    train(model, train_loader, num_epochs=10, learning_rate=0.001)

    torch.save(model.state_dict(), "language_model.pth")

if __name__ == "__main__":
    main()