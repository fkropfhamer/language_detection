import nltk
import torch
from nltk.corpus import udhr
from torch.utils.data import Dataset, DataLoader
from torch import nn


languages = ['English', 'German_Deutsch', 'French_Francais', 'Italian_Italiano', 'Spanish_Espanol']

alphabet = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","p","q","r","s","t","u","v","w","x","y","z","À","Á","Â","Ã","Ä","Å","Æ","Ç","È","É","Ê","Ë","Ì","Í","Î","Ï","Ð","Ñ","Ò","Ó","Ô","Õ","Ö","×","Ø","Ù","Ú","Û","Ü","Ý","Þ","ß","à","á","â","ã","ä","å","æ","ç","è","é","ê","ë","ì","í","î","ï","ð","ñ","ò","ó","ô","õ","ö","÷","ø","ù","ú","û","ü","ý","þ","ÿ"]

class UDHRDataset(Dataset):
  def __init__(self, transform=None, target_transform=None):
    self.labels = []

    self.sentences = []

    for idx, language in enumerate(languages):
      for sentence in udhr.sents(f"{language}-Latin1"):
        self.labels.append(idx)
        self.sentences.append(sentence)

    self.features = [self.get_features(sentence) for sentence in self.sentences]

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    feature = torch.tensor(self.features[idx])
    label = self.labels[idx]

    sample = feature, label

    return sample

  @staticmethod
  def get_features(sentence):
    features = []
    char_sum = sum([len(word) for word in sentence])
    for _, character in enumerate(alphabet): 
      features.append(sum([word.count(character) / char_sum for word in sentence]))

    return features




training_data = UDHRDataset()

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()

    self.linear_relu_stack = nn.Sequential(
            nn.Linear(len(alphabet), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, len(languages)),
            nn.ReLU(),
    )

  def forward(self, inputs):
    logits = self.linear_relu_stack(inputs) 
    return logits 

model = NeuralNetwork()

loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        #X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            #X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5 
for t in range(epochs):
  print(f"Epoch {t+1}\n-------------------------------")
  train(train_dataloader, model, loss_fn, optimizer)
  test(train_dataloader, model)
