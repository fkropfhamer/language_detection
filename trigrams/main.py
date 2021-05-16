import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

# https://towardsdatascience.com/deep-neural-network-language-identification-ae1c158f6a7d

data = pd.read_csv('../data/sentences.csv', sep='\t', encoding='utf8', index_col=0, names=['language', 'text'])

len_cond = [True if 20 <= len(s) <= 200 else False for s in data['text']]
data = data[len_cond]

languages = ['deu', 'eng', 'fra', 'ita', 'por', 'spa']
data = data[data['language'].isin(languages)]

data_trimmed = pd.DataFrame(columns=['language', 'text'])

for language in languages:
  language_trimmed = data[data['language'] == language].sample(50000, random_state=100)
  data_trimmed = data_trimmed.append(language_trimmed)

data_shuffled = data_trimmed.sample(frac=1)

train = data_shuffled[0:210000]
valid = data_shuffled[210000:270000]
tests = data_shuffled[270000:300000]

print(data.head())
print(data.count())

def get_trigrams(corpus, n_feat=200):
  vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3), max_features=n_feat)
  X = vectorizer.fit_transform(corpus)
  
  feature_names = vectorizer.get_feature_names()

  return feature_names


features = {}
features_set = set()

for language in languages:
  corpus = train[train.language==language]['text']
  trigrams = get_trigrams(corpus)

  features[language] = trigrams
  features_set.update(trigrams)


vocabulary = dict()
for idx, feature in enumerate(features_set):
  vocabulary[feature] = idx

vectorizer = CountVectorizer(analyzer='char', ngram_range=(3,3), vocabulary=vocabulary)

corpus = train['text']

X = vectorizer.fit_transform(corpus)

feature_names = vectorizer.get_feature_names()

train_feat = pd.DataFrame(data=X.toarray(), columns=feature_names)

train_min = train_feat.min()
train_max = train_feat.max()
train_feat = (train_feat - train_min)/(train_max-train_min)

#Add target variable 
train_feat['language'] = list(train['language'])
train_feat.dropna()

print(train_feat.head())
print(train_feat.shape)

class LanguageDataset(Dataset):
  def __init__(self, df, transform=None, target_transform=None):
    self.df = df

    self.features = df.drop(['language'], axis=1)
    self.labels = df['language']

  def __len__(self):
    return self.labels.shape[0]

  def __getitem__(self, idx):
    feature = torch.tensor(self.features.iloc[idx])
    label = torch.tensor(languages.index(self.labels[idx]))

    sample = feature, label

    return sample

training_data = LanguageDataset(train_feat)

train_dataloader = DataLoader(training_data, batch_size=100, shuffle=True)

class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()

    self.linear_relu_stack = nn.Sequential(
            nn.Linear(train_feat.shape[1] - 1, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, len(languages)),
            nn.ReLU(),
    )

  def forward(self, inputs):
    logits = self.linear_relu_stack(inputs) 
    return logits 

model = NeuralNetwork()

loss_fn = nn.CrossEntropyLoss()
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
