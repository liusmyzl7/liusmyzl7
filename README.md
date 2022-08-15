- 👋 Hi, I’m @liusmyzl7

- This is our group work on Word Representation in Biomedical Domain.
- Setup and Prerequisites
  Recommended environment

  Python 3.7 or newer
  Free disk space: 100GB

# Data Preparation

## Upload data on the cloud and unzip

import torch

!wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2021-07-26/document_parses.tar.gz

!tar -xf document_parses.tar.gz

## Parse the data
import json
import os
text1=[]
text3=[]
comtext=[]
for jsonfile in os.listdir("/content/document_parses/pdf_json")[:10]:
    jsonfile_path = os.path.join("/content/document_parses/pdf_json", jsonfile)
    with open(jsonfile_path, "r") as infile:
            content = json.load(infile)
            text0 = content.get("abstract", [])
            text1 = text1+text0

for jsonfile in os.listdir("/content/document_parses/pmc_json")[:10]:
    jsonfile_path = os.path.join("/content/document_parses/pmc_json", jsonfile)
    with open(jsonfile_path, "r") as infile:
            content = json.load(infile)
            text2 = content.get("abstract", [])
            text3 = text3+text2

comtext=text1+text3
comstr = " ".join('%s' %a for a in comtext)

## Preprocess
import re
from collections import Counter

def preprocess(text):
    # Convert all text to lowercase
    text = text.lower()

    # Replace punctuation with tokens so we can use them in our model
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace(':', ' <COLON> ')
    
    return text
    
### Output a list of text
text=preprocess(comstr)

# Tokenization

## Use split()
a=text.split(" ")
print(a)

## Use NLTK
!pip install nltk

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
b=word_tokenize(text)
print(b)

## Use Byte-Pair Encoding (BPE)
#Statistics the frequency of adjacent character pairs
import re, collections
def get_vocab(comstr):
    vocab = collections.defaultdict(int)
    for word in comstr.strip().split():
        vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab
print(get_vocab(comstr))

import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

vocab = get_vocab(comstr)
num_merges = 1000
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)
    
##### reference: [1] https://leimao.github.io/blog/Byte-Pair-Encoding/ 
##### [2] https://blog.csdn.net/qq_41020633/article/details/123622667?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165953811016782388024467%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165953811016782388024467&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-5-123622667-null-null.142^v39^pc_rank_34_2&utm_term=BPE&spm=1018.2226.3001.4187


# Build Word Representations

## Use N-gram Language Modeling
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence = a
### we should tokenize the input, but we will ignore that for now
### build a list of tuples.
### Each tuple is ([ a_i-CONTEXT_SIZE, ..., a_i-1 ], target word)
ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]
### Print the first 3, just so you can see what they look like.
print(ngrams[:3])

vocab = set(test_sentence)
a_to_ix = {a: i for i, a in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in ngrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([a_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor([a_to_ix[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!

### To get the embedding of a particular word, e.g. "the"
print(model.embeddings.weight[a_to_ix["the"]])

alphabet_upr = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
alphabet_lwr = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
alphabet = alphabet_upr+alphabet_lwr
alphabet

import pandas as pd
import numpy as np
import tensorflow as tf

df = pd.DataFrame(columns=['Words','Attribute1','Attribute2','Attribute3','Attribute4','Attribute5','Attribute6','Attribute7','Attribute8','Attribute9','Attribute10'])
counter = 0
number = 1
for counter in range(0,len(a)):
  judge = list(a[counter])
  num = 0
  for num in range(0,len(judge)):
    if judge[num] in alphabet:
      if num == len(judge)-1:
        df.loc[number] = [a[counter]]+(model.embeddings.weight[a_to_ix[a[counter]]].detach().numpy()).tolist()
        number = number+1
      else:
        num =num
    else:
     break
df
##### reference: https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

## Use Skip-gram with Negative Sampling
import torch
!wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2021-07-26/document_parses.tar.gz
!tar -xf document_parses.tar.gz

import json
import os
text1=[]
text3=[]
comtext=[]
for jsonfile in os.listdir("/content/document_parses/pdf_json")[:20000]:
    jsonfile_path = os.path.join("/content/document_parses/pdf_json", jsonfile)
    with open(jsonfile_path, "r") as infile:
            content = json.load(infile)
            text0 = content.get("abstract", [])
            text1 = text1+text0

for jsonfile in os.listdir("/content/document_parses/pmc_json")[:20000]:
    jsonfile_path = os.path.join("/content/document_parses/pmc_json", jsonfile)
    with open(jsonfile_path, "r") as infile:
            content = json.load(infile)
            text2 = content.get("abstract", [])
            text3 = text3+text2

comtext=text1+text3
import re
from collections import Counter

def preprocess(text):

    # Convert all text to lowercase
    text = text.lower()

    # Replace punctuation with tokens so we can use them in our model
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()
    
    # Remove all words with 5 or less occurences
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > 5]
    return trimmed_words
### get vocabulary 
text = " ".join('%s' %a for a in comtext)
words = preprocess(text)
#print(text[:100])
#print(words[:30])
print("Total # of words: {}".format(len(words)))
print("# of unique words: {}".format(len(set(words))))
### create dictionary
def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    - words: list of words
    - return: 2 dictionaries, vocab_to_int, int_to_vocab
    """
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True) # sorted the words by descending freq order
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)} #let the most frequent word be the first
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab
vocab_to_int, int_to_vocab = create_lookup_tables(words)
int_words = [vocab_to_int[word] for word in words]
print(vocab_to_int)
### Perform Word Subsampling
import random
import numpy as np

def subsample_words(int_words, threshold = 1e-5):
    #Count the frequency of words
    word_counts = Counter(int_words) 
    total_n_words = len(int_words)
  
    #calculate the frequency ratio of each word
    freq_ratios = {word: count/total_n_words for word, count in word_counts.items()} 
    #Calculate the probability of being deleted
    p_drop = {word: 1 - np.sqrt(threshold/freq_ratios[word]) for word in word_counts} 

    return [word for word in int_words if random.random() < (1 - p_drop[word])]
    #we discard the word if random.random() < probs[word]
train_words = subsample_words(int_words)
print(len(int_words))
print(len(train_words))
print(len(train_words)/len(int_words))
### Generate Context Targets
import random
def get_target(words, idx, max_window_size=5):
    R = random.randint(1, max_window_size)
    start = max(0,idx-R)
    end = min(idx+R,len(words)-1)
    targets = words[start:idx] + words[idx+1:end+1] # +1 since doesn't include this idx
    return targets
int_text = [i for i in range(10)]
print('Input: ', int_text)
idx=5 # word index of interest

for _ in range(5):
    target = get_target(int_text, idx=idx, max_window_size=5)
    print('Target: ', target)  # you should get some indices around the idx
### Generate Batches
def get_batches(words, batch_size, max_window_size=5):
    # only full batches
    n_batches = len(words)//batch_size
    words = words[:n_batches*batch_size]
    for i in range(0, len(words), batch_size):
        batch_of_center_words = words[i:i+batch_size]   # current batch of words
        batch_x, batch_y = [], []  

        for ii in range(len(batch_of_center_words)):  # range(batch_size) unless truncated at the end
            x = [batch_of_center_words[ii]]             # single word
            y = get_target(words=batch_of_center_words, idx=ii, max_window_size=max_window_size)  # list of context words

            batch_x.extend(x * len(y)) # repeat the center word (n_context_words) times
            batch_y.extend(y)
  
        yield batch_x, batch_y       # ex) [1,1,2,2,2,2,3,3,3,3], [0,2,0,1,3,4,1,2,4,5]
int_text = [i for i in range(20)]
x,y = next(get_batches(int_text, batch_size=4, max_window_size=5))

print('x\n', x)
print('y\n', y)
### Define COSINE SIMILARITY Function for Validation Metric
def cosine_similarity(embedding, n_valid_words=16, valid_window=100):
    """ Returns the cosine similarity of validation words with words in the embedding matrix.
        embedding: PyTorch embedding module
        n_valid_words: # of validation words (recommended to have even numbers)
    """
    all_embeddings = embedding.weight  # (n_vocab, n_embed) 
  ### sim = (a . b) / |a||b|
    magnitudes = all_embeddings.pow(2).sum(dim=1).sqrt().unsqueeze(0) # (1, n_vocab)
  
  ### Pick validation words from 2 ranges: (0, window): common words & (1000, 1000+window): uncommon words 
    valid_words = random.sample(range(valid_window), n_valid_words//2) + random.sample(range(1000, 1000+valid_window), n_valid_words//2)
    valid_words = torch.LongTensor(np.array(valid_words)).to(device) # (n_valid_words, 1)

    valid_embeddings = embedding(valid_words) # (n_valid_words, n_embed)
  ### (n_valid_words, n_embed) * (n_embed, n_vocab) --> (n_valid_words, n_vocab) / 1, n_vocab)
    similarities = torch.mm(valid_embeddings, all_embeddings.t()) / magnitudes  # (n_valid_words, n_vocab)
  
    return valid_words, similarities
    
### Define SkipGram model with Negative Sampling
import torch
from torch import nn
import torch.optim as optim
class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist=None):
        super().__init__()
        
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist
        
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)
        
        # Initialize both embedding tables with uniform distribution
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
        

    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors  # input vector embeddings
    

    def forward_target(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors  # output vector embeddings
    

    def forward_noise(self, batch_size, n_samples=5):
        """ Generate noise vectors with shape (batch_size, n_samples, n_embed)"""
        # If no Noise Distribution specified, sample noise words uniformly from vocabulary
        if self.noise_dist is None:
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist
            
        # torch.multinomial :
        # Returns a tensor where each row contains (num_samples) **indices** sampled from 
        # multinomial probability distribution located in the corresponding row of tensor input.
        noise_words = torch.multinomial(input       = noise_dist,           # input tensor containing probabilities
                                        num_samples = batch_size*n_samples, # number of samples to draw
                                        replacement = True)
        noise_words = noise_words.to(device)
        
        # use context matrix for embedding noise samples
        noise_vectors = self.out_embed(noise_words).view(batch_size, n_samples, self.n_embed)
        
        return noise_vectors
        
### Define Loss Class
class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
                input_vectors, 
                output_vectors, 
                noise_vectors):
      
        batch_size, embed_size = input_vectors.shape
    
        input_vectors = input_vectors.view(batch_size, embed_size, 1)   # batch of column vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_size) # batch of row vectors
    
        # log-sigmoid loss for correct pairs
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log().squeeze()
    
        # log-sigmoid loss for incorrect pairs
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors

        return -(out_loss + noise_loss).mean()  # average batch loss

### Define Noise Distribution
### As defined in the paper by Mikolov et all.
freq = Counter(int_words)
freq_ratio = {word:cnt/len(vocab_to_int) for word, cnt in freq.items()}        
freq_ratio = np.array(sorted(freq_ratio.values(), reverse=True))
unigram_dist = freq_ratio / freq_ratio.sum() 
noise_dist = torch.from_numpy(unigram_dist**0.75 / np.sum(unigram_dist**0.75))

### Define Model, Loss, & Optimizer
from torch import optim
embedding_dim = 300
model = SkipGramNeg(len(vocab_to_int), 
                                 embedding_dim, 
                                 noise_dist )
criterion = NegativeSamplingLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.003)

### Train!!
device = 'cuda' if torch.cuda.is_available else 'cpu'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
def train_skipgram(model,
                   criterion,
                   optimizer,
                   int_words,
                   n_negative_samples=5,
                   batch_size=256,
                   n_epochs=5,
                   print_every=1500,
                   ):
  model.to(device)
  
  step = 0
  for epoch in range(n_epochs):
      for inputs, targets in get_batches(int_words, batch_size=batch_size):
          step += 1
          inputs = torch.LongTensor(inputs).to(device)    # [b*n_context_words]
          targets = torch.LongTensor(targets).to(device)  # [b*n_context_words]
      
          embedded_input_words = model.forward_input(inputs)
          embedded_target_words = model.forward_target(targets)
          embedded_noise_words = model.forward_noise(batch_size=inputs.shape[0], 
                                n_samples=n_negative_samples)

          loss = criterion(embedded_input_words, embedded_target_words, embedded_noise_words)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      
          if (step % print_every) == 0:
              print("Epoch: {}/{}".format((epoch+1), n_epochs))
              print("Loss: {:.4f}".format(loss.item()))
              valid_idxs, similarities = cosine_similarity(model.in_embed)
              _, closest_idxs = similarities.topk(6)
              valid_idxs, closest_idxs = valid_idxs.to('cpu'), closest_idxs.to('cpu')
        
              for ii, v_idx in enumerate(valid_idxs):
                  closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                  print(int_to_vocab[v_idx.item()] + " | "+ ", ".join(closest_words))

              print("\n...\n")
train_skipgram(model,
       criterion,
       optimizer,
       int_words,
       n_negative_samples=5)

##### Reference: [1] https://github.com/lukysummer/SkipGram_with_NegativeSampling_Pytorch [2] http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

# Explore the Word Representations

## Visualise the word representations by t-SNE
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

### By input embeddings
### getting embeddings from the embedding layer of our model, by name
embeddings = model.in_embed.weight.to('cpu').data.numpy()

viz_words = 1000
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

fig, ax = plt.subplots(figsize=(16, 16))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)

### getting embeddings from the embedding layer of our model, by name
embeddings = model.in_embed.weight.to('cpu').data.numpy()

viz_words = 1000
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

fig, ax = plt.subplots(figsize=(16, 16))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)

### getting output embeddings
embeddings = model.out_embed.weight.to('cpu').data.numpy()

viz_words = 1000
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

fig, ax = plt.subplots(figsize=(16, 16))
for idx in range(viz_words):
    
    plt.scatter(*embed_tsne[idx, :])
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)

embeddings = model.out_embed.weight.to('cpu').data.numpy()

viz_words = 1000
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

fig, ax = plt.subplots(figsize=(16, 16))
for idx in range(viz_words):
    
    plt.scatter(*embed_tsne[idx, :])
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)

##### Reference: https://github.com/lukysummer/SkipGram_with_NegativeSampling_Pytorch



