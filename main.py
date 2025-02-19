# This models is using embedding feature vector in 30 dimensional space

# Words with similar meaning stay close to each other (17000 possible words)

# We're dealing with 3 words and predicting the 4th word

# Look up table (17000,30) -> 3 words so it would be 90 numbers 

# Last layer would have 17000 neurons represeting the 17000 words and Softmax exponentiation of logit then normalized to 1 to get probability dsitribution for the next word in the sequence  

# The logit refers to the raw, unnormalized output before applying an activation function like softmax or sigmoid.
# We take the log of logits in certain loss functions (like cross-entropy loss) to improve numerical stability and ensure effective gradient updates

# During training we have the label or the identity of the next word in a sequence. That word or its index is used to pluck out the probability of that word and then we are maximizing the probability of that word with respect to the parameters of this neural net so the parameters are the weights and biases of this output layer.
# We'll try to maximize the probability of the 4th word by respect to all parametrs in the neural network

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from MLP_Training import mlp_training



words = open('D:\\Data Science\\Python Assignment\\LLM\\LLM From Scratch\\names.txt', 'r').read().splitlines()
len(words)



chars=sorted(set("".join(words)))

#mapping from integer to string and the oposite
string_to_int={ch:i+1 for i,ch in enumerate(chars)}
string_to_int["."]=0
int_to_string={i:ch for ch,i in string_to_int.items()}
int_to_string



# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
  X, Y = [], []
  for w in words:

    #0 representing "."
    context = [0] * block_size
    for ch in w + '.':
      #convertng each character of the word to string
      ix = string_to_int[ch]
      X.append(context)
      Y.append(ix)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # crop and append the 4th character

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y


random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

#80% of the words for training
Xtr, Ytr = build_dataset(words[:n1])
#10% for validation
Xval, Yval = build_dataset(words[n1:n2])
#10% for testing
Xte, Yte = build_dataset(words[n2:])



#3 character index and the 4th character (prediction label)
print(f"Training data: {Xtr.shape}, Training labels: {Ytr.shape}")
print(f"Validation data: {Xval.shape}, Validation labels: {Yval.shape}")
print(f"Test data: {Xte.shape}, Test labels: {Yte.shape}")


#running the model
model=mlp_training(Xtr,Ytr,Xval,Yval,32)
model.mlp_architecture()
model.training(100,0.01)
model.inference()
