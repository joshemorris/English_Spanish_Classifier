# This is a script to classify spanish and enlish words from a bag of words.
# It is from the tutorial found here: 
# http://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html#sphx-glr-beginner-nlp-deep-learning-tutorial-py
# original author: Robert Guthrie


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Data for this model
data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
				("Give it to me".split(), "ENGLISH"),
				("No creo que sea una buena idea".split(), "SPANISH"),
				("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
							("it is lost on me".split(), "ENGLISH")]

# create a mapping of words to index in BOW

word_to_ix = {}
for sent, _ in data + test_data:
	for word in sent:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)

print("vocab:")
print(word_to_ix)

label_to_ix = {"SPANISH": 0, "ENGLISH": 1}

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = len(label_to_ix)

class BoWClassifier(nn.Module):

	def __init__(self, num_labels, vocab_size):
		# call init of nn.module
		super(BoWClassifier, self).__init__()

		# define parameters
		# nn.Linear provides the Affine mapping
		# we're going from our vocabulary to the two language labels
		self.linear = nn.Linear(vocab_size, num_labels)

		#Log softmax is not defined here because it doesn't have params

	def forward(self, bow_vec):
		#define the forward pass
		#we go through the linear layer and then the log_softmax
		return F.log_softmax(self.linear(bow_vec))

def make_bow_vector(sentence, word_to_ix):
	vec = torch.zeros(len(word_to_ix))

	for word in sentence:
		vec[word_to_ix[word]] += 1

	return vec.view(1, -1)

def make_target(label, label_to_ix):
	return torch.LongTensor([label_to_ix[label]])

model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

#this is the probabilities of the test data before training
print("\nbefore training:")
for instance, label in test_data:
	bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
	log_probs = model(bow_vec)
	print(log_probs)

# print matrix column corresponding to creo
print("matrix column creo:")
print(next(model.parameters())[:,word_to_ix["creo"]])

#set the loss function and optimizer
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

#TRAIN!
for epoch in range(100):
	for instance, label in data:
		#clear out gradients from previous epoch
		model.zero_grad()

		#make BoW vector
		bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))

		#turn targets into indexes for proper interfacing with the 
		#loss function
		target = autograd.Variable(make_target(label, label_to_ix))

		#perform forward pass
		log_probs = model(bow_vec)

		#compute loss and gradient
		loss = loss_function(log_probs, target)
		loss.backward()

		#update parameters
		optimizer.step()

#test model using test data
print("after training:")
for instance, label in test_data:
	bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
	log_probs = model(bow_vec)
	print(log_probs)

# print matrix column corresponding to creo
print("matrix column creo:")
print(next(model.parameters())[:,word_to_ix["creo"]])