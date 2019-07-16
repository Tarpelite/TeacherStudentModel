import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, num_layers, emnbedding_length):
        super(LSTMClassifier, self).__init__()

        '''
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BUcketIterator
        output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
        '''
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_length = emnbedding_length
        self.lstm = nn.LSTM(emnbedding_length, hidden_size, num_layers)
        self.label = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_sentence, batch_size=None):

        """ 
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

		Returns
		-------
		Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
		final_output.shape = (batch_size, output_size)
		"""

        # input = self.word_embedding(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        # input = input.permute(1, 0, 2)

        h0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda())
        c0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda())
        output, (final_hidden_state, final_cell_state) = self.lstm(input_sentence, (h0, c0))
        final_output = self.label(final_hidden_state[-1])  # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
        return final_output


