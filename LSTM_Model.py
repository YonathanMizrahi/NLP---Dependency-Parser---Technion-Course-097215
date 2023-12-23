import torch
import torch.nn as nn
import numpy as np

def getEmbeddingFromGlove(GloveFile):
    # GloveFile format : word x1 x2 x3 x4 ... xn

    # First, we create a dictionary to store the word embeddings
    word_embeddings = {}

    # Open the file and read the lines
    with open(GloveFile, "r", encoding="utf8") as f:
        lines = f.readlines()

    # Iterate over each line
    for line in lines:
        # Split the line into the word and the embedding values
        word_and_embedding = line.strip().split(" ")
        # The first element is the word
        word = word_and_embedding[0]
        # The remaining elements are the values of the embedding
        embedding = np.array([float(x) for x in word_and_embedding[1:]])
        # Add the word and embedding to the dictionary
        word_embeddings[word] = embedding

    # Convert the dictionary to a matrix
    embedding_matrix = np.array([word_embeddings[word] for word in word_embeddings])


    # Convert the embedding matrix to a PyTorch tensor
    glove_embedding = torch.from_numpy(embedding_matrix).float()

    return glove_embedding


class LSTM_Model(nn.Module):
    """
    An LSTM Model to predict the score of each edges of the dependency tree
    """
    def __init__(self, batch_size, word_emb_dim, pos_emb_dim, hidden_dim, word_vocab_size, tag_vocab_size):
        """
        This class initializes the LSTM_Model class with the given parameters.
        It creates 2 bi-LSTM encoders, a word and POS embedding layer, and an MLP to provide the final output.
        Args:
            batch_size: The size of the batch
            word_emb_dim: The dimension of the word embedding.
            pos_emb_dim: The dimension of the POS tag embedding.
            hidden_dim: The dimension of the LSTM's hidden size
            word_vocab_size: The number of words in our vocabulary.
            tag_vocab_size: The number of tags in our vocabulary
        """
        super(LSTM_Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.emb_dim = word_emb_dim + pos_emb_dim
        self.dropout = 0.5

        #self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim)

        # Create an embedding layer using the GloVe embedding tensor
        print('Loading pre-trained glove model from glove.6B.300d.txt ....')
        glove_embedding = getEmbeddingFromGlove(GloveFile='glove.6B.300d.txt')
        self.word_embedding = nn.Embedding.from_pretrained(glove_embedding)
        print('GLOVE EMBEDDED ....')

        # Implement embedding layer for POS tags
        self.tag_embedding = nn.Embedding(tag_vocab_size, pos_emb_dim)

        # First bi-LSTM encoder
        self.encoder = nn.LSTM(input_size=self.emb_dim, hidden_size=hidden_dim, num_layers=2, bidirectional=True,
                               batch_first=True, dropout=self.dropout)

        """
        Let's add an additional layer of bi-LSTM to the model. 
        In the paper https://arxiv.org/abs/1611.01734, 
        the authors use a double layer of bi-LSTM as part of their model for dependency parsing. 
        “We find that using three or four layers gets significantly better performance than two layers, 
        and increasing the LSTM sizes from 200 to 300 or 400 dimensions likewise significantly improves performance”. 
        However, we decided to add only one layer since we are assuming that it will be more than enough. 
        """
        self.encoder_2 = nn.LSTM(input_size=hidden_dim * 2,  hidden_size=hidden_dim, num_layers=2, bidirectional=True,
                               batch_first=True, dropout=self.dropout)

        # MLP layer (2FC layer with TanH activation)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )

    def forward(self, words_idx_tensor, pos_idx_tensor, sen_length):

        # Pass word_idx and pos_idx through their embedding layers
        words_embedding = self.word_embedding(words_idx_tensor[:, :sen_length].to(self.device, non_blocking=True))
        tag_embedding = self.tag_embedding(pos_idx_tensor[:, :sen_length].to(self.device, non_blocking=True))

        # Concat both embedding outputs
        embeds = torch.cat([words_embedding, tag_embedding], 2)

        # Get Bi-LSTM hidden representation for each word+pos in sentence
        lstm_out, _ = self.encoder(embeds)
        lstm_out, _ = self.encoder_2(lstm_out)

        # Initialize empty list to store features
        features = []

        # Loop over lstm_out
        for i in range(lstm_out.shape[0]):
            # Concatenate lstm_out[i] with itself, repeated to have the same shape as lstm_out
            repeated_lstm_out = lstm_out[i].repeat(sen_length, 1, 1)
            # Concatenate repeated_lstm_out with lstm_out[i] unsqueezed and repeated
            concatenated = torch.cat([lstm_out[i].unsqueeze(1).repeat(1, sen_length, 1), repeated_lstm_out], -1)
            # Add concatenated tensor to features list as a single element tensor
            features.append(concatenated.unsqueeze(1))

        # Concatenate all elements in the features list along the second dimension
        features = torch.cat(features, 1)

        # Get score for each possible edge in the parsing graph, construct score matrix
        score_matrix = self.mlp(features)

        # Next TODO: Calculate the negative log likelihood loss described above
        # Next TODO: Use Chu-Liu-Edmonds to get the predicted parse tree T' given the calculated score matrix

        return score_matrix

