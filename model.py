import torch
import torch.nn as nn
import torchvision.models as models


# class EncoderCNN(nn.Module):
#     def __init__(self, image_embed_size):
#         super(EncoderCNN, self).__init__()
#         resnet = models.resnet50(pretrained=True)
#
#         # freeze network parameters
#         for param in resnet.parameters():
#             param.requires_grad_(False)
#
#         # remove the last FC layer
#         modules = list(resnet.children())[:-1]
#         self.resnet = nn.Sequential(*modules)
#         self.embed = nn.Linear(resnet.fc.in_features, image_embed_size)
#
#     def forward(self, images):
#         features = self.resnet(images)
#         features = features.view(features.size(0), -1)
#         features = self.embed(features)
#         return features
#
#
# class DecoderRNN(nn.Module):
#     def __init__(self, word_embed_size, hidden_size, vocab_size, num_layers=1):
#         super(DecoderRNN, self).__init__()
#         # save params
#         self.word_embed_size = word_embed_size
#         self.hidden_size = hidden_size
#         self.vocab_size = vocab_size
#         self.num_layers = num_layers
#         # define layers
#         self.word_embeddings = nn.Embedding(vocab_size, word_embed_size)
#         self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers, batch_first=True)
#         self.hidden2out = nn.Linear(hidden_size, vocab_size)
#         self.softmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, features, captions):
#         captions = self.word_embeddings(captions[:, :-1])  # remove the <end> character
#
#         # [10, 256] => [10, 1, 256] concat [10, 11, 256] => [10, 12, 256] add encoded image (features) as t=0
#         embed = torch.cat((features.unsqueeze(1), captions),
#                           dim=1)  # batch_size,cap_length -> batch_size,cap_length-1,embed_size
#
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         hidden = (torch.zeros(1, embed.size(0), self.hidden_size, device=device),
#                   torch.zeros(1, embed.size(0), self.hidden_size, device=device))
#         lstm_out, hidden = self.lstm(embed,
#                                      hidden)  # lstm_out: [10, 12, 256] # print (lstm_out.shape) -> batch_size, caplength, hidden_size
#         # lstm_out, _ = self.lstm(embed)
#         outputs = self.hidden2out(
#             lstm_out)  # output: [10, 12, vocabsize]  # print (outputs.shape) -> batch_size, caplength, vocab_size
#         return outputs
#
#     def sample(self, inputs, states=None, max_len=20):
#         """
#         Accepts pre-processed image tensor (inputs) and returns predicted sentence
#         (list of tensor ids of length max_len)
#         """
#         res = []
#
#         # Now we feed the LSTM output and hidden states back into itself to get the caption
#         for i in range(max_len):
#             lstm_out, states = self.lstm(inputs, states)  # hiddens: (1, 1, hidden_size)
#             outputs = self.hidden2out(lstm_out.squeeze(1))  # outputs: (1, vocab_size)
#             _, predicted = outputs.max(dim=1)  # predicted: (1, 1)
#             res.append(predicted.item())
#
#             inputs = self.word_embeddings(predicted)  # inputs: (1, word_embed_size)
#             inputs = inputs.unsqueeze(1)  # inputs: (1, 1, word_embed_size)
#
#         return res


#### MODEL PK5
# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size):
#         super(EncoderCNN, self).__init__()
#         resnet = models.resnet50(pretrained=True)
#         for param in resnet.parameters():
#             param.requires_grad_(False)
#
#         modules = list(resnet.children())[:-1]
#         self.resnet = nn.Sequential(*modules)
#         self.embed = nn.Linear(resnet.fc.in_features, embed_size)
#
#     def forward(self, images):
#         features = self.resnet(images)
#         features = features.view(features.size(0), -1)
#         features = self.embed(features)
#         return features
#
#
# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         # The LSTM takes word embeddings as inputs, and
#         # outputs hidden states have a dimension equal to hidden_dim.
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
#                             dropout=0.4 if num_layers > 1 else 0,
#                             batch_first=True)
#         # Add a linear layer that maps from hidden state space to vocab space
#         self.linear = nn.Linear(hidden_size, vocab_size)
#
#     def forward(self, features, captions):
#         captions = captions[:,:-1]
#         embeds = self.embedding(captions)
#         embeds = torch.cat((features.unsqueeze(1), embeds), 1)
#         outputs, hiddens = self.lstm(embeds)
#         outputs = self.linear(outputs)
#         return outputs
#
#
#     def sample(self, inputs, states=None, max_len=20):
#         #" accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
#         res = []
#
#         for i in range(max_len):
#             hiddens, states = self.lstm(inputs, states)
#             outputs = self.linear(hiddens)
#             _, predicted = torch.max(outputs,2)
#             inputs = self.embedding(predicted)
#             predicted_idx = predicted.item()
#             res.append(predicted_idx)
#             # if the predicted idx is the stop index, the loop stops
#             if predicted_idx == 1:
#                 break
#         return res


# ---- MODELS PK-3


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        # Assigning hidden dimension
        self.hidden_dim = hidden_size

        # getting embed from nn.Embedding()
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Creating LSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # Initializing Lineear linear to apply at last of RNN layer for further prediction
        self.linear = nn.Linear(hidden_size, vocab_size)

        # Initializing valuews for hidden and cell state
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))

    def forward(self, features, captions):
        cap_embedding = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), 1)

        # Getting output i.e score and hidden layer
        lstm_out, self.hidden = self.lstm(embeddings)
        outputs = self.linear(lstm_out)

        return outputs

    # def sample(self, inputs, states=None, hidden=None, max_len=20):
    #     ''' accepts pre-processed image tensor (inputs) and returns predicted
    #     sentence (list of tensor ids of length max_len) '''
    #     res = []
    #     for i in range(max_len):
    #         outputs, hidden = self.lstm(inputs, hidden)
    #         #
    #         outputs = self.linear(outputs.squeeze(1))
    #         #
    #         target_index = outputs.max(1)[1]
    #         #
    #         res.append(target_index.item())
    #         inputs = self.embed(target_index).unsqueeze(1)
    #     #
    #     return res

    def sample(self, inputs, states=None, max_len=20):
        """
        Accepts pre-processed image tensor (inputs) and returns predicted sentence
        (list of tensor ids of length max_len)
        """
        res = []

        # Now we feed the LSTM output and hidden states back into itself to get the caption
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)  # hiddens: (1, 1, hidden_size)
            outputs = self.linear(lstm_out.squeeze(1))  # outputs: (1, vocab_size)
            _, predicted_idx = outputs.max(dim=1)  # predicted: (1, 1)
            res.append(predicted_idx.item())
            # if the predicted idx is the stop index, the loop stops
            if predicted_idx == 1:
                break
            inputs = self.embed(predicted_idx)  # inputs: (1, word_embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (1, 1, word_embed_size)

        return res
