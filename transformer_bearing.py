import torch
import Dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class TransformerTimeSeries(torch.nn.Module):
    """
    Time Series application of transformers based on paper

    causal_convolution_layer parameters:
        in_channels: the number of features per time point
        out_channels: the number of features outputted per time point
        kernel_size: k is the width of the 1-D sliding kernel

    nn.Transformer parameters:
        d_model: the size of the embedding vector (input)

    PositionalEncoding parameters:
        d_model: the size of the embedding vector (positional vector)
        dropout: the dropout to be used on the sum of positional+embedding vector

    """

    def __init__(self):
        super(TransformerTimeSeries, self).__init__()
        self.input_embedding = causal_convolution_layer.context_embedding(1, 256, 9)
        self.positional_embedding = torch.nn.Embedding(512, 256)

        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=3)

        self.fc1 = torch.nn.Linear(256, 1)

        self.fc2 = torch.nn.Linear(500, 4)

    def forward(self, x, attention_masks):
        # concatenate observed points and time covariate
        # (B*feature_size*n_time_points)
        # z = torch.cat((y.unsqueeze(1),x.unsqueeze(1)),1)
        z = x.unsqueeze(1)
        # input_embedding returns shape (Batch size,embedding size,sequence len) -> need (sequence len,Batch size,embedding_size)
        z_embedding = self.input_embedding(z).permute(2, 0, 1)

        # get my positional embeddings (Batch size, sequence_len, embedding_size) -> need (sequence len,Batch size,embedding_size)
        # positional_embeddings = self.positional_embedding(x.type(torch.long)).permute(1,0,2)

        input_embedding = z_embedding  # +positional_embeddings

        transformer_embedding = self.transformer_decoder(input_embedding, attention_masks)

        output1 = self.fc1(transformer_embedding.permute(1, 0, 2))

        output = self.fc2(output1[:, :, 0])

        # F.log_softmax(output,dim = 1)
        # F.softmax(output,dim = 1)
        # return F.softmax(output,dim = 1) #output
        return output