import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18



class CNNLSTMModel(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, layer_dim, num_classes):
        super(CNNLSTMModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        # print('in features for resnet: ', resnet.fc.in_features)
        
        # Unfreeze specific layers (fine-tuning) for resnet18
        ct_child = 0
        for child in self.resnet.children():
            ct_child += 1
            check = 9 - args.unfreeze_layers
            if ct_child > check:
                for param in child.parameters():
                    param.requires_grad = True

        # Defining the LSTM
        self.RNN_cell = nn.LSTM(
            input_size=resnet.fc.in_features,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True,
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        # FC layer
        self.fc1 = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, x_cnn, seq_lengths):
        # print('sequence lengths are: ', seq_lengths)
        cnn_embed_seq = torch.Tensor([]).cuda()
        #cnn_embed_seq = []
        for t in range(x_cnn.size(1)):
            x = self.resnet(x_cnn[:, t, :, :, :])  # [b, 512, 1, 1]
            #cnn_embed_seq.append(x)
            cnn_embed_seq = torch.cat((cnn_embed_seq, x.unsqueeze(0)), 0)
        #cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).cuda()
        #print('size of cnn_embed_seq is: ', cnn_embed_seq.size())
        cnn_embed_seq = cnn_embed_seq.squeeze(-1).squeeze(-1)
        cnn_embed_seq = cnn_embed_seq.transpose(0, 1)
        # print('size of cnn_embed_seq is: ', cnn_embed_seq.size())

        self.RNN_cell.flatten_parameters()
        RNN_out, (h_n, h_c) = self.RNN_cell(cnn_embed_seq, None)
        # print('size of RNN_out is: ', RNN_out.size())
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        #TODO: clip to sequence length
        # To Do: sum/average the output from the rnn cell
        rnn_seq = torch.Tensor([]).cuda()
        #rnn_seq = []
        for t in range(RNN_out.size(0)):
            '''
            print('size of RNN_out is: ', RNN_out.size())
            print(f'seq_lengths: {seq_lengths}')
            print(f'seq_lengths[t]: {seq_lengths[t]-1}')
            '''
            input_to_fc = RNN_out[t, seq_lengths[t]-1, :]
            #rnn_seq.append(self.fc1(input_to_fc))
            rnn_seq = torch.cat((rnn_seq, self.fc1(input_to_fc).unsqueeze(0)), 0)
        #rnn_seq = torch.stack(rnn_seq, dim=0).cuda()
        #print('size rnn_seq: ', rnn_seq.size())
        # print('size after fc1 is: ', x.size())
        return rnn_seq