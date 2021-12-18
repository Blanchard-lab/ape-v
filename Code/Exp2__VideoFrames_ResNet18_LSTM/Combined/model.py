import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, num_classes):
        super(CNNLSTMModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        # print('in features for resnet: ', resnet.fc.in_features)

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

        # for FC layers
        input_to_fc = torch.Tensor([]).cuda()
        for t in range(RNN_out.size(0)):
            input_to_fc = torch.cat((input_to_fc, RNN_out[t, seq_lengths[t]-1, :]), 1)
        
        return input_to_fc
        
class CNNLSTMModelEnsemble(nn.Module):
    def __init__(self, model_center, hidden_dim_center, model_left, hidden_dim_left, model_right, hidden_dim_right, num_classes):
        super(CNNLSTMModelEnsemble, self).__init__()
        self.num_classes = num_classes
        
        # model_center
        model_center_modules = list(model_center.children())[:-1]
        self.model_center = nn.Sequential(*model_center_modules)
        for param in self.model_center.parameters():
            param.requires_grad = False
        
        # model_left
        model_left_modules = list(model_left.children())[:-1]
        self.model_left = nn.Sequential(*model_left_modules)
        for param in self.model_left.parameters():
            param.requires_grad = False
        
        # model_right
        model_right_modules = list(model_right.children())[:-1]
        self.model_right = nn.Sequential(*model_right_modules)
        for param in self.model_right.parameters():
            param.requires_grad = False
        
        # FC layer (Classifier)
        self.fc1 = nn.Linear(hidden_dim_center+hidden_dim_left+hidden_dim_right, self.num_classes)
        
    def model_forward(self, model, x_cnn, seq_lengths):
        cnn_embed_seq = torch.Tensor([]).cuda()
        rnn_input_to_fc = torch.Tensor([]).cuda()
        ct = 0
        for child in model.children():
            ct += 1
            if ct == 1:
                for t in range(x_cnn.size(1)):
                    x = child(x_cnn[:, t, :, :, :])  # [b, 512, 1, 1]
                    cnn_embed_seq = torch.cat((cnn_embed_seq, x.unsqueeze(0)), 0)
                cnn_embed_seq = cnn_embed_seq.squeeze(-1).squeeze(-1)
                cnn_embed_seq = cnn_embed_seq.transpose(0, 1)
            else:
                child.flatten_parameters()
                RNN_out, (h_n, h_c) = child(cnn_embed_seq, None)
                # for FC layers
                for t in range(RNN_out.size(0)):
                    rnn_input_to_fc = torch.cat((rnn_input_to_fc, RNN_out[t, seq_lengths[t]-1, :].unsqueeze(1)), 1)
        
        return rnn_input_to_fc
    
    def forward(self, frames_center, lengths_center, frames_left, lengths_left, frames_right, lengths_right):
        frames_center_op = self.model_forward(self.model_center, frames_center, lengths_center) # Forward pass for model center
        frames_left_op = self.model_forward(self.model_left, frames_left, lengths_left) # Forward pass for model left
        frames_right_op = self.model_forward(self.model_right, frames_right, lengths_right) # Forward pass for model right
        
        # Concatenate outputs for fc layer
        input_to_fc = torch.Tensor([]).cuda()
        for t in range(frames_center_op.size(1)):
            input_to_fc = torch.cat((input_to_fc, torch.cat((frames_center_op[:, t], frames_left_op[:, t], frames_right_op[:, t]), dim=0).unsqueeze(1)), dim=1)
        input_to_fc = torch.transpose(input_to_fc, 0, 1)
        
        # FC Layer
        op = torch.Tensor([]).cuda()
        for inp in input_to_fc:
            op = torch.cat((op, self.fc1(inp).unsqueeze(0)), 0)
        
        return op