import torch
import torch.nn as nn
import torchvision
import os
from torchvision.models import resnet18, resnet50
from tsm import TSN
from temporal_shift import *

# https://arxiv.org/abs/1611.05267

# TSN Code explanation: https://www.programmersought.com/article/13951076995/
class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, num_classes):
        super(CNNLSTMModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
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
        #print('sequence lengths are: ', seq_lengths)
        cnn_embed_seq = []
        for t in range(x_cnn.size(1)):
            x = self.resnet(x_cnn[:, t, :, :, :])  # [b, 512, 1, 1]
            cnn_embed_seq.append(x)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # print('size of cnn_embed_seq is: ', cnn_embed_seq.size())
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
        rnn_seq = []
        for t in range(RNN_out.size(0)):
            #print('sequence length[t] is; ', seq_lengths[t])
            input_to_fc = RNN_out[t, seq_lengths[t], :]
            rnn_seq.append(self.fc1(input_to_fc))
        rnn_seq = torch.stack(rnn_seq, dim=0)
        # print('size after fc1 is: ', x.size())
        return rnn_seq


class VideoGlobalModel(nn.Module):
    """
    This model contains only TSM module
    """

    def __init__(self, args, input_dim, num_classes):
        super(VideoGlobalModel, self).__init__()

        self.resnet50 = resnet50(pretrained=True)
        self.nr_actions = num_classes
        self.nr_frames = 16
        self.img_feature_dim = input_dim
        
        self.coord_feature_dim = args.coord_feature_dim
        '''
        self.include_oie = opt.include_oie
        if self.include_oie:
            print('Including Object Identity embeddings: ')

        if opt.freeze_model:
            print('Freezing TSM model')
        '''
        # Editing parameters of TSM model
        # start of the code
        #print('Initializing TSM model')
        self.tsm = TSN(self.nr_actions, self.nr_frames, modality='RGB',
                       dropout=0,
                       img_feature_dim=self.img_feature_dim,
                       partial_bn=False,
                       is_shift=True, shift_div=8,
                       temporal_pool=False,
                       non_local=False, freeze_model=False)

        if args.tsm_resume is not None:
            args.tsm_resume = r'pretrained_models/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment16_e45.pth'
            if args.temporal_pool:  # early temporal pool so that we can load the state_dict
                make_temporal_pool(self.tsm.module.new_base_model, self.nr_frames)
            if os.path.isfile(args.tsm_resume):
                #print(("=> loading checkpoint '{}'".format(args.tsm_resume)))
                checkpoint = torch.load(args.tsm_resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
                state_dict = checkpoint['state_dict']

                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    k = k.replace('module.', '')  # remove 'module.' of dataparallel
                    new_state_dict[k] = v
                # removing the 'fc' keys
                for key in ['new_fc.weight', 'new_fc.bias']:
                    del new_state_dict[key]

                self.tsm.load_state_dict(new_state_dict, strict=False)
                #print("=> loaded checkpoint '{}' ".format(args.tsm_resume))

                # # freeze the TSM
                if args.freeze_model:
                    #print('Freezing TSM')
                    for param in self.tsm.parameters():
                        param.requires_grad = False
            else:
                print(("=> no checkpoint found at '{}'".format(args.tsm_resume)))

        #print('TSM model loaded')
        
        # Unfreeze specific layers (fine-tuning)
        ct = 0
        for child in self.tsm.children():
            ct += 1
            ct_child = 0
            if ct == 1:
                for sub_child in child:
                    ct_child += 1
                    check = 10 - args.unfreeze_layers
                    if ct_child > check:
                        for param in sub_child.parameters():
                            param.requires_grad = True
                            
        # end of the code
        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv = nn.Conv3d(2048, 512, kernel_size=(1, 1, 1), stride=1)
        self.fc = nn.Linear(512, self.nr_actions)

        self.classifier = nn.Sequential(
            #nn.Linear(2 * self.img_feature_dim, self.coord_feature_dim),
            nn.Linear(512, self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_actions)
        )
        '''
        if args.fine_tune:
            self.fine_tune(args.fine_tune)
        '''

    def forward(self, global_img_input, seq_lengths):
        """
        V: num of videos
        T: num of frames
        P: num of proposals
        :param videos: [V x 3 x T x 224 x 224]
        :param proposals_t: [V x T] List of BoxList (size of num_boxes each)
        :return:
        """
        global_img_input = global_img_input.reshape(-1, 3*self.nr_frames, self.img_feature_dim, self.img_feature_dim)# (-1, num_channels*num_frames, self.input_dim, self.input_dim)
        # org_features - [V x 2048 x T / 2 x 14 x 14]
        bs, _, _, _ = global_img_input.size()

        org_features = self.tsm(global_img_input)  # [b, 2048, num_segments, 1, 1]
        videos_features = self.conv(org_features)  # [b, 512, num_segments, 1, 1]
        # # Get global features - [V x 512]
        _gf = self.avgpool(videos_features).squeeze()  # [b, 512]

        video_features = _gf  # [b, 512]
        # print('size of video features after concat operation:', video_features.size())
        if torch.any(torch.isnan(org_features)):
            print('Nan seen in org_features')
            print(torch.any(torch.isnan(videos_features)), torch.any(torch.isnan(org_features)))

        cls_output = self.classifier(video_features)  # (b, num_classes)
        if len(seq_lengths) == 1:
            return cls_output.unsqueeze(0) # if batch_size = 1
        else:
            return cls_output

