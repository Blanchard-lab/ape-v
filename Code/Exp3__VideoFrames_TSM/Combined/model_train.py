'''
https://github.com/YuxinZhaozyx/pytorch-VideoDataset
https://discuss.pytorch.org/t/remove-last-two-layers-fc-and-average-pooling-of-resnet152-and-add-conv2d-at-the-end/51566/5
https://github.com/mit-han-lab/temporal-shift-module
'''

'''
https://stackoverflow.com/questions/54478779/passing-supplementary-parameters-to-hyperopt-objective-function
'''

import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch import Tensor

from torch.cuda.amp import autocast, GradScaler
from logger import FileLogger

import torchvision
import datasets_preproc
import transforms_preproc

from hyperopt import *
from hyperopt.fmin import generate_trials_to_calculate
from functools import partial
import time
import random
import statistics
from model import CNNLSTMModel, VideoGlobalModel, VideoGlobalModelEnsemble
from torch.nn import functional as F
from dependencies import *

from opts import parser

from sampler import BalancedBatchSampler

model_count = 0
# Increment model count
def count_model():
    global model_count
    model_count += 1

args = parser.parse_args()
if args.gpu is not None:
    print("Use GPU: {} for training".format(args.gpu))

print('Logging dir is: ', args.logdir)
# Add logger object
is_master = (not args.distributed) or (int(args.rank) == 0)
is_rank0 = args.local_rank == 0
log = FileLogger(args.logdir, is_master=is_master, is_rank0=is_rank0)
# CNN-LSTM model class
 
# Creates a file for storing hyperparameters and results
def create_hyperparameters_results_file():    
    # Create file to store results
    headings = f'Model,Seed,Batch_Size,Num_Epochs,Learning_Rate,Training_Loss,Training_Accuracy,Testing_Loss,Testing_Accuracy,True_positive,True_negative,False_positive,False_negative,Precision_PositivePredictedValue,Recall_TruePositiveRate,F1_score_positive_class,NegativePredictedValue,Specificity_TrueNegativeRate,F1_score_negative_class,ROC_AUC_score,Cohen_kappa_score,Balanced_accuracy,Matthews_correlation_coefficient'
    file_path = fr'hyperparameters_athletics_video_guided_TSM.csv'
    f = open(file_path, 'a+')
    f.write(headings)
    f.write('\n')
    f.close()
    return 
 
# Hyperparameter search space 
space = {
    'batch_size': hp.choice('batch_size', [32, 64]), # 64, 128, 256, 512
    'num_epochs': hp.choice('num_epochs', range(15, 80, 5)),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(0.05)) # LR bw 0.0001 and 0.05
}
'''
points = [{
    'batch_size': 0,
    'num_epochs': 10,
    'unfreeze_layers': 2,
    'learning_rate': 0.001143345
}]
'''

def adjust_learning_rate(optimizer, epoch, learning_rate, args):
    """Sets the learning rate to the initial LR decayed by 5 every 15 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 15))
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * 1#param_group['lr_mult']
        param_group['weight_decay'] = decay * 1#param_group['decay_mult']

def load_data(data_train, data_valid, batch_size):
    train_lengths = get_frame_length(data_train)
    max_len_train = max(train_lengths)
    train_dataset = datasets.VideoLabelDataset(
        data_train,
        train_lengths,
        transform=torchvision.transforms.Compose([
            transforms.VideoFilePathToTensor(max_len=max_len_train, fps=15, padding_mode='last'),
            #transforms.VideoResize([256, 256]),
            transforms.VideoCenterCrop([224, 224]),  #TODO: change to random crop later
            transforms.VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # default values for imagenet
        ])
    )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=args.workers, shuffle=True)
    
    test_lengths = get_frame_length(data_valid)
    max_len_test = max(test_lengths)
    test_dataset = datasets.VideoLabelDataset(
        data_valid,
        test_lengths,
        transform=torchvision.transforms.Compose([
            transforms.VideoFilePathToTensor(max_len=max_len_test, fps=15, padding_mode='last'),
            #transforms.VideoResize([256, 256]),
            transforms.VideoCenterCrop([224, 224]),
            transforms.VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # default values for imagenet
        ])
    )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=args.workers)
    
    return train_loader, test_loader

def objective(h):
    count_model() # Update model_count
    # Initialize seed here
    '''
    if model_count == 1:
        seed = 3427
    else:
    '''
    random.seed(random.random())
    seed = random.randint(1, 10000)

    # Initialize Dataloaders
    batch_size = h['batch_size']
    
    # Fixed
    print(f'Model: {model_count}, Hyperparameters: {h}') #, Total iterations: {num_iters}')
    criterion = nn.CrossEntropyLoss()
    
    learning_rate = h['learning_rate']
    num_epochs = h['num_epochs']
    
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    
    k = 5 # Number of folds for cross-validation
    # Load whole dataset
    data = pd.read_csv(r"/s/luffy/b/nobackup/chairoy/Experiment_2/Exp_2.1/video_paths_labels_full_rsz_fps15_center_10.csv")
    
    pid_videos = []
    #list_of_P = [1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 16, 22, 24, 25, 28, 33, 37, 38, 41, 43, 44, 46]
    #for p in list_of_P:
    for p in range(1, 90):
        pid_videos.append(f"P{p}")

    # define dictionary to store scores for particular model run
    scores = dict([(key, []) for key in {"TP","TN","FP","FN","precision","recall","f1_positive_class","NPV","TNR","f1_negative_class","accuracy_test","roc_auc","cohen_kappa","balanced_accuracy","mcc","loss_train","accuracy_train","loss_test"}])
    # Check space utilization
    
    for fold in range(k):
        #initializing best loss and accuracy for every fold
        best_loss = 1000000
        best_acc1 = 0

        #gpu_usage(args.gpu)
        # Load data
        # Load data
        data_train_center, data_valid_center = get_k_fold_data(k, fold, data_center, pid_videos) # Get k-fold cross-validation training and verification data
        data_train_left, data_valid_left = get_k_fold_data(k, fold, data_left, pid_videos) # Get k-fold cross-validation training and verification data
        data_train_right, data_valid_right = get_k_fold_data(k, fold, data_right, pid_videos) # Get k-fold cross-validation training and verification data
        #data_train = data.iloc[:15]
        #data_valid = data.iloc[15:]
        #data_valid.reset_index(drop=True, inplace=True)
        '''
        print(f'Train data: {data_train}')
        print(f'Valid data: {data_valid}')
        '''
        st = time.time()
        # Create dataloaders
        train_loader_center, test_loader_center = load_data(data_train_center, data_valid_center, batch_size) # Center angle
        train_loader_left, test_loader_left = load_data(data_train_left, data_valid_left, batch_size) # Left angle
        train_loader_right, test_loader_right = load_data(data_train_right, data_valid_right, batch_size) # Right angle
        
        # Create models
        model_center = VideoGlobalModel(args, args.size, num_classes=args.num_classes)
        model_left = VideoGlobalModel(args, args.size, num_classes=args.num_classes)
        model_right = VideoGlobalModel(args, args.size, num_classes=args.num_classes)
        # Load state dicts
        model_center.load_state_dict(torch.load(fr'best_model_center/athletic_model_video_1_{fold+1}.pt', map_location=lambda storage, loc: storage.cuda(args.gpu)))
        model_left.load_state_dict(torch.load(fr'best_model_left/athletic_model_video_13_{fold+1}.pt', map_location=lambda storage, loc: storage.cuda(args.gpu)))
        model_right.load_state_dict(torch.load(fr'best_model_right/athletic_model_video_8_{fold+1}.pt', map_location=lambda storage, loc: storage.cuda(args.gpu)))
        
        # Create ensemble model
        model = VideoGlobalModelEnsemble(args, args.size, model_center, model_left, model_right, num_classes=args.num_classes)
        
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)  # Optimizer
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=args.weight_decay)  # Optimizer
        #gpu_usage(args.gpu)
        
        # Store loss
        train_loss_values = []
        test_loss_values = []
        
        # Store accuracy
        train_acc_values = []
        test_acc_values = []

        #for epoch in range(args.start_epoch, num_epochs):
        save_epoch = 0
        save_loss_train = 0
        save_acc_train = 0
        save_loss_val = 0
        save_acc_val = 0
        save_predicted_labels = []
        save_true_labels = []
        model_count_fold = f'{model_count}_{fold+1}'
        model_name = f'athletic_model_video_combined_{model_count_fold}.pt'
        model_path = fr'models/{model_name}'
        for epoch in range(num_epochs):
            #adjust_learning_rate(optimizer, epoch, learning_rate, args)
            st = time.time()
            
            loss_train, acc_train = train(train_loader_center, train_loader_left, train_loader_right, model, optimizer, epoch)
            y_true, y_pred, loss_val, acc_val = validate(test_loader_center, test_loader_left, test_loader_right, model, epoch)

            train_loss_values.append(loss_train)
            train_acc_values.append(acc_train)
            test_loss_values.append(loss_val)
            test_acc_values.append(acc_val)

            #print(f"Epoch {epoch + 1} Train: {loss_train:.4f}, {acc_train.item():.4f}, Val: {loss_val:.4f},{acc_val.item():.4f}, time/epoch: {time.time() - st}")
            
            # remember best acc@1 and save checkpoint
            is_best = loss_val < best_loss
            best_loss = min(loss_val, best_loss)

            if is_best:
                save_epoch = epoch + 1
                save_loss_train = loss_train
                save_acc_train = acc_train
                save_loss_val = best_loss
                save_acc_val = acc_val
                save_predicted_labels = y_pred
                save_true_labels = y_true
                torch.save(model.state_dict(), model_path)

            # Calculate metrics
            if (epoch + 1) == num_epochs:
                TP, TN, FP, FN, precision, recall, f1_positive_class, NPV, TNR, f1_negative_class, roc_auc, cohen_kappa, balanced_accuracy, mcc = measure_performance(save_true_labels, save_predicted_labels)
                scores["TP"].append(TP)
                scores["TN"].append(TN)
                scores["FP"].append(FP)
                scores["FN"].append(FN)
                scores["precision"].append(precision)
                scores["recall"].append(recall)
                scores["f1_positive_class"].append(f1_positive_class)
                scores["NPV"].append(NPV)
                scores["TNR"].append(TNR)
                scores["f1_negative_class"].append(f1_negative_class)
                scores["accuracy_test"].append(save_acc_val.item())
                scores["roc_auc"].append(roc_auc)
                scores["cohen_kappa"].append(cohen_kappa)
                scores["balanced_accuracy"].append(balanced_accuracy)
                scores["mcc"].append(mcc)
                scores["loss_train"].append(save_loss_train)
                scores["accuracy_train"].append(save_acc_train.item())
                scores["loss_test"].append(save_loss_val)
        
        true_vs_pred_labels_headings = f'model_{model_count_fold}=Best_Model_at_Epoch:{save_epoch},true_labels,predicted_labels,incorrect_predictions'
        file_path = fr'true_vs_pred_labels/true_vs_pred_labels_{model_count_fold}.csv'
        f = open(file_path, 'a+')
        f.write(true_vs_pred_labels_headings)
        f.write('\n')
        for i, tl, pl in zip(range(len(data_valid_center)), save_true_labels, save_predicted_labels):
            type_jump = data_valid_center.loc[i, "type"].split('-')
            f.write(str(f'{data_valid_center.loc[i, "participant"]}_{type_jump[0]}-combined_{data_valid_center.loc[i, "jump_count"]}'))
            f.write(',')
            f.write(str(tl))
            f.write(',')
            f.write(str(pl))
            f.write(',')
            if tl != pl:
                f.write(str(1))
            else:
                f.write(str(0))
            f.write('\n')
        f.close()
        
        losses_and_accuracies_headings = f'model_{model_count_fold},train_losses,test_losses,train_accuracies,test_accuracies'
        file_path = fr'losses_and_accuracies/losses_and_accuracies_{model_count_fold}.csv'
        f = open(file_path, 'a+')
        f.write(losses_and_accuracies_headings)
        f.write('\n')
        for ep, train_l, test_l, train_acc, test_acc in zip(range(num_epochs), train_loss_values, test_loss_values, train_acc_values, test_acc_values):
            f.write(str(ep+1))
            f.write(',')
            f.write(str(train_l))
            f.write(',')
            f.write(str(test_l))
            f.write(',')
            f.write(str(train_acc))
            f.write(',')
            f.write(str(test_acc))
            f.write('\n')
        f.close()
        
    # average scores for cv
    avg_TP = statistics.mean(scores["TP"])
    avg_TN = statistics.mean(scores["TN"])
    avg_FP = statistics.mean(scores["FP"])
    avg_FN = statistics.mean(scores["FN"])
    avg_precision = statistics.mean(scores["precision"])
    avg_recall = statistics.mean(scores["recall"])
    avg_f1_positive_class = statistics.mean(scores["f1_positive_class"])
    avg_NPV = statistics.mean(scores["NPV"])
    avg_TNR = statistics.mean(scores["TNR"])
    avg_f1_negative_class = statistics.mean(scores["f1_negative_class"])
    avg_accuracy_test = statistics.mean(scores["accuracy_test"])
    avg_roc_auc = statistics.mean(scores["roc_auc"])
    avg_cohen_kappa = statistics.mean(scores["cohen_kappa"])
    avg_balanced_accuracy = statistics.mean(scores["balanced_accuracy"])
    avg_mcc = statistics.mean(scores["mcc"])
    avg_loss_train = statistics.mean(scores["loss_train"])
    avg_accuracy_train = statistics.mean(scores["accuracy_train"])
    avg_loss_test = statistics.mean(scores["loss_test"])     
    
    # Save hyperparameters and metrics
    to_file = f"{model_count},{seed},{h['batch_size']},{h['num_epochs']},{h['learning_rate']},{avg_loss_train},{avg_accuracy_train},{avg_loss_test},{avg_accuracy_test},{avg_TP},{avg_TN},{avg_FP},{avg_FN},{avg_precision},{avg_recall},{avg_f1_positive_class},{avg_NPV},{avg_TNR},{avg_f1_negative_class},{avg_roc_auc},{avg_cohen_kappa},{avg_balanced_accuracy},{avg_mcc}"
    file_path = fr'hyperparameters_athletics_video_guided_TSM.csv'
    f = open(file_path, 'a+')
    f.write(to_file)
    f.write('\n')
    f.close()  

    return avg_loss_test

def load_data_gpu_transpose(frames):
    if args.gpu is not None:
        frames = frames.cuda(args.gpu, non_blocking=True)
    frames = torch.transpose(frames, 1, 2)
    return frames

def train(train_loader_center, train_loader_left, train_loader_right, model, optimizer, epoch):
    losses = AverageMeter('Loss', ':6.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()
    # Iterate through train dataset
    running_loss_train = 0.0
    correct_train = 0
    total_train = 0
    # switch to train mode
    model.train()

    for (i, (frames_center, labels, lengths_center)), (j, (frames_left, _, lengths_left)), (k, (frames_right, _, lengths_right)) in zip(enumerate(train_loader_center), enumerate(train_loader_left), enumerate(train_loader_right)):
        batch_num = i + 1
        # Clear GPU cache
        #torch.cuda.empty_cache()
        # Load frames and labels onto GPU
        frames_center = load_data_gpu_transpose(frames_center)
        frames_left = load_data_gpu_transpose(frames_left)
        frames_right = load_data_gpu_transpose(frames_right)
        if torch.cuda.is_available():
            labels = labels.cuda(args.gpu, non_blocking=True)

        # Runs the forward pass with autocasting.
        with torch.cuda.amp.autocast():
            outputs = model(frames_center, lengths_center, frames_left, lengths_left, frames_right, lengths_right)
            loss = F.cross_entropy(outputs, labels)
            # print('size of outputs is: ', outputs.size())

        acc1 = accuracy(outputs, labels, topk=(1,))

        # measure accuracy and record loss
        losses.update(loss.item(), frames_center.size(0))
        top1.update(acc1[0], frames_center.size(0))

        assert outputs.size()[0] == len(labels)
        
        #optimizer.zero_grad()
        # Efficient way to zero_grad
        for param in model.parameters():
            param.grad = None
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        # Updates the scale for next iteration.
        scaler.update()

        # if i % args.print_freq == 0:
        #     output = (f'Epoch: [{epoch}][{batch_num}/{len(train_loader)}]\t'
        #               f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
        #               f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t')
        #     log.verbose(output)
        # Total number of labels
        #total_train += labels.size(0)
        # Total correct predictions
        #correct_train += (preds == labels).sum()
        #running_loss_train += loss.item()
    #accuracy_train = 100 * correct_train // total_train
    # gpu_usage(args.gpu)
    # loss_val = running_loss_train / len(train_loader.dataset)  # store all losses
    # acc_val = accuracy_train.item()
    return losses.avg, top1.avg#loss_val, acc_val


def validate(test_loader_center, test_loader_left, test_loader_right, model, epoch):
    losses = AverageMeter('Loss', ':6.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # Metrics
    y_true = []
    y_pred = []
    # # Iterate through test dataset
    # running_loss_test = 0.0
    # correct_test = 0
    # total_test = 0
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for (t, (frames_center, labels, lengths_center)), (u, (frames_left, _, lengths_left)), (v, (frames_right, _, lengths_right)) in zip(enumerate(test_loader_center), enumerate(test_loader_left), enumerate(test_loader_right)):
            #batch_num = j+1
            # Load frames and labels onto GPU
            frames_center = load_data_gpu_transpose(frames_center)
            frames_left = load_data_gpu_transpose(frames_left)
            frames_right = load_data_gpu_transpose(frames_right)
            if torch.cuda.is_available():
                labels = labels.cuda(args.gpu, non_blocking=True)

            # Forward pass only to get logits/output
            # Runs the forward pass with autocasting.
            with torch.cuda.amp.autocast():
                outputs = model(frames_center, lengths_center, frames_left, lengths_left, frames_right, lengths_right)
                loss = F.cross_entropy(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                
            # Store labels for confusion matrix
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

            # measure accuracy and record loss
            acc1 = accuracy(outputs, labels, topk=(1,))
            losses.update(loss.item(), outputs.size(0))
            top1.update(acc1[0], outputs.size(0))


            # if j % args.print_freq == 0:
            #     output = (f'Test:  [{epoch}][{batch_num}/{len(test_loader)}]\t'
            #               f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
            #               f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\n')
            #     log.verbose(output)

        #     # Total number of labels
        #     total_test += labels.size(0)
        #     # Total correct predictions
        #     correct_test += (preds == labels).sum()
        #     running_loss_test += loss.item()
        #
        # accuracy_test = 100 * correct_test // total_test
        # # store losses and accuracies for every epoch
        # loss_val = running_loss_test / len(test_loader.dataset)
        # acc_val = accuracy_test.item()

    return y_true, y_pred, losses.avg, top1.avg

def run_search():
    output = fmin(fn=objective,
                space=space,
                algo=partial(tpe.suggest, n_startup_jobs=5), # Guided Search
                #trials=generate_trials_to_calculate(points),
                max_evals=15,
            )
    return output
    
if __name__ == '__main__':
    create_hyperparameters_results_file()
    run_search()