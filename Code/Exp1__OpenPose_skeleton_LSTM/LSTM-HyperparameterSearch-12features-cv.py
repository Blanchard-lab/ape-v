import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

from hyperopt import *
from hyperopt.fmin import generate_trials_to_calculate
from functools import partial
import time
import random
import statistics

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score, balanced_accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score


# Set maximum number of cpu cores to use
torch.set_num_threads(8) 

# Check if GPU available
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

# Gets length (Number of frames) of videos
def get_max_length(skeletons_for_max_length):
    start_index_of_videos = []
    length_of_videos = []

    skeleton_iterator = skeletons_for_max_length.iterrows()
    first = next(skeleton_iterator)
    participant_check = first[1]['participant']
    type_check = first[1]['type']
    jump_count_check = first[1]['jump_count']

    count = 1
    start_index_of_videos.append(1) # not considering title row
    for i, row in skeleton_iterator:
        if row['participant'] == participant_check and row['type'] == type_check and row['jump_count'] == jump_count_check:
            count = count + 1
        else:
            length_of_videos.append(count)
            start_index_of_videos.append(i)
            count = 1
            participant_check = row['participant']
            type_check = row['type']
            jump_count_check = row['jump_count']
        if i == len(skeletons_for_max_length) - 1:
            length_of_videos.append(count)
            
    return start_index_of_videos, length_of_videos
 
# Defines dataset preprocessing before input to model
class AthleticsDataset(Dataset):
    """Athletics dataset."""

    def __init__(self, skeletons, labels, start_index_of_videos, length_of_videos, transform=None):
        self.skeletons = skeletons
        self.labels = labels
        self.max_length = max(length_of_videos)
        self.start_index_of_videos = start_index_of_videos
        self.length_of_videos = length_of_videos

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        all_zeros_index = [0] * self.max_length
        
        # Get start_index and seq_len of video using idx
        start_index = self.start_index_of_videos[idx]
        seq_len = self.length_of_videos[idx]
        
        # Get end_index
        end_index = start_index + seq_len
        
        # Get data
        data = self.skeletons.iloc[start_index:end_index, :]
        
        # Get mean frame for video, and subtract that mean frame from every frame in video (Normalizing)
        data = data.sub(data.mean(axis=0), axis=1)    
        
        # Pad the data for missing frames
        padding = self.max_length - seq_len
        data = np.pad(data, ((0,padding),(0,0)), 'constant') # ((top, bottom), (left, right))
        
        # Get label for current idx
        labels = self.labels.iloc[idx, -1]

        # Converting to torch tensors
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels)#, dtype=torch.float32)

        return (data, labels, seq_len)

# LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape (batch_dim, seq_dim, feature_dim)
        # batch size: b
        # seq_dim: No. of frames
        # input: b/seq_dim/12
        # hidden: h = num of hidden units (1 layer)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True) # Output: b/seq_dim/h
        
        # sum across axis=1: b/h
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim) # Output: b/2

    def forward(self, x, seq_len):
        # Tensors declared in forward() have to be manually transferred on GPU
    
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        
        # Get lstm output
        #self.lstm.flatten_parameters() # For DataParallel
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Get 'out' onto CPU for manipulations
        out.to(torch.device('cpu'))
        out_copy = out.clone() # Copy created for unsqueeze_ inplace operation
        
        #print(f'out.shape 1: {out.shape}') # torch.Size([batch_size, max_length of video, hidden dimension])
        # Clip padded rows from output
        output_list = []
        for i, length in zip(range(len(seq_len)), seq_len):
            output_list.append(out_copy[i,:length,:].unsqueeze_(0))
        # Sum over frames in video to get 1 frame equivalent
        output_list_sum = []
        for i, tensor in zip(range(len(output_list)), output_list):
            output_list_sum.append(torch.sum(tensor, 1))
        # Concatenate individual tensors together
        if len(output_list_sum) == 1:
            out = output_list_sum[0]
        elif len(output_list_sum) >= 2:
            out = torch.cat([output_list_sum[0], output_list_sum[1]])
            if len(output_list_sum) > 2:
                for i in range(2, len(output_list_sum)):
                    out = torch.cat([out, output_list_sum[i]])
        #print(f'out.shape before GPU: {out.shape}') # torch.Size([batch_size, hidden dimension])
        
        # Get 'out' onto GPU
        out.to(device)
        
        out = self.fc(out[:, :]) ## input: batch size * hidden dimension ## output: batch size*2 # [:, -1, :]
        return out
 
# Creates a file for storing hyperparameters and results
def create_hyperparameters_results_file():    
    # Create file to store results
    headings = f'Model,Seed,No_of_Hidden_Layers,No_of_Hidden_Units,Batch_Size,Num_Epochs,Learning_Rate,Training_Loss,Training_Accuracy,Testing_Loss,Testing_Accuracy,True_positive,True_negative,False_positive,False_negative,Precision_PositivePredictedValue,Recall_TruePositiveRate,F1_score_positive_class,NegativePredictedValue,Specificity_TrueNegativeRate,F1_score_negative_class,ROC_AUC_score,Cohen_kappa_score,Balanced_accuracy,Matthews_correlation_coefficient'
    file_path = fr'hyperparameters_athletics_12features_guided.csv'
    f = open(file_path, 'a+')
    f.write(headings)
    f.write('\n')
    f.close()
    return 
 
# Hyperparameter search space 
space = {
    'batch_size': hp.choice('batch_size', [8, 16, 24, 32, 40, 48, 56, 64, 128, 256]),
    'num_epochs': hp.choice('num_epochs', range(10, 205, 5)),
    'n_hidden_units': hp.choice('n_hidden_units', range(10, 205, 5)),
    'n_hidden_layers': hp.choice('n_hidden_layers', range(1, 5, 1)),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)) # LR bw 0.0001 and 0.01
}

# Warm start: Best hyperparameters till now [Give index for hp.choice()]
'''
points = [{
    'batch_size': 3,
    'num_epochs': 21,
    'n_hidden_units': 15,
    'n_hidden_layers': 0,
    'learning_rate': 0.012634357
}]
'''

# Performance metrics
def measure_performance(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    precision = recall = f1_positive_class = NPV = TNR = f1_negative_class = 0
    TN, FP, FN, TP = cm.ravel()

    ## Scores for positive class
    # Precision or positive predictive value (PPV)
    if TP+FP != 0:
        precision = TP/(TP+FP)
    #precision = precision_score(y_true, y_pred)
    # Sensitivity, hit rate, recall, or true positive rate (TPR)
    if TP+FN != 0:
        recall = TP/(TP+FN)
    #recall = recall_score(y_true, y_pred)
    # F1 score
    if precision + recall != 0:
        f1_positive_class = 2 * (precision * recall) / (precision + recall)
    #f1_positive_class = f1_score(y_true, y_pred)
    
    ## Scores for negative class
    # Negative predictive value (NPV)
    if TN+FN != 0:
        NPV = TN/(TN+FN)
    # Specificity or true negative rate (TNR)
    if TN+FP != 0:
        TNR = TN/(TN+FP)
    # F1 score
    if NPV + TNR != 0:
        f1_negative_class = 2 * (NPV * TNR) / (NPV + TNR)
      
    '''
    # False discovery rate
    if TP+FP != 0:
        FDR = FP/(TP+FP)
    # Fall out or false positive rate
    if FP+TN != 0:
        FPR = FP/(FP+TN)
    # False negative rate
    if TP+FN != 0:
        FNR = FN/(TP+FN)
    '''   

    # Overall accuracy
    #accuracy_test = (TP+TN)/(TP+FP+FN+TN)
    
    roc_auc = roc_auc_score(y_true, y_pred)
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    
    # Diagonal of 'normalized' cm -> balanced success rate [use this as main metric]
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    
    # Matthew's correlation coefficient (better for binary classification)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return TP, TN, FP, FN, precision, recall, f1_positive_class, NPV, TNR, f1_negative_class, roc_auc, cohen_kappa, balanced_accuracy, mcc

model_count = 1

def get_fold_set(j):
    fold_1 = ['P1', 'P4', 'P7', 'P8', 'P12', 'P18', 'P20', 'P22', 'P27', 'P30', 'P34', 'P40', 'P46', 'P53', 'P58', 'P62', 'P64', 'P69']
    fold_2 = ['P2', 'P6', 'P13', 'P17', 'P24', 'P29', 'P37', 'P43', 'P49', 'P54', 'P57', 'P68', 'P73', 'P78', 'P80', 'P84', 'P87', 'P89']
    fold_3 = ['P5', 'P16', 'P21', 'P23', 'P31', 'P35', 'P38', 'P42', 'P45', 'P50', 'P52', 'P59', 'P66', 'P72', 'P76', 'P81', 'P83', 'P86']
    fold_4 = ['P9', 'P11', 'P19', 'P25', 'P28', 'P32', 'P36', 'P39', 'P44', 'P48', 'P55', 'P60', 'P65', 'P67', 'P71', 'P75', 'P82', 'P88']
    fold_5 = ['P3', 'P10', 'P14', 'P15', 'P26', 'P33', 'P41', 'P47', 'P51', 'P56', 'P61', 'P63', 'P70', 'P74', 'P77', 'P79', 'P85']
    
    if j == 1:
        return fold_1
    elif j == 2:
        return fold_2
    elif j == 3:
        return fold_3
    elif j == 4:
        return fold_4
    elif j == 5:
        return fold_5

# K-fold dataset
def get_k_fold_data(k, fold, X, y, pid_videos):
    #fold_size = len(pid_videos) // k # Number of copies: total number of data/fold (number of groups)
    X_train, y_train = None, None
    for j in range(k):
        #idx = slice(j * fold_size, (j + 1) * fold_size) #slice(start,end,step) slice function
        ##idx valid for each group
        #pid_set = pid_videos[idx]
        pid_set = get_fold_set(j+1)
        # Get labels and frames wrt the pids
        y_part = y[y.set_index(['participant']).index.isin(pid_set)] 
        X_part = X[X.set_index(['participant']).index.isin(pid_set)]
        if j == fold: ###The i-fold is valid
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = pd.concat((X_train, X_part), axis=0) #axis=0; Increase the number of lines, connect vertically
            y_train = pd.concat((y_train, y_part), axis=0)
    
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_valid.reset_index(drop=True, inplace=True)
    y_valid.reset_index(drop=True, inplace=True)
    return X_train, y_train, X_valid, y_valid

## Check for predefined values in hyperparameters
def objective(h):
    global model_count
    # Initialize seed here
    '''
    if model_count == 1:
        seed = 4110
    else:
    '''
    random.seed(random.random())
    seed = random.randint(1, 10000)

    # Initialize Dataloaders
    batch_size = h['batch_size']
    
    # Fixed
    input_dim = 12 # Number of features
    output_dim = 2 # Output dimensions (correct/incorrect)
    
    # Variable (Hyperparameter search)
    num_epochs = h['num_epochs']
    #num_iters = int(num_epochs * (len(train_dataset) / batch_size))
    
    print(f'Model: {model_count}, Hyperparameters: {h}') #, Total iterations: {num_iters}')
 
    hidden_dim = h['n_hidden_units']
    layer_dim = h['n_hidden_layers']
    
    criterion = nn.CrossEntropyLoss()
    learning_rate = h['learning_rate']
    
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    
    k = 5 # Number of folds for cross-validation
    
    # Load whole dataset
    X = pd.read_csv(r"video_frames_12features_center.csv")
    y = pd.read_csv(r"labels_center.csv")
    
    # list of participants in dataset
    pid_videos = []
    #list_of_P = [1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 16, 22, 24, 25, 28, 33, 37, 38, 41, 43, 44, 46]
    #for p in list_of_P:
    for p in range(1, 90):
        pid_videos.append(f"P{p}")
    
    # define dictionary to store scores for particular model run
    scores = dict([(key, []) for key in {"TP","TN","FP","FN","precision","recall","f1_positive_class","NPV","TNR","f1_negative_class","accuracy_test","roc_auc","cohen_kappa","balanced_accuracy","mcc","loss_train","accuracy_train","loss_test"}])
    
    for fold in range(k):
        X_train, y_train, X_valid, y_valid = get_k_fold_data(k, fold, X, y, pid_videos) # Get k-fold cross-validation training and verification data
        
        train_start_index_of_videos, train_length_of_videos = get_max_length(X_train)
        # define the scaler
        scaler = MinMaxScaler()
        # Define features to normalize
        features_to_normalize = ['RHip - x', 'RHip - y', 'RKnee - x', 'RKnee - y', 'RAnkle - x', 'RAnkle - y', 'LHip - x', 'LHip - y', 'LKnee - x', 'LKnee - y', 'LAnkle - x', 'LAnkle - y']   
        # fit on the training dataset
        scaler.fit(X_train[features_to_normalize])
        # scale the training dataset
        X_train = scaler.transform(X_train[features_to_normalize].values)
        X_train = pd.DataFrame.from_records(X_train)
        train_dataset = AthleticsDataset(X_train, y_train, train_start_index_of_videos, train_length_of_videos)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True, num_workers=0, shuffle=False)

        test_start_index_of_videos, test_length_of_videos = get_max_length(X_valid)
        # scale the test dataset
        X_valid = scaler.transform(X_valid[features_to_normalize].values)
        X_valid = pd.DataFrame.from_records(X_valid)
        test_dataset = AthleticsDataset(X_valid, y_valid, test_start_index_of_videos, test_length_of_videos)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, pin_memory=True, num_workers=0, shuffle=False)
        
        model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
        # Use all available GPUs
        #model= nn.DataParallel(model)
        # Place model on GPU
        model.to(device)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    
        accuracy = 0
        loss = torch.tensor([0.0])
        
        # Number of steps to unroll
        train_seq_dim = max(train_length_of_videos) #max_length of frames from videos # time steps (number of frames in video)
        test_seq_dim = max(test_length_of_videos)
        #iter = 0
        
        # Store loss
        train_loss_values = []
        test_loss_values = []
        
        # Store accuracy
        train_acc_values = []
        test_acc_values = []
        
        best_loss = 1000000
        save_epoch = 0
        save_loss_train = 0
        save_acc_train = 0
        save_loss_val = 0
        save_acc_val = 0
        save_predicted_labels = []
        save_true_labels = []
        model_name = f'athletic_model_12features_{model_count}_{fold+1}.pt'
        model_path = fr'models/{model_name}'
        for epoch in range(num_epochs):
            # Metrics
            y_true = []
            y_pred = []
            
            # Iterate through train dataset
            running_loss_train = 0.0
            correct_train = 0
            total_train = 0
            for i, (frames, labels, seq_len) in enumerate(train_loader):
                # Load frames as a torch tensor
                frames = frames.view(-1, train_seq_dim, input_dim)#.requires_grad_() # (num_samples, t, N)
                
                # Load frames and labels onto GPU
                frames, labels = frames.to(device), labels.to(device)

                # Clear gradients w.r.t. parameters
                #optimizer.zero_grad()
                # Efficient way to zero_grad
                for param in model.parameters():
                    param.grad = None
                
                # Forward pass to get output/logits
                outputs = model(frames, seq_len)
                
                # Get training output predictions from the maximum value
                _, predicted = torch.max(outputs, 1)

                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs, labels)

                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                optimizer.step()
                
                # Total number of labels
                total_train += labels.size(0)
                
                # Total correct predictions
                correct_train += (predicted == labels).sum()
                
                running_loss_train += loss.item()
                
            accuracy_train = 100 * correct_train // total_train
            
            # Iterate through test dataset
            running_loss_test = 0.0
            correct_test = 0
            total_test = 0
            for j, (frames, labels, seq_len) in enumerate(test_loader):
                frames = frames.view(-1, test_seq_dim, input_dim)
                
                # Load frames and labels onto GPU
                frames, labels = frames.to(device), labels.to(device)

                # Forward pass only to get logits/output
                outputs = model(frames, seq_len)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs, 1)

                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs, labels)

                # Store labels for confusion matrix
                y_true.extend(labels.tolist())
                y_pred.extend(predicted.tolist())

                # Total number of labels
                total_test += labels.size(0)

                # Total correct predictions
                correct_test += (predicted == labels).sum()
                
                running_loss_test += loss.item()
            
            accuracy_test = 100 * correct_test // total_test
            
            # store losses and accuracies for every epoch
            train_l = running_loss_train / len(train_dataset)
            train_loss_values.append(train_l) # store all losses
            test_l = running_loss_test / len(test_dataset)
            test_loss_values.append(test_l)
            train_acc_values.append(accuracy_train.item())
            test_acc_values.append(accuracy_test.item())
            
            is_best = test_l < best_loss
            best_loss = min(test_l, best_loss)

            if is_best:
                save_epoch = epoch + 1
                save_loss_train = train_l
                save_acc_train = accuracy_train.item()
                save_loss_val = best_loss
                save_acc_val = accuracy_test.item()
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
                scores["accuracy_test"].append(save_acc_val)
                scores["roc_auc"].append(roc_auc)
                scores["cohen_kappa"].append(cohen_kappa)
                scores["balanced_accuracy"].append(balanced_accuracy)
                scores["mcc"].append(mcc)
                scores["loss_train"].append(save_loss_train)
                scores["accuracy_train"].append(save_acc_train)
                scores["loss_test"].append(save_loss_val)
        
        true_vs_pred_labels_headings = f'model_{model_count}_{fold+1}=Best_Model_at_Epoch:{save_epoch},true_labels,predicted_labels,incorrect_predictions'
        file_path = fr'true_vs_pred_labels/true_vs_pred_labels_{model_count}_{fold+1}.csv'
        f = open(file_path, 'a+')
        f.write(true_vs_pred_labels_headings)
        f.write('\n')
        for i, tl, pl in zip(range(len(y_valid)), save_true_labels, save_predicted_labels):
            f.write(str(f'{y_valid.loc[i, "participant"]}_{y_valid.loc[i, "type"]}_{y_valid.loc[i, "jump_count"]}'))
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
        
        losses_and_accuracies_headings = f'model_{model_count}_{fold+1},train_losses,test_losses,train_accuracies,test_accuracies'
        file_path = fr'losses_and_accuracies/losses_and_accuracies_{model_count}_{fold+1}.csv'
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
    
    to_file = f"{model_count},{seed},{h['n_hidden_layers']},{h['n_hidden_units']},{h['batch_size']},{h['num_epochs']},{h['learning_rate']},{avg_loss_train},{avg_accuracy_train},{avg_loss_test},{avg_accuracy_test},{avg_TP},{avg_TN},{avg_FP},{avg_FN},{avg_precision},{avg_recall},{avg_f1_positive_class},{avg_NPV},{avg_TNR},{avg_f1_negative_class},{avg_roc_auc},{avg_cohen_kappa},{avg_balanced_accuracy},{avg_mcc}"
    file_path = fr'hyperparameters_athletics_12features_guided.csv'
    f = open(file_path, 'a+')
    f.write(to_file)
    f.write('\n')
    f.close()  
    
    model_count += 1

    return avg_loss_test
      
def run_search():
    # warm start: initialize with best set of hyperparameters
    output = fmin(fn=objective,
                space=space,
                algo=partial(tpe.suggest, n_startup_jobs=50), # Guided Search
                #trials=generate_trials_to_calculate(points), # Uncomment when using warm start (when defining "points" (defined after search space))
                max_evals=150
            )
    return output
    
if __name__ == '__main__':
    create_hyperparameters_results_file()
    run_search()