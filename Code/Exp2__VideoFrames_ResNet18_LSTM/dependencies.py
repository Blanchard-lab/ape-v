import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score, balanced_accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
import pandas as pd
import cv2
import shutil
from os.path import join

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# Check GPU usage
def gpu_usage(gpu):
    t = torch.cuda.get_device_properties(gpu).total_memory
    r = torch.cuda.memory_reserved(gpu)
    a = torch.cuda.memory_allocated(gpu)
    f = r - a  # free inside reserved
    print(f'Reserved memory: {r}')
    print(f'Allocated memory: {a}')
    print(f'Free memory: {f}')
    return

def get_frame_length(dataframe):
    number_of_frames = []
    video_paths = dataframe.iloc[:].path
    #print(video_paths)
    for path in video_paths:
        try:
            path = join('/s/babbage/b/nobackup/nblancha/public-datasets/APE-V/clipped_athletic_data_rsz_fps15_every_10th', path)
            #print(path)
            cap = cv2.VideoCapture(path)
        except Exception as e:
            print(e)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        number_of_frames.append(num_frames)
    return number_of_frames


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# Performance metrics
def measure_performance(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    precision = recall = f1_positive_class = NPV = TNR = f1_negative_class = 0
    TN, FP, FN, TP = cm.ravel()

    ## Scores for positive class
    # Precision or positive predictive value (PPV)
    if TP + FP != 0:
        precision = TP / (TP + FP)
    # precision = precision_score(y_true, y_pred)
    # Sensitivity, hit rate, recall, or true positive rate (TPR)
    if TP + FN != 0:
        recall = TP / (TP + FN)
    # recall = recall_score(y_true, y_pred)
    # F1 score
    if precision + recall != 0:
        f1_positive_class = 2 * (precision * recall) / (precision + recall)
    # f1_positive_class = f1_score(y_true, y_pred)

    ## Scores for negative class
    # Negative predictive value (NPV)
    if TN + FN != 0:
        NPV = TN / (TN + FN)
    # Specificity or true negative rate (TNR)
    if TN + FP != 0:
        TNR = TN / (TN + FP)
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
    # accuracy_test = (TP+TN)/(TP+FP+FN+TN)

    roc_auc = roc_auc_score(y_true, y_pred)
    cohen_kappa = cohen_kappa_score(y_true, y_pred)

    # Diagonal of 'normalized' cm -> balanced success rate [use this as main metric]
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    '''
    Balanced Accuracy = TPR+TNR/2
    '''

    # Matthew's correlation coefficient (better for binary classification)
    mcc = matthews_corrcoef(y_true, y_pred)

    return TP, TN, FP, FN, precision, recall, f1_positive_class, NPV, TNR, f1_negative_class, roc_auc, cohen_kappa, balanced_accuracy, mcc


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
def get_k_fold_data(k, fold, data, pid_videos):
    #fold_size = len(pid_videos) // k  # Number of copies: total number of data/fold (number of groups)
    data_train = None
    for j in range(k):
        #idx = slice(j * fold_size, (j + 1) * fold_size) #slice(start,end,step) slice function
        ##idx valid for each group
        #pid_set = pid_videos[idx]
        pid_set = get_fold_set(j+1)
        # Get labels and frames wrt the pids
        data_part = data[data.set_index(['participant']).index.isin(pid_set)]
        if j == fold:  ###The i-fold is valid
            data_valid = data_part
        elif data_train is None:
            data_train = data_part
        else:
            data_train = pd.concat((data_train, data_part), axis=0)  # axis=0; Increase the number of lines, connect vertically

    data_train.reset_index(drop=True, inplace=True)
    data_valid.reset_index(drop=True, inplace=True)
    return data_train, data_valid
