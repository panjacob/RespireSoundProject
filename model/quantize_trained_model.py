import torch.quantization
from LungAttn import LungAttnBinary, QuantizedLungAttnBinary, one_hot, myDataset, makedirs
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import numpy as np
import joblib
import argparse
import torch.nn as nn
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_bs', type=int, default=64)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--input', '-i',
                    default="./pack/binary/tqwt1_4_train.p", type=str,
                    help='path to directory with input data archives')
parser.add_argument('--test', default="./pack/binary/tqwt1_4_test.p",
                    type=str, help='path to directory with test data archives')
parser.add_argument('--prunning_amount', default=0.4,
                    type=int)
args = parser.parse_args()

def get_pruned_version(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.identity(module, 'weight')
        if isinstance(module, torch.nn.Linear):
            prune.identity(module, 'weight')
    return model

def calibrate_model(model, device):
    for batch_idx, (cat_stft, labels) in enumerate(train_loader):
        cat_stft = cat_stft.to(device)
        model(cat_stft)


def print_parameters_dtype(model):
    for name, param in model.named_parameters():
        print(name, ":", param.dtype)

def get_mnist_loaders(batch_size=128, test_batch_size = 500, workers = 4, perc=1.0, binary=False):
    # ori, ck, wh, res, label
    ori, stftl, stfth, stftr, labels = joblib.load(open(args.input, mode='rb'))
    stftl, stfth, stftr = np.array(stftl), np.array(stfth), np.array(ori)
    if binary:
        labels = np.array(labels).reshape(-1, 1)
    else:
        labels = one_hot(np.array(labels), 4)
    stft = np.concatenate((stftl[:, np.newaxis], stfth[:, np.newaxis], stftr[:, np.newaxis]), 1)

    ori_tst, stftl_test, stfth_test, stftr_test, labels_test = joblib.load(open(args.test, mode='rb'))
    stftl_test, stfth_test, stftr_test = np.array(stftl_test), np.array( stfth_test), np.array(ori_tst)
    if binary:
        labels_test = np.array(labels_test).reshape(-1, 1)
    else:
        labels_test = one_hot(np.array(labels_test), 4)
    stft_test = np.concatenate((stftl_test[:, np.newaxis], stfth_test[:, np.newaxis], stftr_test[:, np.newaxis]), 1)

    train_loader = DataLoader(
        myDataset(stft, labels), batch_size=batch_size,
        shuffle=True, num_workers=workers, drop_last=True
    )
    train_eval_loader = DataLoader(
        myDataset(stft, labels), batch_size=test_batch_size,
        shuffle=False, num_workers=workers, drop_last=True
    )

    test_loader = DataLoader(
        myDataset(stft_test, labels_test),
        batch_size=test_batch_size, shuffle=False, num_workers=workers, drop_last=False
    )

    return train_loader, train_eval_loader, test_loader

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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

def accuracy(model, dataset_loader,criterion, device):
    total_correct = 0
    targets = []
    outputs = []
    losses = AverageMeter()
    for cat_stft, y in dataset_loader:
        target_class = np.round(y.numpy())
        targets = np.append(targets, target_class)
        with torch.no_grad():
            sigmoid_output = model(cat_stft)
        y = y.type_as(sigmoid_output)
        loss = criterion(sigmoid_output, y)
        losses.update(loss.data, y.size(0))
        predicted_class = np.round(sigmoid_output.cpu().detach().numpy())
        outputs = np.append(outputs, predicted_class)
        total_correct += np.sum(predicted_class == target_class)
    acc = total_correct / len(dataset_loader.dataset)
    Confusion_matrix = sk_confusion_matrix(targets.tolist(), outputs.tolist())
    print('Confusion_matrix:')
    print(Confusion_matrix)
    tn, fp, fn, tp = Confusion_matrix.ravel()
    Sq = tp / (tp + fn)
    Se = tn / (tn + fp)
    return acc, Se, Sq, (Se + Sq) / 2, losses.avg.item(), Confusion_matrix

if __name__ == "__main__":
    model_path = "./log/pruned/model_pruned_0.4/saved_model_params"
    save_dir = "./log/quantized/model_quantized"
    makedirs(save_dir)

    # get loaders
    train_loader, train_eval_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size,
                                                                     test_batch_size=args.test_bs, workers=args.workers,
                                                                     binary=True)
    # prepare criterion
    criterion = nn.BCELoss()

    # select device
    use_cuda = False
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # FP32 Model tests

    fp32_model = LungAttnBinary()
    fp32_model.load_state_dict(torch.load(model_path))
    # fp32_model.to(device)
    fp32_model.eval()
    # print("FP32 scores")
    # train_acc, train_Se, train_Sq, train_Score, _, train_confm = accuracy_binary(fp32_model, train_eval_loader,
    #                                                                              criterion, device)
    # test_acc, test_Se, test_Sq, test_Score, test_loss, test_confm = accuracy_binary(fp32_model, test_loader, criterion, device)
    #
    # print(
    #     "Train Acc {:.4f} |  train Se {:.4f} | train Sq {:.4f} | train Score {:.4f} | Test Loss {:.4f} |Test Acc {:.4f} | test Se {:.4f} | test Sq {:.4f} | test Score {:.4f}".format(
    #         train_acc, train_Se, train_Sq, train_Score, test_loss, test_acc, test_Se,
    #         test_Sq, test_Score))

    # # FP16 Model tests
    #
    # fp16_model = LungAttnBinary().half()
    # if model_pruned:
    #     fp16_model = get_pruned_version(fp16_model)
    # fp16_model.load_state_dict(torch.load(model_path))
    # fp16_model.to(device)
    # fp16_model.eval()
    #
    # print("FP16 scores")
    # train_acc, train_Se, train_Sq, train_Score, _, train_confm = accuracy_binary(fp16_model, train_eval_loader,
    #                                                                              criterion, device)
    # test_acc, test_Se, test_Sq, test_Score, test_loss, test_confm = accuracy_binary(fp16_model, test_loader, criterion,
    #                                                                                 device)
    # print(
    #     "Train Acc {:.4f} |  train Se {:.4f} | train Sq {:.4f} | train Score {:.4f} | Test Loss {:.4f} |Test Acc {:.4f} | test Se {:.4f} | test Sq {:.4f} | test Score {:.4f}".format(
    #         train_acc, train_Se, train_Sq, train_Score, test_loss, test_acc, test_Se,
    #         test_Sq, test_Score))

    # STATIC QUANTIZATION TESTS
    model_quantized = QuantizedLungAttnBinary(fp32_model)
    backend = "fbgemm"
    model_quantized.eval()
    model_quantized.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_prepared = torch.ao.quantization.prepare(model_quantized, inplace=True)

    # Calibrate quantized model with dataset
    calibrate_model(model_prepared, device)

    # convert model to int8
    model_int8 = torch.ao.quantization.convert(model_prepared, inplace=True)

    print("INT8 static quantization scores")
    train_acc, train_Se, train_Sq, train_Score, _, train_confm = accuracy(model_int8, train_eval_loader,
                                                                                 criterion, device)
    test_acc, test_Se, test_Sq, test_Score, test_loss, test_confm = accuracy(model_int8, test_loader, criterion, device)

    print(
        "Train Acc {:.4f} |  train Se {:.4f} | train Sq {:.4f} | train Score {:.4f} | Test Loss {:.4f} |Test Acc {:.4f} | test Se {:.4f} | test Sq {:.4f} | test Score {:.4f}".format(
            train_acc, train_Se, train_Sq, train_Score, test_loss, test_acc, test_Se,
            test_Sq, test_Score))