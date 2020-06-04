import torch
from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary

import numpy as np
import time
import cv2
from tqdm import tqdm
import yaml
from multiprocessing import cpu_count
import Levenshtein

from dataset.dataset import TextRecDataset, get_char_dict_attention, get_char_dict_ctc
from models.crnn import CRNN
from models.loss import CTCFocalLoss
# from models.resnet import resnet18


def eval_ctc(model, dataloader, idx2char):
    model.eval()
    t = time.time()
    total_dist = 0
    total_line_err = 0
    total_ratio = 0
    total_pred_char = 1
    total_label_char = 0
    total_samples = 0
    total_ned = 0

    for j, batch in enumerate(dataloader):
        imgs = batch[0].cuda()
        labels = batch[1].cuda().long()
        labels_length = batch[2]
        labels_str = batch[3]

        with torch.no_grad():
            outputs = model(imgs)

        prob = outputs.softmax(dim=2).cpu().numpy()
        pred = prob.argmax(axis=2)

        for k in range(pred.shape[1]):
            pred_str = ""
            prev = " "
            for t in pred[:,k]:
                if idx2char[t] != prev:
                    pred_str += idx2char[t]
                    prev = idx2char[t]
            
            pred_str = pred_str.strip()
            pred_str = pred_str.replace('-', '')

            dist = Levenshtein.distance(pred_str, labels_str[k])
            total_dist += dist
            ratio = Levenshtein.ratio(pred_str, labels_str[k])
            total_ratio += ratio
            total_ned += float(dist) / max(len(pred_str), len(labels_str[k]))

            total_pred_char += len(pred_str)
            total_label_char += len(labels_str[k])
            total_samples += 1

            if dist != 0: 
                total_line_err += 1
                if k == 0:
                    print('pred: ', pred_str)
                    print('label:', labels_str[k])
                    
    precision = 1.0 - float(total_dist) / total_pred_char
    recall = 1.0 - float(total_dist) / total_label_char
    ave_Levenshtein_ratio = float(total_ratio) / total_samples
    line_acc = 1.0 - float(total_line_err) / total_samples
    rec_score = 1.0 - total_ned / total_samples       
    print("precision: %f" % precision)
    print("recall: %f" % recall)
    print("ave_Levenshtein_ratio: %f" % ave_Levenshtein_ratio)
    print("line_acc: %f" % line_acc)
    print("rec_score: %f" % rec_score)


def eval_attention(model, dataloader, idx2char):
    model.eval()
    t = time.time()
    total_dist = 0
    total_line_err = 0
    total_ratio = 0
    total_pred_char = 1
    total_label_char = 0
    total_samples = 0
    total_ned = 0
    for j, batch in enumerate(dataloader):
        imgs = batch[0].cuda()
        labels = batch[1].cuda().long()
        labels_length = batch[2]
        labels_str = batch[3]

        with torch.no_grad():

            probs = model(imgs, labels, teacher_forcing_ratio=0)


        prob = outputs.softmax(dim=2).cpu().numpy()
        pred = prob.argmax(axis=2)

        for k in range(pred.shape[1]):
            pred_str = ""
            for t in pred[:,k]:
                if idx2char[t] == 'eos':
                    break
                pred_str += idx2char[t]
            
            pred_str = pred_str.strip()
            dist = Levenshtein.distance(pred_str, labels_str[k])
            total_dist += dist
            ratio = Levenshtein.ratio(pred_str, labels_str[k])
            total_ratio += ratio
            total_ned += float(dist) / max(len(pred_str), len(labels_str[k]))

            total_pred_char += len(pred_str)
            total_label_char += len(labels_str[k])
            total_samples += 1

            if dist != 0: 
                total_line_err += 1
                if k == 0:
                    print('pred: ', pred_str)
                    print('label:', labels_str[k])

    precision = 1.0 - float(total_dist) / total_pred_char
    recall = 1.0 - float(total_dist) / total_label_char
    ave_Levenshtein_ratio = float(total_ratio) / total_samples
    line_acc = 1.0 - float(total_line_err) / total_samples
    rec_score = 1.0 - total_ned / total_samples       
    print("precision: %f" % precision)
    print("recall: %f" % recall)
    print("ave_Levenshtein_ratio: %f" % ave_Levenshtein_ratio)
    print("line_acc: %f" % line_acc)
    print("rec_score: %f" % rec_score)


def main():

    print(torch.__version__)

    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(torch.cuda.is_available())
    torch.backends.cudnn.benchmark = True

    char_set = config['char_set']
    if config['method'] == 'ctc':
        char2idx, idx2char = get_char_dict_ctc(char_set)
    else:
        char2idx, idx2char = get_char_dict_attention(char_set)

    config['char2idx'] = char2idx
    config['idx2char'] = idx2char
    batch_size = config['batch_size']

    print(config)

    train_dataset = TextRecDataset(config, phase='train')
    val_dataset = TextRecDataset(config, phase='val')
    test_dataset = TextRecDataset(config, phase='test')
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=cpu_count(),
                                  pin_memory=False)

    valloader = data.DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=cpu_count(),
                                pin_memory=False)

    testloader = data.DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=cpu_count(),
                                 pin_memory=False)

    class_num = len(config['char_set']) + 1
    print('class_num', class_num)
    model = CRNN(class_num=class_num, char2idx=char2idx)
    if config['method'] == 'ctc':
        # criterion = nn.CTCLoss(blank=char2idx['-'], reduction='mean')
        criterion = CTCFocalLoss(blank=char2idx['-'], gamma=0.5)

    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    model = model.cuda()
    summary(model, (1, 32, 400))

    # model = torch.nn.DataParallel(model)

    # optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=1e-2, weight_decay=5e-4)
    optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=0.001, momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 800], gamma=0.1)

    print('train start, total batches %d' % len(trainloader))
    iter_cnt = 0
    for i in range(0, config['epochs']):
        start = time.time()
        model.train()
        for j, batch in enumerate(trainloader):

            iter_cnt += 1
            imgs = batch[0].cuda()
            labels = batch[1].cuda().long()
            labels_length = batch[2].cuda()
            labels_mask = batch[4].cuda().float()

            if config['method'] == 'ctc':
                # CTC loss
                outputs = model(imgs)
                log_prob = outputs.log_softmax(dim=2)
                t,n,c = log_prob.size(0),log_prob.size(1),log_prob.size(2)
                input_length = (torch.ones((n,)) * t).cuda().int()
                loss = criterion(log_prob, labels, input_length, labels_length)
            else:
                # cross_entropy loss
                outputs = model(imgs, labels)
                outputs = outputs.permute(1, 2, 0)
                losses = criterion(outputs, labels)
                losses = losses * labels_mask
                loss = losses.sum() / labels_mask.sum()

            optimizer.zero_grad()            
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            if iter_cnt % config['print_freq'] == 0:
                print('epoch %d, iter %d, train loss %f' % (i + 1, iter_cnt, loss.item()))

        print('epoch %d, time %f' % (i + 1, (time.time() - start)))
        scheduler.step()

        print("validating...")
        
        if config['method'] == 'ctc':
            eval_ctc(model, valloader, idx2char)
        else:
            eval_attention(model, valloader, idx2char)

        if (i + 1) % config['test_freq'] == 0:
            print("testing...")
            if config['method'] == 'ctc':
                eval_ctc(model, testloader, idx2char)
            else:
                eval_attention(model, testloader, idx2char)
        # # if (i + 1) % config['save_freq'] == 0:
        #     save_model(config['save_path'], i + 1, model, optimizer=optimizer)

if __name__ == '__main__':
    main()