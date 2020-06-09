import torch
from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary

import os
import numpy as np
import time
import cv2
from tqdm import tqdm
import yaml
from multiprocessing import cpu_count
import Levenshtein

from dataset.dataset import TextRecDataset, get_char_dict_attention, get_char_dict_ctc
from models.crnn import CRNN, RNNAttentionDecoder, TransformerDecoder, AttentionHead
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
        labels_length = batch[1].cuda()
        labels_str = batch[2]

        with torch.no_grad():
            outputs_ctc, _ = model(imgs)
            # outputs_att = decoder(sqs, label_att)

        prob = outputs_ctc.softmax(dim=2).cpu().numpy()
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
    return line_acc, rec_score


def eval_attention(model, decoder, dataloader, idx2char):
    model.eval()
    decoder.eval()

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
        labels_length = batch[1].cuda()
        labels_str = batch[2]

        with torch.no_grad():
            outputs_ctc, sqs = model(imgs)
            outputs_att = decoder(sqs)

        prob = outputs_att.softmax(dim=2).cpu().numpy()
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
    return line_acc, rec_score


def main():

    print(torch.__version__)

    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(torch.cuda.is_available())
    torch.backends.cudnn.benchmark = True

    char_set = config['char_set']
    # if config['method'] == 'ctc':
    char2idx_ctc, idx2char_ctc = get_char_dict_ctc(char_set)
    char2idx_att, idx2char_att = get_char_dict_attention(char_set)
    config['char2idx_ctc'] = char2idx_ctc
    config['idx2char_ctc'] = idx2char_ctc
    config['char2idx_att'] = char2idx_att
    config['idx2char_att'] = idx2char_att

    batch_size = config['batch_size']

    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])
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
    model = CRNN(class_num)
    # decoder = Decoder(class_num, config['max_string_len'], char2idx_att)
    attention_head = AttentionHead(class_num, config['max_string_len'], char2idx_att)

    # criterion = nn.CTCLoss(blank=char2idx['-'], reduction='mean')
    criterion_ctc = CTCFocalLoss(blank=char2idx_ctc['-'], gamma=0.5)
    criterion_att = nn.CrossEntropyLoss(reduction='none')

    if config['use_gpu']:
        model = model.cuda()
        # decoder = decoder.cuda()
        attention_head = attention_head.cuda()
    summary(model, (1, 32, 400))

    # model = torch.nn.DataParallel(model)

    # optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=1e-2, weight_decay=5e-4)
    optimizer = torch.optim.SGD([{'params': model.parameters()},
                                 {'params': attention_head.parameters()}], lr=0.001, momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 800], gamma=0.1)

    print('train start, total batches %d' % len(trainloader))
    iter_cnt = 0
    for i in range(1, config['epochs']+1):
        start = time.time()
        model.train()
        attention_head.train()
        for j, batch in enumerate(trainloader):

            iter_cnt += 1
            imgs = batch[0].cuda()
            labels_length = batch[1].cuda()
            labels_str = batch[2]
            labels_ctc = batch[3].cuda().long()
            labels_ctc_mask = batch[4].cuda().float()
            labels_att = batch[5].cuda().long()
            labels_att_mask = batch[6].cuda().float()

            if config['method'] == 'ctc':
                # CTC loss
                outputs, cnn_features = model(imgs)
                log_prob = outputs.log_softmax(dim=2)
                t,n,c = log_prob.size(0),log_prob.size(1),log_prob.size(2)
                input_length = (torch.ones((n,)) * t).cuda().int()
                loss_ctc = criterion_ctc(log_prob, labels_ctc, input_length, labels_length)

                # attention loss   
                outputs = attention_head(cnn_features, labels_att)
                probs = outputs.permute(1, 2, 0)
                losses_att = criterion_att(probs, labels_att)
                losses_att = losses_att * labels_att_mask
                losses_att = losses_att.sum() / labels_att_mask.sum()

                loss = loss_ctc + losses_att

            else:
                # cross_entropy loss
                outputs_ctc, sqs = model(imgs)
                outputs_att = decoder(sqs, label_att)

                outputs = outputs_att.permute(1, 2, 0)
                losses = criterion(outputs, label_att)
                losses = losses * labels_att_mask
                loss = losses.sum() / labels_att_mask.sum()
 
                # attention loss   

            optimizer.zero_grad()            
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            if iter_cnt % config['print_freq'] == 0:
                print('epoch %d, iter %d, train loss %f' % (i, iter_cnt, loss.item()))

        print('epoch %d, time %f' % (i, (time.time() - start)))
        scheduler.step()

        print("validating...")
        
        if config['method'] == 'ctc':
            eval_ctc(model, valloader, idx2char_ctc)
        else:
            eval_attention(model, decoder, valloader, idx2char_att)

        if i % config['test_freq'] == 0:
            print("testing...")
            if config['method'] == 'ctc':
                line_acc, rec_score = eval_ctc(model, testloader, idx2char_ctc)
            else:
                line_acc, rec_score = eval_attention(model, decoder, testloader, idx2char_att)

        if i % config['save_freq'] == 0:
            save_file_name = f"epoch_{i}_acc_{line_acc:.3f}_rec_score_{rec_score:.3f}.pth"
            save_file = os.path.join(config['save_path'], save_file_name)
            torch.save(model.state_dict(), save_file)


if __name__ == '__main__':
    main()