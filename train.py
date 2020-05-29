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

from dataset.dataset import TextRecDataset
from models.crnn import CRNN
# from models.resnet import resnet18



def main():
    with open('config.yaml') as f:
        config = yaml.load(f)
    print(config)
    print(torch.cuda.is_available())
    torch.backends.cudnn.benchmark = True

    train_dataset = TextRecDataset(config, phase='train')
    val_dataset = TextRecDataset(config, phase='val')
    test_dataset = TextRecDataset(config, phase='test')
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=cpu_count(),
                                  pin_memory=False)

    valloader = data.DataLoader(val_dataset,
                                batch_size=32,
                                shuffle=False,
                                num_workers=cpu_count(),
                                pin_memory=False)
    class_num = len(config['char_set'])
    print('class_num', class_num)
    model = CRNN(class_num=class_num)
    criterion = nn.CTCLoss(blank=len(config['char_set'])-1, reduction='mean')
    # criterion = nn.CrossEntropyLoss()

    model = model.cuda()
    summary(model, (1, 32, 400))

    # model = torch.nn.DataParallel(model)

    # optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=5e-4, weight_decay=5e-4)
    optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=0.001, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 500], gamma=0.1)

    print('train start, total batches %d' % len(trainloader))
    iter_cnt = 0
    char_set = config['char_set']
    for i in range(0, config['epochs']):
        start = time.time()
        model.train()
        for j, batch in enumerate(trainloader):
            iter_cnt += 1
            imgs = batch[0].cuda()
            labels = batch[1].cuda()
            labels_length = batch[2]

            optimizer.zero_grad()

            outputs = model(imgs)
            log_prob = outputs.log_softmax(dim=2)
            t,n,c = log_prob.size(0),log_prob.size(1),log_prob.size(2)
            input_length = (torch.ones((n,)) * t).cuda().int()
            loss = criterion(log_prob, labels, input_length, labels_length)

            loss.backward()
            optimizer.step()

            if iter_cnt % config['print_freq'] == 0:
                # trainloader.set_description('train loss %f' %(loss.item()))
                print('epoch %d, iter %d, train loss %f' % (i + 1, iter_cnt, loss.item()))
        print('epoch %d, time %f' % (i + 1, (time.time() - start)))
        scheduler.step()
        # if (i + 1) % config['save_freq'] == 0:
        #     save_model(config['save_path'], i + 1, model, optimizer=optimizer)

        model.eval()
        val_loss = 0
        t = time.time()
        char_err = 0
        line_err = 0
        for j, batch in enumerate(valloader):
            imgs = batch[0].cuda()
            labels_str = batch[3]
            labels_length = batch[2]

            # print(labels_str)

            with torch.no_grad():
                outputs = model(imgs)

            prob = outputs.softmax(dim=2).cpu()
            pred = prob.max(dim=2)[1]

            for k in range(pred.size(1)):
                pred_str = ""
                prev = " "
                for t in pred[:,k]:
                    if char_set[t] != prev:
                        pred_str += char_set[t]
                        prev = char_set[t]
                
                pred_str = pred_str.strip()
                pred_str = pred_str.replace('-', '')

                if pred_str != labels_str[k]:
                    line_err += 1
                    print('target:', labels_str[k])
                    print('pred  :', pred_str)


                # # print(pred_str)
                # line_err_flag = False
                # for t in range(labels_length[k]):
                    
                #     if t >= len(pred_str) or labels_str[k][t] != pred_str[t]:
                #         char_err += 1
                #         line_err_flag = True

                # if len(pred_str) > labels_length[k]:
                #     char_err = char_err + len(pred_str) - labels_length[k]
                #     line_err_flag = True

                # if line_err_flag:
                #     line_err += 1
                #     print('target:', labels_str[k])
                #     print('pred  :', pred_str)
                        
        print("line err: %f" % (float(line_err)/val_dataset.samples_num))


                




                    

        #     val_loss += loss.item()
        # val_loss = val_loss / len(valloader)
        # print('epoch %d, val_loss %f, time %f' % (i + 1, val_loss, time.time() - t))

        # save last image
        # t = time.time()
        # print('***********',outputs.shape)
        # y_pred = outputs.transpose(1,0)[0:1].cpu().numpy()
        # print(y_pred.shape)
        # y_true = batch['labels'][0:1]
        # result = decode(y_pred)
        # print('decode time:', time.time() - t)
        # print('******************************')
        # print([result,y_true])

if __name__ == '__main__':
    main()