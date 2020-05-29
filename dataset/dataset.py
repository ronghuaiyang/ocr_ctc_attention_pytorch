import torch.utils.data as data
from torchvision import transforms as T

import numpy as np
import torch
import json
import cv2
import os
import math
import numpy as np
from PIL import Image, ImageFilter
import random


class TextRecDataset(data.Dataset):

    def __init__(self, config, phase='train'):
        self.config = config
        self.phase = phase
        self.img_paths = []
        self.labels_str = [] 
        self.labels_length = []
        self.load_annotation_file()
        self.char_dict = self.get_char_dict(self.config['char_set'])
        self.config['char_dict'] = self.char_dict
        self.labels, self.labels_mask = self.label_process()
        # self.labels_length = [len(label) for label in self.labels]

        # self.annotations = []
        # for i in range(len(self.labels)):
        #     self.annotations.append({'img_path':self.img_paths[i],
        #                              'label':self.labels[i],
        #                              'label_length':self.labels_length[i]})

        self.idx = list(range(len(self.labels)))
        np.random.seed(10101)
        np.random.shuffle(self.idx)
        np.random.seed(None)

        self.phase = phase
        self.trainval_split = 0.95
        self.num_split = int(len(self.idx) * self.trainval_split)

        if self.phase == 'train':
            self.idx = self.idx[:self.num_split]
            self.transform = T.Compose([T.ToTensor(),
                                        # T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                        T.Normalize(mean=[0.5], std=[0.5])

                                        ])
        elif self.phase == 'val':
            self.idx = self.idx[self.num_split:]
            self.transform = T.Compose([T.ToTensor(),
                                        # T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                        T.Normalize(mean=[0.5], std=[0.5])
                                        ])
        else:
            self.transform = T.Compose([T.ToTensor(),
                                        # T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                        T.Normalize(mean=[0.5], std=[0.5])
                                        ])
                                                

        self.samples_num = self.__len__()
        self.char_num = 0
        for idx in self.idx:
            self.char_num += self.labels_length[idx]

        print(self.phase, 'total samples:', self.samples_num)
        print(self.phase, 'total chars:', self.char_num)

    def __getitem__(self, index):

        idx = self.idx[index]

        img_file = self.img_paths[idx]
        label = self.labels[idx]
        label_length = self.labels_length[idx]
        label_str = self.labels_str[idx]
        label_mask = self.labels_mask[idx]

        # assert len(label) == label_length, print("label len error", len(label), label_length)
        assert len(label_str) == label_length, print("label_str len error", len(label_str), label_length)

        # img_file = self.annotations[index]['img_path']
        # label = self.annotations[index]['label']
        # label_length = self.annotations[index]['label_length']

        img = self.get_image(img_file)
        img = self.transform(img.astype(np.uint8))

        return img, label, label_length, label_str

    def __len__(self):
        return len(self.idx)

    def load_annotation_file(self):
        if self.phase == 'train' or self.phase == 'val':
            label_file = self.config['train_label_file']
        else:
            label_file = self.config['test_label_file']

        with open(label_file) as f:
            lines = f.readlines()

        for line in lines:
            splits = line.split('|||')
            self.img_paths.append(splits[0])
            label_str = splits[1].strip()
            self.labels_str.append(label_str)
            self.labels_length.append(len(label_str))

    def get_char_dict(self, char_set):
        char_dict = {}
        for i, char in enumerate(char_set):
            char_dict[char] = i
        return char_dict

    def label_process(self):
        processed_label = []
        labels_mask = []
        for label in self.labels_str:
            label_idx = []
            label_mask = []
            if len(label) > self.config['max_string_len']:
                label = label[:self.config['max_string_len']]
            for char in label:
                if not char in self.char_dict:
                    char = ' '
                label_idx.append(self.char_dict[char])
                label_mask.append(1)
            for i in range(self.config['max_string_len'] - len(label_idx)):
                label_idx.append(self.char_dict[' '])
                label_mask.append(0)
            processed_label.append(np.array(label_idx))
            labels_mask.append(np.array(label_mask))
        return processed_label, labels_mask


    def get_image(self, img_file):
        """
        generate image: goal is img_h * img_w
        """
        if self.phase == 'train' or self.phase == 'val':
            data_path = self.config['train_data_path']
        else:
            data_path = self.config['test_data_path']

        img_path = os.path.join(data_path, img_file)
        img = cv2.imread(img_path, 0)
        if img is None:
            print('read %s error!' % (img_path))
            exit(0)
        h, w = img.shape[0:2]
        target_h, target_w = self.config['img_shape']

        ratio = float(target_h) / float(h)
        dst_w = int(w * ratio)
        dst_h = int(target_h)
        if dst_w > target_w: dst_w = target_w
        img = cv2.resize(img, dsize=(int(dst_w), int(dst_h)))

        dst_img = np.zeros((target_h, target_w))+255
        dst_img[:dst_h, :dst_w] = img
        # add channel axis [h,w,1]
        dst_img = np.expand_dims(dst_img, axis=2)

        return dst_img

    # def labels_to_text(self,labels):
    #     ret = []
    #     for label in labels:
    #         if label == len(opt.charset):
    #             ret.append('-')
    #             continue
    #         ret.append(opt.charset[label])
    #     return u"".join(ret)

    # def text_to_labels(self,text):
    #     labels = []
    #     for c in text:
    #         ll = opt.charset.find(c)
    #         if ll == -1:
    #             ll = opt.charset_len - 1
    #         labels.append(ll)
    #     if labels == []:
    #         labels = " "
    #     return labels

    # def _load_annotation(self, line):
    #     img_path = self.imgdir + line.split("|||")[0]
    #     img_label = line.split("|||")[1]
    #     img = self.get_image(img_path)
    #     if img is None:
    #         print('read %s error!' % (img_path))
    #         exit(0)
        
    #     label = np.ones([1, self.max_string_len]) * (self.charset_len-1)
    #     label[:, 0:len(img_label)] = self.text_to_labels(img_label)
    #     y_len = len(img_label)
    #     x_text = img_label
    #     y_len = np.expand_dims(np.array(y_len), 1)
    #     x_text = np.expand_dims(np.array(x_text), 1)

    #     return img, list(label), y_len, x_text



if __name__ == '__main__':
    stride = 8
    Dataset = TextRecDataset(
        root_dir='/data1/data/mexico_ocr/train_55/',
        annotation_file='/data1/lym/data/mexico_ocr/train_55.txt')
    train_loader = torch.utils.data.DataLoader(
        Dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        # pin_memory=True
    )

    print('total batches %d' % (len(train_loader)))
    for iter_id, batch in enumerate(train_loader):
        img = batch['the_input'].numpy().squeeze(0)
        label = batch['the_labels'].numpy()
        print(label)
        img = img.transpose((1, 2, 0))
        # print(img.shape)
        img += 1.0
        img *= 127.5
        img = img.astype(np.uint8)
        # img = img[:, :, [2, 1, 0]]
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
