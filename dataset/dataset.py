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


def get_char_dict_attention(char_set):
    char2idx = {'eos':0}
    idx2char = {0:'eos'}
    for i, char in enumerate(char_set):
        char2idx[char] = i+1
        idx2char[i+1] = char
    # char_dict['eos'] = i+2
    char2idx['sos'] = i+2
    idx2char[i+2] = 'sos'
    return char2idx, idx2char

def get_char_dict_ctc(char_set):
    char2idx = {'-':0}
    idx2char = {0:'-'}
    for i, char in enumerate(char_set):
        char2idx[char] = i+1
        idx2char[i+1] = char
    return char2idx, idx2char  

class GaussianBlur(object):
    def __init__(self, radius, p=0.5):
        if radius < 1:
            raise ValueError("kernal_size must be positive integer.")
        self.radius = radius
        self.p = p

    @staticmethod
    def get_params(radius):
        return random.randint(1, radius)

    def __call__(self, img):
        if random.random() < self.p:
            return img
        else:
            radius = self.get_params(self.radius)
            im_filter = ImageFilter.GaussianBlur(radius=radius)
            img = img.filter(im_filter)
            return img  


class TextRecDataset(data.Dataset):

    def __init__(self, config, phase='train'):
        self.config = config
        self.phase = phase
        self.img_paths = []
        self.labels_str = [] 
        self.labels_length = []
        self.load_annotation_file()
        if config['method'] == 'ctc':
            self.labels, self.labels_mask = self.label_process_ctc()
        else:
            self.labels, self.labels_mask = self.label_process_attention()

        self.idx = list(range(len(self.labels)))
        np.random.seed(10101)
        np.random.shuffle(self.idx)
        np.random.seed(None)

        self.phase = phase
        self.trainval_split = 0.95
        self.num_split = int(len(self.idx) * self.trainval_split)

        if self.phase == 'train':
            self.idx = self.idx[:self.num_split]
            self.transform = T.Compose([T.ColorJitter(0.2,0.2,0.2,.02),
                                        # GaussianBlur(5),
                                        T.ToTensor(),                                       
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

        assert len(label_str) == label_length

        img = self.get_image(img_file)

        img = self.transform(Image.fromarray(np.uint8(img)).convert('L'))

        return img, label, label_length, label_str, label_mask

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

    def label_process_ctc(self):
        max_string_len = self.config['max_string_len']
        char2idx = self.config['char2idx']
        processed_label = []
        labels_mask = []
        for label in self.labels_str:

            label_idx = np.zeros(max_string_len) + char2idx[' ']
            label_mask = np.zeros(max_string_len)

            if len(label) > max_string_len:
                label = label[:max_string_len]

            for i, char in enumerate(label):
                if not char in char2idx:
                    char = ' '
                label_idx[i] = char2idx[char]
                label_mask[i] = 1

            processed_label.append(label_idx)
            labels_mask.append(label_mask)

        return processed_label, labels_mask

    # use attention
    def label_process_attention(self):
        max_string_len = self.config['max_string_len']
        char2idx = self.config['char2idx']
        processed_label = []
        labels_mask = []
        for label in self.labels_str:

            label_idx = np.zeros(max_string_len) + char2idx['eos']
            label_mask = np.zeros(max_string_len)

            label_len = min(max_string_len-1, len(label))

            for i in range(0, label_len):
                char = label[i]
                if not char in char2idx:
                    char = ' '
                label_idx[i] = char2idx[char]
                label_mask[i] = 1

            label_idx[i+1] = char2idx['eos']
            label_mask[i+1] = 1

            processed_label.append(label_idx)
            labels_mask.append(label_mask)
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
        img = cv2.imread(img_path)
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

        dst_img = np.zeros((target_h, target_w, 3))+255
        dst_img[:dst_h, :dst_w,:] = img
        # add channel axis [h,w,1]
        # dst_img = np.expand_dims(dst_img, axis=2)

        return dst_img


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
