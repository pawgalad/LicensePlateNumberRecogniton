import cv2
from os.path import join

import numpy as np
import os

from keras import backend as K

chars_set = '0123456789ABCDEFGHIJKLMNOPRSTUWVXYZ '

def convert_to_lab(text):

    return list(map(lambda x: chars_set.index(x), text))

class LicensePlateImages:

    def __init__(self,
                 dirpath,
                 img_w, img_h,
                 batch_size,
                 downsample_factor,
                 max_text_len=7):

        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor

        img_dirpath = dirpath
        self.img_dirpath = img_dirpath

        self.img_dir = os.listdir(img_dirpath)

        self.n = len(self.img_dir)
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []

    def size_of_chars_set(self):
        return len(chars_set) + 1

    def load_data(self):

        for i, img_file in enumerate(self.img_dir):
            path = join(self.img_dirpath, img_file)
            name, ext = os.path.splitext(img_file)
            name2 = name.split("_")
            namee = name2[0]
            img = cv2.imread(path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img /= 255
            self.max_text_len = 7
            self.imgs[i, :, :] = img
            self.texts.append(name2[0])

    def next_image(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            # random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def images_next_batch(self):
        while True:

            if K.image_data_format() == 'channels_first':
                image_tens = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                image_tens = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            labels_tens = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []

            for i in range(self.batch_size):
                img, text = self.next_image()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                image_tens[i] = img

                labels_tens[i] = convert_to_lab(text)
                source_str.append(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': image_tens,
                'labels': labels_tens,
                'in_length': input_length,
                'lab_length': label_length,

            }

            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)
