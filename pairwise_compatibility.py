# *!/usr/bin/env python
# coding: utf-8

# In[5]:


from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import numpy as np
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image

import tensorflow
from tensorflow import keras
from utils import Config


class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()
        # self.X_train, self.X_test, self.y_train, self.y_test, self.classes = self.create_dataset()

    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms

    def create_dataset(self):
        meta_file = open(osp.join(self.root_dir, 'train.json'), 'r')
        meta_json = json.load(meta_file)
        ans = {}
        glst = []
        anslst = []

        for i in range(len(meta_json)):
            set_to_item = {}
            items = meta_json[i]["items"]
            set_id1 = meta_json[i]["set_id"]
            item_id = [sub["item_id"] for sub in items]
            # mapping set id to item-ids
            set_to_item[set_id1] = item_id
            ans.update(set_to_item)

        f = open("compatibility_train.txt", "r")
        f = [line.split(' ') for line in f.readlines()]
        g = f
        for i in range(len(g)):
            (g[i][len(g[i]) - 1]) = (g[i][len(g[i]) - 1]).rstrip('\n')
            if g[i][0] == '0':
                g[i].pop(1)

        for i in range(len(g)):
            lst = []
            for j in range(len(g[i])):
                if j == 0:
                    binary_label = int(g[i][j])
                    lst.append(binary_label)

                else:

                    elem = g[i][j]
                    if elem[-2] == '_':
                        set_id = elem[0:-2]
                        idx = int(elem[-1])

                    elif elem[-3] == '_':
                        set_id = elem[0:-3]
                        idx = int(elem[-2:])

                    itemid = ans[set_id][idx - 1]
                    lst.append(itemid + '.jpg')
            glst.append(lst)

        for i in range(len(glst)):
            for j in range(1, len(glst[i]) - 1):
                for k in range(j + 1, len(glst[i])):
                    flst = []
                    flst.append(glst[i][0])
                    flst.append(glst[i][j])
                    flst.append(glst[i][k])
                    anslst.append(flst)

        files = os.listdir(self.image_dir)
        Xf = []
        y = []
        for i in range(len(anslst)):
            Xf.append(anslst[i][1:3])
            y.append(anslst[i][0])
        X_train, X_test, y_train, y_test = train_test_split(Xf, y, test_size=0.2)
        return X_train, X_test, y_train, y_test, max(y) + 1


class DataGenerator(keras.utils.Sequence):

    def __init__(self, dataset, dataset_size, params):
        self.batch_size = params['batch_size']
        self.shuffle = params['shuffle']
        self.n_classes = params['n_classes']
        self.X, self.y, self.transform = dataset
        self.image_dir = osp.join(Config['root_path'], 'images')
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        X, y = np.stack(X), np.stack(y)
        a = np.moveaxis(X,1,3)
        return a, y

    def __data_generation(self, indexes):
        X = []
        y = []

        for idx in indexes:
            file_path = osp.join(self.image_dir, self.X[idx][0])
            file_path1 = osp.join(self.image_dir, self.X[idx][1])
            a = (self.transform(Image.open(file_path)))
            b = (self.transform(Image.open(file_path1)))
            x = np.stack([a,b])
            X.append(np.reshape(x, [6, 224, 224]))
            y.append(self.y[idx])
        return X, y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)






