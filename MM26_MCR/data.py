import os

import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import clip
from PIL import Image
import pandas as pd
import random


def id2name(id_list, dataset=None, subset=None, dataset_root=None):
    name_list = []
    if dataset == 'coco':
        for id in id_list:
            name_list.append('COCO_2014_' + str(id).rjust(12, '0') + '.jpg')
    elif dataset == 'f30k':
        image_name = []
        for line in open(os.path.join(dataset_root, f'annotations/scan_split/image_name.txt'), 'r', encoding='utf-8'):
            image_name.append(line.strip())
        for id in id_list:
            name_list.append(image_name[int(id)])
    elif dataset == 'cc':
         for id in id_list:
              name_list.append(str(id)+'.jpg')
    return name_list


class LoadDataset(Dataset):
    def __init__(self, args, dataset_root, preprocess, subset=None, logger=None):
        logger.info("========== Initial the %s set ==========", subset)
        self.args = args
        self.dataset = args.dataset  # coco/f30k/cc
        self.dataset_root = dataset_root
        self.image_root = os.path.join(dataset_root, 'images/')
        self.noise_ratio = args.noise_ratio
        self.noise_file = os.path.join(dataset_root, args.noise_file)

        self.preprocess = preprocess
        self.subset = subset
        self.num_anns = args.num_anns

        self.images_id = []
        self.captions = []
        self.label = []

        self.cap_length = 0

        # if subset == 'dev':
        #     self.length = args.dev_length

        if subset == 'dev' or subset == 'test':
            logger.info('Loading %s set.', self.dataset)
            # get images' id(repeat 5 times cause each image has 5 annotation caps)
            for line in open(os.path.join(dataset_root, 'annotations/scan_split/%s_ids.txt' % subset), 'r'):
                if self.dataset == 'cc':
                    self.images_id.append(str(line.strip()))
                else:
                    self.images_id.append(
                        int(line.strip()))  # "int(line.strip())" for MSCOCO and Flickr30K; "str(line.strip())" for CC120K
            # get annotation sentences
            for line in open(os.path.join(dataset_root, 'annotations/scan_split/%s_caps.txt' % subset), 'r',
                             encoding='utf-8'):
                self.captions.append(line.strip())
            logger.info('%d captions have been loaded.', len(self.captions))
            # get images' name list
            self.image_name = id2name(self.images_id, self.dataset, 'dev', self.dataset_root)
            # tokenize annotation captions
            self.cap_token = clip.clip.tokenize(self.captions)
            self.img_length = len(set(self.images_id))
            self.cap_length = len(self.captions)
            logger.info('%d images have been loaded.', self.img_length)
            logger.info("%s set initialization completed!", subset)

        elif subset == 'train':
            if self.noise_ratio != 0:
                logger.info('Loading {} {} set.'.format(self.noise_ratio, self.dataset))
                # get images' id(repeat 5 times cause each image has 5 annotation caps)
                for line in open(os.path.join(dataset_root, 'annotations/scan_split/train_ids.txt'), 'r'):
                    for i in range(self.num_anns):
                        if self.dataset == 'cc':
                            self.images_id.append(str(line.strip()))
                        else:
                            self.images_id.append(
                                int(line.strip()))  # "int(line.strip())" for MSCOCO and Flickr30K; "str(line.strip())" for CC120K
                # get noisy annotation sentences
                for line in open(os.path.join(dataset_root, 'annotations/scan_split/{}_noise_train_caps.txt'.format(
                        self.noise_ratio)), 'r', encoding='utf-8'):
                    self.captions.append(line.strip())
                logger.info('%d captions have been loaded.', len(self.captions))

                # get memory bank
                logger.info('loading frozen_memory_bank/{}_mbank_idx.npy'.format(self.noise_ratio))
                self.mbank_i = np.load(
                    os.path.join(dataset_root,
                                 'annotations/frozen_memory_bank/{}_mbank_img_idx.npy'.format(self.noise_ratio)),
                    allow_pickle=True).item()
                self.mbank_t = np.load(
                    os.path.join(dataset_root,
                                 'annotations/frozen_memory_bank/{}_mbank_txt_idx.npy'.format(self.noise_ratio)),
                    allow_pickle=True).item()

                # get images' name list
                self.image_name = id2name(self.images_id, self.dataset, 'train', self.dataset_root)

                # tokenize annotation captions
                self.cap_token = clip.clip.tokenize(self.captions)
                self.img_length = len(set(self.images_id))
                self.cap_length = len(self.captions)
                logger.info('%d images have been loaded.', self.img_length)
                logger.info("%s set initialization completed!", subset)
            else:
                logger.info('Loading %s set.', self.dataset)
                # get images' id(repeat 5 times cause each image has 5 annotation caps)
                for line in open(os.path.join(dataset_root, 'annotations/scan_split/%s_ids.txt' % subset), 'r'):
                    for i in range(self.num_anns):
                        if self.dataset == 'cc':
                            self.images_id.append(str(line.strip()))
                        else:
                            self.images_id.append(
                                int(line.strip()))  # "int(line.strip())" for MSCOCO and Flickr30K; "str(line.strip())" for CC120K
                logger.info('%d images have been loaded.', len(set(self.images_id)))
                # get annotation sentences
                for line in open(os.path.join(dataset_root, 'annotations/scan_split/%s_caps.txt' % subset), 'r',
                                 encoding='utf-8'):
                    self.captions.append(line.strip())
                logger.info('%d captions have been loaded.', len(self.captions))
                # get memory bank
                self.mbank_i = np.load(
                    os.path.join(dataset_root,
                                 'annotations/frozen_memory_bank/0_mbank_img_idx.npy'.format(self.noise_ratio)),
                    allow_pickle=True).item()
                self.mbank_t = np.load(
                    os.path.join(dataset_root,
                                 'annotations/frozen_memory_bank/0_mbank_txt_idx.npy'.format(self.noise_ratio)),
                    allow_pickle=True).item()
                # get images' name list 
                self.image_name = id2name(self.images_id, self.dataset, 'train', self.dataset_root)
                # tokenize annotation captions
                self.cap_token = clip.clip.tokenize(self.captions)
                self.img_length = len(set(self.images_id))
                self.cap_length = len(self.captions)
                logger.info('%d images have been loaded.', self.img_length)
                logger.info("%s set initialization completed!", subset)
        elif subset == 'train_base':
            if self.noise_ratio != 0:
                logger.info('Loading {} {} set.'.format(self.noise_ratio, self.dataset))
                # 加载图像数据路径
                for line in open(os.path.join(dataset_root, 'annotations/scan_split/train_ids.txt'), 'r'):
                    for i in range(self.num_anns):  # repeat 5 times cause each image has 5 annotation caps
                        if self.dataset == 'cc':
                            self.images_id.append(str(line.strip()))
                        else:
                            self.images_id.append(
                                int(line.strip()))  # "int(line.strip())" for MSCOCO and Flickr30K; "str(line.strip())" for CC120K
                # get images' name list
                self.image_name = id2name(self.images_id, self.dataset, 'train', self.dataset_root)
                self.img_length = len(set(self.images_id))
                logger.info('%d images have been loaded.', self.img_length)

                # 加载文本数据
                # for line in open(self.noise_file, 'r', encoding='utf-8'):
                #     self.captions.append(line.strip())
                captions_pd = pd.read_csv(self.noise_file, sep='\t', header=None, names=['flag', 'caption'])
                self.captions = captions_pd['caption'].tolist()
                self.noise_flag = captions_pd['flag'].tolist()
                logger.info('%d captions have been loaded.', len(self.captions))

                # tokenize annotation captions
                self.cap_token = clip.clip.tokenize(self.captions)
                self.cap_length = len(self.captions)
                logger.info("%s set initialization completed!", subset)
            else:
                logger.info('Loading %s set.', self.dataset)
                # get images' id(repeat 5 times cause each image has 5 annotation caps)
                for line in open(os.path.join(dataset_root, 'annotations/scan_split/train_ids.txt'), 'r'):
                    for i in range(self.num_anns):
                        if self.dataset == 'cc':
                            self.images_id.append(str(line.strip()))
                        else:
                            self.images_id.append(
                                int(line.strip()))  # "int(line.strip())" for MSCOCO and Flickr30K; "str(line.strip())" for CC120K
                        # self.images_id.append(int(line.strip()))
                logger.info('%d images have been loaded.', len(set(self.images_id)))
                # get images' name list
                self.image_name = id2name(self.images_id, self.dataset, 'train', self.dataset_root)

                # get annotation sentences
                for line in open(os.path.join(dataset_root, 'annotations/scan_split/train_caps.txt'), 'r',
                                 encoding='utf-8'):
                    self.captions.append(line.strip())
                logger.info('%d captions have been loaded.', len(self.captions))

                # tokenize annotation captions
                self.cap_token = clip.clip.tokenize(self.captions)
                self.img_length = len(set(self.images_id))
                self.cap_length = len(self.captions)
                self.noise_flag = [1] * self.cap_length

                logger.info("%s set initialization completed!", subset)

    def __len__(self):
        return self.cap_length

    def __getitem__(self, idx):
        if self.subset == 'train':
            image = self.preprocess(
                Image.open(os.path.join(self.image_root, self.image_name[idx])))  # Image from PIL module
            text = self.cap_token[idx]
            img_id = self.images_id[idx]     # 当前样本的标识id
            # mbank_i_idx = int(self.mbank_i[str(int(img_id))])  # f30k training
            mbank_i_idx = int(self.mbank_i[str(int(idx))])           # coco training
            mbank_img = self.preprocess(Image.open(os.path.join(self.image_root, self.image_name[mbank_i_idx])))
            mbank_i2txt = self.cap_token[mbank_i_idx]

            mbank_t_idx = int(self.mbank_t[str(int(idx))])           # coco training
            # mbank_t_idx = int(self.mbank_t[str(int(img_id))])      # f30k training
            mbank_txt = self.cap_token[mbank_t_idx]
            mbank_t2img = self.preprocess(Image.open(os.path.join(self.image_root, self.image_name[mbank_t_idx])))
            return image, text, idx, img_id, mbank_img, mbank_i2txt, mbank_txt, mbank_t2img
        elif self.subset == 'train_base':
            image = self.preprocess(
                Image.open(os.path.join(self.image_root, self.image_name[idx])))  # Image from PIL module
            text = self.cap_token[idx]
            img_id = self.images_id[idx]
            is_clean = self.noise_flag[idx]
            return image, text, is_clean, idx
        else:
            image = self.preprocess(
                Image.open(os.path.join(self.image_root, self.image_name[idx])))  # Image from PIL module
            text = self.cap_token[idx]
            img_id = self.images_id[idx]
            return image, text, img_id


def load_data(args, data, subset):
    if subset == 'train':
        dataloader = DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=(subset == 'train'),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=(subset == 'train'),
        )
    elif subset == 'train_base':
        dataloader = DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=(subset == 'train_base'),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=(subset == 'train_base'),
        )
    else:
        dataloader = DataLoader(
            data,
            batch_size=args.eval_batch_size,
            shuffle=(subset == 'train'),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=(subset == 'train'),
        )
    return dataloader, len(data)


def prepare_dataloader(args, dataset_root, preprocess, logger, subset, subset_size=None):
    dataloders = {}
    data = LoadDataset(args, dataset_root, preprocess, subset=subset, logger=logger)
    if subset_size is not None:
        # 截取数据集的一部分
        # 1. 按原dataset中的数据顺序取前 subset_size 个样本
        data = torch.utils.data.Subset(data, range(subset_size))
        # 2. 从原dataset中随机抽取 subset_size 个样本
        # indices = list(range(len(data)))
        # random_indices = random.sample(indices, subset_size)
        # data_subset = torch.utils.data.Subset(data, random_indices)
    dataloders[subset] = load_data(args, data, subset=subset)
    return dataloders
