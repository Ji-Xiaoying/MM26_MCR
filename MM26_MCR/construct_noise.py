import numpy as np


def make_noise(original_caption, noise_ratio, dataset_name):
    '''
    original_caption: The original caption file path of the dataset. Usually, it is a ".txt" file.
    noise_ratio: The noise proportion you need. (e.g., 0.2, 0.4, and 0.6)
    '''
    captions = []
    for line in open(original_caption, 'r', encoding='utf-8'):
        captions.append(line.strip())
    length = len(captions)
    idx = np.arange(length)
    np.random.shuffle(idx)
    # get noise's number
    noise_length = int(noise_ratio * length)
    shuffle_cap = np.array(captions)[idx[:noise_length]]
    np.random.shuffle(shuffle_cap)
    noise_cap = np.array(captions)
    noise_cap[idx[:noise_length]] = shuffle_cap
    with open('/data/user8/dcw/data/f30k/annotations/scan_split/{}_noise_train_caps.txt' .format(noise_ratio), mode='a', encoding='utf-8') as f:
            for cap in list(noise_cap):
                f.write(cap + '\n')
    return


def get_noise_aware(noisy_txt, ori_clean_txt, noisy_aware_txt):
    """
    """
    with open(noisy_txt, 'r', encoding='utf-8') as noisy_f, open(ori_clean_txt, 'r', encoding='utf-8') as clean_f:
        noisy_line = noisy_f.readlines()
        clean_line = clean_f.readlines()

    if len(noisy_line) != len(clean_line):
        raise ValueError("The two files must have the same number of lines.")

    flag = []
    for line1, line2 in zip(noisy_line, clean_line):
        if line1.strip() == line2.strip():
            flag.append(1)
        else:
            flag.append(0)

    with open(noisy_aware_txt, 'w', encoding='utf-8') as f_out:
        for flag_val, noisy_line in zip(flag, noisy_line):
            f_out.write(str(flag_val) + '\t' + noisy_line)

    print(sum(flag) / len(flag))
    return


import pandas as pd

if __name__ == '__main__':
    # 1. 构造特定比例的噪声数据集
    origin_file = "your_path/train_caps.txt"
    make_noise(original_caption=origin_file, noise_ratio=0.8, dataset_name='f30k')
    
    # 2. 对于一个给定的噪声数据集{x}_noise_train_caps.txt', 标记出其中的噪声样本对
    noisy_txt = 'your_path/0.8_noise_train_caps.txt'
    ori_clean_txt = 'your_path/train_caps.txt'
    noisy_aware_txt = 'your_path/0.8_noise_aware_train_caps.txt'
    
    get_noise_aware(noisy_txt, ori_clean_txt, noisy_aware_txt)
    
    # 3. 使用pd.read_csv() 读取noisy_aware_txt文本
    # df_noisy_aware = pd.read_csv(noisy_aware_txt, sep='\t', header=None, names=['flag', 'caption'])
    # print(df_noisy_aware.head())
