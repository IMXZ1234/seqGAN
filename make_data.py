import pickle

import numpy as np
import random
import torch

MAX_SEQ_LEN = 20
POS_NEG_SAMPLES = 10000


new_samples_path = r'./new_samples'


def sample(num, device, length=3, interval_range=(2, 6)):
    all_samples = []
    for sample_idx in range(num):
        pos = 0
        seq = []
        while pos < MAX_SEQ_LEN:
            itv = min(MAX_SEQ_LEN - pos, random.randint(*interval_range))
            pos += itv
            seq.extend([0] * itv)
            symbol_num = min(length, MAX_SEQ_LEN - pos)
            pos += symbol_num
            seq.extend([1] * symbol_num)
        all_samples.append(seq)
    all_samples = torch.tensor(all_samples, dtype=torch.long, device=device)
    return all_samples


def make_data(length=3, interval_range=(2, 6)):
    all_samples = sample(POS_NEG_SAMPLES, length, interval_range)
    print(all_samples[0])
    print(all_samples.shape)
    torch.save(all_samples, new_samples_path)


if __name__ == '__main__':
    all_samples = torch.load(new_samples_path)
    print(all_samples[0])
    print(all_samples.shape)
    # make_data()
