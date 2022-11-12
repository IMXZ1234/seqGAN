import torch


CUDA = True
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 20
START_LETTER = 0
BATCH_SIZE = 32
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 50
POS_NEG_SAMPLES = 10000

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

oracle_samples_path = './oracle_samples.trc'
oracle_state_dict_path = './oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_gen_path = './gen_MLEtrain_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_dis_path = './dis_pretrain_EMBDIM_64_HIDDENDIM64_VOCAB5000_MAXSEQLEN20.trc'


oracle_samples = torch.load(oracle_samples_path).type(torch.LongTensor)
print(oracle_samples.shape)
print(oracle_samples[0])
# torch.Size([10000, 20])
# tensor([  87, 4410, 3560, 1699, 3485, 1407, 4982, 3391, 1144, 2960, 3784, 2351,
#         3609,   92, 3391, 2187,  168, 4767, 4973,  619])
