import numpy as np
import argparse
import logging
import os, sys
from time import strftime, localtime
import copy
import pickle as pickle
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from transformers import BertModel, BertTokenizer, AutoTokenizer
from mymodel import MultimodalFusion
import warnings

import utils.data_preprocess_gossipcop as data_preprocess
warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
bert_uncased_model_path = "../uncased"
bert_chinese_model_path = "../chinese"

class RumorData(Dataset):
    def __init__(self, data):
        self.text = torch.from_numpy(np.array(data['post_text']))
        self.image = list(data['image'])
        self.mask = torch.from_numpy(np.array(data['mask']))
        self.label = torch.from_numpy(np.array(data['label']))
        self.event_label = torch.from_numpy(np.array(data['event_label']))
        print('TEXT: %d, Image: %d, label: %d, Event: %d'
              % (len(self.text), len(self.image), len(self.label), len(self.event_label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.text[idx], self.image[idx], self.mask[idx]), self.label[idx], self.event_label[idx]

def tensor_to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def tensor_to_numpy(x):
    return x.data.cpu().numpy()

def sample_selection(data, selected_indices):
    temp = []
    for i in range(len(data)):
        print("length is " + str(len(data[i])))
        print(i)
        ele = list(data[i])
        temp.append([ele[i] for i in selected_indices])
    return temp

def run_inference(args):
    print('Loading dataset...')
    test = load_data(args)
    test_id = test['post_id']

    test_dataset = RumorData(test)

    # Data Loader (Input Pipeline)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.number_workers, shuffle=False)

    model = MultimodalFusion(args)

    if torch.cuda.is_available():
        print("CUDA")
        model.cuda()

    # Test the Model
    model.eval()
    test_score = []
    test_pred = []
    test_true = []
    for i, (test_data, test_labels, event_labels) in enumerate(test_loader):
        test_text, test_image, test_mask, test_labels = tensor_to_var(
            test_data[0]), tensor_to_var(test_data[1]), tensor_to_var(test_data[2]), tensor_to_var(test_labels)
        test_outputs, image_outputs, text_outputs, _, _, _ = model(test_text, test_image, test_mask)
        _, test_argmax = torch.max(test_outputs, 1)
        if i == 0:
            test_score = tensor_to_numpy(test_outputs.squeeze())
            test_pred = tensor_to_numpy(test_argmax.squeeze())
            test_true = tensor_to_numpy(test_labels.squeeze())
        else:
            test_score = np.concatenate((test_score, tensor_to_numpy(test_outputs.squeeze())), axis=0)
            test_pred = np.concatenate((test_pred, tensor_to_numpy(test_argmax.squeeze())), axis=0)
            test_true = np.concatenate((test_true, tensor_to_numpy(test_labels.squeeze())), axis=0)

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')

    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

    logger.info("Classification Accuracy: %.4f, AUC-ROC: %.4f" % (test_accuracy, test_aucroc))
    logger.info("Classification Report:\n%s\n" % (metrics.classification_report(test_true, test_pred, digits=4)))
    logger.info("Confusion Matrix:\n%s\n" % (test_confusion_matrix))

def extract_top_n_posts(output, label, test_id, top_n=500):
    filter_output = []
    filter_id = []
    for i, l in enumerate(label):
        if np.argmax(output[i]) == l and int(l) == 1:
            filter_output.append(output[i][1])
            filter_id.append(test_id[i])

    filter_output = np.array(filter_output)

    top_n_indices = filter_output.argsort()[-top_n:][::-1]

    top_n_id = np.array(filter_id)[top_n_indices]
    top_n_id_dict = {}
    for i in top_n_id:
        top_n_id_dict[i] = True

    pickle.dump(top_n_id_dict, open("../Data/weibo/top_n_id.pickle", "wb"))

    return top_n_id


def process_sentences(flag, max_length, dataset):
    if dataset == 'weibo':
        tokenizer = AutoTokenizer.from_pretrained(bert_chinese_model_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_uncased_model_path)
    tokenized_texts = []
    original_texts = flag['original_post']
    for sentence in original_texts:
        tokenized_text = tokenizer.encode(sentence)[:max_length]
        tokenized_texts.append(tokenized_text)
    flag['post_text'] = tokenized_texts


def gather_text(train, validate, test):
    all_text = list(train['post_text']) + list(validate['post_text']) + list(test['post_text'])
    return all_text


def process_data_for_training(flag, args):
    text = []
    mask = []
    for sentence in flag['post_text']:
        sen_embedding = []
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word)

        while len(sen_embedding) < args.sequence_len:
            sen_embedding.append(0)

        text.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
    flag['post_text'] = text
    flag['mask'] = mask


def load_data(args):
    test = data_preprocess.get_data(args.text_only)
    process_sentences(test, max_length=args.max_length, dataset=args.dataset)
    all_text = gather_text(test)
    max_len = len(max(all_text, key=len))
    args.sequence_len = max_len
    process_data_for_training(test, args)
    return test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='weibo', help='')
    parser.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--filter_num', type=int, default=5, help='')
    parser.add_argument('--lambd', type=int, default=1, help='')
    parser.add_argument('--text_only', type=bool, default=False, help='')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--number_workers', type=int, default=4, help='')
    parser.add_argument('--max_length', type=int, default=200, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--early_stop_epoch', type=int, default=10, help='')
    parser.add_argument('--temp', type=float, default=0.2, help='')
    parser.add_argument('--gamma', type=float, default=0.0,help='')
    parser.add_argument('--balanced', type=float, default=0.01, help='')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help='')
    parser.add_argument("--weight_decay", default=0.0, type=float, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--bert_lr', type=float, default=0.00003, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')
    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')

    args = parser.parse_args()
    args.output_file = '../Data/' + args.dataset + '/RESULT_text_image/'
    args.id = '{}-{}.log'.format(args.dataset, strftime("%y%m%d-%H%M", localtime()))
    log_file = '../log/' + args.id
    logger.addHandler(logging.FileHandler(log_file))

    # write arguments into the logger
    logger.info('> testing arguments:')
    for arg in vars(args):
        logger.info('>>> {0}: {1}'.format(arg, getattr(args, arg)))

    run_inference(args)
