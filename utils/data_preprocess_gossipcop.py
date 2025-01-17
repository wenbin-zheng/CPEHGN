# encoding=utf-8
try:
    import cPickle as pickle
except ImportError:
    import pickle
import random
from random import *
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
from collections import defaultdict
import sys, re
import csv
import pandas as pd
from PIL import Image
import math
from types import *
import jieba
import os.path
from googletrans import Translator
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


def load_stopwords(filepath='../Data/weibo/stop_words.txt'):
    stopwords_dict = {}
    for line in open(filepath, 'r').readlines():
        line = line.strip()
        stopwords_dict[line] = 1
    return stopwords_dict


def clean_str(input_str):
    """
    Tokenization/string cleaning for the dataset
    """
    input_str = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", input_str)
    return input_str.strip().lower()


def load_images():
    images_dict = {}
    image_folders = ['../Data/AAAI_dataset/Images/gossip_train/', '../Data/AAAI_dataset/Images/gossip_test/']
    for folder in image_folders:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for i, filename in enumerate(os.listdir(folder)):
            try:
                img = Image.open(folder + filename).convert('RGB')
                img = data_transforms(img)
                images_dict[filename.split('/')[-1].split(".")[0].lower()] = img
            except:
                print(filename)
    print("Total images loaded: " + str(len(images_dict)))
    return images_dict

def save_to_txt(data):
    with open("../Data/weibo/top_n_data.txt", 'wb') as f:
        for line in data:
            for l in line:
                f.write(l + "\n")
            f.write("\n")
            f.write("\n")


text_dict = {}


def save_data(flag, image_data, text_only):
    def load_posts(flag, images):
        base_path = "../Data/AAAI_dataset/"
        all_data = []
        if flag == "train":
            file_name = "../Data/AAAI_dataset/gossip_train.csv"
        elif flag == "test" or "validate":
            file_name = "../Data/AAAI_dataset/gossip_test.csv"  # TODO: change the validation file path here!
        else:
            print('Error')
            return

        with open(file_name, encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                all_data.append(row)

        post_content = []
        data = []
        column_names = ['post_id', 'image_id', 'original_post', 'post_text', 'label', 'event_label']
        event_mapping = {}
        for i, line in enumerate(all_data):
            if i == 0: continue  # skip the header
            line_data = []
            image_id = line[2].split('.')[0].lower()

            post_id = line[0]
            text = line[1]

            label = 0 if line[-1] == '1' else 1

            event_name = re.sub(u'fake', '', image_id)
            event_name = re.sub(u'real', '', event_name)
            event_name = re.sub(u'[0-9_]', '', event_name)
            if event_name not in event_mapping:
                event_mapping[event_name] = len(event_mapping)
                event = event_mapping[event_name]
            else:
                event = event_mapping[event_name]
            line_data.append(post_id)
            line_data.append(image_id)
            post_content.append(text)
            line_data.append(text)
            line_data.append([])
            line_data.append(label)
            line_data.append(event)

            data.append(line_data)

        data_df = pd.DataFrame(data, columns=column_names)

        return post_content, data_df, len(event_mapping)

    post_content, post, event_num = load_posts(flag, image_data)
    print(f"Original {flag} post content length: {len(post_content)}")
    print(f"Original {flag} data frame shape: {post.shape}")

    def find_most_frequent(db):
        max_count = max(len(v) for v in db.values())
        return [k for k, v in db.items() if len(v) == max_count]

    def select_data(train, selected_indices):
        temp = []
        for i in range(len(train)):
            ele = list(train[i])
            temp.append([ele[i] for i in selected_indices])
        return temp

    def prepare_paired_data(text_only=False):
        ordered_images = []
        ordered_texts = []
        ordered_original_posts = []
        ordered_events = []
        labels = []
        post_ids = []
        image_ids = []

        for i, post_id in enumerate(post['post_id']):
            image_id = post.iloc[i]['image_id']

            if text_only or image_id in image_data:
                if not text_only:
                    image_name = image_id
                    image_ids.append(image_name)
                    ordered_images.append(image_data[image_name])
                ordered_texts.append(post.iloc[i]['original_post'])
                ordered_original_posts.append(post.iloc[i]['post_text'])
                ordered_events.append(post.iloc[i]['event_label'])
                post_ids.append(post_id)

                labels.append(post.iloc[i]['label'])

        labels = np.array(labels, dtype=int)
        ordered_events = np.array(ordered_events, dtype=int)

        print(f"Total labels: {len(labels)}")
        print(f"Rumors count: {sum(labels)}")
        print(f"Non-rumors count: {len(labels) - sum(labels)}")

        data = {"post_text": np.array(ordered_original_posts),
                "original_post": np.array(ordered_texts),
                "image": ordered_images, "social_feature": [],
                "label": np.array(labels),
                "event_label": ordered_events, "post_id": np.array(post_ids),
                "image_id": image_ids}

        print(f"Total data size: {len(data['post_text'])}")

        return data

    paired_data = prepare_paired_data(text_only)

    print(f"Paired post content length: {len(paired_data['post_text'])}")
    print(f"Paired data has {len(paired_data)} dimensions")
    return paired_data


def build_vocabulary(train, validate, test):
    vocab = defaultdict(float)
    all_text = list(train['post_text']) + list(validate['post_text']) + list(test['post_text'])
    for sentence in all_text:
        for word in sentence:
            vocab[word] += 1
    return vocab, all_text


def build_data_cv(data_folder, cv=10, clean_string=True):
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 1,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 0,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)
    return revs, vocab


def get_word_matrix(word_vecs, k=32):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    word_idx_map = dict()
    W = np.zeros(shape=(len(word_vecs) + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=32):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def get_data(text_only):
    if text_only:
        print("Text only")
        image_list = []
    else:
        print("Text and image")
        image_list = load_images()

    train_data = save_data("train", image_list, text_only)
    validate_data = save_data("validate", image_list, text_only)
    test_data = save_data("test", image_list, text_only)

    return train_data, validate_data, test_data


if __name__ == '__main__':
    train, validate, test = get_data(text_only=False)
    print('Finish!')
