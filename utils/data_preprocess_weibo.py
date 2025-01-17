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
import pandas as pd
from PIL import Image
import math
from types import *
import jieba
import os.path


def load_stopwords(filepath='../Data/weibo/stop_words.txt'):
    stopwords_dict = {}
    for line in open(filepath, 'r', encoding="utf-8").readlines():
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
    image_folders = ['../Data/weibo/nonrumor_images/', '../Data/weibo/rumor_images/']
    for folder in image_folders:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for i, filename in enumerate(os.listdir(folder)):  # assuming gif
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
    def load_posts(flag):
        stopwords = load_stopwords()
        base_path = "../Data/weibo/tweets/"
        file_paths = [base_path + "test_nonrumor.txt", base_path + "test_rumor.txt",
                      base_path + "train_nonrumor.txt", base_path + "train_rumor.txt"]
        if flag == "train":
            id_dict = pickle.load(open("../Data/weibo/train_id.pickle", 'rb'))
        elif flag == "validate":
            id_dict = pickle.load(open("../Data/weibo/validate_id.pickle", 'rb'))
        elif flag == "test":
            id_dict = pickle.load(open("../Data/weibo/test_id.pickle", 'rb'))

        post_content = []
        labels = []
        image_ids = []
        twitter_ids = []
        all_data = []
        column_names = ['post_id', 'image_id', 'original_post', 'post_text', 'label', 'event_label']
        event_mapping = {}
        data_map = []

        for idx, file_path in enumerate(file_paths):
            with open(file_path, 'r', encoding="utf-8") as f:
                label = 0 if (idx + 1) % 2 == 1 else 1  # real is 0, fake is 1
                twitter_id = 0
                line_data = []
                for i, line in enumerate(f.readlines()):
                    if (i + 1) % 3 == 1:
                        line_data = []
                        twitter_id = line.split('|')[0]
                        line_data.append(twitter_id)
                    elif (i + 1) % 3 == 2:
                        line_data.append(line.lower())
                    elif (i + 1) % 3 == 0:
                        clean_line = clean_str(line)
                        seg_list = jieba.cut_for_search(clean_line)
                        filtered_seg_list = [word for word in seg_list if word not in stopwords]
                        clean_seg = " ".join(filtered_seg_list)

                        if len(clean_seg) > 10 and line_data[0] in id_dict:
                            post_content.append(line)
                            line_data.append(line)
                            line_data.append(clean_seg)
                            line_data.append(label)
                            event_id = int(id_dict[line_data[0]])
                            if event_id not in event_mapping:
                                event_mapping[event_id] = len(event_mapping)
                                event_id = event_mapping[event_id]
                            else:
                                event_id = event_mapping[event_id]

                            line_data.append(event_id)

                            all_data.append(line_data)

        df = pd.DataFrame(np.array(all_data), columns=column_names)
        save_to_txt(data_map)

        return post_content, df

    post_content, post_data = load_posts(flag)
    print(f"Original post content length: {len(post_content)}")
    print(f"Original data frame shape: {post_data.shape}")

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

        for i, post_id in enumerate(post_data['post_id']):
            image_id = ""
            for image_id in post_data.iloc[i]['image_id'].split('|'):
                image_id = image_id.split("/")[-1].split(".")[0]
                if image_id in image_data:
                    break

            if text_only or image_id in image_data:
                if not text_only:
                    image_name = image_id
                    image_ids.append(image_name)
                    ordered_images.append(image_data[image_name])
                ordered_texts.append(post_data.iloc[i]['original_post'])
                ordered_original_posts.append(post_data.iloc[i]['post_text'])
                ordered_events.append(post_data.iloc[i]['event_label'])
                post_ids.append(post_id)

                labels.append(post_data.iloc[i]['label'])

        labels = np.array(labels, dtype=int)
        ordered_events = np.array(ordered_events, dtype=int)

        print(f"Label number is: {len(labels)}")
        print(f"Rumor count: {sum(labels)}")
        print(f"Non-rumor count: {len(labels) - sum(labels)}")

        if flag == "test":
            y = np.zeros(len(ordered_original_posts))
        else:
            y = []

        paired_data = {
            "post_text": np.array(ordered_original_posts),
            "original_post": np.array(ordered_texts),
            "image": ordered_images,
            "social_feature": [],
            "label": np.array(labels),
            "event_label": ordered_events,
            "post_id": np.array(post_ids),
            "image_id": image_ids
        }

        print(f"Data size: {len(paired_data['post_text'])}")

        return paired_data

    paired_data = prepare_paired_data(text_only)

    print(f"Paired posts length: {len(paired_data['post_text'])}")
    print(f"Paired data has {len(paired_data)} dimensions")

    return paired_data


def load_vocabulary(train, validate, test):
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
    word_idx_map = dict()
    W = np.zeros(shape=(len(word_vecs) + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_binary_vectors(fname, vocab):
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
    print('Finished!')
