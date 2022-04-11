import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision import models
from torchvision.models import resnet34
from tensorboardX import SummaryWriter
import time
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
from matplotlib import cm
import pandas as pd
import os
import matplotlib.pyplot as plt
from urllib import  parse
import numpy as np
from torch.nn.utils.rnn import  pad_sequence

import warnings



# Define the transform
train_transform = transforms.Compose([
        transforms.Resize((224,224)),             # takes PIL image as input and outputs PIL image
        transforms.ToTensor(),              # takes PIL image as input and outputs torch.tensor
        transforms.Normalize(mean=[0.4280, 0.4106, 0.3589],  # takes tensor and outputs tensor
                             std=[0.2737, 0.2631, 0.2601]),  # see next step for mean and std
    ])

valid_transform = transforms.Compose([
        transforms.Resize((224,224)),
#         transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4280, 0.4106, 0.3589],
                             std=[0.2737, 0.2631, 0.2601]),
    ])

test_transform = transforms.Compose([
        transforms.Resize((224,224)),
#         transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4280, 0.4106, 0.3589],
                             std=[0.2737, 0.2631, 0.2601]),
    ])


def preprocessing(query_path):
    """
    query_path : path of the annoation csv file
    return the DataFrame type data

    """
    query_data = pd.read_csv(query_path)

    query_expand = query_data
    queries = query_data.iloc[:, 0]
    queries = queries.drop_duplicates()
    for index, query in enumerate(queries):

        q_data = query_data.loc[query_data["query"] == query]
        start_index = q_data.index.tolist()[0]
        last_index = q_data.index.tolist()[-1]

        len_q = len(q_data)

        if len_q < 199:
            diff_len = 199 - len_q
            num_iter = int(199 / len_q)
            if diff_len > len_q:
                added_length = len_q
            else:
                added_length = diff_len
            count = 0
            for j in range(num_iter):
                for i in range(added_length):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        query_expand = query_expand.append(query_data.iloc[i + start_index, :], ignore_index=True)
                    count += 1
                    if len_q + count == 199:
                        break

    return query_expand


class dataset(Dataset):

    def __init__(self, csv_file, root_dir, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the frames.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.query_frame_train = preprocessing(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.query_frame_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.query_frame_train.iloc[idx,2].split("/")[-2] + self.query_frame_train.iloc[idx,2].split("/")[-1]
        img_path = self.img_dir + '/' + img_name

        image = io.imread(img_path)
        query = self.query_frame_train.iloc[idx, 0]
        score_annotations = self.query_frame_train.iloc[idx, 3:]
        score_annotations = np.array([score_annotations])

        score_annotations = score_annotations.astype('float').reshape(-1, )

        sample = {'image': image, 'query': query, 'score_annotations': score_annotations}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image']))
            sample['score_annotations'] = torch.from_numpy(sample['score_annotations'])
        return sample


import gensim
from t2i import T2I
from nltk.tokenize import word_tokenize
# import nltk
# # nltk.data.path.append("F:/Anaconda/Lib/site-packages/nltk_data")
# nltk.download('punkt')
import logging

def embedding_model():
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Load the queries from every dataset
    queries_test = pd.read_csv('query-VS/dataset/videos/video_name_test.csv', engine='python')
    queries_train = pd.read_csv('query-VS/dataset/videos/video_name_train.csv', engine='python')
    queries_val = pd.read_csv('query-VS/dataset/videos/video_name_val.csv', engine='python')


    # Stack train/val/test queries together
    df_queries = pd.concat([queries_train, queries_val, queries_test], axis=0)
    queries = df_queries.values.tolist()

    # Tokenize
    token = [word_tokenize(q[0]) for q in queries]

    # Build the dictionary
    word_index = T2I.build(token)

    # Build word2vec model
    model = gensim.models.Word2Vec(token, min_count=1)
    print(model)
    # summarize vocabulary
    words = list(model.wv.key_to_index)
    return model


model_emb = embedding_model()

import re

def encode_queries(queries, model):
    emb = []
    for q in queries:
        word = q.split()
        emb.append(model.wv[word])
    return emb

def query_embedding(queries, max_length):
    one_hot_x_list = encode_queries(queries, model_emb)
    one_hot_x_tensor = []
    for i in one_hot_x_list:
        one_hot_x_tensor.append(torch.FloatTensor(i))

    one_hot_x_tensor_padded = pad_sequence(one_hot_x_tensor, batch_first=True, padding_value=0)

    one_hot_x_tensor_padded_with_same_max_length = []
    for i in one_hot_x_tensor_padded:
        if len(i) < max_length:
            i = torch.cat((i, torch.zeros((max_length - len(i), 100))), dim=0)
        else:
            i = i[:8]

        one_hot_x_tensor_padded_with_same_max_length.append(i)

    return torch.transpose(torch.stack(one_hot_x_tensor_padded_with_same_max_length), 1,2)


class QVSmodel(nn.Module):

    def __init__(self):
        super(QVSmodel, self).__init__()

        self.model = resnet34(pretrained='imagenet')
        self.model = models.resnet34(pretrained=True)
        self.fc1 = torch.nn.Linear(512, 4)

        self.fc_text1 = torch.nn.Linear(8, 1)
        self.fc_text2 = torch.nn.Linear(100, 512)

    def forward(self, x, y):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = F.avg_pool2d(x, 7)

        # reshape x
        x = x.view(x.size(0), -1)

        y = F.relu(self.fc_text1(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc_text2(y))

        # Combine x and y by element-wise multiplication. The output dimension is still (1, 512).
        t1 = torch.mul(x, y)

        # Computes the second fully connected layer
        relevance_class_prediction = self.fc1(t1)

        return relevance_class_prediction


def train(train_loader, model, optimizer, criterion, device):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        model: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0
    count_relevance = 0
    total = 0

    # Iterate through batches
    for i, sample_batched in enumerate(tqdm(train_loader)):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, query, labels = sample_batched['image'], sample_batched['query'], sample_batched['score_annotations']

        labels_relevance = labels[:, 0]
        queries = query_embedding(query, 8)
        queries = queries.to(device)
        inputs = inputs.to(device)
        labels_relevance = labels_relevance.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs, queries)
        loss = criterion(outputs, labels_relevance.long())
        loss.backward()
        optimizer.step()

        # Keep track of loss and accuracy
        avg_loss += loss.item()
        max_values_relevance, arg_maxs_relevance = torch.max(outputs, dim=1)
        num_correct_relevance = torch.sum(labels_relevance.long() == arg_maxs_relevance.long())
        count_relevance = count_relevance + num_correct_relevance.item()

        # total += labels.size(0)

    return avg_loss / len(train_loader), count_relevance


def test(test_loader, model, criterion, device):
    """
    Evaluates network in batches.

    Args:
        test_loader: Data loader for test set.
        model: Neural network model.
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0
    count_relevance = 0
    total = 0

    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # Iterate through batches
        for i, sample_batched in enumerate(tqdm(test_loader)):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, query, labels = sample_batched['image'], sample_batched['query'], sample_batched[
                'score_annotations']

            labels_relevance = labels[:, 0]
            queries = query_embedding(query, 8)
            queries = queries.to(device)
            inputs = inputs.to(device)
            labels_relevance = labels_relevance.to(device)

            # Zero the parameter gradients


            # Forward + backward + optimize
            outputs = model(inputs, queries)
            loss = criterion(outputs, labels_relevance.long())
            # loss.backward()
            # optimizer.step()

            # Keep track of loss and accuracy
            avg_loss += loss.item()
            max_values_relevance, arg_maxs_relevance = torch.max(outputs, dim=1)
            num_correct_relevance = torch.sum(labels_relevance.long() == arg_maxs_relevance.long())
            count_relevance = count_relevance + num_correct_relevance.item()

            # total += labels.size(0)

    return avg_loss / len(test_loader), count_relevance


def run(train_dataset,test_dataset,epochs =1):
    """


    Args:
        rnn_type: can be either vanilla, gru or lstm
        epochs: number of epochs to run
        hidden_size: dimension of hidden state of rnn cell
    """
    # Create a writer to write to Tensorboard
    writer = SummaryWriter()

    # Create classifier model
    model_qvs = QVSmodel()
    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_qvs.parameters(), 0.0001)

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_qvs = model_qvs.to(device)

    for epoch in tqdm(range(epochs)):
        # Train on data
        train_loss, train_count = train(train_loader,
                                        model_qvs,
                                        optimizer,
                                        criterion,
                                        device)

        # Test on data
        test_loss, test_count = test(test_loader,
                                     model_qvs,
                                     criterion,
                                     device)

        train_acc_relevance = (float(train_count) / len(train_dataset))
        test_acc_relevance = (float(test_count) / len(test_dataset) )

        # Write metrics to Tensorboard
        print("Train loss: ", train_loss, "Test loss: ", test_loss)
        print("Train acc: ", train_acc_relevance, "Test acc: ", test_acc_relevance)
        # Write metrics to Tensorboard
        writer.add_scalars('Loss', {
            'Train': train_loss,
            'Test': test_loss
        }, epoch)
        writer.add_scalars('Accuracy', {
            'Train': train_acc_relevance,
            'Test': test_acc_relevance
        }, epoch)

    # torch.save(model_qvs, "models")

    print('\nFinished.')
    writer.flush()
    writer.close()

train_path = "annotations/query_frame_annotations_train_major.csv"
test_path = "annotations/query_frame_annotations_test_major.csv"
val_path = "annotations/query_frame_annotations_val_major.csv"
root_dir = "https://data.vision.ee.ethz.ch/arunv/AMT_VideoFrames/"

train_dataset = dataset(train_path, root_dir,'train_data', train_transform)
test_dataset = dataset(test_path, root_dir, 'test_data',test_transform)
# val_dataset = dataset(val_path, root_dir, valid_transform)



train_loader =  DataLoader(train_dataset, batch_size = 199, shuffle=True)
test_loader =  DataLoader(test_dataset, batch_size = 199, shuffle=True)

run(train_dataset,test_dataset,epochs=25)
