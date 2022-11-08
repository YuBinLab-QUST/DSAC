import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import math
import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from torch.utils.data import random_split
from Datasets.DataGenerator import Dataset_690


class Constructor:
    """
        Using CNN and self-attention mechanism to extract features in an interactive way
    """

    def __init__(self, model, model_name='dsac'):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5, verbose=1)
        self.loss_function = nn.BCELoss()

        self.batch_size = 64
        self.epochs = 15

    def train(self, TrainLoader, ValidateLoader):
        path = os.path.abspath(os.curdir)
        best = 1
        for epoch in range(self.epochs):
            self.model.train()
            ProgressBar = tqdm(TrainLoader)
            for data in ProgressBar:
                self.optimizer.zero_grad()

                ProgressBar.set_description("Epoch %d" % epoch)
                seq, label = data
                output1, output2 = self.model(seq.unsqueeze(1).to(self.device))  #
                loss = 0.5 * self.loss_function(output1, label.float().to(self.device)) + 0.5 * self.loss_function(
                    output2, label.float().to(self.device))
                #output=self.model(seq.unsqueeze(1).to(self.device))
                #loss = self.loss_function(output, label.float().to(self.device))
                ProgressBar.set_postfix(loss=loss.item())

                loss.backward()
                self.optimizer.step()

            validate_loss = self.validate(ValidateLoader)
            if validate_loss < best:
                best = validate_loss
                model_name = path + '\\' + self.model_name + 'epoch' + str(epoch) + '.pth'
                
        torch.save(self.model.state_dict(), model_name)
        return model_name

        print('Complete training and validation!\n')

    def validate(self, ValidateLoader):
        path = os.path.abspath(os.curdir)
        valid_loss = []
        predicted=[]
        true=[]
        self.model.eval()
        with torch.no_grad():
            for valid_seq, valid_labels in ValidateLoader:
                valid_output1, valid_output2 = self.model(valid_seq.unsqueeze(1).to(self.device))
                '''print((0.5*valid_output1+0.5*valid_output2).size())
                predicted.append((0.5*valid_output1+0.5*valid_output2).squeeze(dim=1))
                true.append(valid_labels)'''
                #valid_output = self.model(valid_seq.unsqueeze(1).to(self.device))
                valid_labels = valid_labels.float().to(self.device)
                #valid_loss.append(self.loss_function(valid_output, valid_labels))
                valid_loss.append((0.5 * self.loss_function(valid_output1, valid_labels) +
                0.5 * self.loss_function(valid_output2, valid_labels)).item())
            valid_loss_avg = torch.mean(torch.Tensor(valid_loss))
            self.scheduler.step(valid_loss_avg)
            #a,r,p=self.calculate(predicted,true)
            #print(a)
        return valid_loss_avg

    def test(self, TestLoader, model_name):

        self.model.load_state_dict(torch.load(model_name))
        predicted_value = []
        true_label = []
        self.model.eval()
        for seq, label in TestLoader:
            output1, output2 = self.model(seq.unsqueeze(1))  #
            output = 0.5 * output1 + 0.5 * output2
            #output=self.model(seq)#.unsqueeze(1)
            predicted_value.append(output.squeeze(dim=0).squeeze(dim=0).detach().numpy())
            true_label.append(label.squeeze(dim=0).squeeze(dim=0).detach().numpy())
        print('Complete test!\n')
        return predicted_value, true_label

    def calculate(self, predicted_value, true_label):
        accuracy = accuracy_score(y_pred=np.array(predicted_value).round(), y_true=true_label)
        roc_auc = roc_auc_score(y_score=predicted_value, y_true=true_label)

        precision, recall, _ = precision_recall_curve(probas_pred=predicted_value, y_true=true_label)
        pr_auc = auc(recall, precision)

        return accuracy, roc_auc, pr_auc

    def run(self, file_name, ratio=0.8):

        Train_Validate_Set = Dataset_690(file_name, False)

        """divide Train samples and Validate samples"""
        Train_Set, Validate_Set = random_split(dataset=Train_Validate_Set,
                                               lengths=[math.ceil(len(Train_Validate_Set) * ratio),
                                                        len(Train_Validate_Set) -
                                                        math.ceil(len(Train_Validate_Set) * ratio)],
                                               generator=torch.Generator().manual_seed(0))

        TrainLoader = loader.DataLoader(dataset=Train_Set, drop_last=True,
                                        batch_size=self.batch_size, shuffle=True, num_workers=0)
        ValidateLoader = loader.DataLoader(dataset=Validate_Set, drop_last=True,
                                           batch_size=self.batch_size, shuffle=False, num_workers=0)

        TestLoader = loader.DataLoader(dataset=Dataset_690(file_name, True),
                                       batch_size=1, shuffle=False, num_workers=0)

        model_name = self.train(TrainLoader, ValidateLoader)

        predicted_value, true_label = self.test(TestLoader, model_name)

        accuracy, roc_auc, pr_auc = self.calculate(predicted_value, true_label)


        return accuracy, roc_auc, pr_auc


from models.DSAC import dsac

Train = Constructor(model=dsac())

Train.run(file_name='wgEncodeAwgTfbsBroadDnd41Ezh239875UniPk')