import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.optim as optim
# import transformers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os, time
import numpy as np


classification = 52


def evaluate_model(y_pred, y):
    y_pred = torch.argmax(y_pred, dim=1)
    #     y = torch.argmax(y, dim=1)
    good = 0
    for i in range(len(y)):
        if y_pred[i] == y[i]:
            good = good + 1
    return (good / len(y)) * 100.0


class torch_trainer():

    def __init__(self, name="result"):
        self.path = name + "/"

        if not os.path.exists(name):
            os.makedirs(name)
            print("Created folder " + name)
        else:
            print("Warning!! folder " + name + " is existed")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("will use " + str(self.device))

        self.model_teacher, self.model_student = None, None

        self.fc_mu = nn.Linear(classification, 64).to(self.device)
        self.fc_var = nn.Linear(classification, 64).to(self.device)
        # self.fc_stu_out = nn.Linear(128, 52).to(self.device)

        self.fc_out = nn.Linear(64, classification).to(self.device)

    def set_model_cls(self, model_teacher, model_student):
        '''
        Input:
            model_cls: torch class
        '''
        self.model_teacher = model_teacher
        self.model_student = model_student
        print("model is setted")

    def set_model_parameter(self, args):
        '''
        Input:
            arg: argumentation for creating torch model (type: dict)
        '''
        self.parameter = args
        print("parameter is setted")

    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std

    def do_cross_validation(self, dataset, k, batch, epochs):
        '''
        doing cross_validation 
        Input:
            dataset: torch dataset
            k: split dataset into k flods
            batch: batch size
            batch_fn: collate_fn of dataloader
            epochs: training epochs
        '''
        # KFold from sklearn
        kfold = KFold(n_splits=k, shuffle=True)

        # for storing score of each flod's test
        Score = []

        # just for print the other flod's idx
        flod_id = [i for i in range(1, k + 1)]

        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

            # create/recreate a new model to prevent using the weight of previous training step
            print("initaial a model ...", end="")
            modelTeacher = self.model_teacher.to(self.device)
            model = self.model_student.to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
            print("done")

            # create dataloader with ids from KFold
            train_subsampler = SubsetRandomSampler(train_ids)
            test_subsampler = SubsetRandomSampler(test_ids)

            train_loader = DataLoader(dataset, batch_size=batch, shuffle=True)
            val_loader = DataLoader(dataset, batch_size=batch, shuffle=False)

            # start training
            show_idxs = flod_id[::]
            show_idxs.remove(fold + 1)
            print('Fold ' + str(show_idxs) + ' as traing set')
            print('start training...')

            for epoch in range(epochs):
                train_res = self.train(train_loader, model, optimizer)
                print("Epoch: " + str(epoch + 1) + " Train Loss: " + str(train_res))

            print("... done")

            # test after training
            print("start testing...", end="")
            val_res, acc = self.test(val_loader, model)
            print("done")

            print('Fold ' + str(fold + 1) + ' Val Loss: ' + str(val_res))
            print('Fold ' + str(fold + 1) + ' Val Acc: ' + str(acc))

            # record the score
            Score.append(acc)
            print("-" * 80)

        # return the mean of Score
        print("Score:", np.mean(Score))
        return np.mean(Score)

    def train_find_best_epoch(self, train_ds, test_ds, batch, epochs):
        ''' 
        Input:
            train_ds: torch dataset
            test_ds: torch dataset
            k: split dataset into k flods
            batch: batch size
            batch_fn: collate_fn of dataloader
            epochs: training epochs
        '''

        # creating a model
        print("Creating model ...", end="")
        modelTeacher = self.model_teacher.to(self.device)
        model = self.model_student.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
        print("done")

        # creat dataloader
        print("Creating dataloader ...", end="")

        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)

        test_loader = DataLoader(test_ds, batch_size=batch * 2, shuffle=False)
        print("done")

        # store the train loss and test loss for each epoch
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        # record the best epoch
        best = 0
        best_acc = 0.0

        # start training 
        print("Starting training ...")
        print("-" * 80)
        for epoch in range(epochs):

            # train step
            train_res, train_acc = self.train(train_loader, modelTeacher, model, optimizer)

            # test step
            test_res, test_acc = self.test(test_loader, model)

            # if (epoch + 1) % 10 == 0:
            print("Epoch: " + str(epoch + 1) + " Train Loss: " + str(train_res))
            print("Epoch: " + str(epoch + 1) + " Val Loss: " + str(test_res))

            # record loss and accuracy
            train_losses.append(train_res)
            train_accs.append(train_acc)
            test_losses.append(test_res)
            test_accs.append(test_acc)


            # if test_res is the lowest in current losses, store the model
            if test_acc >= max(test_accs):
                # if (epoch + 1) % 10 == 0:
                print("Epoch " + str(epoch + 1) + " is current best,  test acc: " + str(test_acc))

                # torch.save(model.state_dict(), self.path + "TCT_pretrained.pt")
                # torch.save(model.state_dict(), "saved_models/TCT_pretrained.pt")
                # print("save model to " + self.path + "best.pt")
                best = epoch + 1
                best_acc = test_acc
            # print("-" * 80)
            # self.draw_loss(train_losses, test_losses)

        # train_losses = np.array(train_losses).reshape(-1, 1)
        # train_accs = np.array(train_accs).reshape(-1, 1)
        # test_losses = np.array(test_losses).reshape(-1, 1)
        # test_accs = np.array(test_accs).reshape(-1, 1)
        #
        # # result = int(np.hstack((train_losses, train_accs, test_losses, test_accs)))
        # result = np.concatenate([train_losses, train_accs, test_losses, test_accs], axis=1).reshape(-1, 4)
        # np.savetxt('results/111.csv', result, fmt='%f', delimiter=',')

        print("...Epoch " + str(best) + " is best, acc: " + str(best_acc))

    def train(self, loader, modelTeacher, model, optimizer):
        '''
        normal training step for pytorch model
        '''

        # set the model to train mode
        global acc
        model.train()
        preds = []
        targets = []

        # record loss for each batch
        losses = []
        loss_fn = nn.CrossEntropyLoss()
        soft_loss_fn = nn.KLDivLoss(reduction='batchmean')

        for data in loader:
            # retrive input of model from dataloader
            input_tensor, target_tensor = data

            # put input into same device as model is (cpu or gpu)
            input_tensor = input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)
            # mask_tensor = mask_tensor.to(self.device)

            predTeacher = modelTeacher(input_tensor)
            pred = model(input_tensor)
            # preStudent_out = self.fc_stu_out(pred)

            # pred_hidden = predTeacher + pred
            mu = self.fc_mu(predTeacher)
            log_var = self.fc_var(predTeacher)
            z = self.reparameterize(mu, torch.exp(log_var))
            variational_out = self.fc_out(z)

            hard_loss = loss_fn(pred, target_tensor)
            soft_loss = soft_loss_fn(
                F.log_softmax(pred / 5, dim=1),
                F.softmax(predTeacher / 5, dim=1)
            )
            variational_loss = loss_fn(variational_out, target_tensor)
            loss = hard_loss + soft_loss + variational_loss

            optimizer.zero_grad()

            losses.append(loss.item())
            preds += pred.tolist()
            targets += target_tensor.tolist()
            acc = evaluate_model(pred, target_tensor)

            loss.backward()

            optimizer.step()
            m_loss = np.mean(losses)

        return np.mean(losses), acc

    def test(self, loader, model):
        model.eval()

        preds = []
        targets = []
        losses = []
        loss_fn = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data in loader:
                input_tensor, target_tensor = data

                input_tensor = input_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)
                # mask_tensor = mask_tensor.to(self.device)

                pred = model(input_tensor)
                loss = loss_fn(pred, target_tensor)

                losses.append(loss.item())
                preds += pred.tolist()
                targets += target_tensor.tolist()
                acc = evaluate_model(pred, target_tensor)

                break

        #         print(classification_report(targets, preds))

        # acc = sum(1 for x, y in zip(preds, targets) if x == y) / float(len(preds))
        return np.mean(losses), acc

    def draw_loss(self, y1, y2):
        x = [i + 1 for i in range(len(y1))]
        plt.plot(x, y1, label='train loss')
        plt.plot(x, y2, label='eval loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(self.path + "loss_s223.png")
        plt.close()
