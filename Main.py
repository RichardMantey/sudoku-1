
# coding: utf-8

# In[54]:

import os
import torch
import copy
import tensorflow as tf
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
# from Downloads import LSTMSudokuClassifier as LSTMC
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from random import randint

from tqdm import tqdm


# In[55]:

class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_label, batch_size, n_layers, use_gpu):
        super(LSTMClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.n_layers = n_layers
               
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, n_layers, bidirectional=True)
        self.lstm2 = nn.LSTM(input_dim, hidden_dim, n_layers, bidirectional=True)
        self.lstm3 = nn.LSTM(input_dim, hidden_dim, n_layers, bidirectional=True)
        
        self.hidden2label = nn.Sequential(torch.nn.Linear(2*hidden_dim, n_label),
                                          torch.nn.Softmax(dim = -1))
        self.r_hidden = self.init_hidden()
        self.c_hidden = self.init_hidden()
        self.s_hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(2*n_layers, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(2*n_layers, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(2*n_layers, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(2*n_layers, self.batch_size, self.hidden_dim))
        return (h0, c0)
    
    def ManipulateSquare(self, lstm_out_square):
        counter = 0
        inner_stack = None
        outer_stack = None
        for i in range(3):
            for j in range(3):
                x = lstm_out_square[counter,:,:]
                counter += 1
                x = x.view(3,3,2*hidden_dim)#change to 2*hidden_dim
                if inner_stack is None:
                    inner_stack = x
                else:
                    inner_stack = torch.cat((inner_stack,x),1)        
            if outer_stack is None:
#                 print(inner_stack.shape)
                outer_stack = inner_stack
                inner_stack = None
#                 print('B = ',outer_stack)
            else:
#                 print(inner_stack.shape)
                outer_stack = torch.cat((outer_stack,inner_stack),0)
#                 print('B = ',outer_stack)
                inner_stack = None
        return outer_stack

    def forward(self, row_tensor, col_tensor, square_tensor):
        
        lstm_out_row, hidden_row = self.lstm1(row_tensor, self.r_hidden)
        lstm_out_col, hidden_col = self.lstm2(col_tensor, self.c_hidden)
        lstm_out_square, hidden_square = self.lstm3(square_tensor, self.s_hidden)
        
#         self.r_hidden = hidden_row
#         self.c_hidden = hidden_col
#         self.s_hidden = hidden_square
        
        # Do manipulation Here
        #Row & Col Manipulation
        row_col = lstm_out_row + lstm_out_col +lstm_out_square 
        #Square Manipulation
        lstm_out = row_col 
        lstm_out_tensor = lstm_out
#         if self.use_gpu:
#             lstm_out_tensor = lstm_out.cuda()
#         print("final tensor", lstm_out_tensor.shape, lstm_out_tensor)
        
        #Sum Square Row Col
        y  = self.hidden2label(lstm_out_tensor)
#         print('Output Shape ', y.shape)
        return y


# In[56]:

class InputGrid():
    
    def __init__(self, Grid):
        self.Grid = Grid
        self.gridLength = len(Grid[0])
#         self.labels = labels
        
    def getlims(self, i):
        if 0 <= i <= 2:
            rowlims = [0,3]
        elif 3 <= i <= 5:
            rowlims = [3,6]
        elif 6 <= i <= 8:
                rowlims = [6,9]
        return rowlims

    def get_square(self, i):
        grid_square = None
        if i < 3:
            if i%3 == 0:
                grid_square = self.Grid[0:3,0:3]
            elif i%3 == 1:
                grid_square = self.Grid[0:3,3:6]
            elif i%3 == 2:
                grid_square = self.Grid[0:3,6:9]
        elif i < 6:
            if i%3 == 0:
                grid_square = self.Grid[3:6,0:3]
            elif i%3 == 1:
                grid_square = self.Grid[3:6,3:6]
            elif i%3 == 2:
                grid_square = self.Grid[3:6,6:9]
        elif i < 9:
            if i%3 == 0:
                grid_square = self.Grid[6:9,0:3]
            elif i%3 == 1:
                grid_square = self.Grid[6:9,3:6]
            elif i%3 == 2:
                grid_square = self.Grid[6:9,6:9]
        return grid_square.flatten()
                
    
    # takes 1D returns 2D
    def one_hot(self, vec):
        one_hot_matrix = []
        for val in vec:
            hot_vec = [0 for _ in range(9)]
            if val > 0:
                hot_vec[int(val)-1] = 1
            one_hot_matrix.append(hot_vec)
        real_one_hot = np.array(one_hot_matrix)
        return real_one_hot
    
    #takes 2D returns 3D
    def getInput(self):
#         print(self.Grid)
        Rows = np.zeros((9,9,9))
        Columns = np.zeros((9,9,9))
        Squares = np.zeros((9,9,9))
        Labels = np.zeros((9,9,9))
        for i in range(self.gridLength):
            hot_row = self.one_hot(self.Grid[i,:])
            hot_column = self.one_hot(self.Grid[:,i])
            hot_square = self.one_hot(self.get_square(i))
#             hot_label = self.one_hot(self.labels[i,:])
            
            Rows[i] = hot_row
            Columns[i] = hot_column
            Squares[i] = hot_square
#             Labels[i] = hot_label
        
        Rows = np.reshape (Rows , (1, 81, 9))
        Columns = np.reshape (Columns, (1, 81, 9))
        Squares = np.reshape (Squares, (1, 81, 9))
        Rows = np.transpose (Rows, ((1, 0, 2)))
        Columns = np.transpose (Columns, ((1, 0, 2)))
        Squares = np.transpose (Squares, ((1, 0, 2)))

        row_tensor = Variable(torch.FloatTensor(Rows))
        col_tensor = Variable(torch.FloatTensor(Columns))
        square_tensor = Variable(torch.FloatTensor(Squares))
#         label_tensor = Variable(torch.LongTensor(Labels))
#         print(row_tensor)
#         print(col_tensor)
#         print(square_tensor)
#         print(label_tensor)
                 
        return row_tensor, col_tensor, square_tensor


# In[57]:

# use_plot = True
# use_save = True
# if use_save:
#     import pickle
#     from datetime import datetime

DATA_DIR = 'data'
TRAIN_FILE = 'kaggle_sudoku.csv'
TEST_FILE = 'sudoku_test.txt'
TRAIN_LABEL = 'train_label.txt'
TEST_LABEL = 'test_label.txt'


# In[58]:

class SudokuDataset(Dataset):
    def __init__(self, fpath):
        
        print(fpath)
        lines = open(fpath, 'r').read().splitlines()[1:]
        nsamples = 5000

        X = np.zeros((nsamples, 9*9), np.float32)  
        Y = np.zeros((nsamples, 9*9), np.int32) 

        for i, line in enumerate(lines):
            quiz, solution = line.split(",")
            for j, (q, s) in enumerate(zip(quiz, solution)):
                X[i, j], Y[i, j] = q, s
            if i>=nsamples-1:
                break

        X = np.reshape(X, (-1, 9, 9))
        Y = np.reshape(Y, (-1, 9, 9))
        Y -= 1
        self.X = X
        self.Y = Y
#         print(Y)
#         print(Y.shape)

    def __getitem__(self, index):
        quiz = self.X[index]
        sol = self.Y[index]
        return quiz, sol
    
    def __len__(self):
        return len(self.X)


# In[59]:

if __name__=='__main__':
    
    ## parameter setting
    epochs = 1
    batch_size = 1
    use_gpu = torch.cuda.is_available()
    learning_rate = 0.001

    input_dim = 9
    hidden_dim = 200
    n_label = 9
    n_layers = 3
    
    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    test_path = os.path.join(DATA_DIR, TEST_FILE)
    save_filepath = 'rnn.pt'


     ### ********************create model**************************
    model = LSTMClassifier(input_dim, hidden_dim, n_label, batch_size, n_layers, use_gpu)
    if use_gpu:
        model = model.cuda()
    
    training_set = SudokuDataset(train_path)
    train_loader = DataLoader(training_set,
                          batch_size=1,
                          shuffle=True,
                          num_workers=8
                          )
    
    print('TRAINING')

    optimizer = optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    
    
    def adjust_learning_rate(optimizer, epoch):
        lr = learning_rate * (0.1 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer
    
#     import pdb
### training procedure

#     t = trange(100, desc='Bar desc', leave=True)
    for epoch in range(epochs):
#         optimizer = adjust_learning_rate(optimizer, epoch)
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0

#         model.hidden = model.init_hidden()

        for i, traindata in zip(tqdm(range(5000)), train_loader):
            model.zero_grad()
            train_inputs_raw, train_labels_raw = traindata
#             print(train_inputs_raw.shape)

            train_data_numpy = Variable(train_inputs_raw).data.numpy()
            train_labels_numpy = Variable(train_labels_raw).data.numpy()
            train_labels = Variable(torch.LongTensor(train_labels_numpy[0]))

#             pdb.set_trace ()
            input_grid = InputGrid(train_data_numpy[0])
            row_tensor, col_tensor, square_tensor = input_grid.getInput()

            if use_gpu:
                row_tensor, col_tensor, square_tensor, train_labels =                         row_tensor.cuda(), col_tensor.cuda(), square_tensor.cuda(), train_labels.cuda()

            model.hidden = model.init_hidden()
            model.batch_size =  batch_size 
            output = model(row_tensor, col_tensor, square_tensor)
            output = output.view(81,9)
            loss = loss_function(output, train_labels.view(81,))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
#             print(predicted.shape)
#             print('TrainPred =  ',predicted, 'While TrainLabel = ' ,train_labels.data)
            acc = (predicted == train_labels.data.view(81)).sum()/81
#             print(acc)
            total += 81
#             print(loss.data[0])
#             print (acc)

#        print("loss:", loss.data[0], "acc", total_acc)
            train_loss_.append(loss.data[0])
            train_acc_.append(acc)

        print('[Epoch: %3d/%3d] Training Loss: %.3f, Training Acc: %.3f' 
                  % (epoch, epochs, np.mean(train_loss_), np.mean (train_acc_)))
    
    torch.save(model.state_dict(), save_filepath)

    #Later to restore:
#     model.load_state_dict(torch.load(save_filepath))
#     model.eval()


# In[ ]:



