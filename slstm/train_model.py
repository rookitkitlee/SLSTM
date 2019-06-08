import torch
import torch.nn as nn
import torch.utils.data as Data
import json

from torch.autograd import Variable
from model import SLSTMCell
from data import w2i
from data import read_json
from data import create_data_loader
from data import get_batch_data

torch.manual_seed(1)

base_size = 128
batch_size = 3
target_size = 17
interval_size = 5

e_poch = 10

d2 = read_json("data/d2.json")
d4 = []
d8 = []
d16 = []
d32 = []


def train(cell, data, rnn_length):
    loader = create_data_loader(data, batch_size)
    train_loss = Variable(torch.FloatTensor([0.]))
    for step, (batch_x, batch_y) in enumerate(loader):
        batch_length, intervals, states = get_batch_data(data, batch_x, rnn_length)
        step_loss = train_step(cell, batch_length, intervals, states, rnn_length)
        train_loss = train_loss + step_loss * batch_length
    average_loss = train_loss / len(data)
    print("train local rnn_length : "+str(rnn_length) +" loss is : "+str(average_loss))
    return train_loss

def train_step(cell, batch_length, intervals, states, rnn_length):
    init_ss = Variable(torch.randn(batch_length, base_size), requires_grad=True)
    last_step_loss = Variable(torch.FloatTensor([0.]))
    for j in range(e_poch + 1):
        operate_cs = cell.init_c
        operate_hs = cell.init_h
        operate_xs = cell.init_x
        operate_ss = init_ss
        operate_is = cell.interval_embedding(torch.LongTensor(intervals[0]))
        step_loss = Variable(torch.FloatTensor([0.]), requires_grad=True)
        for m in range(rnn_length):
            operate_cs, operate_hs, operate_ss, rc = cell(operate_cs, operate_hs, operate_ss, operate_is, operate_xs)
            loss = cell.compute_loss(rc, torch.LongTensor(states[m]))
            step_loss = step_loss + loss
            if m != rnn_length - 1:
                operate_is = cell.interval_embedding(torch.LongTensor(intervals[m + 1]))
                operate_xs = cell.target_embedding(torch.LongTensor(states[m + 1]))
        if j < e_poch:
            step_loss.backward()
            init_ss = cell.backward_e(step_loss, 0.5, init_ss)
            init_ss = Variable(init_ss, requires_grad=True)
            cell.clear_grad()
        else:
            step_loss.backward()
            cell.backward_m(step_loss, 0.5)
            last_step_loss = step_loss
    return last_step_loss



if __name__ == '__main__':
    cell = SLSTMCell(base_size, target_size, interval_size)
    for i in range(1):
        total_loss = Variable(torch.FloatTensor([0.]))
        total_loss = total_loss + train(cell, d2, 2)
        # train(cell, d4, 4)
        # train(cell, d8, 8)
        # train(cell, d16, 16)
        #train(cell, d32, 32)
        average_loss = total_loss / (len(d2) + len(d4) + len(d8) + len(d16) + len(32))
        print("train global poch is "+str(i)+" loss is : "+str(average_loss))