import torch
import torch.nn as nn
import torch.utils.data as Data
import json

from torch.autograd import Variable
from slstm.model import SLSTMCell
from slstm.data import w2i
from slstm.data import read_json
from slstm.data import create_data_loader
from slstm.data import get_batch_data
from slstm.data import try_cuda
from slstm.data import create_0_value
from slstm.data import create_random_variable
from slstm.data import create_variable_by_longTensor
from slstm.data import create_ss_variable


torch.manual_seed(1)

base_size = 32
batch_size = 128
target_size = 17
interval_size = 5

e_poch = 5
e_lr = 0.1
m_lr = 0.05
poch = 201

# poch 100   loss 3452

d2 = read_json("data/d2.json")
d4 = read_json("data/d4.json")
d8 = read_json("data/d8.json")
d16 = read_json("data/d16.json")
d32 = read_json("data/d32.json")

for d in d2:
    d['ss'] = create_ss_variable(base_size)

for d in d4:
    d['ss'] = create_ss_variable(base_size)

for d in d8:
    d['ss'] = create_ss_variable(base_size)

for d in d16:
    d['ss'] = create_ss_variable(base_size)

for d in d32:
    d['ss'] = create_ss_variable(base_size)


def train(cell, data, rnn_length):
    loader = create_data_loader(data, batch_size)
    train_loss = create_0_value()
    for step, (batch_x, batch_y) in enumerate(loader):
        batch_length, intervals, states = get_batch_data(data, batch_x, rnn_length)
        init_ss_numpy = [data[int(i)]['ss'].detach().cpu().numpy() for i in batch_x]
        init_ss = Variable(torch.FloatTensor(init_ss_numpy).cuda(), requires_grad=True)
        step_loss,init_ss, operate_ss, operate_cs = train_step(cell, init_ss, batch_length, intervals, states, rnn_length)
        j = 0
        for i in batch_x:
            data[int(i)]['iss'] = init_ss[j]
            data[int(i)]['oss'] = operate_ss[j]
            data[int(i)]['cs'] = operate_cs[j]
            j = j+1
        train_loss = train_loss + step_loss * batch_length
    average_loss = train_loss / len(data)
    print("train local rnn_length : "+str(rnn_length) +" loss is : "+str(average_loss))
    return train_loss

def train_step(cell, init_ss, batch_length, intervals, states, rnn_length):

    # print('--------------------------------------------------')
    # print(init_ss)
    # init_ss = create_random_variable(batch_length, base_size)
    # print(init_ss)
    last_step_loss = create_0_value()
    operate_cs = cell.init_c
    operate_ss = init_ss
    for j in range(e_poch + 1):
        operate_cs = cell.init_c
        operate_hs = cell.init_h
        operate_xs = cell.init_x
        operate_ss = init_ss
        interval0 = create_variable_by_longTensor(torch.LongTensor(intervals[0]))
        operate_is = cell.interval_embedding(interval0)
        step_loss = create_0_value()
        for m in range(rnn_length):
            operate_cs, operate_hs, operate_ss, rc = cell(operate_cs, operate_hs, operate_ss, operate_is, operate_xs)
            statesm = create_variable_by_longTensor(torch.LongTensor(states[m]))
            loss = cell.compute_loss(rc, statesm)
            step_loss = step_loss + loss
            if m != rnn_length - 1:
                intervalm1 = create_variable_by_longTensor(torch.LongTensor(intervals[m+1]))
                statesm1 = create_variable_by_longTensor(torch.LongTensor(states[m+1]))
                operate_is = cell.interval_embedding(intervalm1)
                operate_xs = cell.target_embedding(statesm1)
        if j < e_poch:
            step_loss.backward()
            init_ss = cell.backward_e(step_loss, e_lr, init_ss)
            init_ss = Variable(init_ss, requires_grad=True)
            cell.clear_grad()
        else:
            step_loss.backward()
            cell.backward_m(step_loss, m_lr / rnn_length)
            last_step_loss = step_loss

    return last_step_loss / rnn_length, Variable(init_ss, requires_grad=True),  Variable(operate_ss, requires_grad=True) , Variable(operate_cs, requires_grad=True)



def save_data():
    cp = []
    for d in d2:
        ss = d['oss']
        cs = d['cs']
        cp.append(
            {'bid': d['bid'], 'bout': d['bout'], 'ret': d['ret'], 'level': 2, 'ss': str(ss.detach().cpu().numpy()),
             'cs': str(cs.detach().cpu().numpy())})
    for d in d4:
        ss = d['oss']
        cs = d['cs']
        cp.append(
            {'bid': d['bid'], 'bout': d['bout'], 'ret': d['ret'], 'level': 4, 'ss': str(ss.detach().cpu().numpy()), 'cs':str(cs.detach().cpu().numpy())})
    for d in d8:
        ss = d['oss']
        cs = d['cs']
        cp.append(
            {'bid': d['bid'], 'bout': d['bout'], 'ret': d['ret'], 'level': 8, 'ss': str(ss.detach().cpu().numpy()), 'cs':str(cs.detach().cpu().numpy())})
    for d in d16:
        ss = d['oss']
        cs = d['cs']
        cp.append(
            {'bid': d['bid'], 'bout': d['bout'], 'ret': d['ret'], 'level': 16, 'ss': str(ss.detach().cpu().numpy()), 'cs':str(cs.detach().cpu().numpy())})
    for d in d32:
        ss = d['oss']
        cs = d['cs']
        cp.append(
            {'bid': d['bid'], 'bout': d['bout'], 'ret': d['ret'], 'level': 32, 'ss': str(ss.detach().cpu().numpy()), 'cs':str(cs.detach().cpu().numpy())})

    with open("checkpoint2/data.json", "w") as f:
        json.dump(cp, f)


if __name__ == '__main__':
    cell = SLSTMCell(base_size, target_size, interval_size)
    cell = try_cuda(cell)
    loss = Variable(torch.FloatTensor([10000.]).cuda())

    for i in range(poch):
        total_loss = create_0_value()
        total_loss = total_loss + train(cell, d2, 2)
        total_loss = total_loss + train(cell, d4, 4)
        total_loss = total_loss + train(cell, d8, 8)
        total_loss = total_loss + train(cell, d16, 16)
        total_loss = total_loss + train(cell, d32, 32)
        average_loss = total_loss /( len(d2) +  len(d4) + len(d8) + len(d16) + len(d32))
        print("train global poch : "+str(i)+" loss is : "+str(average_loss))

        if average_loss < loss:
            loss = average_loss
            torch.save(cell, "checkpoint2/model.pth")

            if i % 10 == 0:
                save_data()
            if i > 180:
                save_data()




