import json
import torch
import torch.utils.data as Data

from torch.autograd import Variable

w2i = {}
w2i['N'] = 0
w2i['M'] = 1
w2i['E'] = 2
w2i['A'] = 3
w2i['R'] = 4
w2i['C'] = 5
w2i['D'] = 6
w2i['V'] = 7
w2i['Y'] = 8
w2i['S'] = 9
w2i['H'] = 10
w2i['F'] = 11
w2i['W'] = 12
w2i['L'] = 13
w2i['P'] = 14
w2i['Q'] = 15
w2i['Z'] = 16

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def create_data_loader(data, batch_size):
    t = torch.linspace(0, len(data)-1, len(data))
    torch_dataset = Data.TensorDataset(t, t)
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    return loader

def get_batch_data(data, batch_x, rnn_length):

    batch = [data[int(i)] for i in batch_x]
    batch_length = len(batch)

    intervals = []  # rnn_length * batch_size * base_size
    states = []  # rnn_length * batch_size * base_size

    for b in batch:
        binterval = b['interval']
        tintervals = []
        for m in range(rnn_length):
            tintervals.append(binterval[m])
        intervals.append(tintervals)

        bstates = b['states']
        tstates = []
        for m in range(rnn_length):
            s = w2i[bstates[m]]
            tstates.append(s)
        states.append(tstates)

    intervals_revert = []
    states_revert = []
    for m in range(rnn_length):
        tintervals = []
        tstates = []
        for n in range(len(intervals)):
            tintervals.append(intervals[n][m])
            tstates.append(states[n][m])
        intervals_revert.append(tintervals)
        states_revert.append(tstates)

    intervals = intervals_revert
    states = states_revert

    return batch_length, intervals, states


def try_cuda(target):
    if torch.cuda.is_available():
        return target.cuda()
    else:
        return target

def create_0_value():
    if torch.cuda.is_available():
        return Variable(torch.FloatTensor([0.]).cuda())
    else:
        return Variable(torch.FloatTensor([0.]))

def create_ss_variable(s1):
    if torch.cuda.is_available():
        return Variable(torch.randn(s1).cuda(), requires_grad=True)
    else:
        return Variable(torch.randn(s1), requires_grad=True)

def create_random_variable(s1, s2):
    if torch.cuda.is_available():
        return Variable(torch.randn(s1, s2).cuda(), requires_grad=True)
    else:
        return Variable(torch.randn(s1, s2), requires_grad=True)

def create_variable_by_longTensor(data):
    if torch.cuda.is_available():
        return Variable(data.cuda())
    else:
        return Variable(data)


