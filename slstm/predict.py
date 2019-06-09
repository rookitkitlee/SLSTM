import torch
import torch.nn as nn
import torch.utils.data as Data
import json
from torch import optim
from torch.autograd import Variable
from slstm.predict_model import classifer
from slstm.data import create_0_value

torch.manual_seed(1)

base_size = 64
batch_size = 128
lr = 0.0025
poch = 5000




data = []
train = []
test2 = []
test4 = []
test8 = []
test16 = []
test32 = []


with open("checkpoint2/data.json", 'r') as f:
    data = json.load(f)

print(len(data))

xxx = 0

for d in data:


    float_ss = []

    ss = d['ss']
    ss = ss[1:len(ss)-1]
    for s in ss.replace('\n','').replace("  "," ").replace("  "," ").split(' '):
        if s != ' ' and s != '':
            float_ss.append(float(s))

    cs = d['cs']
    cs = cs[1:len(cs) - 1]
    for s in cs.replace('\n', '').replace("  ", " ").replace("  ", " ").split(' '):
        if s != ' ' and s != '':
            float_ss.append(float(s))


    d['ss'] = float_ss

    if float(d['ret']) < 0.8:
        train.append(d)
    elif int(d['level']) == 2:
        test2.append(d)
    elif int(d['level']) == 4:
        test4.append(d)
    elif int(d['level']) == 8:
        test8.append(d)
    elif int(d['level']) == 16:
        test16.append(d)
    elif int(d['level']) == 32:
        test32.append(d)

    if int(d['bout']) == 1:
        xxx = xxx + 1
print(xxx)


def train_work():

    model = classifer(base_size)
    if torch.cuda.is_available():
        model.cuda()
    loss_fn = nn.CrossEntropyLoss()  # 定义损失函数
    optimiser = optim.SGD(params=model.parameters(), lr=lr)  # 定义优化器

    global_loss = Variable(torch.FloatTensor([1000.]).cuda())
    for i in range(poch):

        t = torch.linspace(0, len(train) - 1, len(train))
        torch_dataset = Data.TensorDataset(t, t)
        loader = Data.DataLoader(
            dataset=torch_dataset,  # torch TensorDataset format
            batch_size=batch_size,
            shuffle=True,
            num_workers=1
        )

        total_loss = create_0_value()
        for step, (batch_x, batch_y) in enumerate(loader):
            optimiser.zero_grad()  # 梯度清零

            x0 = [train[int(i)]['ss'] for i in batch_x]
            y0 = [train[int(i)]['bout'] for i in batch_x]

            x = Variable(torch.FloatTensor(x0).cuda())
            y = Variable(torch.LongTensor(y0).cuda())


            out = model(x)  # 前向传播
            loss = loss_fn(out, y)  # 计算损失
            loss.backward()  # 反向传播
            optimiser.step()  # 随机梯度下降
            total_loss = total_loss + loss


        print('poch : '+str(i)+' loss : '+ str(total_loss))

        if global_loss > loss:
            global_loss = total_loss
            torch.save(model, "predict/model.pth")

        if i % 10 == 0:
            predict_work(model)


def predict_step(model, data, rnn_length):


    t = torch.linspace(0, len(data) - 1, len(data))
    torch_dataset = Data.TensorDataset(t, t)
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )

    right = 0
    err = 0
    xx = 0
    for step, (batch_x, batch_y) in enumerate(loader):

        x0 = [data[int(i)]['ss'] for i in batch_x]
        y0 = [data[int(i)]['bout'] for i in batch_x]


        x = Variable(torch.FloatTensor(x0).cuda())

        out = model(x)  # 前向传播


        i = -1
        for o in out:
            i = i+1
            flag = 0


            if o[1] > o[0]:
                flag = 1
                xx = xx + 1

            if flag == y0[i]:
                right = right + 1
            else:
                err = err + 1

    print("rnn_length : "+str(rnn_length) +" total : "+str(len(data))+" right : "+str(right)+" err : "+str(err)+" xx : "+str(xx))
    return right



def predict_work(model):


    #model = torch.load("predict/model.pth")
    total_right = 0
    total_right = total_right + predict_step(model, test2, 2)
    total_right = total_right + predict_step(model, test4, 4)
    total_right = total_right + predict_step(model, test8, 8)
    total_right = total_right + predict_step(model, test16, 16)
    total_right = total_right + predict_step(model, test32, 32)
    print("predict total "+str(len(test2) + len(test4)+len(test8)+len(test16)+len(test32))+" right "+str(total_right))



if __name__ == '__main__':
    train_work()
    #predict_work()


