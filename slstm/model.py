import torch
import torch.nn as nn
from torch.autograd import Variable

class SLSTMCell(nn.Module):

    def __init__(self, base_size, target_size, interval_size):
        super(SLSTMCell, self).__init__()

        self.base_size = base_size
        self.target_size = target_size
        self.interval_size = interval_size

        self.weight_i_s = self.create_base_parameter()
        self.weight_i_h = self.create_base_parameter()
        self.weight_i_i = self.create_base_parameter()
        self.weight_i_x = self.create_base_parameter()
        self.bias_i_s = self.create_base_parameter()
        self.bias_i_h = self.create_base_parameter()
        self.bias_i_i = self.create_base_parameter()
        self.bias_i_x = self.create_base_parameter()

        self.weight_f_s = self.create_base_parameter()
        self.weight_f_h = self.create_base_parameter()
        self.weight_f_i = self.create_base_parameter()
        self.weight_f_x = self.create_base_parameter()
        self.bias_f_s = self.create_base_parameter()
        self.bias_f_h = self.create_base_parameter()
        self.bias_f_i = self.create_base_parameter()
        self.bias_f_x = self.create_base_parameter()

        self.weight_g_s = self.create_base_parameter()
        self.weight_g_h = self.create_base_parameter()
        self.weight_g_i = self.create_base_parameter()
        self.weight_g_x = self.create_base_parameter()
        self.bias_g_s = self.create_base_parameter()
        self.bias_g_h = self.create_base_parameter()
        self.bias_g_i = self.create_base_parameter()
        self.bias_g_x = self.create_base_parameter()

        self.weight_o_s = self.create_base_parameter()
        self.weight_o_h = self.create_base_parameter()
        self.weight_o_i = self.create_base_parameter()
        self.weight_o_x = self.create_base_parameter()
        self.bias_o_s = self.create_base_parameter()
        self.bias_o_h = self.create_base_parameter()
        self.bias_o_i = self.create_base_parameter()
        self.bias_o_x = self.create_base_parameter()

        self.weight_m_s = self.create_base_parameter()
        self.weight_m_h = self.create_base_parameter()
        self.weight_m_i = self.create_base_parameter()
        self.weight_m_x = self.create_base_parameter()
        self.bias_m_s = self.create_base_parameter()
        self.bias_m_h = self.create_base_parameter()
        self.bias_m_i = self.create_base_parameter()
        self.bias_m_x = self.create_base_parameter()

        self.weight_n_s = self.create_base_parameter()
        self.weight_n_h = self.create_base_parameter()
        self.weight_n_i = self.create_base_parameter()
        self.weight_n_x = self.create_base_parameter()
        self.bias_n_s = self.create_base_parameter()
        self.bias_n_h = self.create_base_parameter()
        self.bias_n_i = self.create_base_parameter()
        self.bias_n_x = self.create_base_parameter()

        self.init_c = self.create_base_parameter()
        self.init_h = self.create_base_parameter()
        self.init_x = self.create_base_parameter()

        self.trans = nn.Linear(self.base_size, self.target_size)
        self.interval_embedding = nn.Embedding(self.interval_size, self.base_size)
        self.target_embedding = nn.Embedding(self.target_size, self.base_size)





    def forward(self, ct_1, ht_1, st_1, it, xt):

        rit = torch.sigmoid(self.weight_i_s * st_1 + self.bias_i_s
                            + self.weight_i_h * ht_1 + self.bias_i_h
                            + self.weight_i_i * it + self.bias_i_i
                            + self.weight_i_x * xt + self.bias_i_x)
        rft = torch.sigmoid(self.weight_f_s * st_1 + self.bias_f_s
                            + self.weight_f_h * ht_1 + self.bias_f_h
                            + self.weight_f_i * it + self.bias_f_i
                            + self.weight_f_x * xt + self.bias_f_x)
        rgt = torch.tanh(self.weight_g_s * st_1 + self.bias_g_s
                         + self.weight_g_h * ht_1 + self.bias_g_h
                         + self.weight_g_i * it + self.bias_g_i
                         + self.weight_g_x * xt + self.bias_g_x)
        rot = torch.sigmoid(self.weight_o_s * st_1 + self.bias_o_s
                            + self.weight_o_h * ht_1 + self.bias_o_h
                            + self.weight_o_i * it + self.bias_o_i
                            + self.weight_o_x * xt + self.bias_o_x)
        rmt = torch.sigmoid(self.weight_m_s * st_1 + self.bias_m_s
                            + self.weight_m_h * ht_1 + self.bias_m_h
                            + self.weight_m_i * it + self.bias_m_i
                            + self.weight_m_x * xt + self.bias_m_x)
        rnt = torch.sigmoid(self.weight_n_s * st_1 + self.bias_n_s
                            + self.weight_n_h * ht_1 + self.bias_n_h
                            + self.weight_n_i * it + self.bias_n_i
                            + self.weight_n_x * xt + self.bias_n_x)

        rct = rft * ct_1 + rit * rgt
        rht = rot * torch.tanh(rct)
        rst = rmt * st_1 + rnt * torch.tanh(ct_1)
        rc = self.trans(rht)

        return rct, rht, rst, rc



    def compute_loss(self, rc, target):

        loss = nn.CrossEntropyLoss()
        output = loss(rc, target)
        return output.mean()

    def backward_e(self, loss, lr, st):

        return st - lr*st.grad*loss

    def backward_m(self, loss, lr):

        self.weight_i_s = self.create_parameter(self.weight_i_s - lr*self.weight_i_s.grad*loss)
        self.weight_i_h = self.create_parameter(self.weight_i_h - lr*self.weight_i_h.grad*loss)
        self.weight_i_i = self.create_parameter(self.weight_i_i - lr*self.weight_i_i.grad*loss)
        self.weight_i_x = self.create_parameter(self.weight_i_x - lr*self.weight_i_x.grad*loss)
        self.bias_i_s = self.create_parameter(self.bias_i_s - lr*self.bias_i_s.grad*loss)
        self.bias_i_h = self.create_parameter(self.bias_i_h - lr*self.bias_i_h.grad*loss)
        self.bias_i_i = self.create_parameter(self.bias_i_i - lr*self.bias_i_i.grad*loss)
        self.bias_i_x = self.create_parameter(self.bias_i_x - lr*self.bias_i_x.grad*loss)

        self.weight_f_s = self.create_parameter(self.weight_f_s - lr*self.weight_f_s.grad*loss)
        self.weight_f_h = self.create_parameter(self.weight_f_h - lr*self.weight_f_h.grad*loss)
        self.weight_f_i = self.create_parameter(self.weight_f_i - lr*self.weight_f_i.grad*loss)
        self.weight_f_x = self.create_parameter(self.weight_f_x - lr*self.weight_f_x.grad*loss)
        self.bias_f_s = self.create_parameter(self.bias_f_s - lr*self.bias_f_s.grad*loss)
        self.bias_f_h = self.create_parameter(self.bias_f_h - lr*self.bias_f_h.grad*loss)
        self.bias_f_i = self.create_parameter(self.bias_f_i - lr*self.bias_f_i.grad*loss)
        self.bias_f_x = self.create_parameter(self.bias_f_x - lr*self.bias_f_x.grad*loss)

        self.weight_g_s = self.create_parameter(self.weight_g_s - lr*self.weight_g_s.grad*loss)
        self.weight_g_h = self.create_parameter(self.weight_g_h - lr*self.weight_g_h.grad*loss)
        self.weight_g_i = self.create_parameter(self.weight_g_i - lr*self.weight_g_i.grad*loss)
        self.weight_g_x = self.create_parameter(self.weight_g_x - lr*self.weight_g_x.grad*loss)
        self.bias_g_s = self.create_parameter(self.bias_g_s - lr*self.bias_g_s.grad*loss)
        self.bias_g_h = self.create_parameter(self.bias_g_h - lr*self.bias_g_h.grad*loss)
        self.bias_g_i = self.create_parameter(self.bias_g_i - lr*self.bias_g_i.grad*loss)
        self.bias_g_x = self.create_parameter(self.bias_g_x - lr*self.bias_g_x.grad*loss)

        self.weight_o_s = self.create_parameter(self.weight_o_s - lr*self.weight_o_s.grad*loss)
        self.weight_o_h = self.create_parameter(self.weight_o_h - lr*self.weight_o_h.grad*loss)
        self.weight_o_i = self.create_parameter(self.weight_o_i - lr*self.weight_o_i.grad*loss)
        self.weight_o_x = self.create_parameter(self.weight_o_x - lr*self.weight_o_x.grad*loss)
        self.bias_o_s = self.create_parameter(self.bias_o_s - lr*self.bias_o_s.grad*loss)
        self.bias_o_h = self.create_parameter(self.bias_o_h - lr*self.bias_o_h.grad*loss)
        self.bias_o_i = self.create_parameter(self.bias_o_i - lr*self.bias_o_i.grad*loss)
        self.bias_o_x = self.create_parameter(self.bias_o_x - lr*self.bias_o_x.grad*loss)

        self.weight_m_s = self.create_parameter(self.weight_m_s - lr*self.weight_m_s.grad*loss)
        self.weight_m_h = self.create_parameter(self.weight_m_h - lr*self.weight_m_h.grad*loss)
        self.weight_m_i = self.create_parameter(self.weight_m_i - lr*self.weight_m_i.grad*loss)
        self.weight_m_x = self.create_parameter(self.weight_m_x - lr*self.weight_m_x.grad*loss)
        self.bias_m_s = self.create_parameter(self.bias_m_s - lr*self.bias_m_s.grad*loss)
        self.bias_m_h = self.create_parameter(self.bias_m_h - lr*self.bias_m_h.grad*loss)
        self.bias_m_i = self.create_parameter(self.bias_m_i - lr*self.bias_m_i.grad*loss)
        self.bias_m_x = self.create_parameter(self.bias_m_x - lr*self.bias_m_x.grad*loss)

        self.weight_n_s = self.create_parameter(self.weight_n_s - lr*self.weight_n_s.grad*loss)
        self.weight_n_h = self.create_parameter(self.weight_n_h - lr*self.weight_n_h.grad*loss)
        self.weight_n_i = self.create_parameter(self.weight_n_i - lr*self.weight_n_i.grad*loss)
        self.weight_n_x = self.create_parameter(self.weight_n_x - lr*self.weight_n_x.grad*loss)
        self.bias_n_s = self.create_parameter(self.bias_n_s - lr*self.bias_n_s.grad*loss)
        self.bias_n_h = self.create_parameter(self.bias_n_h - lr*self.bias_n_h.grad*loss)
        self.bias_n_i = self.create_parameter(self.bias_n_i - lr*self.bias_n_i.grad*loss)
        self.bias_n_x = self.create_parameter(self.bias_n_x - lr*self.bias_n_x.grad*loss)

        self.init_c = self.create_parameter(self.init_c - lr*self.init_c.grad*loss)
        self.init_h = self.create_parameter(self.init_h - lr*self.init_h.grad*loss)
        self.init_x = self.create_parameter(self.init_x - lr*self.init_x.grad*loss)

        self.trans.weight = nn.Parameter(self.trans.weight - lr * self.trans.weight.grad * loss)
        self.trans.bias = nn.Parameter(self.trans.bias - lr * self.trans.bias.grad * loss)

        self.interval_embedding.weight = nn.Parameter(self.interval_embedding.weight - lr*self.interval_embedding.weight.grad*loss)
        self.target_embedding.weight = nn.Parameter(self.target_embedding.weight - lr * self.target_embedding.weight.grad * loss)

        self.try_cuda()


    def create_base_parameter(self):
        if torch.cuda.is_available():
            return Variable(torch.randn(self.base_size).cuda(), requires_grad=True)
        else:
            return Variable(torch.randn(self.base_size), requires_grad=True)

    def create_parameter(self, data):

        return Variable(data, requires_grad=True)



    def try_cuda(self):

        if torch.cuda.is_available():
            self.trans = self.trans.cuda()
            self.interval_embedding = self.interval_embedding.cuda()
            self.target_embedding = self.target_embedding.cuda()

    def clear_grad(self):

        self.weight_i_s.grad.zero_()
        self.weight_i_h.grad.zero_()
        self.weight_i_i.grad.zero_()
        self.weight_i_x.grad.zero_()
        self.bias_i_s.grad.zero_()
        self.bias_i_h.grad.zero_()
        self.bias_i_i.grad.zero_()
        self.bias_i_x.grad.zero_()

        self.weight_f_s.grad.zero_()
        self.weight_f_h.grad.zero_()
        self.weight_f_i.grad.zero_()
        self.weight_f_x.grad.zero_()
        self.bias_f_s.grad.zero_()
        self.bias_f_h.grad.zero_()
        self.bias_f_i.grad.zero_()
        self.bias_f_x.grad.zero_()

        self.weight_g_s.grad.zero_()
        self.weight_g_h.grad.zero_()
        self.weight_g_i.grad.zero_()
        self.weight_g_x.grad.zero_()
        self.bias_g_s.grad.zero_()
        self.bias_g_h.grad.zero_()
        self.bias_g_i.grad.zero_()
        self.bias_g_x.grad.zero_()

        self.weight_o_s.grad.zero_()
        self.weight_o_h.grad.zero_()
        self.weight_o_i.grad.zero_()
        self.weight_o_x.grad.zero_()
        self.bias_o_s.grad.zero_()
        self.bias_o_h.grad.zero_()
        self.bias_o_i.grad.zero_()
        self.bias_o_x.grad.zero_()

        self.weight_m_s.grad.zero_()
        self.weight_m_h.grad.zero_()
        self.weight_m_i.grad.zero_()
        self.weight_m_x.grad.zero_()
        self.bias_m_s.grad.zero_()
        self.bias_m_h.grad.zero_()
        self.bias_m_i.grad.zero_()
        self.bias_m_x.grad.zero_()

        self.weight_n_s.grad.zero_()
        self.weight_n_h.grad.zero_()
        self.weight_n_i.grad.zero_()
        self.weight_n_x.grad.zero_()
        self.bias_n_s.grad.zero_()
        self.bias_n_h.grad.zero_()
        self.bias_n_i.grad.zero_()
        self.bias_n_x.grad.zero_()

        self.init_c.grad.zero_()
        self.init_h.grad.zero_()
        self.init_x.grad.zero_()

        self.trans.weight.grad.zero_()
        self.trans.bias.grad.zero_()

        self.interval_embedding.weight.grad.zero_()
        self.target_embedding.weight.grad.zero_()



