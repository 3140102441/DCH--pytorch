import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def pairwise_loss(u, v, label_u, label_v, hashbit = 48,gamma=1.0, q_lambda=1.0,normed=True):
    label_ip = Variable(torch.mm(label_u.data.float(), label_v.data.float().t()) > 0).float()
    s = torch.clamp(label_ip, 0.0, 1.0)

    if normed:
        ip_1 = torch.mm(u, v.t())

        def reduce_shaper(t):
            return torch.sum(t,1).view(t.size(0),1)
        mod_1 = torch.sqrt(torch.mm(reduce_shaper(torch.mul(u,u)), (reduce_shaper(
            torch.mul(v,v)) + 0.000001).t()))
        dist = torch.tensor(hashbit / 2.0 * \
            (1.0 - torch.div(ip_1, mod_1) +0.000001))
    else:
        r_u = torch.sum(u * u, 1).view(-1,1)
        r_v = torch,sum(v * v, 1).view(-1,1)
        dist = r_u - 2 * torch.mm(u, v.t()) + \
            r_v.t() + 0.001

    cauchy = gamma / (dist + gamma)

    s_t = torch.mul(torch.add(s,-0.5),  2.0)
    sum_1 = torch.sum(s)
    sum_all = torch.sum(torch.abs(s_t))
    balance_param = torch.add(
        torch.abs(torch.add(s, -1.0)), torch.mul(torch.div(sum_all, sum_1), s))

    mask = torch.eye(u.size(0)).data == 0

    cauchy_mask = torch.masked_select(cauchy, Variable(mask).cuda())
    s_mask = torch.masked_select(s, Variable(mask).cuda())
    balance_p_mask = torch.masked_select(balance_param, Variable(mask).cuda())


    all_loss = - s_mask * \
        torch.log(cauchy_mask) - (1.0 - s_mask) * \
        torch.log(1.0 - cauchy_mask)

    cos_loss = torch.mean(torch.mul(all_loss, balance_p_mask))
    q_loss_temp = torch.add(torch.abs(u), -1.0)
    q_loss = q_lambda*torch.mean(torch.mul(q_loss_temp,q_loss_temp))

    return cos_loss + q_loss

