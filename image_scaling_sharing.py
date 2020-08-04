import numpy as np
import copy
from math import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

import matplotlib.pyplot as plt
import time

from torchvision.datasets import CIFAR100
from torchvision.datasets import MNIST

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--ops", type=int, default=8)
parser.add_argument("--pop", type=int, default=8)
args = parser.parse_args()

N_OPS = args.ops
N_POP = args.pop

MAX_OPS = 2 * N_OPS
MIN_OPS = N_OPS

BS = 400  # Sub-batch size for memory savings
MBS = 400  # Full batch size

TEST_BS = 50
SEQLEN = 5
TEST_STEPS = 50
SPARSE = 0.001
EXCHANGE_STEPS = 20
DELETE_STEPS = 20

MNIST_DIR = "mnist/"
CIFAR_DIR = "cifar100/"

data_dir = "data_%d_%d" % (N_OPS, N_POP)
ckpt_dir = "checkpoints/%s" % data_dir

try:
    os.mkdir(data_dir)
except:
    pass

try:
    os.mkdir(ckpt_dir)
except:
    pass

DEVICE = "cuda"

""" Definition of network and operators """


class Operator(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.Conv2d(16 + 32, 32, 5, padding=2)
        self.l2 = nn.Conv2d(32, 32, 5, padding=2)
        self.l4 = nn.Conv2d(32, 16, 5, padding=2)

        self.skip = np.random.randint(2)
        self.act = np.random.randint(2)

        self.B = nn.Parameter(torch.randn(2, 16) * 4 * pi)

    def copy_from(self, other):
        for p1, p2 in zip(self.parameters(), other.parameters()):
            p1.data = p2.data.clone()
        self.skip = other.skip
        self.act = other.act

    def setfield(self):
        pass

    def forward(self, x, detach=False):
        xx, yy = np.meshgrid(np.arange(32) / 16.0 - 1.0, np.arange(32) / 16.0 - 1.0)
        xx = torch.FloatTensor(xx).to(DEVICE)
        yy = torch.FloatTensor(yy).to(DEVICE)
        zz = torch.cat([xx.unsqueeze(0).unsqueeze(1), yy.unsqueeze(0).unsqueeze(1)], 1)

        zz = zz.view((2, 32 * 32)).transpose(1, 0).contiguous()
        zz = torch.matmul(zz, self.B)
        zz = torch.cat([torch.cos(zz), torch.sin(zz)], 1)
        zz = zz.view((1, 32, 32, 32)).permute(0, 3, 1, 2).contiguous()
        zz = zz.expand(x.size(0), 32, 32, 32)

        if self.act == 0:
            z = F.elu(self.l1(torch.cat([x, zz], 1)))
            z = F.elu(self.l2(z))
            z = self.l4(z)
        else:
            z = torch.sin(self.l1(torch.cat([x, zz], 1)))
            z = torch.sin(self.l2(z))
            z = self.l4(z)

        if self.skip == 1:
            z = 1.2 * torch.tanh(x + z)
        else:
            z = 1.2 * torch.tanh(z)

        # w = self.getfield()

        return z  # *w+x*(1-w)


class Operators(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.operators = nn.ModuleList([Operator() for i in range(N)])
        self.optim = torch.optim.Adam(
            self.parameters(), lr=1e-4
        )  # [p for o in operators for p in o.parameters()], lr=1e-4)
        self.stats = np.zeros(N)  # Summed usage pattern over the operators
        self.mu = 0

    def get_initial(self, LBS):
        return torch.zeros((LBS, 16, 32, 32)).to(DEVICE)

    def forward(self, zz, seq, dummy, detach=False):
        allzs = []

        for op in self.operators:
            allzs.append(op(zz, detach).unsqueeze(4))

        allzs = torch.cat(allzs, 4)

        return torch.sum(
            F.softmax(seq, dim=1).view(seq.size(0), 1, 1, 1, len(self.operators)) * allzs, 4
        )

    def setfield(self):
        for op in self.operators:
            op.setfield()


class ProgramSet(nn.Module):
    def __init__(self, N_images, N_seq, N_ops):
        super().__init__()

        self.programs = nn.Parameter(torch.zeros(N_seq, N_images, N_ops))
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-1)


# A program here is a sequence of vectors giving weights for each operator which sum to 1

# seq is Length x Batch x Ops
def runPrograms(ops, seq, detach=False):
    zz = ops.get_initial(seq.size(1))

    for i in range(seq.shape[0]):
        dummy = torch.zeros(1).to(DEVICE)
        dummy.requires_grad = True
        zz = checkpoint.checkpoint(lambda x, y, z: ops.forward(x, y, z, detach), zz, seq[i], dummy)

    return zz


def rankfit(x):
    npx = np.array(x)
    ranks = np.zeros(npx.shape[0])
    ranks[np.argsort(npx)] = np.arange(npx.shape[0])
    ranks = 2 * (ranks / npx.shape[0] - 0.5)

    return ranks


def optimize(ops, program, data, optimize_ops=True):
    ops.optim.zero_grad()
    program.optim.zero_grad()

    losses = []

    metabatches = data.shape[0] // BS
    if metabatches < 1:
        metabatches = 1

    for i in range(metabatches):
        x = torch.FloatTensor(data[i * BS : i * BS + BS]).to(DEVICE)
        results = runPrograms(ops, program.programs[:, i * BS : i * BS + BS])
        loss = torch.mean((x - results[:, 0:3, :, :]) ** 2)
        losses.append(loss.cpu().detach().item())

        # Entropy term, promotes sparseness
        p = torch.log_softmax(program.programs[:, i * BS : i * BS + BS], dim=2)
        H = -torch.sum(p * torch.exp(p), 2).mean()

        lloss = loss + SPARSE * H
        torch.autograd.backward(lloss)

    for p in ops.parameters():
        if p.grad is not None:
            p.grad.data.mul_(1.0 / metabatches)
        else:
            print(p.size())

    for p in program.parameters():
        if p.grad is not None:
            p.grad.data.mul_(1.0 / metabatches)
        else:
            print(p.size())

    if optimize_ops:
        ops.optim.step()
    program.optim.step()

    return np.mean(losses)


def attempt_exchange(i):
    global operators, programs

    N = len(operators[i].operators)
    newops1 = Operators(N=N + 1).to(DEVICE)
    newprog1 = ProgramSet(MBS, SEQLEN, N + 1).to(DEVICE)

    newops2 = Operators(N=N).to(DEVICE)
    newprog2 = ProgramSet(MBS, SEQLEN, N).to(DEVICE)

    newprog1.programs.data[:, :, :N] = programs[i].programs.data.clone()
    newprog2.programs.data = programs[i].programs.data.clone()

    for k in range(N):
        newops1.operators[k].copy_from(operators[i].operators[k])
        newops2.operators[k].copy_from(operators[i].operators[k])

    k = np.random.randint(len(operators))
    l = np.random.randint(len(operators[k].operators))

    newops1.operators[-1].copy_from(operators[k].operators[l])

    avg_logit = torch.mean(newprog2.programs, 2).detach()
    newprog1.programs.data[:, :, -1] = (
        avg_logit.data.clone() - 1
    )  # Make the new operator a little weaker to avoid immediate interference

    for step in range(EXCHANGE_STEPS):
        loss1 = optimize(newops1, newprog1, data_x[i * MBS : i * MBS + MBS])
        loss2 = optimize(newops2, newprog2, data_x[i * MBS : i * MBS + MBS])

    print("Exchange (new, original): %.6g, %.6g" % (loss1, loss2))
    if loss1 < loss2:  # Replace
        operators = operators[:i] + [newops1] + operators[i + 1 :]
        programs = programs[:i] + [newprog1] + programs[i + 1 :]

        return True
    else:
        # Lets still benefit from the epochs of training we performed
        operators = operators[:i] + [newops2] + operators[i + 1 :]
        programs = programs[:i] + [newprog2] + programs[i + 1 :]

        return False


def attempt_delete(i, j):
    global operators, programs

    N = len(operators[i].operators)
    newops1 = Operators(N=N - 1).to(DEVICE)
    newprog1 = ProgramSet(MBS, SEQLEN, N - 1).to(DEVICE)

    newops2 = Operators(N=N).to(DEVICE)
    newprog2 = ProgramSet(MBS, SEQLEN, N).to(DEVICE)

    newprog1.programs.data[:, :, :j] = programs[i].programs.data[:, :, :j].clone()
    newprog1.programs.data[:, :, j:] = programs[i].programs.data[:, :, j + 1 :].clone()

    newprog2.programs.data = programs[i].programs.data.clone()

    for k in range(N):
        if k < j:
            newops1.operators[k].copy_from(operators[i].operators[k])
        elif k > j:
            newops1.operators[k - 1].copy_from(operators[i].operators[k])
        newops2.operators[k].copy_from(operators[i].operators[k])

    for step in range(DELETE_STEPS):
        loss1 = optimize(newops1, newprog1, data_x[i * MBS : i * MBS + MBS])
        loss2 = optimize(newops2, newprog2, data_x[i * MBS : i * MBS + MBS])

    print("Delete (new, original): %.6g, %.6g" % (loss1, loss2))
    if loss1 < loss2 * 1.05:  # Replace if within 5%
        operators = operators[:i] + [newops1] + operators[i + 1 :]
        programs = programs[:i] + [newprog1] + programs[i + 1 :]

        return True
    else:
        # Lets still benefit from the epochs of training we performed
        operators = operators[:i] + [newops2] + operators[i + 1 :]
        programs = programs[:i] + [newprog2] + programs[i + 1 :]

        return False


###################
# Initializations #
###################

""" Load the relevant datasets """

cifar = CIFAR100(CIFAR_DIR, download=True)

data_x = []
data_y = []

for x, y in cifar:
    data_x.append((np.array(x) - 128.0) / 128.0)
    data_y.append(y)

data_x = np.array(data_x)
data_y = np.array(data_y)
data_x = data_x.transpose(0, 3, 1, 2)

mnist = MNIST(MNIST_DIR, download=True)

data_mn = []

for x, y in mnist:
    data_mn.append((np.array(x) - 128.0) / 128.0)

data_mn = np.array(data_mn)[:, np.newaxis, :, :].repeat(3, axis=1)
data_mn = np.pad(data_mn, [[0, 0], [0, 0], [2, 2], [2, 2]], constant_values=-1)

""" Create operators """

operators = [Operators(N=N_OPS).to(DEVICE) for i in range(N_POP)]
programs = [ProgramSet(MBS, SEQLEN, N_OPS).to(DEVICE) for i in range(N_POP)]

tr_err = []
exchanges = []
exchange_epochs = []
deletions = []
opcount = []
opstats = []

######################
# Main training loop #
######################

for epoch in range(1501):
    #########################################################
    # Log test set performance                              #
    #                                                       #
    # This is expensive so we only do it for one repertoire #
    #########################################################

    if epoch % 100 == 0:
        x1 = data_x[:TEST_BS].transpose(0, 2, 3, 1)
        x2 = data_x[-TEST_BS:].transpose(0, 2, 3, 1)
        x3 = data_mn[:TEST_BS].transpose(0, 2, 3, 1)

        results = runPrograms(operators[0], programs[0].programs)

        results = results.cpu().detach().numpy().transpose(0, 2, 3, 1)
        testset1 = ProgramSet(TEST_BS, SEQLEN, len(operators[0].operators)).to(DEVICE)
        testset2 = ProgramSet(TEST_BS, SEQLEN, len(operators[0].operators)).to(DEVICE)

        test1_curve = []
        test2_curve = []

        for step in range(TEST_STEPS):
            loss = optimize(operators[0], testset1, data_x[-TEST_BS:], optimize_ops=False)
            test1_curve.append(loss)

        results1 = runPrograms(operators[0], testset1.programs)
        results1 = results1.cpu().detach().numpy().transpose(0, 2, 3, 1)

        for step in range(TEST_STEPS):
            loss = optimize(operators[0], testset2, data_mn[:TEST_BS], optimize_ops=False)
            test2_curve.append(loss)

        results2 = runPrograms(operators[0], testset2.programs)
        results2 = results2.cpu().detach().numpy().transpose(0, 2, 3, 1)

        np.savetxt("%s/op0_epoch%d_cifar_err.txt" % (data_dir, epoch), test1_curve)
        np.savetxt("%s/op0_epoch%d_mnist_err.txt" % (data_dir, epoch), test2_curve)

        plt.clf()
        for j in range(16):
            plt.subplot(12, 8, 1 + j % 4 + 8 * (j // 4))
            if j % 4 == 0:
                plt.ylabel("Train")
            plt.imshow(0.5 + 0.5 * results[j, :, :, 0:3])
            plt.xticks([])
            plt.yticks([])
            if j < 4:
                plt.title("Reconstruction")

        for j in range(16):
            plt.subplot(12, 8, 1 + 32 + j % 4 + 8 * (j // 4))
            if j % 4 == 0:
                plt.ylabel("Test-CIFAR")
            plt.imshow(0.5 + 0.5 * results1[j, :, :, 0:3])
            plt.xticks([])
            plt.yticks([])

        for j in range(16):
            plt.subplot(12, 8, 1 + 64 + j % 4 + 8 * (j // 4))
            if j % 4 == 0:
                plt.ylabel("Test-MNIST")
            plt.imshow(0.5 + 0.5 * results2[j, :, :, 0:3])
            plt.xticks([])
            plt.yticks([])

        for j in range(16):
            plt.subplot(12, 8, 5 + j % 4 + 8 * (j // 4))
            if j < 4:
                plt.title("Reference")
            plt.imshow(0.5 + 0.5 * x1[j, :, :, 0:3])
            plt.xticks([])
            plt.yticks([])

        for j in range(16):
            plt.subplot(12, 8, 5 + 32 + j % 4 + 8 * (j // 4))
            plt.imshow(0.5 + 0.5 * x2[j, :, :, 0:3])
            plt.xticks([])
            plt.yticks([])

        for j in range(16):
            plt.subplot(12, 8, 5 + 64 + j % 4 + 8 * (j // 4))
            plt.imshow(0.5 + 0.5 * x3[j, :, :, 0:3])
            plt.xticks([])
            plt.yticks([])

        plt.gcf().set_size_inches((16, 24))
        plt.gcf().set_facecolor("white")
        plt.savefig("%s/op0_epoch%d.png" % (data_dir, epoch))
        torch.save(operators, open("%s/operator_ckpt%d.torch" % (ckpt_dir, epoch), "wb"))
        torch.save(programs, open("%s/program_ckpt%d.torch" % (ckpt_dir, epoch), "wb"))

    op_errs = []

    # Exchange mechanism
    # Every 20 steps, attempt to add a new operator and delete an old one, keeping between N_OPS and 2*N_OPS
    # When deleting, focus on the least-used

    if epoch % 20 == 0:
        n_exch = 0
        n_del = 0
        n_ops = 0

        for i in range(len(operators)):
            if len(operators[i].operators) < MAX_OPS:
                result = attempt_exchange(i)

                if result:
                    n_exch += 1

            if len(operators[i].operators) > N_OPS:
                result = attempt_delete(i, np.random.randint(len(operators[i].operators)))
                if result:
                    n_del += 1

            n_ops += len(operators[i].operators) / N_POP

        exchange_epochs.append(epoch)
        exchanges.append(n_exch)
        deletions.append(n_del)
        opcount.append(n_ops)

        np.savetxt(
            "%s/exchanges.txt" % (data_dir),
            np.concatenate(
                [
                    np.array(exchange_epochs)[:, np.newaxis],
                    np.array(exchanges)[:, np.newaxis],
                    np.array(deletions)[:, np.newaxis],
                    np.array(opcount)[:, np.newaxis],
                ],
                axis=1,
            ),
        )
    lstats = []
    i = 0

    # Training mechanism
    for op, prog in zip(operators, programs):
        op.setfield()
        op.optim.zero_grad()
        prog.optim.zero_grad()

        errs = []
        for j in range(MBS // BS):
            datarange = np.arange(i * MBS + j * BS, i * MBS + j * BS + BS)

            x = torch.FloatTensor(data_x[datarange]).to(DEVICE)
            results = runPrograms(op, prog.programs[:, j * BS : j * BS + BS])
            loss = torch.mean((x - results[:, 0:3, :, :]) ** 2)
            torch.autograd.backward(loss)
            errs.append(loss.cpu().detach().item())
        op_errs.append(np.mean(errs))

        for p in op.parameters():
            if p.grad is not None:
                p.grad.data.mul_(BS / MBS)
            else:
                print(p.size())

        for p in prog.parameters():
            if p.grad is not None:
                p.grad.data.mul_(BS / MBS)
            else:
                print(p.size())

        op.optim.step()
        prog.optim.step()
        i += 1

        stats = torch.softmax(prog.programs, dim=2).sum(1).mean(0).cpu().detach().numpy()
        lstats.append(stats)

    opstats.append(lstats)
    for i in range(N_POP):
        f = open("%s/op%d_stats.txt" % (data_dir, i), "w")

        maxops = np.max([len(opstats[j][i]) for j in range(len(opstats))])

        for j in range(len(opstats)):
            for k in range(maxops):
                if k < len(opstats[j][i]):
                    f.write("%.6g " % (opstats[j][i][k]))
                else:
                    f.write("0 ")
            f.write("\n")

        f.close()

    tr_err.append(np.array(op_errs))
    np.savetxt("%s/training_errors.txt" % (data_dir), np.array(tr_err))
