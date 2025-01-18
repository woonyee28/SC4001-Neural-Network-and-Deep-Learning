#
# Tutorial 4, Question 3b
#

import torch
from torch import nn

import numpy as np
import pylab as plt
import multiprocessing as mp



# parameters
no_epochs = 5000

seed = 100
np.random.seed(seed)
torch.manual_seed(seed)


# generate data
X = np.zeros((9*9, 2)).astype(np.float32)
p = 0
for i in np.arange(-1, 1.001, 0.25):
    for j in np.arange(-1, 1.001, 0.25):
        X[p] = [i, j]
        p += 1
        
np.random.shuffle(X)
Y = np.zeros((9*9, 1)).astype(np.float32)
Y[:,0] = 0.8*X[:,0]**2 - X[:,1]**3 + 2.5*X[:,0]*X[:,1]

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_stack = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )

    def forward(self, x):
        logits = self.relu_stack(x)
        return logits

def train_loop(X, Y, model, loss_fn, optimizer):
    
    pred = model(X)
    loss = loss_fn(pred, Y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    return loss.item()

model = FFN()

loss_fn = nn.MSELoss()


def my_train(rate):
    
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    
    train_loss_ = []
    for epoch in range(no_epochs):
        train_loss = train_loop(torch.tensor(X), torch.tensor(Y), model, loss_fn, optimizer)
        train_loss_.append(train_loss)
               
    return(train_loss_)


def main():

    rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    no_threads = mp.cpu_count()
    p = mp.Pool(processes = no_threads)
    costs = p.map(my_train, rates)

    plt.figure(1)
    for r in range(len(rates)):
      plt.plot(range(no_epochs), costs[r], label='lr = {}'.format(rates[r]))

    plt.xlabel('iterations')
    plt.ylabel('mean square error')
    plt.title('gradient descent learning')
    plt.legend()
    plt.savefig('./figures/t4q3b_1.png')

#    plt.show()


if __name__ == '__main__':
  main()

