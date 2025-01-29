import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse

class MLP(nn.Module):
    def __init__(self, width=512, output_scale=1, weight_scale=5, dropout=0):
        super(MLP, self).__init__()
        self.nonlin = nn.Tanh()
        self.output_scale = output_scale
        self.l1 = nn.Linear(784,width)
        self.l2 = nn.Linear(width,width)
        self.l3 = nn.Linear(width,width)
        self.l4 = nn.Linear(width,10)
        if dropout > 0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = nn.Identity()

        for w, l in zip((784,width, width,width), [self.l1, self.l2, self.l3, self.l4]):
            nn.init.normal_(l.weight,std=weight_scale/np.sqrt(w))

    def forward(self, x):
        x = self.nonlin(self.l1(x))
        x = self.nonlin(self.l2(x))
        x = self.nonlin(self.l3(x))
        x = self.output_scale * self.l4(self.drop(x))
        return x

def get_loader(train, data_cnt=1000, num_classes=10, batch_size=128, target_scale=1, input_scale=1):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the image
    ])

    # Load MNIST dataset
    dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)

    # Randomly select 1000 datapoints from the training set
    indices = np.random.choice(len(dataset), size=data_cnt, replace=False)
    subset_images = input_scale * torch.stack([dataset[i][0] for i in indices])
    subset_labels = target_scale * torch.eye(num_classes)[torch.tensor([dataset[i][1] for i in indices])].float()

    # Dataset with one-hot vectors
    dataset = TensorDataset(subset_images, subset_labels)

    return DataLoader(dataset, batch_size=128, shuffle=train)

def compute_accuracy(outputs, labels):
    with torch.no_grad():
        pred_classes = torch.argmax(outputs, dim=1)  # Get index of max value (predictions)
        true_classes = torch.argmax(labels, dim=1)   # Get index of max value (true labels)

        correct = (pred_classes == true_classes).sum().detach().item()  # Count correct predictions

    return correct, labels.shape[0]

def train(model, loader, optimizer, criterion):
    tr_acc = 0
    cnt = 0
    model.train()
    device = list(model.parameters())[0].device
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
        logit = model(x)
        loss = criterion(model(x),y)
        loss.backward()
        optimizer.step()
        a, c = compute_accuracy(logit, y)
        tr_acc += a
        cnt += c
    return tr_acc/cnt

def test(model, loader):
    te_acc = 0
    cnt = 0
    model.train()
    device = list(model.parameters())[0].device
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
        logit = model(x)
        a, c = compute_accuracy(logit, y)
        te_acc += a
        cnt += c
    return te_acc/cnt

def run(args, name, weight_scale, target_scale, input_scale, dropout, output_scale):
    trs = []
    tes = []
    train_loader = get_loader(True, target_scale=target_scale, input_scale=input_scale)
    test_loader = get_loader(False, data_cnt=10000, batch_size=10000, target_scale=target_scale, input_scale=input_scale)
    model = MLP(output_scale=output_scale, weight_scale=weight_scale,dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss(reduction='mean')
    for epoch in range(args.epochs):
        trs.append(test(model, train_loader))
        tes.append(test(model, test_loader))
        train(model, train_loader, optimizer, criterion)
        print('epoch: ', epoch, ' train: ',trs[-1], ' test: ', tes[-1])
    np.save(f'./data/grokking_{name}', np.stack([np.array(trs), np.array(tes)]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="epochs", type=int, default=2000)
    parser.add_argument("-l", "--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("-w", "--weight_decay", help="weight decay", type=float, default=1e-4)
    parser.add_argument("-f", "--subfig", help="sub_fig", type=int, default=1)
    args = parser.parse_args()

    subfig_arr =[
        {'name': 'a', 'weight_scale': 5, 'target_scale' : 3, 'input_scale': 1, 'dropout':0, 'output_scale':1 },
        {'name': 'b', 'weight_scale': 1, 'target_scale': 3, 'input_scale': 1, 'dropout': 0, 'output_scale': 1},
        {'name': 'c', 'weight_scale': 5, 'target_scale': 30, 'input_scale': 1, 'dropout': 0, 'output_scale': 1},
        {'name': 'd', 'weight_scale': 5, 'target_scale': 3, 'input_scale': 0.01, 'dropout': 0, 'output_scale': 1},
        {'name': 'e', 'weight_scale': 5, 'target_scale': 3, 'input_scale': 1, 'dropout': 0.6, 'output_scale': 1},
        {'name': 'f', 'weight_scale': 5, 'target_scale': 3, 'input_scale': 1, 'dropout': 0, 'output_scale': 0.1},
    ]
    run(args, **(subfig_arr[args.subfig]))