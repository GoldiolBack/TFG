from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from make_ds import train_loader
from make_ds_test import test_loader


def show_tensor(x):
    im = transforms.ToPILImage()(x)
    im.save("asdf.png")
    image = Image.open(r"asdf.png")
    image.show()


class Net(nn.Module):
    def __init__(self, feature_size=6, kernel_size=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(feature_size, feature_size, kernel_size, stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(feature_size, 3, kernel_size, 1, 1)
        self.rBlock = ResBlock(feature_size, kernel_size)

    def forward(self, input10, input20, num_layers=6):
        sentinel = torch.cat((input10, input20), 1)
        x = sentinel
        x = self.conv1(x)
        x = F.relu(x)
        for i in range(num_layers):
            x = self.rBlock(x)
        x = self.conv2(x)
        x += input20
        return x


class ResBlock(nn.Module):
    def __init__(self, channels=3, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv3 = nn.Conv2d(6, 6, kernel_size, 1, 1)

    def forward(self, x, scale=0.1):
        tmp = self.conv3(x)
        tmp = F.relu(tmp)
        tmp = self.conv3(tmp)
        tmp = tmp * scale
        tmp += x
        return tmp


def train(args, model, device, optimizer, epoch):
    model.train()
    for batch_idx, (rs, gt) in enumerate(train_loader):
        print(f'batch {batch_idx+1}:')
        rs, gt = rs.to(device), gt.to(device)
        optimizer.zero_grad()
        output = model(gt, rs)
#        gt = gt.long()
        loss_function = nn.L1Loss()
#        loss = F.nll_loss(output, gt)
        loss = loss_function(output, gt)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(rs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for rs_test, gt_test in test_loader:
            rs_test, gt_test = rs_test.to(device), gt_test.to(device)
            output = model(gt_test, rs_test)
            show_tensor(output[0])
#            test_loss += F.nll_loss(output, gt_test, reduction='sum').item()  # sum up batch loss
            test_loss_function = nn.L1Loss(reduction='sum')
            test_loss = test_loss_function(output, gt_test).item()
            out = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += (out.cpu() == gt_test.cpu()).sum()
#            correct += pred.eq(gt_test.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch TFG Net')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, optimizer, epoch)
        test(args, model, device)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "net.pt")


if __name__ == '__main__':
    main()

