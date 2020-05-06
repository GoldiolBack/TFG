from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from create_dataset import train_ds, val_ds, test_ds
import skimage.metrics as skm
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, feature_size=10, kernel_size=3):
        super(Net, self).__init__()
        self.ups = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv1 = nn.Conv2d(feature_size, feature_size, kernel_size, stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(feature_size, 6, kernel_size, 1, 1)
        self.rBlock = ResBlock(feature_size, kernel_size)

    def forward(self, input10, input20, num_layers=6):
        upsamp20 = self.ups(input20)
        sentinel = torch.cat((input10, upsamp20), 1)
        x = sentinel
        x = self.conv1(x)
        x = F.relu(x)
        for i in range(num_layers):
            x = self.rBlock(x)
        x = self.conv2(x)
        x += upsamp20
        return x


class ResBlock(nn.Module):
    def __init__(self, channels=3, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv3 = nn.Conv2d(10, 10, kernel_size, 1, 1)

    def forward(self, x, scale=0.1):
        tmp = self.conv3(x)
        tmp = F.relu(tmp)
        tmp = self.conv3(tmp)
        tmp = tmp * scale
        tmp += x
        return tmp


def train(args, train_loader, model, device, optimizer, epoch):
    model.train()
    for batch_idx, (hr, lr, target) in enumerate(train_loader):
        print(f'batch {batch_idx+1}:')
        lr, hr, target = lr.to(device), hr.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(hr, lr)
#        gt = gt.long()
        loss_function = nn.L1Loss()
#        loss = F.nll_loss(output, gt)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(lr), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, test_loader, model, device):
    model.eval()
    test_loss = 0
    rmse = 0
    psnr = 0
    ssim = 0
    with torch.no_grad():
        for hr, lr, target in test_loader:
            lr, hr, target = lr.to(device), hr.to(device), target.to(device)
            output = model(hr, lr)
            test_loss_function = nn.L1Loss(reduction='sum')
            test_loss = test_loss_function(output, target).item()
            real = np.moveaxis(target.cpu().numpy(), 1, 3)
            predicted = np.moveaxis(output.cpu().numpy(), 1, 3)
            rmse += skm.normalized_root_mse(real, predicted)
            psnr += skm.peak_signal_noise_ratio(real, predicted, data_range=real.max()-real.min())
            for i in range(5):
                ssim += skm.structural_similarity(real[i], predicted[i], multichannel=True,
                                                  data_range=real.max() - real.min())

    test_loss /= len(test_loader.dataset)
    rmse /= len(test_loader.dataset)
    psnr /= len(test_loader.dataset)
    ssim /= len(test_loader.dataset)

    print('\nTest set: Average values --> Loss: {:.4f}, RMSE: ({:.2f}), PSNR: ({:.2f}dB),'
          ' SSIM: ({:.2f})\n'.format(test_loss, rmse, psnr, ssim))


def validation(args, val_loader, model, device):
    model.eval()
    val_loss = 0
    rmse = 0
    psnr = 0
    ssim = 0
    with torch.no_grad():
        for hr, lr, target in val_loader:
            lr, hr, target = lr.to(device), hr.to(device), target.to(device)
            output = model(hr, lr)
            val_loss_function = nn.L1Loss(reduction='sum')
            val_loss = val_loss_function(output, target).item()
            real = np.moveaxis(target.cpu().numpy(), 1, 3)
            predicted = np.moveaxis(output.cpu().numpy(), 1, 3)
            rmse += skm.normalized_root_mse(real, predicted)
            psnr += skm.peak_signal_noise_ratio(real, predicted, data_range=real.max()-real.min())
            for i in range(5):
                ssim += skm.structural_similarity(real[i], predicted[i], multichannel=True,
                                                  data_range=real.max() - real.min())

    val_loss /= len(val_loader.dataset)
    rmse /= len(val_loader.dataset)
    psnr /= len(val_loader.dataset)
    ssim /= len(val_loader.dataset)

    print('\nValidation set: Average values --> Loss: {:.4f}, RMSE: ({:.2f}), PSNR: ({:.2f}dB),'
          ' SSIM: ({:.2f})\n'.format(val_loss, rmse, psnr, ssim))

    np.savetxt('val_input.csv', (np.moveaxis(lr.cpu().numpy(), 1, 3)).reshape(5, 32*32*6), delimiter=',')
    np.savetxt('val_real.csv', real.reshape(5, 64*64*6), delimiter=',')
    np.savetxt('val_output.csv', predicted.reshape(5, 64*64*6), delimiter=',')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch TFG Net')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=5, metavar='N',
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
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    train_loader = DataLoader(train_ds.dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds.dataset, batch_size=args.test_batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=True)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.004)

    scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)
    model = model.type(dst_type=torch.float64)
    for epoch in range(1, args.epochs + 1):
        train(args, train_loader, model, device, optimizer, epoch)
        if epoch % 5 == 0:
            validation(args, val_loader, model, device)
        test(args, test_loader, model, device)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "net.pt")


if __name__ == '__main__':
    main()

