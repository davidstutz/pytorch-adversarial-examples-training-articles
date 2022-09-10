import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse
import common.summary
import common.train
import torch
import torch.utils.tensorboard
import torchvision
import datetime


# see https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5)
        self.fc1 = torch.nn.Linear(64 * 5 * 5, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Main:
    def __init__(self, args=None):
        """
        Initialize.

        :param args: optional arguments if not to use sys.argv
        :type args: [str]
        """

        self.args = None
        """ Arguments of program. """

        parser = self.get_parser()
        if args is not None:
            self.args = parser.parse_args(args)
        else:
            self.args = parser.parse_args()

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=128, shuffle=True, num_workers=2)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=128, shuffle=False, num_workers=2)

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Train a model with monitoring.')
        parser.add_argument('--directory', type=str, default='./checkpoints/')
        parser.add_argument('--no-cuda', action='store_false', dest='cuda', default=True, help='do not use cuda')

        return parser

    def main(self):
        """
        Main.
        """

        dt = datetime.datetime.now()
        # writer = common.summary.SummaryPickleWriter('%s/logs/%s/' % (self.args.directory, dt.strftime('%d%m%y%H%M%S')), max_queue=100)
        writer = torch.utils.tensorboard.SummaryWriter('%s/logs/%s/' % (self.args.directory, dt.strftime('%d%m%y%H%M%S')), max_queue=100)

        self.model = Net()
        if self.args.cuda:
            self.model = self.model.cuda()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainloader), gamma=0.9)
        trainer = common.train.NormalTraining(self.model, self.trainloader, self.testloader, optimizer, scheduler,
                                              writer=writer, cuda=self.args.cuda)
        #trainer.summary_histograms = True

        epochs = 50
        for epoch in range(0, epochs):
            trainer.step(epoch)
            writer.flush()

        model_file = '%s/classifier.pth.tar' % self.args.directory
        torch.save(self.model.state_dict(), model_file)


if __name__ == '__main__':
    program = Main()
    program.main()