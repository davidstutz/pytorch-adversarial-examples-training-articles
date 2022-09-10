import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse
import common.summary
import common.train
from common.log import log
from imgaug import augmenters as iaa
import torch
import torchvision


# see https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def find_incomplete_state_file(model_file):
    """
    State file.

    :param model_file: base state file
    :type model_file: str
    :return: state file of ongoing training
    :rtype: str
    """

    base_directory = os.path.dirname(os.path.realpath(model_file))
    file_name = os.path.basename(model_file)

    if os.path.exists(base_directory):
        state_files = []
        files = [os.path.basename(f) for f in os.listdir(base_directory) if os.path.isfile(os.path.join(base_directory, f))]

        for file in files:
            if file.find(file_name) >= 0 and file != file_name:
                state_files.append(file)

        if len(state_files) > 0:
            epochs = [state_files[i].replace(file_name, '').replace('.pth.tar', '').replace('.', '') for i in range(len(state_files))]
            epochs = [epoch for epoch in epochs if epoch.isdigit()]
            epochs = list(map(int, epochs))
            epochs = [epoch for epoch in epochs if epoch >= 0]

            if len(epochs) > 0:
                # list is not ordered by epochs!
                i = numpy.argmax(epochs)
                return os.path.join(base_directory, file_name + '.%d' % epochs[i])


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
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=4, shuffle=True, num_workers=2)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=4, shuffle=False, num_workers=2)

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

        def get_augmentation(crop=True, flip=True):
            augmenters = []
            if crop:
                augmenters.append(iaa.CropAndPad(
                    px=((0, 4), (0, 4), (0, 4), (0, 4)),
                    pad_mode='constant',
                    pad_cval=(0, 0),
                ))
            if flip:
                augmenters.append(iaa.Fliplr(0.5))

            return iaa.Sequential(augmenters)

        # writer = common.summary.SummaryPickleWriter('%s/logs/' % self.args.directory, max_queue=100)
        writer = torch.utils.tensorboard.SummaryWriter('%s/logs/' % self.args.directory, max_queue=100)

        crop = True
        flip = True

        epochs = 100
        snapshot = 10

        model_file = '%s/classifier.pth.tar' % self.args.directory
        incomplete_model_file = find_incomplete_state_file(model_file)
        load_file = model_file
        if incomplete_model_file is not None:
            load_file = incomplete_model_file

        start_epoch = 0
        if os.path.exists(load_file):
            self.model = Net()
            self.model.load_state_dict(torch.load(load_file))
        else:
            self.model = Net()

        if self.args.cuda:
            self.model = self.model.cuda()

        augmentation = get_augmentation(crop=crop, flip=flip)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.075, momentum=0.9)
        scheduler = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=len(self.trainloader),
                                                           gamma=0.97)
        trainer = common.train.NormalTraining(self.model, self.trainloader, self.testloader, optimizer, scheduler,
                                              augmentation=augmentation, writer=writer, cuda=self.args.cuda)

        for epoch in range(start_epoch, epochs):
            trainer.step(epoch)
            writer.flush()

            snapshot_model_file = '%s/classifier.pth.tar.%d' % (self.args.directory, epoch)
            torch.save(self.model.state_dict(), snapshot_model_file)

            previous_model_file = '%s/classifier.pth.tar.%d' % (self.args.directory, epoch - 1)
            if os.path.exists(previous_model_file) and (epoch - 1) % snapshot > 0:
                os.unlink(previous_model_file)

        previous_model_file = '%s/classifier.pth.tar.%d' % (self.args.directory, epoch - 1)
        if os.path.exists(previous_model_file) and (epoch - 1) % snapshot > 0:
            os.unlink(previous_model_file)

        torch.save(self.model.state_dict(), model_file)


if __name__ == '__main__':
    program = Main()
    program.main()