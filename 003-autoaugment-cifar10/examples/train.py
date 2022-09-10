import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse
import models
import common.autoaugment
from common.log import log
import common.summary
import common.train
import common.state
import common.test
import torch
import numpy
import torch.utils.tensorboard
import torchvision
import datetime


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

        cutout = 16
        mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
        # has to be tensor
        data_mean = torch.tensor(mean)
        # has to be tuple
        data_mean_int = []
        for c in range(data_mean.numel()):
            data_mean_int.append(int(255 * data_mean[c]))
        data_mean_int = tuple(data_mean_int)
        data_resolution = 32
        train_transform = torchvision.transforms.Compose([
            #common.autoaugment.Debug(),
            #torchvision.transforms.ToPILImage(),
            #common.autoaugment.Debug(),
            torchvision.transforms.RandomCrop(data_resolution, padding=int(data_resolution * 0.125), fill=data_mean_int),
            #common.autoaugment.Debug(),
            torchvision.transforms.RandomHorizontalFlip(),
            #common.autoaugment.Debug(),
            common.autoaugment.CIFAR10Policy(fillcolor=data_mean_int),
            #common.autoaugment.Debug(),
            torchvision.transforms.ToTensor(),
            #common.autoaugment.Debug(),
            common.autoaugment.CutoutAfterToTensor(n_holes=1, length=cutout, fill_color=data_mean),
            #common.autoaugment.Debug(),
        ])
        test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Train a model with AutoAugment and CutOut.')
        parser.add_argument('--architecture', type=str, default='resnet50')
        parser.add_argument('--directory', type=str, default='./checkpoints/')
        parser.add_argument('--no-cuda', action='store_false', dest='cuda', default=True, help='do not use cuda')

        return parser

    def checkpoint(self, model_file, model, epoch=None):
        """
        Save file and check to delete previous file.

        :param model_file: path to file
        :type model_file: str
        :param model: model
        :type model: torch.nn.Module
        :param epoch: epoch of file
        :type epoch: None or int
        """

        if epoch is not None:
            checkpoint_model_file = '%s.%d' % (model_file, epoch)
            common.state.State.checkpoint(checkpoint_model_file, model, self.optimizer, self.scheduler, epoch)
        else:
            epoch = self.epochs
            checkpoint_model_file = model_file
            common.state.State.checkpoint(checkpoint_model_file, model, self.optimizer, self.scheduler, epoch)

        previous_model_file = '%s.%d' % (model_file, epoch - 1)
        if os.path.exists(previous_model_file):
            os.unlink(previous_model_file)

    def main(self):
        """
        Main.
        """

        dt = datetime.datetime.now()
        # writer = common.summary.SummaryPickleWriter('%s/logs/%s/' % (self.args.directory, dt.strftime('%d%m%y%H%M%S')), max_queue=100)
        writer = torch.utils.tensorboard.SummaryWriter('%s/logs/%s/' % (self.args.directory, dt.strftime('%d%m%y%H%M%S')), max_queue=100)

        N_class = 10
        resolution = [3, 32, 32]
        dropout = False
        architecture = self.args.architecture
        normalization = 'bn'

        state = None
        start_epoch = 0
        self.epochs = 250

        model_file = '%s/model.pth.tar' % self.args.directory
        incomplete_model_file = find_incomplete_state_file(model_file)
        if os.path.exists(model_file):
            state = common.state.State.load(model_file)
            log('loaded %s' % model_file)
            self.model = state.model
            start_epoch = self.epochs
        elif incomplete_model_file is not None:
            state = common.state.State.load(incomplete_model_file)
            log('loaded %s' % incomplete_model_file)
            self.model = state.model
            start_epoch = state.epoch + 1
        else:
            if architecture == 'resnet18':
                self.model = models.ResNet(N_class, resolution, blocks=[2, 2, 2, 2], channels=64,
                                      normalization=normalization)
            elif architecture == 'resnet20':
                self.model = models.ResNet(N_class, resolution, blocks=[3, 3, 3], channels=64,
                                      normalization=normalization)
            elif architecture == 'resnet34':
                self.model = models.ResNet(N_class, resolution, blocks=[3, 4, 6, 3], channels=64,
                                      normalization=normalization)
            elif architecture == 'resnet50':
                self.model = models.ResNet(N_class, resolution, blocks=[3, 4, 6, 3], block='bottleneck', channels=64,
                                      normalization=normalization)
            elif architecture == 'wrn2810':
                self.model = models.WideResNet(N_class, resolution, channels=16, normalization=normalization,
                                          dropout=dropout)
            elif architecture == 'simplenet':
                self.model = models.SimpleNet(N_class, resolution, dropout=dropout,
                                         channels=64, normalization=normalization)
            else:
                assert False

        if self.args.cuda:
            self.model = self.model.cuda()

        print(self.model)
        lr = 0.05
        momentum = 0.9
        weight_decay = 0.0005
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, nesterov=True,
                                         weight_decay=weight_decay)

        if state is not None:
            # fine-tuning should start with fresh optimizer and learning rate
            self.optimizer.load_state_dict(state.optimizer)
            log('loaded optimizer')

        milestones = [2 * self.epochs // 5, 3 *self.epochs // 5, 4 *self.epochs // 5]
        lr_factor = 0.1
        batches_per_epoch = len(self.trainloader)
        self.scheduler = common.train.get_multi_step_scheduler(self.optimizer, batches_per_epoch=batches_per_epoch, milestones=milestones, gamma=lr_factor)
        if state is not None:
            # will lead to errors when fine-tuning pruned models
            self.scheduler.load_state_dict(state.scheduler)
            log('loaded scheduler')

        trainer = common.train.NormalTraining(self.model, self.trainloader, self.testloader, self.optimizer, self.scheduler,
                                              writer=writer, cuda=self.args.cuda)

        for epoch in range(start_epoch, self.epochs):
            trainer.step(epoch)
            writer.flush()
            self.checkpoint(model_file, self.model, epoch)
        self.checkpoint(model_file, self.model)

        self.model.eval()
        error = common.test.test(self.model, self.testloader, cuda=self.args.cuda)
        log('error: %g' % error)

if __name__ == '__main__':
    program = Main()
    program.main()