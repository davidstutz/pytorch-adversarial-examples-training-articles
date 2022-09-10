import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse
import attacks
import numpy
import attacks.norms
import attacks.projections
import attacks.initializations
import attacks.objectives
import common.state
import common.test
import common.utils
import common.eval
import torchvision
import torch.utils.data


class CleanDataset(torch.utils.data.Dataset):
    """
    General, clean dataset used for training, testing and attacking.
    """

    def __init__(self, images, labels, indices=None):
        """
        Constructor.

        :param images: images/inputs
        :type images: str or numpy.ndarray
        :param labels: labels
        :type labels: str or numpy.ndarray
        :param indices: indices
        :type indices: numpy.ndarray
        :param resize: resize in [channels, height, width
        :type resize: resize
        """

        self.images_file = None
        """ (str) File images were loaded from. """

        self.labels_file = None
        """ (str) File labels were loaded from. """

        if isinstance(images, str):
            self.images_file = images
            images = common.utils.read_hdf5(self.images_file)
            print('read %s' % self.images_file)
        if not images.dtype == numpy.float32:
            images = images.astype(numpy.float32)

        if isinstance(labels, str):
            self.labels_file = labels
            labels = common.utils.read_hdf5(self.labels_file)
            print('read %s' % self.labels_file)
        labels = numpy.squeeze(labels)
        if not labels.dtype == int:
            labels = labels.astype(int)

        assert isinstance(images, numpy.ndarray)
        assert isinstance(labels, numpy.ndarray)
        assert images.shape[0] == labels.shape[0]

        if indices is None:
            indices = range(images.shape[0])
        assert numpy.min(indices) >= 0
        assert numpy.max(indices) < images.shape[0]

        images = images[indices]
        labels = labels[indices]

        self.images = images
        """ (numpy.ndarray) Inputs. """

        self.labels = labels
        """ (numpy.ndarray) Labels. """

        self.indices = indices
        """ (numpy.ndarray) Indices. """

        self.transform = None
        """ (torchvision.transforms.Transform) Transforms. """

    def __getitem__(self, index):
        assert index < len(self)
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        assert self.images.shape[0] == self.labels.shape[0]
        return self.images.shape[0]

    def __add__(self, other):
        return torch.utils.data.ConcatDataset([self, other])


class RandomTestSet(CleanDataset):
    def __init__(self, N, size):
        test_images_file = 'random_images.h5'
        test_labels_file = 'random_labels.h5'

        if not os.path.exists(test_images_file):
            test_images = numpy.random.uniform(0, 1, size=[N] + list(size))
            common.utils.write_hdf5(test_images_file, test_images)

        if not os.path.exists(test_labels_file):
            test_labels = numpy.random.randint(0, 9, size=(N, 1))
            common.utils.write_hdf5(test_labels_file, test_labels)

        super(RandomTestSet, self).__init__(test_images_file, test_labels_file)


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

        test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=128, shuffle=False, num_workers=2)

        random_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.randomset = RandomTestSet(1000, [32, 32, 3])
        self.randomset.transform = random_transform
        self.randomloader = torch.utils.data.DataLoader(self.randomset, batch_size=128, shuffle=False, num_workers=2)

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Compute distal adversarial examples.')
        parser.add_argument('--iterations', type=int, default=10)
        parser.add_argument('--attempts', type=int, default=1)
        parser.add_argument('--model_file', type=str)
        parser.add_argument('--no-cuda', action='store_false', dest='cuda', default=True, help='do not use cuda')

        return parser

    def main(self):
        """
        Main.
        """

        state = common.state.State.load(self.args.model_file)
        print('read %s' % self.args.model_file)

        model = state.model
        model.eval()
        print(model)

        if self.args.cuda:
            model = model.cuda()

        epsilon = 0.03
        attack = attacks.BatchGradientDescent()
        attack.max_iterations = self.args.iterations
        attack.base_lr = 0.005
        attack.momentum = 0.9
        attack.c = 0
        attack.lr_factor = 1
        attack.normalized = True
        attack.backtrack = False
        attack.initialization = attacks.initializations.SequentialInitializations([
            attacks.initializations.LInfUniformNormInitialization(epsilon),
            attacks.initializations.SmoothInitialization(),
        ])
        attack.projection = attacks.projections.SequentialProjections([attacks.projections.LInfProjection(epsilon), attacks.projections.BoxProjection()])
        attack.norm = attacks.norms.LInfNorm()

        objective = attacks.objectives.UntargetedF0Objective(loss=common.torch.max_p_loss)  # max_log_loss
        probabilities = common.test.test(model, self.testloader, cuda=self.args.cuda)
        _, adversarial_probabilities, adversarial_errors = common.test.attack(model, self.randomloader, attack, objective, attempts=self.args.attempts, cuda=self.args.cuda)

        labels = self.testset.targets
        adversarial_eval = common.eval.DistalEvaluation(probabilities, adversarial_probabilities, labels)
        print('distal adversarial examples: false positive rate at 99%%TPR: %g' % adversarial_eval.fpr_at_99tpr())


if __name__ == '__main__':
    program = Main()
    program.main()