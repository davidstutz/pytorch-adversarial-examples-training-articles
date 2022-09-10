import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse
import attacks
import numpy
import attacks.objectives
import common.state
import common.test
import common.eval
import common.mask
import common.plot
import torchvision
import torch.utils.data
from matplotlib import pyplot as plt


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

        test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.adversarialset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        self.adversarialset = torch.utils.data.Subset(self.adversarialset, range(0, 1000))
        self.adversarialloader = torch.utils.data.DataLoader(self.adversarialset, batch_size=128, shuffle=False, num_workers=2)

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Compute adversarial patches or frames.')
        parser.add_argument('--attack', type=str, default='patches')
        parser.add_argument('--iterations', type=int, default=10)
        parser.add_argument('--attempts', type=int, default=5)
        parser.add_argument('--size', type=int, default=8)
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

        attack = attacks.BatchPatches()
        img_shape = (32, 32)
        mask_dims = (self.args.size, self.args.size)
        frame_size = self.args.size
        if self.args.attack == 'patches':
            mask_gen = common.mask.PatchGenerator(img_shape, mask_dims, exclude_list=numpy.array([[12, 12, 8, 8]]))
        elif self.args.attack == 'frames':
            mask_gen = common.mask.FrameGenerator(img_shape, frame_size)
        else:
            raise ValueError('Choose "patches" or "frames" as --attack.')
        attack.mask_gen = mask_gen
        attack.base_lr = 0.05
        attack.max_iterations = self.args.iterations

        objective = attacks.objectives.UntargetedF0Objective()
        probabilities = common.test.test(model, self.testloader, cuda=self.args.cuda)
        adversarial_perturbations, adversarial_probabilities, adversarial_errors = common.test.attack(model, self.adversarialloader, attack, objective, attempts=self.args.attempts, cuda=self.args.cuda)

        for _, (images, _) in enumerate(self.testloader):
            break

        images = images.numpy()
        # Take first attempt adversarial patches only.
        n_images = min(adversarial_perturbations.shape[1], images.shape[0])
        adversarial_perturbations = adversarial_perturbations[0][:n_images]
        images = images[:n_images]
        images[adversarial_perturbations > 0] = adversarial_perturbations[adversarial_perturbations > 0]

        images = images.transpose((0, 2, 3, 1))
        common.plot.mosaic(images)
        plt.savefig('adversarial_patches.png', dpi=300)

        labels = numpy.array(self.testset.targets)
        eval = common.eval.CleanEvaluation(probabilities, labels)
        adversarial_eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels)

        print('error: %g' % eval.test_error())
        print('adversarial error: %g' % adversarial_eval.robust_test_error())
        print('confidence-thresholded adversarial error (99%%TPR): %g' % adversarial_eval.robust_test_error_at_99tpr())
        print('threshold (99%%TPR): %.4f' % adversarial_eval.confidence_at_99tpr())


if __name__ == '__main__':
    program = Main()
    program.main()