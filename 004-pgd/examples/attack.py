import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse
import attacks
import attacks.norms
import attacks.projections
import attacks.initializations
import attacks.objectives
import common.state
import common.test
import torchvision
import torch.utils.data

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
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        testset = torch.utils.data.Subset(testset, range(0, 1000))
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Attack a model using PGD.')
        parser.add_argument('--attack', type=str, default='linf')
        parser.add_argument('--model_file', type=str, default='simplenet_bn.pth.tar')
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

        if self.args.attack == 'linf':
            epsilon = 0.03
            attack = attacks.BatchGradientDescent()
            attack.norm = attacks.norms.LInfNorm()
            attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
            attack.projection = attacks.projections.SequentialProjections([
                attacks.projections.BoxProjection(),
                attacks.projections.LInfProjection(epsilon),
            ])
            attack.base_lr = 0.05
            attack.lr_factor = 1
            attack.max_iterations = 7
            attack.normalized = True
            attack.backtrack = False
            attack.c = 0
            attack.momentum = 0.9
        elif self.args.attack == 'l2':
            epsilon = 0.5
            attack = attacks.BatchGradientDescent()
            attack.norm = attacks.norms.L2Norm()
            attack.initialization = attacks.initializations.L2UniformNormInitialization(epsilon)
            attack.projection = attacks.projections.SequentialProjections([
                attacks.projections.BoxProjection(),
                attacks.projections.L2Projection(epsilon),
            ])
            attack.base_lr = 0.5
            attack.lr_factor = 1
            attack.max_iterations = 7
            attack.normalized = True
            attack.backtrack = False
            attack.c = 0
            attack.momentum = 0.9
        elif self.args.attack == 'l1':
            epsilon = 10
            attack = attacks.BatchGradientDescent()
            attack.norm = attacks.norms.L1Norm()
            attack.initialization = attacks.initializations.L1UniformNormInitialization(epsilon)
            attack.projection = attacks.projections.SequentialProjections([
                attacks.projections.BoxProjection(),
                attacks.projections.L1Projection(epsilon),
            ])
            attack.base_lr = 15
            attack.lr_factor = 1
            attack.max_iterations = 20
            attack.normalized = True
            attack.backtrack = False
            attack.c = 0
            attack.momentum = 0.9
        else:
            assert False

        objective = attacks.objectives.UntargetedF0Objective()
        error = common.test.test(model, self.testloader, cuda=self.args.cuda)
        adversarial_error = common.test.attack(model, self.testloader, attack, objective, cuda=self.args.cuda)

        print('error: %g' % error)
        print('adversarial error: %g' % adversarial_error)


if __name__ == '__main__':
    program = Main()
    program.main()