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
import common.eval
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

        parser = argparse.ArgumentParser(description='Evaluate adversarial robustness.')
        parser.add_argument('--attacks', type=str, default='pgd_linf')
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

        attack_names = self.args.attacks.split(',')
        for attack_name in attack_names:
            assert attack_name in [
                'pgd_linf', 'pgd_conf_linf', 'pgd_l2', 'pgd_conf_l2', 'pgd_l1', 'pgd_conf_l1',
                'aa_linf', 'aa_l2', 'aa_conf_linf', 'aa_conf_l2'
            ]

        labels = numpy.array(self.testset.targets)
        probabilities = common.test.test(model, self.testloader, cuda=self.args.cuda)
        eval = common.eval.CleanEvaluation(probabilities, labels)

        for attack_name in attack_names:
            if attack_name == 'pgd_linf':
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
                attack.max_iterations = 200
                attack.normalized = True
                attack.backtrack = False
                attack.c = 0
                attack.momentum = 0.9
                attempts = 50
                objective = attacks.objectives.UntargetedF0Objective()
            elif attack_name == 'pgd_l2':
                epsilon = 1
                attack = attacks.BatchGradientDescent()
                attack.norm = attacks.norms.L2Norm()
                attack.initialization = attacks.initializations.L2UniformNormInitialization(epsilon)
                attack.projection = attacks.projections.SequentialProjections([
                    attacks.projections.BoxProjection(),
                    attacks.projections.L2Projection(epsilon),
                ])
                attack.base_lr = 0.5
                attack.lr_factor = 1
                attack.max_iterations = 200
                attack.normalized = True
                attack.backtrack = False
                attack.c = 0
                attack.momentum = 0.9
                attempts = 50
                objective = attacks.objectives.UntargetedF0Objective()
            elif attack_name == 'pgd_l1':
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
                attack.max_iterations = 200
                attack.normalized = True
                attack.backtrack = False
                attack.c = 0
                attack.momentum = 0.9
                attemtps = 50
                objective = attacks.objectives.UntargetedF0Objective()
            elif attack_name == 'pgd_conf_linf':
                epsilon = 0.03
                attack = attacks.BatchGradientDescent()
                attack.norm = attacks.norms.LInfNorm()
                attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
                attack.projection = attacks.projections.SequentialProjections([
                    attacks.projections.BoxProjection(),
                    attacks.projections.LInfProjection(epsilon),
                ])
                attack.base_lr = 0.005
                attack.lr_factor = 1.1
                attack.max_iterations = 1000
                attack.normalized = True
                attack.backtrack = True
                attack.c = 0
                attack.momentum = 0.9
                attempts = 10
                objective = attacks.objectives.UntargetedF7PObjective()
            elif attack_name == 'pgd_conf_l2':
                epsilon = 1
                attack = attacks.BatchGradientDescent()
                attack.norm = attacks.norms.L2Norm()
                attack.initialization = attacks.initializations.L2UniformNormInitialization(epsilon)
                attack.projection = attacks.projections.SequentialProjections([
                    attacks.projections.BoxProjection(),
                    attacks.projections.L2Projection(epsilon),
                ])
                attack.base_lr = 0.05
                attack.lr_factor = 1.1
                attack.max_iterations = 1000
                attack.normalized = True
                attack.backtrack = True
                attack.c = 0
                attack.momentum = 0.9
                attempts = 10
                objective = attacks.objectives.UntargetedF7PObjective()
            elif attack_name == 'pgd_conf_l1':
                epsilon = 10
                attack = attacks.BatchGradientDescent()
                attack.norm = attacks.norms.L1Norm()
                attack.initialization = attacks.initializations.L1UniformNormInitialization(epsilon)
                attack.projection = attacks.projections.SequentialProjections([
                    attacks.projections.BoxProjection(),
                    attacks.projections.L1Projection(epsilon),
                ])
                attack.base_lr = 0.5
                attack.lr_factor = 1.1
                attack.max_iterations = 1000
                attack.normalized = True
                attack.backtrack = True
                attack.c = 0
                attack.momentum = 0.9
                attempts = 10
                objective = attacks.objectives.UntargetedF7PObjective()
            elif attack_name == 'aa_linf':
                attack = attacks.BatchAutoAttack()
                attack.epsilon = 0.03
                attack.version = 'standard'
                attack.norm = 'Linf'
                attempts = 1
                objective = attacks.objectives.UntargetedF0Objective()
            elif attack_name == 'aa_l2':
                attack = attacks.BatchAutoAttack()
                attack.epsilon = 1
                attack.version = 'standard'
                attack.norm = 'L2'
                attempts = 1
                objective = attacks.objectives.UntargetedF0Objective()
            elif attack_name == 'aa_conf_linf':
                attack = attacks.BatchConfidenceAutoAttack()
                attack.epsilon = 0.03
                attack.version = 'standard-conf'
                attack.norm = 'Linf'
                attempts = 1
                objective = attacks.objectives.UntargetedF7PObjective()
            elif attack_name == 'aa_conf_l2':
                attack = attacks.BatchConfidenceAutoAttack()
                attack.epsilon = 1
                attack.version = 'standard-conf'
                attack.norm = 'L2'
                attempts = 1
                objective = attacks.objectives.UntargetedF7PObjective()
            else:
                assert False

            _, adversarial_probabilities, adversarial_errors = common.test.attack(
                model, self.adversarialloader, attack, objective, attempts=attempts, cuda=self.args.cuda, progress=False)
            adversarial_eval = common.eval.AdversarialEvaluation(probabilities, adversarial_probabilities, labels)

            print('%s error: %g' % (attack_name, eval.test_error()))
            print('%s adversarial error: %g' % (attack_name, adversarial_eval.robust_test_error()))
            print('%s confidence-thresholded adversarial error (99%%TPR): %g' % (attack_name, adversarial_eval.robust_test_error_at_99tpr()))
            print('%s threshold (99%%TPR): %.4f' % (attack_name, adversarial_eval.confidence_at_99tpr()))


if __name__ == '__main__':
    program = Main()
    program.main()