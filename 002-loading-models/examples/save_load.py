import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import argparse
import models
import common.state


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

    def get_parser(self):
        """
        Get parser.

        :return: parser
        :rtype: argparse.ArgumentParser
        """

        parser = argparse.ArgumentParser(description='Save and load a model.')
        parser.add_argument('--architecture', type=str, default='resnet50')

        return parser

    def main(self):
        """
        Main.
        """

        N_class = 10
        resolution = [3, 32, 32]
        dropout = False
        architecture = self.args.architecture
        normalization = 'bn'

        if architecture == 'resnet18':
            model = models.ResNet(N_class, resolution, blocks=[2, 2, 2, 2], channels=64,
                                  normalization=normalization)
        elif architecture == 'resnet20':
            model = models.ResNet(N_class, resolution, blocks=[3, 3, 3], channels=64,
                                  normalization=normalization)
        elif architecture == 'resnet34':
            model = models.ResNet(N_class, resolution, blocks=[3, 4, 6, 3], channels=64,
                                  normalization=normalization)
        elif architecture == 'resnet50':
            model = models.ResNet(N_class, resolution, blocks=[3, 4, 6, 3], block='bottleneck', channels=64,
                                  normalization=normalization)
        elif architecture == 'wrn2810':
            model = models.WideResNet(N_class, resolution, channels=16, normalization=normalization,
                                      dropout=dropout)
        elif architecture == 'simplenet':
            model = models.SimpleNet(N_class, resolution, dropout=dropout,
                                    channels=64, normalization=normalization)
        else:
            assert False

        common.state.State.checkpoint('model.pth.tar', model)
        print('wrote model.pth.tar')

        state = common.state.State.load('model.pth.tar')
        print('read model.pth.tar')

        model = state.model
        print(model)


if __name__ == '__main__':
    program = Main()
    program.main()