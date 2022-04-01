from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='test', help='train, validation, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        self.parser.add_argument('--target', type=str, default='Ki67 LI', help='target for prediction (ki67, sox2, or cd68')
        self.parser.add_argument('--roisize', type=int, default=16, help='size of the roi around bx location')
        self.parser.add_argument('--input_seq', type=list, default=['T1GD', 'T2'], help='sequences that are used for training')
        self.parser.add_argument('--criterion', type=str, default='input', help='clinical or median or input')


        self.isTrain = False
