from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, deploy')
        self.parser.add_argument('--criterion', type=str, default='clinical', help='clinical or median or input')
        self.parser.add_argument('--threshold', type=float, default=0.5, help='threshold for high-low manual setting')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=40, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=40, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--weight_decay', type=float, default=0, help='weight decay ratio')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--input_seq', type=list, default=['T1GD', 'T2'], help='input sequences (T1GD, T2, FLAIR, T1, ADC, DWI, SWI)')
        self.parser.add_argument('--roisize', type=int, default=16, help='size of the roi around bx location')
        self.parser.add_argument('--target', type=str, default='Ki67', help='target for prediction could be Ki67, SOX2, or abnormal')
        self.parser.add_argument('--lambda_u', type=float, default=0.0, help='weight for unlabeled loss')
        self.parser.add_argument('--class_weight', action='store_true', help='if specified, use class ratio to balance sample contribution.')
        self.parser.add_argument('--rotate', action='store_true', help='if specified, use rotation in augmentation.')
        self.parser.add_argument('--pool_size', type=int, default=10, help='the size of image buffer that stores previously generated images')

        self.isTrain = True
