from argparse import ArgumentParser

def get_parser():
    parser = ArgumentParser(description='StyleGANv2')
    # General options.
    parser.add_argument('--exp_name', help='Experiment name', required=False,
                        default='default_name')
    parser.add_argument('--outdir', help='Where to save the results',
                        required=False, default='training-runs', metavar='DIR')
    parser.add_argument('--slurm_logdir', help='Where to save the logs from SLURM',
                        required=False, default='training-runs', metavar='DIR')
    parser.add_argument('--partition', help='Partition name for SLURM', required=False,
                        default='learnlab')
    parser.add_argument('--slurm_time', help='Time in minutes that an experiment runs in SLURM',
                        default=3200, type=int, metavar='INT')
    parser.add_argument('--gpus', help='Number of GPUs to use [default: 1]',
                        type=int, metavar='INT')
    parser.add_argument('--nodes', help='Number of nodes to use [default: 1]',
                        type=int, metavar='INT', default = 1)
    parser.add_argument('--snap', help='Snapshot interval [default: 50 ticks]',
                        type=int, metavar='INT')
    parser.add_argument('--seed', help='Random seed [default: 0]', type=int,
                        metavar='INT')
    parser.add_argument('--dry-run',
                         help='Print training options and exit', type=bool, metavar='BOOL')

    # Dataset.
    parser.add_argument('--data', help='Training data (directory or zip)',
                        metavar='PATH', required=False,
                        default='datasets/cocostuff_128.zip')
    parser.add_argument('--cond',
                        help='Train conditional model based on dataset labels [default: false]',
                        type=bool, metavar='BOOL')
    parser.add_argument('--subset',
                        help='Train with only N images [default: all]',
                        type=int, metavar='INT')
    parser.add_argument('--mirror',
                        help='Enable dataset x-flips [default: false]',
                        type=bool, metavar='BOOL')
    # Base config.
    parser.add_argument('--cfg', help='Base config [default: auto]',
                        choices=
                        ['auto', 'stylegan2', 'paper256', 'paper512',
                         'paper1024', 'cifar'])
    parser.add_argument('--gamma', help='Override R1 gamma', type=float)
    parser.add_argument('--kimg', help='Override training duration', type=int,
                        metavar='INT')
    parser.add_argument('--batch', help='Override batch size', type=int,
                        metavar='INT')
    parser.add_argument('--lrate', help='Override lrate', type=float)
    parser.add_argument('--num_channel_g', help='Override width of generator', type=int)
    parser.add_argument('--num_channel_d', help='Override width of discriminator', type=int)

    # Discriminator augmentation.
    parser.add_argument('--aug', help='Augmentation mode [default: ada]',
                        choices=['noaug', 'ada', 'fixed'])
    parser.add_argument('--p', help='Augmentation probability for --aug=fixed',
                        type=float)
    parser.add_argument('--target', help='ADA target value for --aug=ada',
                        type=float)
    parser.add_argument('--augpipe',
                        help='Augmentation pipeline [default: bgc]',
                        choices=['blit', 'geom', 'color', 'filter', 'noise',
                                 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc'])

    # Transfer learning.
    parser.add_argument('--resume', help='Resume training [default: noresume]',
                        metavar='PKL')
    parser.add_argument('--freezed', help='Freeze-D [default: 0 layers]',
                        type=int, metavar='INT')

    # Performance options.
    parser.add_argument('--fp32', help='Disable mixed-precision training',
                        type=bool, metavar='BOOL')
    parser.add_argument('--nhwc', help='Use NHWC memory format with FP16',
                        type=bool, metavar='BOOL')
    parser.add_argument('--nobench', help='Disable cuDNN benchmarking',
                        type=bool, metavar='BOOL')
    parser.add_argument('--allow-tf32',
                        help='Allow PyTorch to use TF32 internally', type=bool,
                        metavar='BOOL')
    parser.add_argument('--workers',
                        help='Override number of DataLoader workers', type=int,
                        metavar='INT')

    ## Experiment setup
    parser.add_argument('--slurm', help='Using SLURM to launch the experiment in a cluster',
                        type=bool, metavar='BOOL')
    return parser
#----------------------------------------------------------------------------