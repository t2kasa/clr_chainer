import argparse
import os
import random
from datetime import datetime

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import training
from chainer.backends import cuda
from chainer.iterators import SerialIterator
from chainer.training import extensions

from clr.training.extensions.clr import CLR


def ensure_reproducibility(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if cuda.available:
        cuda.cupy.random.seed(seed)
    chainer.global_config.cudnn_deterministic = True


def build_mlp(n_units, n_out):
    return chainer.Sequential(L.Linear(None, n_units), F.relu,
                              L.Linear(None, n_units), F.relu,
                              L.Linear(None, n_out))


def train_main(args, out_dir, policy):
    model = L.Classifier(build_mlp(args.unit, 10))
    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.SGD().setup(model)

    batch_size = args.batch_size
    train, test = chainer.datasets.get_mnist()
    train_iter = SerialIterator(train, batch_size)
    test_iter = SerialIterator(test, batch_size, repeat=False, shuffle=False)

    updater = training.updaters.StandardUpdater(train_iter, optimizer,
                                                device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=out_dir)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))

    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    trainer.extend(extensions.LogReport())

    iter_per_epoch = len(train) // args.batch_size
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['lr'], 'iteration', trigger=(iter_per_epoch, 'iteration'),
            file_name='lr.png'))
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'], 'epoch',
            file_name='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'], 'epoch',
            file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'lr', 'elapsed_time']))

    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.observe_lr())
    # Apply cyclical learning rate
    trainer.extend(CLR('lr', (0.01, 0.1), 2 * iter_per_epoch, policy))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


def main():
    ensure_reproducibility()
    policy_names = list(CLR.policy_choices.keys())

    parser = argparse.ArgumentParser(description='Cyclic Learning Rate Example')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--policy', type=str, default=policy_names[0],
                        choices=policy_names, help='CLR policy')
    args = parser.parse_args()

    out_dir = os.path.join(
        'results',
        '{}_policy={}'.format(datetime.now().strftime('%Y%m%d_%H%M%S'),
                              args.policy))
    train_main(args, out_dir, args.policy)


if __name__ == '__main__':
    main()
