import time
import os
from functools import reduce
import argparse
import torch
import torch.nn as nn
try:
    from tensorboardX import SummaryWriter
except:
    pass

import util
import alexnet
import files
from util import TotalAverage, MovingAverage,accuracy
from data import get_standard_data_loader_pairs



class Probes(nn.Module):
    """Linear probing container."""
    def __init__(self, trunk, probed_layers, num_classes=1000):
        super(Probes, self).__init__()
        x = torch.zeros(2,3,int(args.crop_size),int(args.crop_size))
        num_classes = num_classes
        cnvs = [
                nn.MaxPool2d(6, stride=6, padding=3),
                nn.MaxPool2d(4, stride=4, padding=0),
                nn.MaxPool2d(3, stride=3, padding=1),
                nn.MaxPool2d(3, stride=3, padding=1),
                nn.MaxPool2d(2, stride=2, padding=0)]
        # yields sized of [9600, 9216, 9600, 9600,9216]
        self.probed_layers = probed_layers
        self.trunk = trunk
        self.probes = nn.ModuleList()
        self.deepest_layer_index = 0
        j=0
        layer_list = self.trunk.modules()
        for index, (name) in enumerate(list(layer_list)[0].children()):
            x = name.forward(x)
            print(f'Visiting layer {index: 3d}: {name} [shape: {x.shape}] ()')
            if index > max(self.probed_layers):
                break
            if index in self.probed_layers or name in self.probed_layers:
                self.deepest_layer_index = index
                # Downsampler
                x_volume = reduce(lambda x, y: x * y, x.shape[1:])
                downsampler = cnvs[j] #
                j+=1
                y = downsampler(x)
                y_volume = reduce(lambda x, y: x * y, y.shape[1:])

                # Linear classifier
                bn = nn.BatchNorm2d(y.shape[1], affine=False)
                predictor = nn.Conv2d(y.shape[1], num_classes, y.shape[2:4], bias=True)
                torch.nn.init.xavier_uniform_(predictor.weight, gain=1)
                torch.nn.init.constant_(predictor.bias, 0)
                # Probe
                self.probes.append(nn.Sequential(downsampler, bn, predictor))
                print(f"Attaching linear probe to layer {index: 3d}: "
                      f"{name}) with size reduction {x_volume} -> {y_volume} ({downsampler})")

    def forward(self, x):
        outputs = []
        for index, (name, layer) in enumerate(self.trunk.named_children()):
            x = layer.forward(x)
            probe_index = None
            if index in self.probed_layers:
                probe_index = self.probed_layers.index(index)
            elif name in self.probed_layers:
                probe_index = self.probed_layers.index(name)
            if probe_index is not None:
                y = self.probes[probe_index](x).squeeze()
                outputs += [y]
            if index == self.deepest_layer_index:
                break
        return outputs

    def lp_parameters(self):
        return self.probes.parameters()


def model_with_probes(model_path=None, which='Imagenet'):
    if which == 'Imagenet':
        nc = 1000
    elif which == 'Places':
        nc = 205
    state_dict = torch.load(model_path)['state_dict']
    # automatically determine size of last FC layer that is not needed anyhow.
    ncls = []
    for q in (state_dict.keys()):
        if 'top_layer' in q:
            if 'weight' in q:
                ncl = state_dict[q].shape[0]
                ncls.append(ncl)
    outs = ncls
    model = alexnet.alexnet(out=outs)
    model.load_state_dict(state_dict)
    layers = [1, 4, 7, 9, 11]  # because BN.
    util.search_absorb_bn(model)
    model = util.sequential_skipping_bn_cut(model)
    for relu in filter(lambda x: issubclass(x.__class__, nn.ReLU), model.children()):
        relu.inplace = False
    model = Probes(model, layers, num_classes=nc)
    return model


class LinearProbesOptimizer():
    def __init__(self):
        self.num_epochs = 36
        self.lr = 0.01
        def zheng_lr_schedule(epoch):
            if epoch < 10:
                return 1e-2
            elif epoch < 20:
                return 1e-3
            elif epoch < 30:
                return 1e-4
            else:
                return 1e-5
        self.lr_schedule = lambda epoch: zheng_lr_schedule(epoch)
        self.criterion = nn.CrossEntropyLoss()
        self.momentum = 0.9
        self.weight_decay = 1e-5
        self.nesterov = False
        self.validate_only = False
        self.resume = True
        self.checkpoint_dir = None
        self.writer = None

    def optimize(self, model, train_loader, val_loader=None, optimizer=None):
        """Perform full optimization."""
        # Initialize
        criterion = self.criterion
        metrics = {'train':[], 'val':[]}
        first_epoch = 0
        model_path = None

        # Send models to device
        criterion = criterion.to('cuda:0')
        model = model.to('cuda:0')

        # Get optimizer (after sending to device)
        if optimizer is None:
            optimizer = self.get_optimizer(model)

        # Resume from checkpoint
        if self.checkpoint_dir is not None:
            model_path = os.path.join(self.checkpoint_dir, 'model.pth')
            if self.resume:
                first_epoch, metrics = files.load_checkpoint(self.checkpoint_dir, model, optimizer)

        print(f"{self.__class__.__name__}:"
            f" epochs:{self.num_epochs}"
            f" momentum:{self.momentum}"
            f" weight_decay:{self.weight_decay}"
            f" nesterov:{self.nesterov}")

        # Perform epochs
        if not self.validate_only:
            for epoch in range(first_epoch, 1 if self.validate_only else self.num_epochs):
                print(optimizer)
                m = self.optimize_epoch(model, criterion, optimizer, train_loader, epoch, is_validation=False)
                metrics["train"].append(m)
                if epoch > 25:
                    if val_loader:
                        with torch.no_grad():
                            m = self.optimize_epoch(model, criterion, optimizer, val_loader, epoch, is_validation=True)
                            metrics["val"].append(m)
                files.save_checkpoint(self.checkpoint_dir, model, optimizer, metrics, epoch)
        else:
            print('only evaluating!', flush=True)
            with torch.no_grad():
                m = self.optimize_epoch(model, criterion, optimizer, val_loader, 0, is_validation=True)
                metrics["val"].append(m)
        print(f"Model optimization completed. Saving final model to {model_path}")
        files.save_model(model, model_path)

        return model, metrics

    def get_optimizer(self, model):
        return torch.optim.SGD(model.lp_parameters(), # <- all lp_parameters!
                               lr=self.lr_schedule(0),
                               momentum=self.momentum,
                               weight_decay=self.weight_decay,
                               nesterov=self.nesterov)

    def optimize_epoch(self, model, criterion, optimizer, loader, epoch, is_validation=False):
        top1 = []
        top5 = []
        loss_value = []
        for i in range(len(model.probes)):
            top1.append(TotalAverage())
            top5.append(TotalAverage())
            loss_value.append(TotalAverage())
        batch_time = MovingAverage(intertia=0.9)
        now = time.time()

        if is_validation is False:
            model.train()
            lr = self.lr_schedule(epoch)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            print(f"Starting epoch {epoch} with learning rate {lr}")
        else:
            model.eval()
        for iter, (input, label) in enumerate(loader):
            input = input.to('cuda:0')
            label = label.to('cuda:0')
            mass = input.size(0)
            total_loss = None
            if is_validation and args.tencrops:
                bs, ncrops, c, h, w = input.size()
                input_tensor = input.view(-1, c, h, w)
                input = input_tensor.cuda()
            else:
                input = input.cuda()

            predictions = model(input)
            if is_validation and args.tencrops:
                predictions = [torch.squeeze(p.view(bs, ncrops, -1).mean(1)) for p in predictions]
            for i, prediction in enumerate(predictions):
                loss = criterion(prediction, label)
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss = total_loss + loss
                top1_, top5_ = accuracy(prediction, label, topk=(1, 5))
                top1[i].update(top1_.item(), mass)
                top5[i].update(top5_.item(), mass)
                loss_value[i].update(loss.item(), mass)

            if is_validation is False:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            batch_time.update(time.time() - now)
            now = time.time()

        top1_str = 'top1 val' if is_validation else 'top1 train'
        top5_str = 'top5 val' if is_validation else 'top5 train'
        if self.writer is not None:
            self.writer.add_scalars(top1_str, {f"depth_{k+1}":top1[k].avg for k in range(len(model.probes))}, epoch)
            self.writer.add_scalars(top5_str, {f"depth_{k+1}":top5[k].avg for k in range(len(model.probes))}, epoch)
            self.writer.add_scalars('losses', {f"depth_{k+1}": loss_value[k].avg for k in range(len(model.probes))}, epoch)
        if is_validation:
            print('VAL:')
            for i in range(len(model.probes)):
                print(f" [{i}] t1:{top1[i].avg:04.2f} loss:{loss_value[i].avg:.2f}",end='')
            print()
        else:
            print('TRAIN:')
            for i in range(len(model.probes)):
                print(f" [{i}] t1:{top1[i].avg:04.2f} loss:{loss_value[i].avg:.2f}", end='')
            print()

        return {"loss": [x.avg for x in loss_value],
            "top1": [x.avg for x in top1],
            "top5": [x.avg for x in top5]}


def get_parser():
    parser = argparse.ArgumentParser(description='AlexNet standard linear separability tests')
    parser.add_argument('--data', default='Imagenet', type=str, help='Dataset Imagenet or Places (default: Imagenet)')
    parser.add_argument('--ckpt-dir', default='./test', metavar='DIR', help='path to checkpoints (default: ./test)')
    parser.add_argument('--device', default="0", type=str, metavar='d', help='GPU device')
    parser.add_argument('--modelpath',default='.ckpt400.pth',type=str, help='path to model')
    parser.add_argument('--workers', default=6, type=int, metavar='N', help='number of data loading workers (default: 6)')

    # optimization params
    parser.add_argument('--epochs', default=36, type=int, metavar='N', help='number of epochs (default: 36)')
    parser.add_argument('--batch-size', default=192, type=int, metavar='N', help='batch size (default: 192)')
    parser.add_argument('--learning-rate', default=0.01, type=float, metavar='FLOAT', help='initial learning rate (default: 0.01)')
    parser.add_argument('--tencrops', dest='tencrops', action='store_false', help='flag to not use tencrops (default: on)')

    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='flag to evaluate only (default: off)')
    parser.add_argument('--img-size', default=256, type=int, help='imagesize (default: 256)')
    parser.add_argument('--crop-size', default=224, type=int, help='cropsize for CNN (default: 224)')
    parser.add_argument('--imagenet-path', default='/data/imagenet', type=str,
                        help='path to imagenet folder, where train and val are located')
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    args.device = [int(args.device)]
    # Setup CUDA and random seeds
    util.setup_runtime(seed=2, cuda_dev_id=args.device)
    name = args.name.replace('/', '_')
    print(name, flush=True)



    model = model_with_probes(model_path=args.modelpath)
    train_loader, val_loader = get_standard_data_loader_pairs(dir_path=args.imagenet_path,
                                                              batch_size=args.batch_size,
                                                              num_workers=args.workers,
                                                              tencrops=args.tencrops)

    o = LinearProbesOptimizer()
    if args.tensorboard:
        writer = SummaryWriter('./runs_LP/%s' % (name))
        writer.add_text('args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))
        o.writer = writer
    o.lr = args.learning_rate
    o.validate_only = args.evaluate
    o.num_epochs = args.epochs
    o.checkpoint_dir = args.ckpt_dir
    o.resume = True
    o.optimize(model, train_loader, val_loader)