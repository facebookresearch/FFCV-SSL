# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/libffcv/ffcv-imagenet to support SSL

import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)
import torchvision.transforms as transforms
import torchmetrics
import numpy as np
from tqdm import tqdm
import subprocess
import os
import time
import json
import uuid
import ffcv
import submitit
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config, set_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from utils import LARS, cosine_scheduler, learning_schedule

Section('model', 'model details').params(
    arch=Param(str, 'model to use', default='resnet50'),
    remove_head=Param(int, 'remove the projector? (1/0)', default=0),
    mlp=Param(str, 'number of projector layers', default="2048-512"),
    mlp_coeff=Param(float, 'number of projector layers', default=1),
    patch_keep=Param(float, 'Proportion of patches to keep with VIT training', default=1.0),
    fc=Param(int, 'remove the projector? (1/0)', default=0),
    proj_relu=Param(int, 'Proj relu? (1/0)', default=0),
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=64),
    max_res=Param(int, 'the maximum (starting) resolution', default=224),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=30),
    start_ramp=Param(int, 'when to start interpolating resolution', default=10)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', default=""),
    num_workers=Param(int, 'The number of workers', default=10),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True),
)

Section('vicreg', 'Vicreg').params(
    sim_coeff=Param(float, 'VicREG MSE coefficient', default=25),
    std_coeff=Param(float, 'VicREG STD coefficient', default=25),
    cov_coeff=Param(float, 'VicREG COV coefficient', default=1),
)

Section('simclr', 'simclr').params(
    temperature=Param(float, 'SimCLR temperature', default=0.5),
)

Section('barlow', 'barlow').params(
    lambd=Param(float, 'Barlow Twins Lambd parameters', default=0.0051),
)

Section('byol', 'byol').params(
    momentum_teacher=Param(float, 'Momentum Teacher value', default=0.996),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True),
    log_level=Param(int, '0 if only at end 1 otherwise', default=2),
    checkpoint_freq=Param(int, 'When saving checkpoints', default=5)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=256),
    resolution=Param(int, 'final resized validation image size', default=224),
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    eval_freq=Param(float, 'number of epochs', default=1),
    batch_size=Param(int, 'The batch size', default=512),
    num_crops=Param(int, 'number of crops?', default=1),
    optimizer=Param(And(str, OneOf(['sgd', 'adamw', 'lars'])), 'The optimizer', default='adamw'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=30),
    base_lr=Param(float, 'number of epochs', default=0.0005),
    end_lr_ratio=Param(float, 'number of epochs', default=0.001),
    label_smoothing=Param(float, 'label smoothing parameter', default=0),
    distributed=Param(int, 'is distributed?', default=0),
    clip_grad=Param(float, 'sign the weights of last residual block', default=0),
    use_ssl=Param(int, 'use ssl data augmentations?', default=0),
    loss=Param(str, 'use ssl data augmentations?', default="simclr"),
    train_probes_only=Param(int, 'load linear probes?', default=0),
)

Section('dist', 'distributed training options').params(
    use_submitit=Param(int, 'enable submitit', default=0),
    world_size=Param(int, 'number gpus', default=1),
    ngpus=Param(int, 'number of gpus per nodes', default=8),
    nodes=Param(int, 'number of nodes', default=1),
    comment=Param(str, 'comment for slurm', default=''),
    timeout=Param(int, 'timeout', default=2800),
    partition=Param(str, 'partition', default="learnlab"),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='58492')
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

################################
##### Some Miscs functions #####
################################

def get_shared_folder() -> Path:
    user = os.getenv("USER")
    path = "/checkpoint/"
    if Path(path).is_dir():
        p = Path(f"{path}{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")

def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

def exclude_bias_and_norm(p):
    return p.ndim == 1

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def gather_center(x):
    x = batch_all_gather(x)
    x = x - x.mean(dim=0)
    return x

def batch_all_gather(x):
    x_list = GatherLayer.apply(x.contiguous())
    return ch.cat(x_list, dim=0)

class GatherLayer(ch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [ch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = ch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

################################
##### Loss definitions #####
################################

class SimCLRLoss(nn.Module):
    """
    SimCLR Loss:
    When using a batch size of 2048, use LARS as optimizer with a base learning rate of 0.5, 
    weight decay of 1e-6 and a temperature of 0.15.
    When using a batch size of 256, use LARS as optimizer with base learning rate of 1.0, 
    weight decay of 1e-6 and a temperature of 0.15.
    """
    @param('simclr.temperature')
    def __init__(self, batch_size, world_size, gpu, temperature):
        super(SimCLRLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size).to(gpu)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = ch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.size(0)
        N = 2 * batch_size * self.world_size

        if self.world_size > 1:
            z_i = ch.cat(GatherLayer.apply(z_i), dim=0)
            z_j = ch.cat(GatherLayer.apply(z_j), dim=0)
        
        z = ch.cat((z_i, z_j), dim=0)

        features = F.normalize(z, dim=1)
        sim = ch.matmul(features, features.T)/ self.temperature

        sim_i_j = ch.diag(sim, batch_size * self.world_size)
        sim_j_i = ch.diag(sim, -batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = ch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        logits = ch.cat((positive_samples, negative_samples), dim=1)
        logits_num = logits
        logits_denum = ch.logsumexp(logits, dim=1, keepdim=True)
        num_sim = (- logits_num[:, 0]).sum() / N
        num_entropy = logits_denum[:, 0].sum() / N
        return num_sim, num_entropy

class VicRegLoss(nn.Module):
    """
    ViCREG Loss:
    When using a batch size of 2048, use LARS as optimizer with a base learning rate of 0.5, 
    weight decay of 1e-4 and a sim and std coeff of 25 with a cov coeff of 1.
    When using a batch size of 256, use LARS as optimizer with base learning rate of 1.5, 
    weight decay of 1e-4 and a sim and std coeff of 25 with a cov coeff of 1.
    """
    @param('vicreg.sim_coeff')
    @param('vicreg.std_coeff')
    @param('vicreg.cov_coeff')
    def __init__(self, sim_coeff, std_coeff, cov_coeff):
        super(VicRegLoss, self).__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, z_i, z_j, return_only_loss=True):
        # Repr Loss
        repr_loss = self.sim_coeff * F.mse_loss(z_i, z_j)
        std_loss = 0.
        cov_loss = 0.

        # Std Loss z_i
        x = gather_center(z_i)
        std_x = ch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = std_loss + self.std_coeff * ch.mean(ch.relu(1 - std_x))
        # Cov Loss z_i
        cov_x = (x.T @ x) / (x.size(0) - 1)
        cov_loss = cov_loss + self.cov_coeff * off_diagonal(cov_x).pow_(2).sum().div(z_i.size(1))
        
        # Std Loss z_j
        x = gather_center(z_j)
        std_x = ch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = std_loss + self.std_coeff * ch.mean(ch.relu(1 - std_x))
        # Cov Loss z_j
        cov_x = (x.T @ x) / (x.size(0) - 1)
        cov_loss = cov_loss + self.cov_coeff * off_diagonal(cov_x).pow_(2).sum().div(z_j.size(1))

        std_loss = std_loss / 2.

        loss = std_loss + cov_loss + repr_loss
        if return_only_loss:
            return loss
        else:
            return loss, repr_loss, std_loss, cov_loss

class BarlowTwinsLoss(nn.Module):
    @param('barlow.lambd')
    def __init__(self, bn, batch_size, world_size, lambd):
        super(BarlowTwinsLoss, self).__init__()
        self.bn = bn
        self.lambd = lambd
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size * self.world_size)
        ch.distributed.all_reduce(c)

        on_diag = ch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

class ByolLoss(nn.Module):
    @param('byol.momentum_teacher')
    def __init__(self, momentum_teacher):
        super().__init__()
        self.momentum_teacher = momentum_teacher

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output.chunk(2)
        teacher_out = teacher_output.detach().chunk(2)

        student_out_1, student_out_2 = student_out
        student_out_1 = F.normalize(student_out_1, dim=-1, p=2)
        student_out_2 = F.normalize(student_out_2, dim=-1, p=2)
        teacher_out_1, teacher_out_2 = teacher_out
        teacher_out_1 = F.normalize(teacher_out_1, dim=-1, p=2)
        teacher_out_2 = F.normalize(teacher_out_2, dim=-1, p=2)
        loss_1 = 2 - 2 * (student_out_1 * teacher_out_2.detach()).sum(dim=-1)
        loss_2 = 2 - 2 * (student_out_2 * teacher_out_1.detach()).sum(dim=-1)
        return (loss_1 + loss_2).mean()

################################
##### SSL Model Generic CLass ##
################################

class SSLNetwork(nn.Module):
    @param('model.arch')
    @param('model.remove_head')
    @param('model.mlp')
    @param('model.patch_keep')
    @param('model.fc')
    @param('training.loss')
    def __init__(
        self, arch, remove_head, mlp, patch_keep, fc, loss
    ):
        super().__init__()
        if "resnet" in arch:
            import torchvision.models.resnet as resnet
            self.net = resnet.__dict__[arch]()
            if fc:
                self.net.fc = nn.Linear(2048, 256)
            else:
                self.net.fc = nn.Identity()
        elif "vgg" in arch:
            import torchvision.models.vgg as vgg
            self.net = vgg.__dict__[arch]()
            self.net.classifier = nn.Identity()
        else:
            print("Arch not found")
            exit(0)

        # Compute the size of the representation
        self.representation_size = self.net(ch.zeros((1,3,224,224))).size(1)
        print("REPR SIZE:", self.representation_size)
        # Add a projector head
        self.mlp = mlp
        if remove_head:
            self.num_features = self.representation_size
            self.projector = nn.Identity()
        else:
            self.num_features = int(self.mlp.split("-")[-1])
            self.projector = self.MLP(self.representation_size)
        self.loss = loss
        if loss == "barlow":
            self.bn = nn.BatchNorm1d(self.num_features, affine=False)
        elif loss == "byol":
            self.predictor = self.MLP(self.num_features)

    @param('model.proj_relu')
    @param('model.mlp_coeff')
    def MLP(self, size, proj_relu, mlp_coeff):
        mlp_spec = f"{size}-{self.mlp}"
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        f[-2] = int(f[-2] * mlp_coeff)
        print("MLP:", f)
        for i in range(len(f) - 2):
            layers.append(nn.Sequential(nn.Linear(f[i], f[i + 1]), nn.BatchNorm1d(f[i + 1]), nn.ReLU(True)))
        if proj_relu:
            layers.append(nn.Sequential(nn.Linear(f[-2], f[-1], bias=False), nn.ReLU(True)))
        else:
            layers.append(nn.Linear(f[-2], f[-1], bias=False))
        return nn.Sequential(*layers)

    def forward(self, inputs, embedding=False, predictor=False):
        if embedding:
            embedding = self.net(inputs)
            return embedding
        else:
            representation = self.net(inputs)
            embeddings = self.projector(representation)
            list_outputs = [representation.detach()]
            outputs_train = representation.detach()
            for l in range(len(self.projector)):
                outputs_train = self.projector[l](outputs_train).detach()
                list_outputs.append(outputs_train)
            if self.loss == "byol" and predictor:         
                embeddings = self.predictor(embeddings)
            return embeddings, list_outputs

class LinearsProbes(nn.Module):
    @param('model.mlp_coeff')
    def __init__(self, model, num_classes, mlp_coeff):
        super().__init__()
        print("NUM CLASSES", num_classes)
        mlp_spec = f"{model.module.representation_size}-{model.module.mlp}"
        f = list(map(int, mlp_spec.split("-")))
        f[-2] = int(f[-2] * mlp_coeff)
        self.probes = []
        for num_features in f:
            self.probes.append(nn.Linear(num_features, num_classes))
        self.probes = nn.Sequential(*self.probes)

    def forward(self, list_outputs, binary=False):
        return [self.probes[i](list_outputs[i]) for i in range(len(list_outputs))]


################################
##### Main Trainer ############
################################

class ImageNetTrainer:
    @param('training.distributed')
    @param('training.batch_size')
    @param('training.label_smoothing')
    @param('training.loss')
    @param('training.train_probes_only')
    @param('training.epochs')
    @param('data.train_dataset')
    @param('data.val_dataset')
    def __init__(self, gpu, ngpus_per_node, world_size, dist_url, distributed, batch_size, label_smoothing, loss, train_probes_only, epochs, train_dataset, val_dataset):
        self.all_params = get_current_config()
        ch.cuda.set_device(gpu)
        self.gpu = gpu
        self.rank = self.gpu + int(os.getenv("SLURM_NODEID", "0")) * ngpus_per_node
        self.world_size = world_size
        self.seed = 50 + self.rank
        self.dist_url = dist_url
        self.batch_size = batch_size
        self.uid = str(uuid4())
        if distributed:
            self.setup_distributed()
        self.start_epoch = 0
        # Create DataLoader
        self.train_dataset = train_dataset
        self.index_labels = 1
        self.train_loader = self.create_train_loader_ssl(train_dataset)
        self.num_train_exemples = self.train_loader.indices.shape[0]
        self.num_classes = 1000
        self.val_loader = self.create_val_loader(val_dataset)
        print("NUM TRAINING EXEMPLES:", self.num_train_exemples)
        # Create SSL model
        self.model, self.scaler = self.create_model_and_scaler()
        self.num_features = self.model.module.num_features
        self.n_layers_proj = len(self.model.module.projector) + 1
        print("N layers in proj:", self.n_layers_proj)
        self.initialize_logger()
        self.classif_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.create_optimizer()
        # Create lineares probes
        self.loss = nn.CrossEntropyLoss()
        self.probes = LinearsProbes(self.model, num_classes=self.num_classes)
        self.probes = self.probes.to(memory_format=ch.channels_last)
        self.probes = self.probes.to(self.gpu)
        self.probes = ch.nn.parallel.DistributedDataParallel(self.probes, device_ids=[self.gpu])
        self.optimizer_probes = ch.optim.AdamW(self.probes.parameters(), lr=1e-4)
        # Load models if checkpoints
        self.load_checkpoint()
        # Define SSL loss
        self.do_ssl_training = False if train_probes_only else True
        self.teacher_student = False
        self.supervised_loss = False
        self.loss_name = loss
        if loss == "simclr":
            self.ssl_loss = SimCLRLoss(batch_size, world_size, self.gpu).to(self.gpu)
        elif loss == "vicreg":
            self.ssl_loss = VicRegLoss()
        elif loss == "barlow":
            self.ssl_loss = BarlowTwinsLoss(self.model.module.bn, batch_size, world_size)
        elif loss == "byol":
            self.ssl_loss = ByolLoss()
            self.teacher_student = True
            self.teacher, _ = self.create_model_and_scaler()
            self.teacher.module.load_state_dict(self.model.module.state_dict())
            self.momentum_schedule = cosine_scheduler(self.ssl_loss.momentum_teacher, 1, epochs, len(self.train_loader))
            for p in self.teacher.parameters():
                p.requires_grad = False
        elif loss == "supervised":
            self.supervised_loss = True
        else:
            print("Loss not available")
            exit(1)

    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param('training.use_ssl')
    @param('data.train_dataset')
    def get_dataloader(self, use_ssl, train_dataset):
        if use_ssl:
            return self.create_train_loader_ssl(160), self.create_val_loader()
        else:
            return self.create_train_loader_supervised(160), self.create_val_loader()

    def setup_distributed(self):
        dist.init_process_group("nccl", init_method=self.dist_url, rank=self.rank, world_size=self.world_size)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing):
        assert optimizer == 'sgd' or optimizer == 'adamw' or optimizer == "lars"

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]
        if optimizer == 'sgd':
            self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        elif optimizer == 'adamw':
            # We use a big eps value to avoid instabilities with fp16 training
            self.optimizer = ch.optim.AdamW(param_groups, lr=1e-4)
        elif optimizer == "lars":
            self.optimizer = LARS(param_groups)  # to use with convnet and large batches
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optim_name = optimizer

    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    @param('data.in_memory')
    def create_train_loader_ssl(self, train_dataset, num_workers, batch_size,
                            distributed, in_memory):
        this_device = f'cuda:{self.gpu}'
        train_path = Path(train_dataset)
        assert train_path.is_file()
        # First branch of augmentations
        self.decoder = ffcv.transforms.RandomResizedCrop((224, 224))
        image_pipeline_big: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            ffcv.transforms.RandomGrayscale(0.2),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))
        ]

        # Second branch of augmentations
        self.decoder2 = ffcv.transforms.RandomResizedCrop((224, 224))
        image_pipeline_big2: List[Operation] = [
            self.decoder2,
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            ffcv.transforms.RandomGrayscale(0.2),
            ffcv.transforms.RandomSolarization(0.2, 128),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        # SSL Augmentation pipeline
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True)
        ]

        pipelines={
            'image': image_pipeline_big,
            'label': label_pipeline,
            'image_0': image_pipeline_big2
        }

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        custom_field_mapper={"image_0": "image"}

        # Create data loader
        loader = ffcv.Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines=pipelines,
                        distributed=distributed,
                        custom_field_mapper=custom_field_mapper)


        return loader

    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution, distributed):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
        ]

        order = OrderOption.SEQUENTIAL

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        return loader

    @param('training.epochs')
    @param('logging.log_level')
    def train(self, epochs, log_level):
        # We scale the number of max steps w.t the number of examples in the training set
        self.max_steps = epochs * self.num_train_exemples // (self.batch_size * self.world_size)
        for epoch in range(self.start_epoch, epochs):
            res = self.get_resolution(epoch)
            self.res = res
            self.decoder.output_size = (res, res)
            self.decoder2.output_size = (res, res)
            train_loss, stats = self.train_loop(epoch)
            if log_level > 0:
                extra_dict = {
                    'train_loss': train_loss,
                    'epoch': epoch
                }
                self.log(dict(stats,  **extra_dict))
            self.eval_and_log(stats, extra_dict)
            # Run checkpointing
            self.checkpoint(epoch + 1)
        if self.gpu == 0:
            ch.save(self.model.state_dict(), self.log_folder / 'final_weights.pt')

    def eval_and_log(self, stats, extra_dict={}):
        stats = self.val_loop()
        self.log(dict(stats, **extra_dict))
        return stats

    @param('training.loss')
    def create_model_and_scaler(self, loss):
        scaler = GradScaler()
        model = SSLNetwork()
        if loss == "supervised":
            model.fc = nn.Linear(model.num_features, self.num_classes)
        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
        return model, scaler

    @param('training.train_probes_only')
    def load_checkpoint(self, train_probes_only):
        if (self.log_folder / "model.pth").is_file():
            if self.rank == 0:
                print("resuming from checkpoint")
            ckpt = ch.load(self.log_folder / "model.pth", map_location="cpu")
            self.start_epoch = ckpt["epoch"]
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if not train_probes_only:
                self.probes.load_state_dict(ckpt["probes"])
                self.optimizer_probes.load_state_dict(ckpt["optimizer_probes"])
            else:
                self.start_epoch = 0

    @param('logging.checkpoint_freq')
    @param('training.train_probes_only')
    def checkpoint(self, epoch, checkpoint_freq, train_probes_only):
        if self.rank != 0 or epoch % checkpoint_freq != 0:
            return
        if train_probes_only:
            state = dict(
                epoch=epoch, 
                probes=self.probes.state_dict(), 
                optimizer_probes=self.optimizer_probes.state_dict()
            )
            save_name = f"probes.pth"
        else:
            state = dict(
                epoch=epoch, 
                model=self.model.state_dict(), 
                optimizer=self.optimizer.state_dict(),
                probes=self.probes.state_dict(), 
                optimizer_probes=self.optimizer_probes.state_dict()
            )
            save_name = f"model.pth"
        ch.save(state, self.log_folder / save_name)

    @param('logging.log_level')
    @param('training.base_lr')
    @param('training.end_lr_ratio')
    def train_loop(self, epoch, log_level, base_lr, end_lr_ratio):
        """
        Main training loop for SSL training with VicReg criterion.
        """
        model = self.model
        model.train()
        losses = []

        iterator = tqdm(self.train_loader)
        for ix, loaders in enumerate(iterator, start=epoch * len(self.train_loader)):

            # Get lr
            lr = learning_schedule(
                global_step=ix,
                batch_size=self.batch_size * self.world_size,
                base_lr=base_lr,
                end_lr_ratio=end_lr_ratio,
                total_steps=self.max_steps,
                warmup_steps=10 * self.num_train_exemples // (self.batch_size * self.world_size),
            )
            for g in self.optimizer.param_groups:
                 g["lr"] = lr

            # Get data
            images_big_0 = loaders[0]
            labels_big = loaders[1]
            batch_size = loaders[1].size(0)
            images_big_1 = loaders[2]
            images_big = ch.cat((images_big_0, images_big_1), dim=0)

            # SSL Training
            if self.do_ssl_training:
                self.optimizer.zero_grad(set_to_none=True)
                with autocast():
                    if self.teacher_student:
                        with ch.no_grad():
                            teacher_output, _ = self.teacher(images_big)
                            teacher_output = teacher_output.view(2, batch_size, -1)
                        embedding_big, _ = model(images_big, predictor=True)
                    elif self.supervised_loss:
                        embedding_big, _ = model(images_big_0.repeat(2,1,1,1))
                    else:
                        # Compute embedding in bigger crops
                        embedding_big, _ = model(images_big)
                    
                    # Compute SSL Loss
                    if self.teacher_student:
                        embedding_big = embedding_big.view(2, batch_size, -1)
                        loss_train = self.ssl_loss(embedding_big, teacher_output)
                    elif self.supervised_loss:
                        output_classif_projector = model.module.fc(embedding_big)
                        loss_train = self.classif_loss(output_classif_projector, labels_big.repeat(2))
                    else:
                        embedding_big = embedding_big.view(2, batch_size, -1)
                        if "simclr" in self.loss_name:
                            loss_num, loss_denum = self.ssl_loss(embedding_big[0], embedding_big[1])
                            loss_train = loss_num + loss_denum
                        else:
                            loss_train = self.ssl_loss(embedding_big[0], embedding_big[1])
                            
                    self.scaler.scale(loss_train).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss_train = ch.tensor(0.)
            if self.teacher_student:
                m = self.momentum_schedule[ix]  # momentum parameter
                for param_q, param_k in zip(model.module.parameters(), self.teacher.module.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # Online linear probes training
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_probes.zero_grad(set_to_none=True)
            # Compute embeddings vectors
            with ch.no_grad():
                with autocast():
                    _, list_representation = model(images_big_0)
            # Train probes
            with autocast():
                # Real value classification
                list_outputs = self.probes(list_representation)
                loss_classif = 0.
                for l in range(len(list_outputs)):
                    # Compute classif loss
                    current_loss = self.loss(list_outputs[l], labels_big)
                    loss_classif += current_loss
                    self.train_meters['loss_classif_layer'+str(l)](current_loss.detach())
                    for k in ['top_1_layer'+str(l), 'top_5_layer'+str(l)]:
                        self.train_meters[k](list_outputs[l].detach(), labels_big)
            self.scaler.scale(loss_classif).backward()
            self.scaler.step(self.optimizer_probes)
            self.scaler.update()

            # Logging
            if log_level > 0:
                self.train_meters['loss'](loss_train.detach())
                losses.append(loss_train.detach())
                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.5f}')

                names = ['ep', 'iter', 'shape', 'lrs']
                values = [epoch, ix, tuple(images_big.shape), group_lrs]
                if log_level > 1:
                    names += ['loss']
                    values += [f'{loss_train.item():.3f}']
                    names += ['loss_c']
                    values += [f'{loss_classif.item():.3f}']

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)

        # Return epoch's log
        if log_level > 0:
            self.train_meters['time'](ch.tensor(iterator.format_dict["elapsed"]))
            loss = ch.stack(losses).mean().cpu()
            assert not ch.isnan(loss), 'Loss is NaN!'
            stats = {k: m.compute().item() for k, m in self.train_meters.items()}
            [meter.reset() for meter in self.train_meters.values()]
            return loss.item(), stats

    def val_loop(self):
        model = self.model
        model.eval()
        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader):
                    _, list_representation = model(images)
                    list_outputs = self.probes(list_representation)
                    loss_classif = 0.
                    for l in range(len(list_outputs)):
                        # Compute classif loss
                        current_loss = self.loss(list_outputs[l], target)
                        loss_classif += current_loss
                        self.val_meters['loss_classif_val_layer'+str(l)](current_loss.detach())
                        for k in ['top_1_val_layer'+str(l), 'top_5_val_layer'+str(l)]:
                            self.val_meters[k](list_outputs[l].detach(), target)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    @param('logging.folder')
    def initialize_logger(self, folder):
        self.train_meters = {
            'loss': torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu),
            'time': torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu),
        }

        for l in range(self.n_layers_proj):
            self.train_meters['loss_classif_layer'+str(l)] = torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu)
            self.train_meters['top_1_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, compute_on_step=False).to(self.gpu)
            self.train_meters['top_5_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, compute_on_step=False).to(self.gpu)

        self.val_meters = {}
        for l in range(self.n_layers_proj):
            self.val_meters['loss_classif_val_layer'+str(l)] = torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu)
            self.val_meters['top_1_val_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, compute_on_step=False).to(self.gpu)
            self.val_meters['top_5_val_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, compute_on_step=False).to(self.gpu)

        if self.gpu == 0:
            if Path(folder + 'final_weights.pt').is_file():
                self.uid = ""
                folder = Path(folder)
            else:
                folder = Path(folder)
            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)
        self.log_folder = Path(folder)

    @param('training.train_probes_only')
    def log(self, content, train_probes_only):
        print(f'=> Log: {content}')
        if self.rank != 0: return
        cur_time = time.time()
        name_file = 'log_probes' if train_probes_only else 'log'
        with open(self.log_folder / name_file, 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()

    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    @param('dist.port')
    def launch_from_args(cls, distributed, world_size, port):
        if distributed:
            ngpus_per_node = ch.cuda.device_count()
            world_size = int(os.getenv("SLURM_NNODES", "1")) * ngpus_per_node
            if "SLURM_JOB_NODELIST" in os.environ:
                cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
                host_name = subprocess.check_output(cmd).decode().splitlines()[0]
                dist_url = f"tcp://{host_name}:"+port
            else:
                dist_url = "tcp://localhost:"+port
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=ngpus_per_node, join=True, args=(None, ngpus_per_node, world_size, dist_url))
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        if args[1] is not None:
            set_current_config(args[1])
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('training.distributed')
    @param('training.eval_only')
    def exec(cls, gpu, config, ngpus_per_node, world_size, dist_url, distributed, eval_only):
        trainer = cls(gpu=gpu, ngpus_per_node=ngpus_per_node, world_size=world_size, dist_url=dist_url)
        if eval_only:
            trainer.eval_and_log()
        else:
            trainer.train()

        if distributed:
            trainer.cleanup_distributed()

class Trainer(object):
    def __init__(self, config, num_gpus_per_node, dump_path, dist_url, port):
        self.num_gpus_per_node = num_gpus_per_node
        self.dump_path = dump_path
        self.dist_url = dist_url
        self.config = config
        self.port = port

    def __call__(self):
        self._setup_gpu_args()

    def checkpoint(self):
        self.dist_url = get_init_file().as_uri()
        print("Requeuing ")
        empty_trainer = type(self)(self.config, self.num_gpus_per_node, self.dump_path, self.dist_url, self.port)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        from pathlib import Path
        job_env = submitit.JobEnvironment()
        self.dump_path = Path(str(self.dump_path).replace("%j", str(job_env.job_id)))
        gpu = job_env.local_rank
        world_size = job_env.num_tasks
        if "SLURM_JOB_NODELIST" in os.environ:
            cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
            host_name = subprocess.check_output(cmd).decode().splitlines()[0]
            dist_url = f"tcp://{host_name}:"+self.port
        else:
            dist_url = "tcp://localhost:"+self.port
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        ImageNetTrainer._exec_wrapper(gpu, config, self.num_gpus_per_node, world_size, dist_url)

# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast SSL training')
    parser.add_argument("folder", type=str)
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()
    return config

@param('logging.folder')
@param('dist.ngpus')
@param('dist.nodes')
@param('dist.timeout')
@param('dist.partition')
@param('dist.comment')
@param('dist.port')
def run_submitit(config, folder, ngpus, nodes,  timeout, partition, comment, port):
    Path(folder).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=folder, slurm_max_num_timeout=30)

    num_gpus_per_node = ngpus
    nodes = nodes
    timeout_min = timeout

    # Cluster specifics: To update accordingly to your cluster
    kwargs = {}
    kwargs['slurm_comment'] = comment
    executor.update_parameters(
        mem_gb=60 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node, 
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="ffcv-ssl")

    dist_url = get_init_file().as_uri()

    trainer = Trainer(config, num_gpus_per_node, folder, dist_url, port)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {folder}")

@param('dist.use_submitit')
def main(config, use_submitit):
    if use_submitit:
        run_submitit(config)
    else:
        ImageNetTrainer.launch_from_args()

if __name__ == "__main__":
    config = make_config()
    main(config)
