import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torchvision_models

import torch.distributed as dist

import habitat_baselines.rl.models.vision_transformer as vits
from habitat_baselines.rl.models.vision_transformer import DINOHead
import habitat_baselines.rl.representation.utils as utils

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, update):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[update]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach()

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class MultiDistill(nn.Module):

    def __init__(self, args, device, num_eposides, max_episode_updates):
        super(MultiDistill, self).__init__()

        self.args = args
        self.num_eposides = num_eposides
        self.max_episode_updates = max_episode_updates
        # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
        if args.arch in vits.__dict__.keys():
            self.student = vits.__dict__[args.arch](
                patch_size=args.patch_size,
                drop_path_rate=args.drop_path_rate,  # stochastic depth
            )
            self.teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
            self.embed_dim = self.student.embed_dim # 384
        # if the network is a XCiT
        elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
            self.student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                     pretrained=False,
                                     drop_path_rate=args.drop_path_rate)
            self.teacher = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                     pretrained=False)
            self.embed_dim = self.student.embed_dim
        # otherwise, we check if the architecture is in torchvision models
        elif args.arch in torchvision_models.__dict__.keys():
            self.student = torchvision_models.__dict__[args.arch]()
            self.teacher = torchvision_models.__dict__[args.arch]()
            self.embed_dim = self.student.fc.weight.shape[1]
        else:
            print(f"Unknow architecture: {args.arch}")

        # multi-crop wrapper handles forward with inputs of different resolutions
        self.student = utils.MultiCropWrapper(self.student, DINOHead(
            self.embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ))
        self.teacher = utils.MultiCropWrapper(
            self.teacher,
            DINOHead(self.embed_dim, args.out_dim, args.use_bn_in_head),
        )

        self.device = device
        self.student, self.teacher = self.student.to(self.device), self.teacher.to(self.device)

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.dino_loss = DINOLoss(
            args.out_dim,
            args.local_crops_number + 2,
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epoches,
            self.num_eposides * self.max_episode_updates
        ).to(self.device)

        self.params_groups = utils.get_params_groups(self.student)
        if args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(self.params_groups)  # to use with ViTs
        elif args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.params_groups, lr=0,
                                        momentum=0.9)  # lr is set by scheduler
        elif args.optimizer == "lars":
            self.optimizer = utils.LARS(self.params_groups)  # to use with convnet and large batches

        # ============ init schedulers ... ============
        # the length of the schedules is the total batches of the train process e.g. 100(epoches) * 3125(batch nums)
        self.lr_schedule = utils.cosine_scheduler(
            args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
            args.min_lr,
            self.num_eposides, self.max_episode_updates,
            warmup_epochs=args.warmup_epochs,
        )  # (312500, )
        self.wd_schedule = utils.cosine_scheduler(
            args.weight_decay,
            args.weight_decay_end,
            self.num_eposides, self.max_episode_updates,
        )
        # momentum parameter is increased to 1. during training with a cosine schedule
        self.momentum_schedule = utils.cosine_scheduler(args.momentum_teacher,
                                                   1,
                                                   self.num_eposides,
                                                   self.max_episode_updates,)
        self.train()

    def forward(self, images, points, update):
        for i, param_group in enumerate(self.params_groups):
            param_group["lr"] = self.lr_schedule[update]
            if i == 0:
                param_group["weight_decay"] = self.wd_schedule[update]

        images = images.to(self.device)# 8
        points = points.to(self.device)

        teacher_output = self.teacher(points)
        student_output = self.student(images)
        loss = self.dino_loss(student_output, teacher_output, update)

        return loss

    def updata_distill(self, images, points, clip_grad, freeze_last_layer, update):
        # update student
        self.optimizer.zero_grad()
        param_norms = None
        loss = self.forward(images, points, update)
        loss.backward()
        if clip_grad:
            param_norms = utils.clip_gradients(self.student, clip_grad)
        utils.cancel_gradients_last_layer(update, self.student, freeze_last_layer)  # 取消最后一层的梯度
        self.optimizer.step()

        # EMA update for the teacher
        with torch.no_grad():
            m = self.momentum_schedule[update]  # momentum parameter
            for param_q, param_k in zip(self.student.parameters(),
                                        self.teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        return {"multi_distill_loss": loss}

    def save(self, save_dir, update):
        checkpoint = {
            'student': self.student.state_dict(),
            'teacher': self.teacher.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update': update + 1,
            'args': self.args,
            'dino_loss': self.dino_loss.state_dict(),
        }
        torch.save(
            checkpoint,
            os.path.join(save_dir, "idm_param_{}.pth".format(int(update)))
        )

    def save_teacher_and_student(self):
        pass

