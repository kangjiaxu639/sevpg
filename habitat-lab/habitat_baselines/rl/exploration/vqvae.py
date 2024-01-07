import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import kl_divergence
from torch.autograd import Function
from typing import List

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class BasicEncBlock(nn.Module):
    def __init__(
            self,
            inplanes: int,
            planes: int,
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 1,
            downsample=None,
            is_last=False
    ):
        super().__init__()

        if is_last:
            self.block = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size, stride, padding),
                nn.BatchNorm2d(planes)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size, stride, padding),
                nn.BatchNorm2d(planes),
                nn.ReLU()
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.block(x)
        if self.downsample is not None:
            identity = self.downsample(x)
            out += identity
        return out

class BasicDecBlock(nn.Module):
    def __init__(
            self,
            inplanes: int,
            planes: int,
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 1,
            upsample=None,
            is_last=False
    ):
        super().__init__()

        if is_last:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(inplanes, planes, kernel_size, stride, padding)
            )
        else:
            self.block = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(inplanes, planes, kernel_size, stride, padding),
                nn.BatchNorm2d(planes),
            )
        self.is_last = is_last
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        out = self.block(x)
        if self.upsample is not None:
            identity = self.upsample(x)
            out += identity
        return out

def build_encoder_res18(input_dim, dim, kernel_size, num_cnn_blocks=5, num_res_blocks=0, learnable=False, return_module_list=False):
    if not learnable:
        downsample = nn.AvgPool2d(kernel_size=4, stride=2, padding=1)
    else:
        downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    blocks = [
        BasicEncBlock(input_dim, dim, kernel_size=kernel_size),
        BasicEncBlock(dim, dim, kernel_size=kernel_size),
    ]

    for i in range(num_cnn_blocks-3):
        blocks.append(BasicEncBlock(dim, dim, kernel_size=kernel_size, downsample=downsample))
    blocks.append(BasicEncBlock(dim, dim, kernel_size=kernel_size, downsample=downsample, is_last=True))

    for i in range(num_res_blocks):
        blocks.append(ResBlock(dim))

    if return_module_list:
        return nn.ModuleList(blocks)

    return nn.Sequential(*blocks)

def build_decoder_res18(input_dim, dim, kernel_size, num_cnn_blocks=5, num_res_blocks=0, learnable=False, return_module_list=False):
    if not learnable:
        upsample = nn.Upsample(mode='bilinear', scale_factor=2)
    else:
        upsample = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

    blocks = []

    for i in range(num_res_blocks):
        blocks.append(ResBlock(dim))

    for i in range(num_cnn_blocks-2):
        blocks.append(BasicDecBlock(dim, dim, kernel_size=kernel_size, upsample=upsample))

    blocks.extend([
        BasicDecBlock(dim, dim, kernel_size=kernel_size),
        BasicDecBlock(dim, input_dim, kernel_size=kernel_size, is_last=True),
        nn.Tanh()
    ])

    if return_module_list:
        return nn.ModuleList(blocks)

    return nn.Sequential(*blocks)
def forward_module_list(module_list, x):
    encodings = []
    x_enc = x
    for i in range(len(module_list)):
        layer = module_list[i]
        x_enc = layer(x_enc)
        encodings.append(x_enc)
    return encodings

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def select(self, indices):
        return torch.index_select(self.embedding.weight, dim=0, index=indices)

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim,
                 K=512, num_cnn_blocks=5, num_res_blocks=3, kernel_size=4,
                 reg_type='l2', arch='basic'):
        super().__init__()
        self.K = K
        self.reg_type = reg_type
        if arch == 'res':
            self.encoder = build_encoder_res18(input_dim, dim, kernel_size, num_cnn_blocks=num_cnn_blocks, num_res_blocks=num_res_blocks, return_module_list=True)
        elif arch == 'resl':
            self.encoder = build_encoder_res18(input_dim, dim, kernel_size, num_cnn_blocks=num_cnn_blocks, num_res_blocks=num_res_blocks, learnable=True, return_module_list=True)

        self.codebook = VQEmbedding(K, dim)

        if arch == 'res':
            self.decoder = build_decoder_res18(input_dim, dim, kernel_size, num_cnn_blocks=num_cnn_blocks, num_res_blocks=num_res_blocks)
        elif arch == 'resl':
            self.decoder = build_decoder_res18(input_dim, dim, kernel_size, num_cnn_blocks=num_cnn_blocks, num_res_blocks=num_res_blocks, learnable=True)

        self.apply(weights_init)

    def encode(self, x, return_reps=False):
        if type(self.encoder) == torch.nn.ModuleList:
            encodings = forward_module_list(self.encoder, x)
            z_e_x = encodings[-1]
        else:
            z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        if return_reps:
            z_q_x_st, _ = self.codebook.straight_through(z_e_x)
            reps = z_e_x
            if type(return_reps) == int:
                reps = encodings[return_reps]
            return latents, (z_q_x_st, reps)
        else:
            return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        if type(self.encoder) == torch.nn.ModuleList:
            encodings = forward_module_list(self.encoder, x)
            z_e_x = encodings[-1]
        else:
            z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x

    def calc_reg_loss(self, reg_type=None):
        loss = 0
        reg_type = reg_type or self.reg_type
        if reg_type == 'l1':
            loss = torch.abs(self.codebook.embedding.weight).mean()
        elif reg_type == 'l2':
            loss = torch.square(torch.norm(self.codebook.embedding.weight, dim=1)).mean()

        return loss

    def calc_losses(self, x):
        x_tilde, z_e_x, z_q_x = self.forward(x)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, x)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss_reg = self.calc_reg_loss()

        return loss_recons, loss_vq, loss_commit, loss_reg

    def select_codes(self, indices):
        if not type(indices) == torch.Tensor:
            indices = torch.LongTensor(indices).to(next(self.codebook.parameters()).device)
        return self.codebook.embedding(indices)

