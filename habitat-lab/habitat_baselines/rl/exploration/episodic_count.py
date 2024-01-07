import os
import numpy as np
import copy
import torch
from torchvision import transforms
from vqvae import VectorQuantizedVAE

MP3D_OBS_SHAPE = (480, 640, 3)

class EpisodicHashMemory:
    def __init__(self, memory_size):
        self.hash = {}
        self.memory_size = memory_size

    def add(self, keys, increment=1):
        if not keys in self.hash:
            self.hash[keys] = 0
        self.hash[keys] += increment
        if len(self.hash) > self.memory_size:
            self.hash.popitem()
        return self.hash(keys)

class VQRepresentation:
    def __init__(self, config, obs_shape=MP3D_OBS_SHAPE):
        self.cfg = config
        self.reg_cfg = self.config.reg_cfg
        self.reg_coef = self.config.reg_coef
        self.arch = self.config.arch
        self.model = VectorQuantizedVAE(3, self.cfg.cell_dim,
                                        K=self.cfg.depth,
                                        num_cnn_blocks=self.cfg.num_cnn_blocks,
                                        num_res_blocks=self.cfg.num_res_blocks,
                                        reg_type=self.reg_cfg, arch=self.arch)
        self.extended = False
        if self.cfg.ext_dim is None:
            self.extended = True

        _transforms = []
        if obs_shape != self.cfg.img_shape:
            _transforms.append(transforms.CenterCrop(obs_shape))
        self.transform = transforms.Compose(_transforms)

    def calc_loss(self, obs, output_dict, ext_data=None):
        obs = self.transforms(obs)
        obs = self.get_obs_norm(obs)

        if self.extended:
            assert ext_data is not None
            loss_recons, loss_vq, loss_commit, loss_reg = self.model.calc_losses(obs, ext_data)
        else:
            loss_recons, loss_vq, loss_commit, loss_reg = self.model.calc_losses(obs)
        loss = loss_recons + loss_vq + loss_commit
        if self.reg_coef > 0:
            loss += self.reg_coef * loss_reg

        output_dict['loss_recons'] = loss_recons
        output_dict['loss_vq'] = loss_vq
        output_dict['loss_reg'] = loss_reg

        return loss

    def get_representations(self, ids):
        return self.model.select_codes(ids) # 根据ids获取Embedding中的离散编码

    def forward(self, obs, ext_data=None, return_reps=False):
        obs = self.transforms(obs)
        obs = self.get_obs_norm(obs)
        if self.extended:
            return self.model.encode(obs, ext_data, return_reps=return_reps)
        else:
            return self.model.encode(obs, return_reps=return_reps)

class EpisodicCount:

    def __init__(self,cfg):
        self.cfg = cfg
        self.memory = EpisodicCount(self.cfg.memory_size)
        self.vq_rep = VQRepresentation(cfg)
        self.optimizer = torch.optim.AdamW(self.vq_rep.model.parameters())

    def _maybe_load_pretrained_cell_rep(self):
        if self.learnable_hash and self.cfg.pretrained_vq_path is not None:
            ckpt = torch.load(self.cfg.pretrained_vq_path)
            self.vq_rep.load_state_dict(ckpt, strict=False)
            print(f'Successfully loaded pretrained vq-rep from {self.cfg.pretrained_vq_path}')

    def calc_loss(self, obs, output_dict, ext_data=None):
        if self._freezing:
            with torch.no_grad():
                return self.vq_rep.calc_loss(obs, output_dict, ext_data=ext_data)
        else:
            return self.vq_rep.calc_loss(obs, output_dict, ext_data=ext_data)

    def obs2cell(self, obs, to_numpy=True, ext_data=None, return_reps=False):
        reps = None
        if self.vq_rep is not None:
            if return_reps:
                cells, reps = self.vq_rep(obs, ext_data=ext_data, return_reps=return_reps)
            else:
                cells = self.vq_rep(obs, ext_data=ext_data)
        else:
            cells = obs

        if to_numpy and type(cells) == torch.Tensor:
            cells = cells.detach().cpu().numpy().astype(np.uint8)

        if return_reps:
            return cells, reps
        return cells

    def get_representations(self, ids):
        return self.vq_rep.get_representations(ids)

    @staticmethod
    def cell2key(cell, act=None, rew=None):
        items = cell.reshape(-1).tolist()
        if act is not None:
            items.append(act)
        if rew is not None:
            items.append(rew)
        key = '_'.join(map(str, items))
        return key

    def prepare_items(self, obs, stats, ext_data=None, actions=None, rewards=None):
        cells = self.obs2cell(obs, stats=stats, ext_data=ext_data)
        if actions is not None:
            actions = actions.detach().cpu().numpy().astype(np.uint8)
        return cells, actions, rewards

    def count(self, cells, actions=None, rewards=None):
        rets = []
        for i, cell in cells:
            act = None if actions is None else actions[i]
            rew = None if rewards is None else rewards[i]
            cell_key = self.cell2key(cell, act=act, rew=rew)
            out = self.memory.add(cell_key)
            rets.append(out)
        return rets

    def add(self, obs, ext_data=None, actions=None, rewards=None):
        cells, actions, rewards = self.prepare_items(obs, ext_data=ext_data, actions=actions, rewards=rewards) # cells: (20, 3, 3) [[6, 1, 5],[7, 1, 5],]
        outs = self.count(cells, actions, rewards)
        return outs

    def cal_init_reward(self, obs, actions=None, rewards=None):
        eposidic_count = self.add(obs, actions=actions, rewards=rewards)
        init_reward = np.sum(1 / np.sqrt(eposidic_count))
        return init_reward

    def update_model(self, obs):
        self.optimizer.zero_grad()
        loss = self.calc_loss(obs, output_dict={})
        loss.backward()
        self.optimizer.step()
        return {"episodic_count_loss": loss}

    def save(self, save_dir, step):
        checkpoint = {
            'vq_vae': self.vq_rep.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(
            checkpoint,
            os.path.join(save_dir, "episodic_count_{}.pth".format(int(step)))
        )







