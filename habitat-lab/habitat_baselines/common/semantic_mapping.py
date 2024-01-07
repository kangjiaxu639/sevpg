import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import habitat_baselines.common.map_utils.depth_utils as du
from habitat_baselines.common.map_utils.model import get_grid, ChannelPool
class Semantic_Mapping(nn.Module):
    """
    Semantic_Mapping
    """

    def __init__(self, args, device):
        super(Semantic_Mapping, self).__init__()
        # print(args.device)
        # exit(0)
        self.device = device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories

        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.camera_height * 100.
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        self.pool = ChannelPool(1)

        vr = self.vision_range

        self.init_grid = torch.zeros(
            args.num_processes, 1 + self.num_sem_categories, vr, vr,
                                self.max_height - self.min_height
        ).float().to(self.device)
        self.feat = torch.ones(
            args.num_processes, 1 + self.num_sem_categories,
                                self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

    def forward(self, obs, pose_obs, maps_last, poses_last, origins,
                observation_points, gl_tree_list, infos, args):

        bs, c, h, w = obs.size()  # c:28 bs:2 h:120 w:160
        depth = obs[:, 3, :, :]  # (bs, h, w)

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device,
            scale=self.du_scale)  # (bs, h, w, 3)

        point_cloud_t_3d = point_cloud_t.clone()  # depth中重构点云

        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, 0,
            self.device)  # 坐标系变换 (bs, h, w, 3)

        agent_view_t_3d = point_cloud_t.clone()

        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device)  # (bs, h, w 3)

        # from matplotlib import pyplot as plt
        # fig = plt.figure()
        # ax1 = plt.axes(projection='3d')
        # ax1.set_xlabel('x', size=20)
        # ax1.set_ylabel('y', size=20)
        # ax1.set_zlabel('z', size=20)
        # ax1.scatter3D(agent_view_centered_t.cpu()[1, :, :, 0], agent_view_centered_t.cpu()[1, :, :, 1], agent_view_centered_t.cpu()[1, :, :, 2],
        #               cmap='Blues')
        # plt.show()

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.

        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
            obs[:, 4:4 + (self.num_sem_categories), :, :]
        ).view(bs, self.num_sem_categories,
               h // self.du_scale * w // self.du_scale)

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[
                                         3])  # (2, 3, 19200)

        voxels = du.splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2,
                                                                  3)  # （bs, 23, 100, 100, 80)

        min_z = int(25 / z_resolution - min_h)  # min_z: 13
        max_z = int(
            (self.agent_height + 1) / z_resolution - min_h)  # max_z: 25

        agent_height_proj = voxels[..., min_z:max_z].sum(
            4)  # (bs, 23, 100, 100)
        all_height_proj = voxels.sum(4)  # (bs, 23, 100, 100) voxels -> map

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        # agent_view size: (bs, 26, 240, 240)
        channel = c
        # if args.dataset == 'mp3d':
        #     channel = channel - 2  # -2 including, entropy, goal
        agent_view = torch.zeros(bs, channel,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred

        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                          torch.sin(pose[:, 2] / 57.29577951308232) \
                          + rel_pose_change[:, 1] * \
                          torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                          torch.cos(pose[:, 2] / 57.29577951308232) \
                          - rel_pose_change[:, 1] * \
                          torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) / \
                         (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat,
                                align_corners=True)  # (bs, 26, 240, 240)
        translated = F.grid_sample(rotated, trans_mat,
                                   align_corners=True)  # (bs, 26, 240, 240)

        points_pose = current_poses.clone()  # (bs, 3)
        points_pose[:, :2] = points_pose[:, :2] + torch.from_numpy(
            origins[:, :2]).to(self.device).float()

        points_pose[:, 2] = points_pose[:, 2] * np.pi / 180
        points_pose[:, :2] = points_pose[:, :2] * 100

        goal_maps = torch.zeros([bs, 1, 240, 240], dtype=float)

        for e in range(bs):

            world_view_t = du.transform_pose_t2(
                agent_view_t_3d[e, ...], points_pose[e, ...].cpu().numpy(),
                self.device).reshape(-1, 3)  # (19200, 3) RGB

            # world_view_sem_t: (19200, 22)
            world_view_sem_t = obs[e, 4:4 + (self.num_sem_categories), :,
                               :].reshape((self.num_sem_categories),
                                          -1).transpose(0, 1)

            # filter 过滤点云
            non_zero_row_1 = torch.abs(
                point_cloud_t_3d[e, ...].reshape(-1, 3)).sum(
                dim=1) > 0  # (19200,)
            non_zero_row_2 = torch.abs(world_view_sem_t).sum(
                dim=1) > 0  # (19200,)
            non_zero_row_3 = torch.argmax(world_view_sem_t,
                                          dim=1) != self.num_sem_categories - 1  # (19200,)

            non_zero_row = non_zero_row_1 & non_zero_row_2 & non_zero_row_3 # (19200,)
            # world_view_sem = world_view_sem_t[
            #     non_zero_row].cpu().numpy()  # (num, 22)
            world_view_sem = world_view_sem_t.cpu().numpy()  # (num, 22)


            if world_view_sem.shape[0] < 50:
                continue

            world_view_label = np.argmax(world_view_sem, axis=1)  # (1600,)

            world_view_rgb = obs[e, :3, :, :].permute(1, 2, 0).reshape(-1, 3)[
                non_zero_row].cpu().numpy()  # (1600, 3)
            world_view_t = world_view_t[
                non_zero_row].cpu().numpy()  # (pixels_num, 3)

            # from world_view of current frame sample 512 points and every point has 8 neighbors
            if world_view_t.shape[0] >= 512:
                indx = np.random.choice(world_view_t.shape[0], 512,
                                        replace=False)  # (512, )
            else:
                indx = np.linspace(0, world_view_t.shape[0] - 1,
                                   world_view_t.shape[0]).astype(np.int32)

            gl_tree = gl_tree_list[e]
            gl_tree.init_points_node(
                world_view_t[indx])  # every point init a octree
            per_frame_nodes = gl_tree.add_points(world_view_t[indx],
                                                 world_view_sem[indx],
                                                 world_view_rgb[indx],
                                                 world_view_label[indx],
                                                 infos[e]['timestep'])
            scene_nodes = gl_tree.all_points()
            gl_tree.update_neighbor_points(per_frame_nodes)

            sample_points_tensor = torch.tensor(
                gl_tree.sample_points())  # local map

            sample_points_tensor[:, :2] = sample_points_tensor[:,
                                          :2] - origins[e, :2] * 100
            sample_points_tensor[:, 2] = sample_points_tensor[:,
                                         2] - 0.88 * 100
            sample_points_tensor[:, :3] = sample_points_tensor[:,
                                          :3] / args.map_resolution

            observation_points[e] = sample_points_tensor.transpose(1, 0)

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)

        return fp_map_pred, map_pred, pose_pred, current_poses, observation_points
