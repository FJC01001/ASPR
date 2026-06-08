import model.resnet as resnet

import torch
from torch import nn
import torch.nn.functional as F
import pdb
import random


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class FiLMBlock(nn.Module):

    def __init__(self, channels, hidden_ratio=0.25):
        super().__init__()
        hidden = max(4, int(channels * hidden_ratio))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, kernel_size=1, bias=True),
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        s = self.gap(x)               # [B,C,1,1]
        h = self.mlp(s)               # [B,2C,1,1]
        gamma, beta = torch.chunk(h, 2, dim=1)
        x_film = (1.0 + gamma) * x + beta
        return x_film, gamma, beta


class GateBlock(nn.Module):

    def __init__(self, channels, hidden_ratio=0.25, bias_init=-1.5):
        super().__init__()
        hidden = max(4, int(channels * hidden_ratio))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),  # ← 从1改为channels
            nn.Sigmoid()
        )

        with torch.no_grad():
            self.mlp[-2].bias.fill_(bias_init)

    def forward(self, x_orig, x_rect):
        diff = x_rect - x_orig                  # [B,C,H,W]
        g = self.mlp(self.gap(diff))            # [B,C,1,1]（通道级门控）
        x_out = x_orig + g * diff               # 广播到 H,W
        return x_out, g




class SSP_MatchingNet(nn.Module):
    def __init__(self, backbone, dim_ls, local_noise_std=0.75):
        super(SSP_MatchingNet, self).__init__()
        backbone = resnet.__dict__[backbone](pretrained=True)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.layer_block = [self.layer0, self.layer1, self.layer2, self.layer3]

        self.local_noise_std = local_noise_std

        self.T_local = 2

        self.tau_local = 0.7

        self.sigma_min = 1e-3

        self.eps_base = 2.0 / 255.0

        self.perturb_layers_idx = {}
        self.perturb_layers = [0, 1, 2]
        self.perturb_layer_dim = dim_ls

        self.DR_Adapter = nn.ModuleList()
        self.FiLM = nn.ModuleList()
        self.Gate = nn.ModuleList()
        for idx, layer in enumerate(self.perturb_layers):
            self.perturb_layers_idx[layer] = idx

            # === Parallel Multi-Scale Adapter ===
            in_ch = self.perturb_layer_dim[idx]

            Adapter = nn.ModuleDict({

                "conv1": nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, dilation=1, padding=1),
                    LayerNorm2d(in_ch),
                    nn.ReLU(inplace=True)
                ),


                "conv2": nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, dilation=2, padding=2),
                    LayerNorm2d(in_ch),
                    nn.ReLU(inplace=True)
                ),


                "conv3": nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, dilation=4, padding=4),
                    LayerNorm2d(in_ch),
                    nn.ReLU(inplace=True)
                ),

                # --- 融合层 ---
                "fuse": nn.Sequential(
                    nn.Conv2d(in_ch * 3, in_ch, kernel_size=1, stride=1),
                    LayerNorm2d(in_ch),
                    nn.ReLU(inplace=True)
                )
            })

            self.DR_Adapter.append(Adapter)

            ch = self.perturb_layer_dim[idx]
            self.FiLM.append(FiLMBlock(ch, hidden_ratio=0.25))
            self.Gate.append(GateBlock(ch, hidden_ratio=0.25))

    def _assp_update(self, x, y_mask, eps_scale, T, tau, step_scale=1.0):

        B, C, H, W = x.shape
        eps = 1e-6


        mu0 = x.mean(dim=(2, 3), keepdim=True).detach()
        var0 = x.var(dim=(2, 3), keepdim=True).detach()
        std0 = torch.sqrt(var0 + eps)


        Fnorm = (x - mu0) / std0
        mu = mu0.clone().detach().requires_grad_(True)
        std = std0.clone().detach().requires_grad_(True)


        def _ce_logits_and_loss(fmap, y):
            B, C, H, W = fmap.shape
            y_ds = F.interpolate(y.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1).long()
            fg_mask = (y_ds == 1).float()
            bg_mask = (y_ds == 0).float()
            fg_proto = self.masked_average_pooling(fmap, fg_mask)
            bg_proto = self.masked_average_pooling(fmap, bg_mask)
            logits = self.similarity_func(fmap, fg_proto[..., None, None], bg_proto[..., None, None])
            y_ce = torch.clamp(y_ds, max=1)
            loss = F.cross_entropy(logits, y_ce, ignore_index=255, reduction='mean')
            with torch.no_grad():
                prob = logits.softmax(1)
                yv = y_ce.clone();
                yv[yv == 255] = 0
                conf_map = prob.gather(1, yv.unsqueeze(1)).squeeze(1)
                valid = (y_ce != 255).float()
                conf = (conf_map * valid).sum() / (valid.sum() + 1e-6)
            return logits, loss, conf


        step = self.eps_base * eps_scale * step_scale  # eps_base=2/255


        for t in range(T):
            x_t = Fnorm * std + mu
            _, loss_style, conf = _ce_logits_and_loss(x_t, y_mask)

            g_mu, g_std = torch.autograd.grad(loss_style, [mu, std], retain_graph=False, create_graph=False)

            with torch.no_grad():

                mu += step * g_mu.sign()
                std += step * g_std.sign()

                std.copy_(torch.clamp(std, min=1e-6))


            if conf <= tau:
                break


        x_adv = Fnorm * std + mu
        return x_adv.detach(), mu.detach(), std.detach()

    def forward(self, img_s_list, mask_s_list, img_q, mask_q, training=True, get_prot=False):

        h, w = img_q.shape[-2:]
        if training:
            p_local = random.random()
        else:
            p_local = 1.0

        feature_s_list = []
        x_params = []
        x_layer_style = []
        x_ori_vec = []


        s_0 = img_s_list[0]
        q_0 = img_q
        for idx, layer in enumerate(self.layer_block):
            s_0 = layer(s_0)
            q_0 = layer(q_0)

            if idx in self.perturb_layers:

                x_q_mean = q_0.mean(dim=(2, 3), keepdim=True).detach()
                x_s_mean = s_0.mean(dim=(2, 3), keepdim=True).detach()
                x_q_var = q_0.var(dim=(2, 3), keepdim=True).detach()
                x_s_var = s_0.var(dim=(2, 3), keepdim=True).detach()
                x_ori_mean = torch.cat((x_q_mean, x_s_mean), dim=0)
                x_ori_var = torch.cat((x_q_var, x_s_var), dim=0)
                x_layer_style.append([x_ori_mean, x_ori_var])
                if get_prot:
                    continue

                x_q_disturb, x_s_disturb = q_0, s_0
                flag = 0
                use_alpha = use_beta = None
                x_ori_mean = x_ori_var = None


                if p_local < 0.7:


                    x_q_disturb, x_s_disturb, alpha, beta = self.local_perturb(
                        x_q_disturb, x_s_disturb, mask_q, mask_s_list[0]
                    )
                    use_alpha, use_beta = alpha.detach(), beta.detach()
                    flag = 1


                    x_q_m0 = x_q_disturb.mean(dim=(2, 3), keepdim=True)
                    x_q_s0 = torch.sqrt(x_q_disturb.var(dim=(2, 3), keepdim=True) + 1e-6)
                    x_s_m0 = s_0.mean(dim=(2, 3), keepdim=True)
                    x_s_s0 = torch.sqrt(s_0.var(dim=(2, 3), keepdim=True) + 1e-6)
                    x_ori_mean = torch.cat((x_q_m0, x_s_m0), dim=0)
                    x_ori_var = torch.cat((x_q_s0, x_s_s0), dim=0)


                if flag == 1 or not training:
                    dist_stats = (
                        x_q_disturb.mean(dim=(2, 3), keepdim=True),
                        torch.sqrt(x_q_disturb.var(dim=(2, 3), keepdim=True) + 1e-6),
                        x_s_disturb.mean(dim=(2, 3), keepdim=True),
                        torch.sqrt(x_s_disturb.var(dim=(2, 3), keepdim=True) + 1e-6),
                    )
                    x_q_rectify, x_s_rectify, x_q_m, x_q_v, x_s_m, x_s_v = self.domain_rectify(
                        x_q_disturb, x_s_disturb, idx, dist_stats
                    )
                    q_0 = x_q_rectify
                    s_0 = x_s_rectify


                if training and flag == 1:
                    sr_q_m, sr_q_v = self.cyclic_rectify(idx, use_alpha, use_beta, x_q_m, x_q_v, q_0)
                    sr_s_m, sr_s_v = self.cyclic_rectify(idx, use_alpha, use_beta, x_s_m, x_s_v, s_0)

                    x_param = (
                        x_ori_mean, x_ori_var,
                        torch.cat((x_q_m, x_s_m), dim=0),
                        torch.cat((x_q_v, x_s_v), dim=0),
                        torch.cat((sr_q_m, sr_s_m), dim=0),
                        torch.cat((sr_q_v, sr_s_v), dim=0),
                    )
                    x_params.append(x_param)

        feature_q = q_0
        feature_s_list.append(s_0)


        if len(img_s_list) > 1:
            for k in range(1, len(img_s_list)):
                s_k = img_s_list[k]
                q_k = img_q
                for idx, layer in enumerate(self.layer_block):
                    s_k = layer(s_k)
                    q_k = layer(q_k)
                    if (not training) and (idx in self.perturb_layers):
                        x_q_m = q_k.mean(dim=(2, 3), keepdim=True)
                        x_q_v = torch.sqrt(q_k.var(dim=(2, 3), keepdim=True) + 1e-6)
                        x_s_m = s_k.mean(dim=(2, 3), keepdim=True)
                        x_s_v = torch.sqrt(s_k.var(dim=(2, 3), keepdim=True) + 1e-6)
                        _, s_k, _, _, _, _ = self.domain_rectify(
                            q_k, s_k, idx, (x_q_m, x_q_v, x_s_m, x_s_v)
                        )
                feature_s_list.append(s_k)


        feature_fg_list, feature_bg_list, supp_out_ls = [], [], []
        for k in range(len(img_s_list)):
            fg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 1).float())[None, :]
            bg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 0).float())[None, :]
            feature_fg_list.append(fg)
            feature_bg_list.append(bg)

            if self.training:
                sim_fg = F.cosine_similarity(feature_s_list[k], fg.squeeze(0)[..., None, None], dim=1)
                sim_bg = F.cosine_similarity(feature_s_list[k], bg.squeeze(0)[..., None, None], dim=1)
                supp_out = torch.cat((sim_bg[:, None], sim_fg[:, None]), dim=1) * 10.0
                supp_out = F.interpolate(supp_out, size=(h, w), mode="bilinear", align_corners=True)
                supp_out_ls.append(supp_out)

        FP = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        BP = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)

        out_0 = self.similarity_func(feature_q, FP, BP)
        SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feature_q, out_0)

        FP_1 = FP * 0.5 + SSFP_1 * 0.5
        BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7
        out_1 = self.similarity_func(feature_q, FP_1, BP_1)
        out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)

        out_ls = [out_1]

        if self.training:
            fg_q = self.masked_average_pooling(feature_q, (mask_q == 1).float())[None, :].squeeze(0)
            bg_q = self.masked_average_pooling(feature_q, (mask_q == 0).float())[None, :].squeeze(0)
            sim_fg = F.cosine_similarity(feature_q, fg_q[..., None, None], dim=1)
            sim_bg = F.cosine_similarity(feature_q, bg_q[..., None, None], dim=1)
            self_out = torch.cat((sim_bg[:, None], sim_fg[:, None]), dim=1) * 10.0
            self_out = F.interpolate(self_out, size=(h, w), mode="bilinear", align_corners=True)
            out_ls.extend([self_out, torch.cat(supp_out_ls, 0)])

        out_ls.extend([x_ori_vec, x_params, x_layer_style])
        return out_ls

    def cyclic_rectify(self, idx, alpha, beta, x_rect_miu, x_rect_sigma, feature):

        eps = 1e-6
        lid = self.perturb_layers_idx[idx]


        second_dist_mean = (1.0 + beta) * x_rect_miu
        second_dist_sigma = (1.0 + alpha) * x_rect_sigma


        second_dist = ((feature - x_rect_miu) / (x_rect_sigma + eps)) * second_dist_sigma + second_dist_mean


        second_dist_mod, _, _ = self.FiLM[lid](second_dist)


        Adapter = self.DR_Adapter[lid]
        x_b1 = Adapter["conv1"](second_dist_mod)
        x_b2 = Adapter["conv2"](second_dist_mod)
        x_b3 = Adapter["conv3"](second_dist_mod)
        x_cat = torch.cat([x_b1, x_b2, x_b3], dim=1)
        second_rect = Adapter["fuse"](x_cat)


        second_rect_beta = second_rect.mean(dim=(2, 3), keepdim=True)
        second_rect_alpha = torch.sqrt(second_rect.var(dim=(2, 3), keepdim=True) + eps)


        second_rect_miu = (1.0 + second_rect_beta) * second_dist_mean
        second_rect_sigma = (1.0 + second_rect_alpha) * second_dist_sigma

        second_rectified = ((second_dist - second_dist_mean) / (second_dist_sigma + eps)) \
                           * second_rect_sigma + second_rect_miu


        second_out, _ = self.Gate[lid](second_dist, second_rectified)
        second_rect_miu = second_out.mean(dim=(2, 3), keepdim=True)
        second_rect_sigma = torch.sqrt(second_out.var(dim=(2, 3), keepdim=True) + eps)


        return second_rect_miu, second_rect_sigma

    def local_perturb(self, x_q_disturb, x_s_disturb, mask_q, mask_s):
        C = x_q_disturb.shape[1]
        eps_scale = 64.0 / float(C)
        eps = 1e-6

        q_adv, q_mu, q_std = self._assp_update(
            x_q_disturb, mask_q, eps_scale=eps_scale, T=self.T_local, tau=self.tau_local, step_scale=1.0
        )
        s_adv, s_mu, s_std = self._assp_update(
            x_s_disturb, mask_s, eps_scale=eps_scale, T=self.T_local, tau=self.tau_local, step_scale=1.0
        )


        q_mu0 = x_q_disturb.mean(dim=(2, 3), keepdim=True)
        q_std0 = torch.sqrt(x_q_disturb.var(dim=(2, 3), keepdim=True) + eps)
        s_mu0 = x_s_disturb.mean(dim=(2, 3), keepdim=True)
        s_std0 = torch.sqrt(x_s_disturb.var(dim=(2, 3), keepdim=True) + eps)

        beta_q = q_mu / (q_mu0 + eps) - 1.0
        alpha_q = q_std / (q_std0 + eps) - 1.0
        beta_s = s_mu / (s_mu0 + eps) - 1.0
        alpha_s = s_std / (s_std0 + eps) - 1.0
        alpha = 0.5 * (alpha_q + alpha_s)
        beta = 0.5 * (beta_q + beta_s)
        return q_adv, s_adv, alpha.detach(), beta.detach()

    def domain_rectify(self, x_q_disturb, x_s_disturb, idx, dist_statistics):

        eps = 1e-6
        x_q_disturb_miu, x_q_disturb_sigma, x_s_disturb_miu, x_s_disturb_sigma = dist_statistics
        lid = self.perturb_layers_idx[idx]


        x_q_mod, _, _ = self.FiLM[lid](x_q_disturb)
        x_s_mod, _, _ = self.FiLM[lid](x_s_disturb)


        Adapter = self.DR_Adapter[lid]

        # --- Query ---
        q_b1 = Adapter["conv1"](x_q_mod)
        q_b2 = Adapter["conv2"](x_q_mod)
        q_b3 = Adapter["conv3"](x_q_mod)
        q_cat = torch.cat([q_b1, q_b2, q_b3], dim=1)
        x_q_rect = Adapter["fuse"](q_cat)

        x_q_rect_beta = x_q_rect.mean(dim=(2, 3), keepdim=True)
        x_q_rect_alpha = torch.sqrt(x_q_rect.var(dim=(2, 3), keepdim=True) + eps)

        # --- Support ---
        s_b1 = Adapter["conv1"](x_s_mod)
        s_b2 = Adapter["conv2"](x_s_mod)
        s_b3 = Adapter["conv3"](x_s_mod)
        s_cat = torch.cat([s_b1, s_b2, s_b3], dim=1)
        x_s_rect = Adapter["fuse"](s_cat)

        x_s_rect_beta = x_s_rect.mean(dim=(2, 3), keepdim=True)
        x_s_rect_alpha = torch.sqrt(x_s_rect.var(dim=(2, 3), keepdim=True) + eps)


        x_q_rect_miu = (1.0 + x_q_rect_beta) * x_q_disturb_miu
        x_q_rect_sigma = (1.0 + x_q_rect_alpha) * x_q_disturb_sigma
        x_q_rectify = ((x_q_disturb - x_q_disturb_miu) /
                       (x_q_disturb_sigma + eps)) * x_q_rect_sigma + x_q_rect_miu

        x_s_rect_miu = (1.0 + x_s_rect_beta) * x_s_disturb_miu
        x_s_rect_sigma = (1.0 + x_s_rect_alpha) * x_s_disturb_sigma
        x_s_rectify = ((x_s_disturb - x_s_disturb_miu) /
                       (x_s_disturb_sigma + eps)) * x_s_rect_sigma + x_s_rect_miu


        x_q_out, _ = self.Gate[lid](x_q_disturb, x_q_rectify)
        x_s_out, _ = self.Gate[lid](x_s_disturb, x_s_rectify)


        return x_q_out, x_s_out, x_q_rect_miu, x_q_rect_sigma, x_s_rect_miu, x_s_rect_sigma

    def SSP_func(self, feature_q, out):
        bs = feature_q.shape[0]
        pred_1 = out.softmax(1)
        pred_1 = pred_1.view(bs, 2, -1)
        pred_fg = pred_1[:, 1]
        pred_bg = pred_1[:, 0]
        fg_ls = []
        bg_ls = []
        fg_local_ls = []
        bg_local_ls = []
        for epi in range(bs):
            fg_thres = 0.7
            bg_thres = 0.6
            cur_feat = feature_q[epi].view(1024, -1)
            f_h, f_w = feature_q[epi].shape[-2:]
            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi] > fg_thres)]
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices]
            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi] > bg_thres)]
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices]
            # global proto
            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

            # local proto
            fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True)
            bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True)
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True)

            cur_feat_norm_t = cur_feat_norm.t()  # N3, 1024
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0

            fg_sim = fg_sim.softmax(-1)
            bg_sim = bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t())
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t())

            fg_proto_local = fg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0)
            bg_proto_local = bg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0)

            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)


        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)


        new_fg_local = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg_local = torch.cat(bg_local_ls, 0)

        return new_fg, new_bg, new_fg_local, new_bg_local

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature
