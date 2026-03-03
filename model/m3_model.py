import torch
from torch import nn
from model import resnet
import torch.nn.functional as F

class SharedMLP(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

class PointNet(nn.Module):
    def __init__(self, n_channel=4, embed_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            SharedMLP(n_channel, 64),
            SharedMLP(64, 64),
            SharedMLP(64, 64),
            SharedMLP(64, 128),
            SharedMLP(128, 1024)
        )
        self.decoder = nn.Sequential(
            SharedMLP(1088, embed_dim),
            # Additional layers can be added if needed
        )

    def forward(self, x):
        bs, _, n_pts = x.size()
        x_list = [x]
        for i in range(5):
            x = self.encoder[i](x_list[-1])
            x_list.append(x)

        global_feat = torch.max(x_list[-1], dim=2, keepdim=True)[0]
        global_feat = global_feat.repeat(1, 1, n_pts)
        out = torch.cat([x_list[2], global_feat], dim=1)
        out = self.decoder(out)
        return torch.max(out, 2)[0]  # bs, 512


class Classifier(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embed_dim // 4, 2)
        )

    def forward(self, x1, x2):
        return self.classifier(torch.cat([x1, x2], dim=1))


class Net(nn.Module):
    def __init__(self, embed_dim=512, dataset='kitti'):
        super().__init__()
        self.dataset = dataset

        self.encode_img = resnet.resnet18(embed_dim=embed_dim)
        self.encode_geo = PointNet(n_channel=2, embed_dim=embed_dim)
        self.encode_pts = PointNet(n_channel=7, embed_dim=embed_dim)

        self.classifier1 = Classifier(embed_dim)
        self.classifier2 = Classifier(embed_dim*2)
        self.classifier3 = Classifier(embed_dim*3)

    def forward(self, det_img, det_geo, det_pts, trk_img, trk_geo, trk_pts):

        det_img = self.encode_img(det_img)
        trk_img = self.encode_img(trk_img)

        det_geo = self.encode_geo(det_geo)
        trk_geo = self.encode_geo(trk_geo)

        pred_score1 = self.classifier1(det_img, trk_img)

        pred_score2 = self.classifier2(torch.cat((det_img, det_geo), dim=1),
                                       torch.cat((trk_img, trk_geo), dim=1))

        if self.dataset != 'kitti':
            return pred_score1, pred_score2

        det_pts = torch.cat((det_pts, F.normalize(det_pts[:, :3, :], dim=-1, p=2)), dim=1)
        trk_pts = torch.cat((trk_pts, F.normalize(trk_pts[:, :3, :], dim=-1, p=2)), dim=1)
        det_pts = self.encode_pts(det_pts)
        trk_pts = self.encode_pts(trk_pts)

        pred_score3 = self.classifier3(torch.cat((det_img, det_geo, det_pts), dim=1),
                                       torch.cat((trk_img, trk_geo, trk_pts), dim=1))


        return pred_score1, pred_score2, pred_score3