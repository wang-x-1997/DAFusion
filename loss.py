import torch
import torch.nn as nn
import torch.nn.functional as F


def cos_simi(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return torch.clamp(sim, min=0.0005, max=0.9995)


def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return 1 - sim


def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.size()

    # embedded_fg = F.normalize(embedded_fg, dim=1)
    # embedded_bg = F.normalize(embedded_bg, dim=1)

    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)

    return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C

# Minimize Similarity, e.g., push representation of foreground and background apart.
class SimMinLoss(nn.Module):
    def __init__(self, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


# Maximize Similarity, e.g., pull representation of background and background together.
class SimMaxLoss(nn.Module):
    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg , embedded_bg1):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError

        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_bg1)
            loss = -torch.log(sim)
            loss[loss < 0] = 0
            _, indices = sim.sort(descending=True, dim=1)
            _, rank = indices.sort(dim=1)
            rank = rank - 1
            rank_weights = torch.exp(-rank.float() * self.alpha)
            loss = loss * rank_weights
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


class Contrastive_Euc_loss(nn.Module):
    def __init__(self, margin=0.4): # 2-> best=0.4
        super(Contrastive_Euc_loss, self).__init__()
        self.margin = margin

    def forward(self, z1, z2, label):
        # print(type(z1))
        dist = torch.pairwise_distance(z1, z2, p=2)
        # dist = torch.cosine_similarity(z1, z2)
        # term1 = (label) * (torch.pow(dist, 2))
        # term2 = (1 - label) * (torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        loss = (1 - label) * torch.pow(dist, 2) + label * torch.pow(torch.clamp(self.margin - dist, min=0),
                                                                             2)
        # return torch.mean(term1 + term2)
        loss = torch.mean(loss)
        return loss

class Contrastive_cos_loss(nn.Module):
    def __init__(self, margin=0.5):
        super(Contrastive_cos_loss, self).__init__()
        self.margin = margin

    def forward(self, z1, z2, label):
        # print(type(z1))
        # dist = torch.pairwise_distance(z1, z2, p=2)
        dist = torch.cosine_similarity(z1, z2)
        # term1 = (label) * (torch.pow(dist, 2))
        # term2 = (1 - label) * (torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        loss = (1 - label) * torch.pow(dist, 2) + label * torch.pow(torch.clamp(self.margin - dist, min=0),
                                                                             2)
        # return torch.mean(term1 + term2)
        loss = torch.mean(loss)
        return loss


# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=1.5):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#
#
#     def forward(self, x, y,label):
#         similariti = torch.cosine_similarity(x, y)
#         loss = (1 - label) * torch.pow(similariti, 2) + label * torch.pow(torch.clamp(self.margin - similariti, min=0),
#                                                                     2)
#         return loss
#
# x1 = torch.randn([1, 5])
# x2 = torch.randn([1, 5])
# label = 1
#
# loss = ContrastiveLoss()(x1, x2, label)
# print(loss)



# class ContrastiveLoss(torch.nn.Module):
#     """
#     Contrastive loss function.
#     Based on:
#     """
#
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#
#     def check_type_forward(self, in_types):
#         assert len(in_types) == 3
#
#         x0_type, x1_type, y_type = in_types
#         assert x0_type.size() == x1_type.shape
#         assert x1_type.size()[0] == y_type.shape[0]
#         assert x1_type.size()[0] > 0
#         assert x0_type.dim() == 2
#         assert x1_type.dim() == 2
#         assert y_type.dim() == 1
#
#     def forward(self, x0, x1, y):
#         self.check_type_forward((x0, x1, y))
#
#         # euclidian distance
#         diff = x0 - x1
#         dist_sq = torch.sum(torch.pow(diff, 2), 1)
#         dist = torch.sqrt(dist_sq)
#
#         mdist = self.margin - dist
#         dist = torch.clamp(mdist, min=0.0)
#         loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
#         loss = torch.sum(loss) / 2.0 / x0.size()[0]
#         return loss
#
# class ContrastiveLoss(torch.nn.Module):
#
#     def __init__(self, margin=1.0):
#         self.margin = margin
#
#
#     def forward(self, x0, x1, y):
#         self.check_type_forward((x0, x1, y))
#
#         # euclidian distance
#         diff = x0 - x1
#         dist_sq = torch.sum(torch.pow(diff, 2), 1)
#         dist = torch.sqrt(dist_sq)
#
#         mdist = self.margin - dist
#         dist = torch.clamp(mdist, min=0.0)
#         loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
#         loss = torch.sum(loss) / 2.0 / x0.size()[0]
#
#         return loss




if __name__ == '__main__':
    fg_embedding = torch.randn((4, 12))
    bg_embedding = torch.randn((4, 12))
    # print(fg_embedding, bg_embedding)

    # neg_contrast = NegContrastiveLoss(metric='cos')
    # neg_loss = neg_contrast(fg_embedding, bg_embedding)
    # print(neg_loss)

    # pos_contrast = PosContrastiveLoss(metric='cos')
    # pos_loss = pos_contrast(fg_embedding)
    # print(pos_loss)

    examplar = torch.tensor([[1, 2, 3, 4], [2, 3, 1, 4], [4, 2, 1, 3]])

    _, indices = examplar.sort(descending=True, dim=1)
    print(indices)
    _, rank = indices.sort(dim=1)
    print(rank)
    rank_weights = torch.exp(-rank.float() * 0.25)
    print(rank_weights)
