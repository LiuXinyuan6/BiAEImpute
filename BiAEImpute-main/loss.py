import torch
import torch.nn as nn


# class DiscriminatorLossReal(nn.Module):
#     def __init__(self, p=1):
#         super().__init__()
#
#
#     def forward(self, d_real):
#         # 创建一个与 d_real 形状相同的目标张量，全部为1
#         labels_real = torch.ones_like(d_real)
#         # 使用二元交叉熵损失函数计算损失
#         criterion = nn.BCEWithLogitsLoss()
#         dc_loss_real = criterion(d_real, labels_real)
#         # 计算平均损失
#         mean_loss = torch.mean(dc_loss_real)
#
#         return mean_loss
#
#
# class DiscriminatorLossFake(nn.Module):
#     def __init__(self, p=1):
#         super().__init__()
#
#     def forward(self, d_fake):
#         # 创建一个与 d_fake 形状相同的目标张量，全部为0
#         labels_fake = torch.zeros_like(d_fake)
#         # 使用二元交叉熵损失函数计算损失
#         criterion = nn.BCEWithLogitsLoss()
#         dc_loss_fake = criterion(d_fake, labels_fake)
#         # 计算平均损失
#         mean_loss = torch.mean(dc_loss_fake)
#
#         return mean_loss
#
#
# class GeneratorLoss(nn.Module):
#     def __init__(self, p=1):
#         super().__init__()
#
#     def forward(self, d_fake):
#         # 创建一个与 d_real 形状相同的目标张量，全部为1
#         labels_real = torch.ones_like(d_fake)
#         # 使用二元交叉熵损失函数计算损失
#         criterion = nn.BCEWithLogitsLoss()
#         dc_loss_real = criterion(d_fake, labels_real)
#         # 计算平均损失
#         mean_loss = torch.mean(dc_loss_real)
#
#         return mean_loss




class SquareRegularizeLoss(nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p

    ##input:samples*features
    ##regularized loss:1/n*(|1-(x1^2+...+xn^2)|^p)
    def forward(self, input):
        feature_num = input.size(1)
        input = torch.pow(input, 2).sum(dim=1)
        if self.p == 1:
            loss = torch.abs(1 - input)
        else:
            loss = torch.pow(1 - input, self.p)
        loss = loss.mean() / feature_num

        return loss
