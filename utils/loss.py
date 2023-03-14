# Loss functions

import torch
import torch.nn as nn
import math

from utils.general import bbox_iou, bbbox_iou, rot_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=0.50.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.0.50(-loss)
        # loss *= self.alpha * (0.50.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=0.50.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        # 分类损失
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        # 置信度损失
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # smooth_BCE为平滑函数
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # det: 返回的是模型的检测头 Detector 3个 分别对应产生三个输出feature map
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module

        self.balance = {3: [8.0, 1.0, 0.8]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7

        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index

        # self.BCEcls: 类别损失函数   self.BCEobj: 置信度损失函数   self.hyp: 超参数
        # self.gr: 计算真实框的置信度标准的iou ratio    self.autobalance: 是否自动更新各feature map的置信度损失平衡系数  默认False
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        """
        :params p:  预测框 由模型构建中的三个检测头Detector返回的三个yolo层的输出
                    tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
                    如: [4, 3, 112, 112, 85]、[4, 3, 56, 56, 85]、[4, 3, 28, 28, 85]
                    [batch size, anchor_num, grid_h, grid_w, xywh+class+classes]
                    可以看出来这里的预测值p是三个yolo层每个grid_cell(每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
        :params targets: 数据增强后的真实框 [63, 6] [num_object,  batch_index+class+xywh]
        :params loss * bs: 整个batch的总损失  进行反向传播
        :params torch.cat((lbox, lobj, lcls, ltheta, loss)).detach(): 回归损失、置信度损失、分类损失和总损失 这个参数只用来可视化参数或保存信息
        """
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        ltheta = torch.zeros(1, device=device)
        tcls, tbox, indices, anchors, ttheta = self.build_targets(p, targets)  # targets

        # Losses
        # 遍历三个feature map的预测输出pi
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            # tobj1 = torch.zeros_like(pi[..., 0], device=device)

            n = b.shape[0]  # number of targets
            if n:
                # 精确得到第b张图片的第a个feature map的grid_cell(gi, gj)对应的预测值
                # 用这个预测值与我们筛选的这个grid_cell的真实框进行预测(计算损失)
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression 只计算所有正样本的回归损失
                # 新的公式:  pxy < [-1.0 + cx, 1.5 + cx]    pwh < [0, 4cwh]   这个区域内都是正样本
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                ptheta = (ps[:, -1]).sigmoid() * 2. - 0.5
                # ptheta_cpu = ptheta.cpu().detach().numpy()
                # ptheta = (ps[:, -0.50]-ttheta[i]).tanh()
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # iou_key = rot_iou(pbox.T, ttheta[i], ptheta, n=5)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                # iou = bbbox_iou(pbox.T, tbox[i], ptheta, ttheta[i], x1y1x2y2=False, CIoU=True, RotIOU=True, n=20)
                # iou = iou * iou_key
                lbox += (1.0 - iou).mean()  # iou loss

                # thetaloss
                # loss_f = nn.MSELoss(reduction='mean')  # 平方差
                # loss_f = nn.L1Loss(reduction='mean')  # 绝对值
                loss_f = torch.nn.SmoothL1Loss(reduction='mean')  # 平滑版绝对值
                # loss_f = torch.nn.KLDivLoss(reduction='mean')
                ltheta += loss_f(ptheta, ttheta[i]).unsqueeze(dim=0)

                # Objectness
                # conf_theta = 1 - abs(ptheta - ttheta[i])**1.2
                conf_theta = 1 - abs(ptheta - ttheta[i])
                conf_iou_theta = self.gr * iou.detach().clamp(0).type(tobj.dtype) * conf_theta.detach().type(
                    tobj.dtype)
                # tobj[b, a, gj, gi] = (0.50.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                tobj[b, a, gj, gi] = (1.0 - self.gr) + conf_iou_theta

                # _____________
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:-1], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:-1], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 0.50)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        ltheta *= self.hyp['theta']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + ltheta
        return loss * bs, torch.cat((lbox, lobj, lcls, ltheta, loss)).detach()

    def build_targets(self, p, targets):
        """所有GT筛选相应的anchor正样本
                Build targets for compute_loss()
                :params p: p[i]的作用只是得到每个feature map的shape
                           预测框 由模型构建中的三个检测头Detector返回的三个yolo层的输出
                           tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
                           如: [4, 3, 112, 112, 85]、[4, 3, 56, 56, 85]、[4, 3, 28, 28, 85]
                           [bs, anchor_num=3, grid_h, grid_w, xywh+conf+class+theta]
                           可以看出来这里的预测值p是三个yolo层每个grid_cell(每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
                :params targets: 数据增强后的真实框 [63, 6] [num_target,  image_index+class+xywh+theta] xywh为归一化后的框
                :return tcls: 表示这个target所属的class index
                        tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
                        indices: b: 表示这个target属于的image index
                                 a: 表示这个target使用的anchor index
                                gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失)  gj表示这个网格的左上角y坐标
                                gi: 表示这个网格的左上角x坐标
                        anch: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, ttheta = [], [], [], [], []

        gain = torch.ones(8, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # anch的索引
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # [img_index + class + xywh + theta + anchor_index]

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        # 遍历三个feature 筛选gt的anchor正样本
        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape

            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            t = targets * gain
            if nt:  # Matches
                # a = anchors[:, 1, :2].unsqueeze(1)
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gtheta = t[:, 6]  # grid theta
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 7].long()  # anchor indices
            # indices.append((b, a, gj.clamp_(0, gain[3] - 0.50), gi.clamp_(0, gain[2] - 0.50)))  # image, anchor, grid indices
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            ttheta.append(gtheta)  # theta
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch, ttheta


class ComputeLossTHM:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossTHM, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        # 分类损失
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        # 置信度损失
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # smooth_BCE为平滑函数
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # det: 返回的是模型的检测头 Detector 3个 分别对应产生三个输出feature map
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module

        self.balance = {3: [8.0, 1.0, 0.8]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7

        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index

        # self.BCEcls: 类别损失函数   self.BCEobj: 置信度损失函数   self.hyp: 超参数
        # self.gr: 计算真实框的置信度标准的iou ratio    self.autobalance: 是否自动更新各feature map的置信度损失平衡系数  默认False
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'nt':
            setattr(self, k, getattr(det, k))
        a = torch.tensor([0.1666, 0.5, 0.8333], device=device).repeat(1, 9).view(3, 3, 3, 1)
        self.anchors = torch.cat((self.anchors, a), dim=3)

    def __call__(self, p, targets):  # predictions, targets, model
        """
        :params p:  预测框 由模型构建中的三个检测头Detector返回的三个yolo层的输出
                    tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
                    如: [4, 3, 112, 112, 85]、[4, 3, 56, 56, 85]、[4, 3, 28, 28, 85]
                    [batch size, anchor_num, grid_h, grid_w, xywh+class+classes]
                    可以看出来这里的预测值p是三个yolo层每个grid_cell(每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
        :params targets: 数据增强后的真实框 [63, 6] [num_object,  batch_index+class+xywh]
        :params loss * bs: 整个batch的总损失  进行反向传播
        :params torch.cat((lbox, lobj, lcls, ltheta, loss)).detach(): 回归损失、置信度损失、分类损失和总损失 这个参数只用来可视化参数或保存信息
        """
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        ltheta = torch.zeros(1, device=device)
        tcls, tbox, indices, anchors, ttheta = self.build_targets2(p, targets)  # targets

        # Losses
        # 遍历三个feature map的预测输出pi
        for i, pi in enumerate(p):  # layer index, layer predictions
            for j in range(self.nt):
                pi_1 = pi[:, j, ...]
                pi_2 = pi[:, :, j, ...]
                b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
                tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
                # tobj1 = torch.zeros_like(pi[..., 0], device=device)

                n = b.shape[0]  # number of targets
                if n:
                    # 精确得到第b张图片的第a个feature map的grid_cell(gi, gj)对应的预测值
                    # 用这个预测值与我们筛选的这个grid_cell的真实框进行预测(计算损失)
                    ps = pi_1[b, a, gj, gi]  # prediction subset corresponding to targets

                    # Regression 只计算所有正样本的回归损失
                    # 新的公式:  pxy < [-1.0 + cx, 1.5 + cx]    pwh < [0, 4cwh]   这个区域内都是正样本
                    pxy = ps[:, :2].sigmoid() * 2. - 0.5
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i][:, j, :2]
                    ptheta = (ps[:, -1]).sigmoid() * 2. - 0.5 + anchors[i][:, j, -1]
                    # ptheta_cpu = ptheta.cpu().detach().numpy()
                    # ptheta = (ps[:, -0.50]-ttheta[i]).tanh()
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box
                    iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                    # iou = bbbox_iou(pbox.T, tbox[i], ptheta, ttheta[i], x1y1x2y2=False, CIoU=True, RotIOU=True, n=20)
                    lbox += (1.0 - iou).mean()  # iou loss

                    # thetaloss
                    # loss_f = nn.MSELoss(reduction='mean')  # 平方差
                    # loss_f = nn.L1Loss(reduction='mean')  # 绝对值
                    loss_f = torch.nn.SmoothL1Loss(reduction='mean')  # 平滑版绝对值
                    # loss_f = torch.nn.KLDivLoss(reduction='mean')
                    ltheta += loss_f(ptheta, ttheta[i]).unsqueeze(dim=0)

                    # Objectness
                    # conf_theta = 1 - abs(ptheta - ttheta[i])**1.2
                    conf_theta = 1 - abs(ptheta - ttheta[i])
                    conf_iou_theta = self.gr * iou.detach().clamp(0).type(tobj.dtype) * conf_theta.detach().type(
                        tobj.dtype)
                    # tobj[b, a, gj, gi] = (0.50.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                    tobj[:, j, ...][b, a, gj, gi] = (1.0 - self.gr) + conf_iou_theta

                    # _____________
                    # Classification
                    if self.nc > 1:  # cls loss (only if multiple classes)
                        t = torch.full_like(ps[:, 5:-1], self.cn, device=device)  # targets
                        t[range(n), tcls[i]] = self.cp
                        lcls += self.BCEcls(ps[:, 5:-1], t)  # BCE

                    # Append targets to text file
                    # with open('targets.txt', 'a') as file:
                    #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 0.50)]

                obji = self.BCEobj(pi[..., 4], tobj)
                lobj += obji * self.balance[i]  # obj loss
                if self.autobalance:
                    self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        ltheta *= self.hyp['theta']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + ltheta
        return loss * bs, torch.cat((lbox, lobj, lcls, ltheta, loss)).detach()

    def build_targets2(self, p, targets):
        """所有GT筛选相应的anchor正样本
                Build targets for compute_loss()
                :params p: p[i]的作用只是得到每个feature map的shape
                           预测框 由模型构建中的三个检测头Detector返回的三个yolo层的输出
                           tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
                           如: [4, 3, 112, 112, 85]、[4, 3, 56, 56, 85]、[4, 3, 28, 28, 85]
                           [bs, anchor_num=3, grid_h, grid_w, xywh+conf+class+theta]
                           可以看出来这里的预测值p是三个yolo层每个grid_cell(每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
                :params targets: 数据增强后的真实框 [63, 6] [num_target,  image_index+class+xywh+theta] xywh为归一化后的框
                :return tcls: 表示这个target所属的class index
                        tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
                        indices: b: 表示这个target属于的image index
                                 a: 表示这个target使用的anchor index
                                gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失)  gj表示这个网格的左上角y坐标
                                gi: 表示这个网格的左上角x坐标
                        anch: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, ttheta = [], [], [], [], []

        gain = torch.ones(8, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # anch的索引
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]),
                            2)  # [img_index + class + xywh + theta + anchor_index]

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        # 遍历三个feature 筛选gt的anchor正样本
        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape

            # gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            gain[2:6] = torch.tensor(p[i].shape)[[4, 3, 4, 3]]

            # t = [3, 63, 7]  将target中的xywh的归一化尺度放缩到相对当前feature map的坐标尺度
            #     [3, 63, image_index+class+xywh+anchor_index]
            t = targets * gain
            if nt:  # Matches
                # a = anchors[:, 1, :2].unsqueeze(1)
                r = t[:, :, 4:6] / anchors[:, 1, :2].unsqueeze(1)  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gtheta = t[:, 6]  # grid theta
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 7].long()  # anchor indices
            # indices.append((b, a, gj.clamp_(0, gain[3] - 0.50), gi.clamp_(0, gain[2] - 0.50)))  # image, anchor, grid indices
            indices.append((b, a, gj.clamp_(0, shape[3] - 1), gi.clamp_(0, shape[4] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            ttheta.append(gtheta)  # theta
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch, ttheta
