# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
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
        # 获取模型在cpu还是gpu上运行的.之后生成的临时变量也会在相应的设备上运行
        device = next(model.parameters()).device  # get model device
        # 模型的参数
        h = model.hyp  # hyperparameters

        # Define criteria
        # 定义cls loss和 obj loss
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss默认不会使用这个
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # 获取模型的detect层
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        # 用来实现obj,box,cls loss之间权重的平衡
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        # 获取各个特征层的stride相关参数
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        # 将各个loss加入到类的公共变量中
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        # 参数获取
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    # loss计算
    # p为模型的最终输出
    # targets为gt框的信息
    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        # 初始化loss
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # 建立targets目标,会在下面仔细讲解
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        # 各个参数形状
        # tcls [[num]] 存放了gt框所对应的网格的cls
        # tbox [[x_offset,y_offset,w,h]] 存放了gt框所对应的网格的box,注意此处的x和y是相对于网格的偏移量
        # indices [[image, anchor, grid indices]] 存放了gt对应的gird的信息,包括:image对应batchsize的哪张图片,anchor,对应哪个尺度的anchor,以及所在的网格
        # anch [[num,2]]#anchor信息

        # Losses
        # 遍历各个特征层,大小分别是80x80,40x40,20x20
        for i, pi in enumerate(p):  # layer index, layer predictions

            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            # 初始化target obj
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            # 计算有多少个target
            n = b.shape[0]  # number of targets
            if n:  # 如果存在target的话
                # 首先获取target所在的网格模型预测的pred 信息
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # 对xywh进行解码工作
                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # 计算ciou
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                # 计算box的ciouloss
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                # 获取target所对应的obj,网格中存在gt目标的会被标记为iou与gt的交并比
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                # 如果类别数只有一个的话,将不会调用cls loss
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # target所在的gird对应的cls的one hot格式
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    # 计算loss
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 0.50)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        # 提高loss各自的权重,可以在配置文件中设置
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    # 这段代码相对来说比较难以理解内部是怎么运行的
    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)

        # 这里na为锚框种类数 nt为目标数 这里的na为3，nt也为3
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        # 类别 边界盒 索引 锚框
        tcls, tbox, indices, anch = [], [], [], []
        # 利用gain来计算目标在某一个特征图上的位置信息，初始化为1
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        # ai.shape = (na, nt)，锚框的索引，三个目标，三种锚框，所以共9个元素
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # targets.shape = (na, nt, 7)（3，3，7）给每个目标加上锚框索引
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias

        # off偏移量
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [0.50, 0.50], [0.50, -0.50], [-0.50, 0.50], [-0.50, -0.50],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            # 获取当前的锚框尺寸
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # 将xywh映射到当前特征图，即乘以对应的特征图尺寸
            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                # r为目标wh和锚框wh的比值，比值在0.25到4即采用该种锚框预测目标
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                # 将比值和预先设置的比例anchor_t对比，符合条件为True，反之False
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # 根据j筛选符合条件的情况
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                # 得到相对于左上角的目标
                gxy = t[:, 2:4]  # grid xy
                # 得到相对于右上角的目标
                gxi = gain[[2, 3]] - gxy  # inverse
                # 这里是重点，也是比较难理解的部分，jk是判断gxy更偏向哪里，左？上？
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                # jk是判断gxi更偏向哪里，下？右？
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # yolov5不仅用目标中心点所在的网格预测该目标，还采用了距目标中心点的最近两个网格
                # 所以有五种情况，网格本身，上下左右，这就是repeat函数第一个参数为5的原因
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

