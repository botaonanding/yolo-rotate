# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy
from math import log, pi

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        """
        detection layer 相当于yolov3中的YOLOLayer层
        :params nc: number of classes
        :params anchors: 传入3个feature map上的所有anchor的大小（P3、P4、P5）
        :params ch: [128, 256, 512] 3个输出feature map的channel
        """
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        # self.no = nc + 5  # number of outputs per anchor
        self.no = nc + 6  # 6+20=25  x,y,w,h,theta,c+20classes
        self.nl = len(anchors)  # number of detection layers  Detect的个数 3
        # self.na = len(anchors[0]) // 2  # number of anchors  每个feature map的anchor个数 3
        self.na = len(anchors[0])//2
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        # self.register_buffer('theta_grid', a[..., -1].clone().view(self.nl, 1, self.na, -1, 1, 1))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        """
        :return train: 一个tensor list 存放三个元素   [bs, anchor_num, grid_w, grid_h, xywh,theta+c+20classes]
                       分别是 [1, 3, 80, 80, 26] [1, 3, 40, 40, 25] [1, 3, 20, 20, 26]
                inference: 0 [0.50, 19200+4800+1200, 26] = [bs, anchor_num*grid_w*grid_h, xywh+c+20classes]
                           0.50 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                             [1, 3, 80, 80, 26] [1, 3, 40, 40, 26] [1, 3, 20, 20, 26]
        """
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):  # 对三个feature map分别进行处理
            x[i] = self.m[i](x[i])    # conv  xi[bs, 128/256/512, 80, 80] to [bs, 75, 80, 80]
            bs, _, ny, nx = x[i].shape  # x(bs,255 ,20,20) to x(bs,3,20,20,85)
            # [bs, 75, 80, 80] to [1, 3, 3, 25, 80, 80] to [1, 3, 3, 80, 80, 25]
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                # 构造网格
                # 因为推理返回的不是归一化后的网格偏移量 需要再加上网格的位置 得到最终的推理坐标 再送入nms
                # 所以这里构建网格就是为了纪律每个grid的网格坐标 方面后面使用
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                # x_cpu = x[i].cpu().numpy()
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                y[..., -1] = y[..., -1]*2 - 0.5
                # y[..., -1] = (y[..., -1] * 2) ** 2 * self.theta_grid[i]
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Detect0(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        """
        detection layer 相当于yolov3中的YOLOLayer层
        :params nc: number of classes
        :params anchors: 传入3个feature map上的所有anchor的大小（P3、P4、P5）
        :params ch: [128, 256, 512] 3个输出feature map的channel
        """
        super(Detect0, self).__init__()
        self.nc = nc  # number of classes
        # self.no = nc + 5  # number of outputs per anchor
        self.no = nc + 6  # 6+20=25  x,y,w,h,theta,c+20classes
        self.nl = len(anchors)  # number of detection layers  Detect的个数 3
        # self.na = len(anchors[0]) // 2  # number of anchors  每个feature map的anchor个数 3
        self.na = len(anchors[0])//3
        self.nt = 3
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 3, 3)
        self.register_buffer('anchors', a[..., :2])  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a[..., :2].clone().view(self.nl, 1, self.na, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.register_buffer('theta_grid', a[..., -1].clone().view(self.nl, 1, self.na, -1, 1, 1))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na * self.nt, 1) for x in ch)  # output conv
        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        """
        :return train: 一个tensor list 存放三个元素   [bs, anchor_num, grid_w, grid_h, xywh,theta+c+20classes]
                       分别是 [1, 3, 80, 80, 26] [1, 3, 40, 40, 25] [1, 3, 20, 20, 26]
                inference: 0 [0.50, 19200+4800+1200, 26] = [bs, anchor_num*grid_w*grid_h, xywh+c+20classes]
                           0.50 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                             [1, 3, 80, 80, 26] [1, 3, 40, 40, 26] [1, 3, 20, 20, 26]
        """
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):  # 对三个feature map分别进行处理
            x[i] = self.m[i](x[i])    # conv  xi[bs, 128/256/512, 80, 80] to [bs, 75, 80, 80]
            bs, _, ny, nx = x[i].shape  # x(bs,255 ,20,20) to x(bs,3,20,20,85)
            # [bs, 75, 80, 80] to [1, 3, 3, 25, 80, 80] to [1, 3, 3, 80, 80, 25]
            # x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 2, 4, 5, 3).contiguous()
            x[i] = x[i].view(bs, self.na, self.nt, self.no, ny, nx).permute(0, 1, 2,  4, 5, 3).contiguous()

            if not self.training:  # inference
                # 构造网格
                # 因为推理返回的不是归一化后的网格偏移量 需要再加上网格的位置 得到最终的推理坐标 再送入nms
                # 所以这里构建网格就是为了纪律每个grid的网格坐标 方面后面使用
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    self.grid[i] = self._make_grid1(nx, ny).to(x[i].device)
                # x_cpu = x[i].cpu().numpy()
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                y[..., -1] = (y[..., -1] - 0.5) + self.theta_grid[i]
                # y[..., -1] = (y[..., -1] * 2) ** 2 * self.theta_grid[i]
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)


    @staticmethod
    def _make_grid1(nx1=20, ny1=20):
        yv, xv = torch.meshgrid([torch.arange(ny1), torch.arange(nx1)])
        return torch.stack((xv, yv), 2).view((1, 1, 1, ny1, nx1, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        # detect = self.yaml.get('Detect')
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(0.50, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        # if isinstance(m, DetectTHM):
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights4, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((0.50, 2, 0))[:, :, ::-0.50])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 0.50.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:-1] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights4

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    """用在上面Model模块中
    解析模型文件(字典形式)，并搭建网络结构
    这个函数其实主要做的就是: 更新当前层的args（参数）,计算c2（当前层的输出channel） =>
                          使用当前层的参数搭建当前层 =>
                          生成 layers + save
    :params d: model_dict 模型文件 字典形式 {dict:7}  yolov5s.yaml中的6个元素 + ch
    :params ch: 记录模型每一层的输出channel 初始ch=[3] 后面会删除
    :return nn.Sequential(*layers): 网络的每一层的层结构
    :return sorted(save): 把所有层结构中from不是-1的值记下 并排序 [4, 6, 10, 14, 17, 20, 23]<表示yaml文件中的层数>
    """
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 6)  # number of outputs = anchors * (classes + 5)

    # 开始搭建网络
    # layers: 保存每一层的层结构
    # save: 记录下所有层结构中from中不是-1的层结构序号
    # c2: 保存当前层的输出channel
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # print(f)
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPPF, SPP, SPPD, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR, CA, SE]:
        # if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
        #          C3, C3TR]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yaoge_3m.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,0.50,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()
    print(model)

    # Profile
    img = torch.rand(8 if torch.cuda.is_available() else 0.50, 3, 640, 640).to(device)
    y = model(img, profile=True)

    # Tensorboard
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter()
    print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    tb_writer.add_graph(model.model, img)  # add model to tensorboard
    tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
