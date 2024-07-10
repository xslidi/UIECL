import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from util.util import NoneLayer
from torch.nn.utils import spectral_norm


###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = NoneLayer
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer



def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,   
            gpu_ids=[], scale=True):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
                                           
    if netG == 'rdnccut':
        net = RDNC_CUT(input_nc, output_nc, ngf, norm_layer=norm_layer, scale=scale)                                                            
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', 
            init_gain=0.02, gpu_ids=[], spectral=True, attention=False, gray=False, only_gray=False):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, spectral=spectral, attention=attention)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, spectral=spectral, gray=gray, only_gray=only_gray)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'no_norm_4':
        net = NoNormDiscriminator(input_nc, ndf, 4, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_F(input_nc, netF, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], nc=None):
    if netF == 'mlp_sample':
        net = PatchSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=nc)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    return init_net(net, init_type, init_gain, gpu_ids)

##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, loss='none'):
        super(GANLoss, self).__init__()
        if loss == 'relative':
            target_fake_label = -1.0

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if loss == 'hinge' or loss == 'wasserstein':
            self.loss = None
        elif use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        if self.loss == None:
            G_loss = - input.mean()
            # print('true')
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
            G_loss = self.loss(input, target_tensor)
        return G_loss


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/




# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, spectral=True, attention=False, gray=False, only_gray=False):
        self.gray = gray
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        input_nc = input_nc + 1 if gray else input_nc
        input_nc = 1 if only_gray else input_nc
        kw = 4
        padw = 1
        if spectral:
            sequence = [
                spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2, True)
            ]

            nf_mult = 1
            nf_mult_prev = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2**n, 8)

                sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]

            nf_mult_prev = nf_mult
            nf_mult = min(2**n_layers, 8)    
            sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

            sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]

        else:
            sequence = [
                    nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)
                ]

            nf_mult = 1
            nf_mult_prev = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2**n, 8)
                sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]

            nf_mult_prev = nf_mult
            nf_mult = min(2**n_layers, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input, gray_img=None):
        if self.gray:
            input = torch.cat((input, gray_img), 1)
        return self.model(input)

class NoNormDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False):
        super(NoNormDiscriminator, self).__init__()

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class Bn_Relu_Conv(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size,
                stride=1, padding=1, norm_layer=nn.BatchNorm2d):
        super(Bn_Relu_Conv, self).__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size, padding=padding, stride=stride)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.norm = norm_layer(output_nc)


    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        return x




class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=True, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class make_dilation_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3, norm_layer=nn.BatchNorm2d):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2+1, bias=True, dilation=2)
        self.norm = norm_layer(growthRate)
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = self.norm(out)
        out = torch.cat((x, out), 1)
        return out

     

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3, norm_layer=nn.BatchNorm2d, dilation=1):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=1, bias=True, dilation=dilation)
        self.norm = norm_layer(growthRate)
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = self.norm(out)
        out = torch.cat((x, out), 1)
        return out

class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, norm_layer=nn.BatchNorm2d, dilation=1):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_dense(nChannels_, growthRate, norm_layer=norm_layer, dilation=dilation))
            nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)    
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)
    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RDNC_CUT(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, scale=True):
        super(RDNC_CUT, self).__init__()
        nDenselayer = 3
        growthRate = 32
        self.scale = scale
        dilation = 1
        
        self.att1_1x1 = nn.Conv2d(ngf, ngf, kernel_size=1, padding=0, bias=True)
        self.att3_1x1 = nn.Conv2d(ngf*2, ngf, kernel_size=1, padding=0, bias=True)
        self.att1_conv = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.att3_conv = nn.Conv2d(ngf, ngf*2, kernel_size=3, padding=1, bias=True)

        # RDBs 3
        p = 1        
        self.upsample1 = UPConv(ngf*4, ngf*4)
        self.upsample2 = UPConv(ngf*2, ngf*2)
        self.conv1_1 = Bn_Relu_Conv(input_nc, ngf, 3, padding=p, norm_layer=norm_layer)
        self.conv1_2 = Bn_Relu_Conv(ngf, ngf, 3, padding=p, norm_layer=norm_layer)
        self.ca1 = CALayer(ngf, reduction=4)
        
        self.conv2_1 = Bn_Relu_Conv(ngf, ngf*2, 3, padding=p, norm_layer=norm_layer, stride=2)
        self.conv2_2 = Bn_Relu_Conv(ngf*2, ngf*2, 3, padding=p, norm_layer=norm_layer)
        self.ca2 = CALayer(ngf*2, reduction=4)

        self.conv3_1 = Bn_Relu_Conv(ngf*2, ngf*4, 3, padding=p, norm_layer=norm_layer, stride=2)
        self.conv3_2 = RDB(ngf*4, nDenselayer, growthRate, norm_layer, dilation)
        self.conv3_3 = RDB(ngf*4, nDenselayer, growthRate, norm_layer, dilation)
        self.conv3_4 = RDB(ngf*4, nDenselayer, growthRate, norm_layer, dilation)
        self.GFF_1x1 = nn.Conv2d(ngf*4*3, ngf*4, kernel_size=1, padding=0, bias=True)


        self.conv4_1 = Bn_Relu_Conv(ngf*6, ngf*2, 3, padding=p, norm_layer=norm_layer)
        self.ca3 = CALayer(ngf*2, reduction=4)
        self.conv5_1 = Bn_Relu_Conv(ngf*3, ngf, 3, padding=p, norm_layer=norm_layer)
        self.ca4 = CALayer(ngf, reduction=4)

        # conv 
        self.conv3 = nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0, bias=True)
        self.relu = nn.LeakyReLU()


    def forward(self, x, layers=[], encode_only=False):

        F_1 = self.conv1_1(x)
        feature_4 = F_1
        F_1 = self.conv1_2(F_1)
        F_1 = self.ca1(F_1)

        F_2 = self.conv2_1(F_1)
        F_2 = self.conv2_2(F_2)
        F_2 = self.ca2(F_2)
        feature_2 = F_2

        F_3 = self.conv3_1(F_2)        
        F_31 = self.conv3_2(F_3)
        feature_1 = F_31
        F_32 = self.conv3_3(F_31)
        F_33 = self.conv3_4(F_32)
        F_3_ = torch.cat((F_31, F_32, F_33), 1)
        F_3 =  self.relu(self.GFF_1x1(F_3_))
        
        F_2_ = self.relu(self.att3_1x1(F_2))
        F_2 = self.att3_conv(F_2_)
        F_4 = self.upsample1(F_3)
        F_4 = torch.cat((F_4, F_2), 1)
        F_4 = self.conv4_1(F_4)
        F_4 = self.ca3(F_4)
        feature_3 = F_4

        
        F_1_ = self.relu(self.att1_1x1(F_1))
        F_1 = self.att1_conv(F_1_)
        F_5 = self.upsample2(F_4)
        F_5 = torch.cat((F_5, F_1), 1)
        F_5 = self.conv5_1(F_5)
        F_5 = self.ca4(F_5)
        feature_5 = F_5


        output = self.conv3(F_5)
        output = output + x
        if self.scale:
            output = torch.tanh(output)

        if len(layers) > 0:
            feats = [feature_1, feature_2, feature_3, feature_4, feature_5]

        if encode_only:
            # print('encoder only return features')
            return feats  # return intermediate features alone; stop in the last layers
        else:
            """Standard forward"""
            return output     



class UPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, use_bias=True):
        super(UPConv, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(self.padding),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias))
    def forward(self, x):
        x = self.main(x)
        return x




if __name__ == "__main__":
    from torchinfo import summary
    model = RDNC_CUT(3, 3, ngf=64, norm_layer=NoneLayer, use_dropout=False, scale=True)
    batch_size = 2
    summary(model, input_size=(batch_size, 3, 512,512))

