import torch
import torch.nn as nn

class ShallowFeatureExtraction(nn.Module):
    def __init__(self, in_channels=1, embed_dim=1, kernel_size=3, stride=1, padding=1, order='cl'):
        super(ShallowFeatureExtraction, self).__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.order = order

        self.conv1 = SingleConv(self.in_channels, self.embed_dim, self.kernel_size, self.stride, 1,1, self.order)
        self.conv2 = SingleConv(self.in_channels, self.embed_dim, self.kernel_size, self.stride, 2,2, self.order)
        self.conv3 = SingleConv(self.in_channels, self.embed_dim, self.kernel_size, self.stride, 3,3, self.order)

        self.conv4 = SingleConv(self.embed_dim*3, self.embed_dim, self.kernel_size, self.stride, self.padding,1, 'c')

    def forward(self, x):
        out = self.conv4(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1))
        return out

class ShallowTemporalFeatureExtraction(nn.Module):
    def __init__(self, in_channels=1, embed_dim=1, kernel_size=3, stride=1, padding=1, order='cl'):
        super(ShallowTemporalFeatureExtraction, self).__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.order = order

        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Sequential(
            SingleConv(self.in_channels, self.embed_dim, self.kernel_size, self.stride, self.padding,  1, self.order),
            SingleConv(self.embed_dim, self.in_channels, self.kernel_size, self.stride, self.padding, 1, 'c')
        )

        self.conv2 = SingleConv(self.in_channels, self.embed_dim, self.kernel_size, self.stride, 1,1, self.order)
        self.conv3 = SingleConv(self.in_channels, self.embed_dim, self.kernel_size, self.stride, 2,2, self.order)
        self.conv4 = SingleConv(self.in_channels, self.embed_dim, self.kernel_size, self.stride, 3,3, self.order)

        self.conv5 = SingleConv(self.embed_dim*3, self.embed_dim, self.kernel_size, self.stride, self.padding,1, 'c')

    def forward(self, x, rTG):
        x = x.mul(self.sigmoid(self.conv1(rTG)))
        out = self.conv5(torch.cat([self.conv2(x), self.conv3(x), self.conv4(x)], dim=1))
        return out

class ConvAfterSwin(nn.Module):
    def __init__(self, embed_dim=1, kernel_size=3, stride=1, padding=1, order='cl'):
        super(ConvAfterSwin, self).__init__()

        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.order = order

        self.conv = nn.Sequential(
            SingleConv(embed_dim, embed_dim // 4, self.kernel_size, self.stride, self.padding, 1, self.order),
            SingleConv(embed_dim // 4, embed_dim // 4, 1, 1, 0, 1, self.order),
            SingleConv(embed_dim // 4, embed_dim, self.kernel_size, self.stride, self.padding, 1, 'c')
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class ConvAfterBody(nn.Module):
    def __init__(self, embed_dim=1, kernel_size=3, stride=1, padding=1, order='cl'):
        super(ConvAfterBody, self).__init__()

        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.order = order

        self.conv = nn.Sequential(
            SingleConv(embed_dim, embed_dim // 4, self.kernel_size, self.stride, self.padding, 1, self.order),
            SingleConv(embed_dim // 4, embed_dim // 4, 1, 1, 0, 1, self.order),
            SingleConv(embed_dim // 4, embed_dim, self.kernel_size, self.stride, self.padding, 1, 'c')
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class ImageReconstruction(nn.Module):
    def __init__(self, embed_dim=1, out_channels=1, kernel_size=3, stride=1, padding=1, order='cl'):
        super(ImageReconstruction, self).__init__()

        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.order = order

        self.conv = nn.Sequential(
            SingleConv(embed_dim, embed_dim // 4, self.kernel_size, self.stride, self.padding, 1, self.order),
            SingleConv(embed_dim // 4, embed_dim // 4, self.kernel_size, self.stride, self.padding,1, self.order),
            SingleConv(embed_dim // 4, self.out_channels, self.kernel_size, self.stride, self.padding, 1, 'c')
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, order='cl'):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, stride, padding, dilation, order):
            self.add_module(name, module)

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, order='cl'):
        super(DoubleConv, self).__init__()
        # conv1
        self.add_module('SingleConv1',
                        SingleConv(in_channels, out_channels, kernel_size, stride, padding, dilation, order))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(out_channels, out_channels, kernel_size, stride, padding, dilation, order))

def create_conv(in_channels, out_channels, kernel_size, stride, padding, dilation, order):
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of gatchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=bias)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm2d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm2d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'r', 'l', 'e', 'c']")

    return modules
