import torch
import torch.nn as nn
from torch.autograd import Variable
from .baseLayers import Encoder, Decoder, FinalConv, DoubleConv, ExtResNetBlock, SingleConv

def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]


class Unet_Lite(nn.Module):
    """
    Based on 3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.
    In this code we changed 3D Conv to 2D Conv.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps = 16, layer_order = 'cbr',
                 num_groups = 8,
                 **kwargs):
        super(Unet_Lite, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps = 4)
            self.f_maps = f_maps

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling = False, basic_module = DoubleConv,
                                  conv_layer_order = layer_order, num_groups = num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module = DoubleConv,
                                  conv_layer_order = layer_order, num_groups = num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if i == 0:
                in_feature_num = reversed_f_maps[i]
            else:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]

            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module = DoubleConv,
                              conv_layer_order = layer_order, num_groups = num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim = 1)
        self.mid_activation = nn.Sigmoid()

    def forward(self, x):
        time_num = x.shape[1]
        # encoder part
        encoders_features = []

        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for i, (decoder, encoder_features) in enumerate(zip(self.decoders, encoders_features)):

            use_skip_connections = i != 0

            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x, use_skip_connections)

        x = self.final_conv(x)

        # if not self.training:
        #     x = self.final_activation(x)
        return x


if __name__ == '__main__':
    # gradient check
    model = Unet_Lite(in_channels = 64,
                      out_channels = 64,
                      final_sigmoid = True,
                      f_maps = [64, 64, 64, 64]).cuda()
    # tsm1 = TemporalShift(model, n_segment=10, n_div=10, inplace=False)

    # print(model)
    loss_fn = torch.nn.MSELoss()

    input = Variable(torch.randn(1, 64, 512, 512)).cuda()
    target = Variable(torch.randn(1, 64, 512, 512)).double().cuda()
    mid_feature = torch.randn(1, 64, 64, 64).cuda()
    output, mid_feature = model(input, 1, mid_feature)
    output = output.double()
    # res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    # print(res)
    # print(model)
