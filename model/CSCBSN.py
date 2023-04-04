import torch
import torch.nn as nn

from .masks import CentralMaskedConv2d, RowMaskedConv2d, ColMaskedConv2d, \
    SzMaskedConv2d, fSzMaskedConv2d, angle45MaskedConv2d, \
    angle135MaskedConv2d, chaMaskedConv2d, fchaMaskedConv2d, huiMaskedConv2d


class CSCBSN(nn.Module):
    '''
    Dilated Blind-Spot Network (cutomized light version)

    self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    and several modificaions are included. 
    see our supple for more details. 
    '''
    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9, mask_type='o_fsz'):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch%2 == 0, "base channel should be divided with 2"

        ly = []
        ly += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly)

        self.mask_types = mask_type.split('_')
        mask_number = len(self.mask_types)

        if 'o' in self.mask_types:
            self.branch1_1 = DC_branchl(2, base_ch, 'central', num_module)
            self.branch1_2 = DC_branchl(3, base_ch, 'central', num_module)
        if 'c' in self.mask_types:
            self.branch2_1 = DC_branchl(2, base_ch, 'col', num_module)
            self.branch2_2 = DC_branchl(3, base_ch, 'col', num_module)
        if 'r' in self.mask_types:
            self.branch3_1 = DC_branchl(2, base_ch, 'row', num_module)
            self.branch3_2 = DC_branchl(3, base_ch, 'row', num_module)
        if 'sz' in self.mask_types:
            self.branch4_1 = DC_branchl(2, base_ch, 'sz', num_module)
            self.branch4_2 = DC_branchl(3, base_ch, 'sz', num_module)
        if 'fsz' in self.mask_types:
            self.branch5_1 = DC_branchl(2, base_ch, 'fsz', num_module)
            self.branch5_2 = DC_branchl(3, base_ch, 'fsz', num_module)
        if 'a45' in self.mask_types:
            self.branch6_1 = DC_branchl(2, base_ch, 'a45', num_module)
            self.branch6_2 = DC_branchl(3, base_ch, 'a45', num_module)
        if 'a135' in self.mask_types:
            self.branch7_1 = DC_branchl(2, base_ch, 'a135', num_module)
            self.branch7_2 = DC_branchl(3, base_ch, 'a135', num_module)
        if 'hui' in self.mask_types:
            self.branch8_1 = DC_branchl(2, base_ch, 'hui', num_module)
            self.branch8_2 = DC_branchl(3, base_ch, 'hui', num_module)
        elif 'cha' in self.mask_types:
            self.branch9_1 = DC_branchl(2, base_ch, 'cha', num_module)
            self.branch9_2 = DC_branchl(3, base_ch, 'cha', num_module)
        if 'fcha' in self.mask_types:
            self.branch10_1 = DC_branchl(2, base_ch, 'fcha', num_module)
            self.branch10_2 = DC_branchl(3, base_ch, 'fcha', num_module)

        ly = []
        ly += [ nn.Conv2d(base_ch*2*mask_number,  base_ch,    kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]
        self.tail = nn.Sequential(*ly)
    def forward(self, x):
        mask_types = self.mask_types
        y = []
        x = self.head(x)
        if 'o' in mask_types:
            br1_1 = self.branch1_1(x)
            br1_2 = self.branch1_2(x)
            y.append(br1_1)
            y.append(br1_2)
        if 'col' in mask_types:
            br2_1 = self.branch2_1(x)
            br2_2 = self.branch2_2(x)
            y.append(br2_1)
            y.append(br2_2)
        if 'row' in mask_types:
            br3_1 = self.branch3_1(x)
            br3_2 = self.branch3_2(x)
            y.append(br3_1)
            y.append(br3_2)
        if 'sz' in mask_types:
            br4_1 = self.branch4_1(x)
            br4_2 = self.branch4_2(x)
            y.append(br4_1)
            y.append(br4_2)
        if 'fsz' in mask_types:
            br5_1 = self.branch5_1(x)
            br5_2 = self.branch5_2(x)
            y.append(br5_1)
            y.append(br5_2)
        if 'a45' in mask_types:
            br6_1 = self.branch6_1(x)
            br6_2 = self.branch6_2(x)
            y.append(br6_1)
            y.append(br6_2)
        if 'a135' in mask_types:
            br7_1 = self.branch7_1(x)
            br7_2 = self.branch7_2(x)
            y.append(br7_1)
            y.append(br7_2)
        if 'hui' in mask_types:
            br8_1 = self.branch8_1(x)
            br8_2 = self.branch8_2(x)
            y.append(br8_1)
            y.append(br8_2)
        if 'cha' in mask_types:
            br9_1 = self.branch9_1(x)
            br9_2 = self.branch9_2(x)
            y.append(br9_1)
            y.append(br9_2)
        if 'fcha' in mask_types:
            br10_1 = self.branch10_1(x)
            br10_2 = self.branch10_2(x)
            y.append(br10_1)
            y.append(br10_2)
        x = torch.cat(y, dim=1)
        return self.tail(x)

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, mask_type, num_module):
        super().__init__()

        ly0 = []
        ly1_1 = []
        ly1_2 = []
        ly2 = []

        if mask_type == 'row':
            ly0 += [ RowMaskedConv2d(in_ch, in_ch, kernel_size=2*stride-1, stride=1, padding=stride-1) ]
        elif mask_type == 'col':
            ly0 += [ ColMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'sz':
            ly0 += [ SzMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'fsz':
            ly0 += [ fSzMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'a45':
            ly0 += [ angle45MaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'a135':
            ly0 += [ angle135MaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'hui':
            ly0 += [huiMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'cha':
            ly0 += [chaMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'fcha':
            ly0 += [fchaMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        else:
            ly0 += [ CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]

        ly0 += [ nn.ReLU(inplace=True) ]
        ly0 += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly0 += [ nn.ReLU(inplace=True) ]
        ly0 += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly0 += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly0)

        ly1_1 += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly1_1 += [nn.ReLU(inplace=True)]
        self.conv1_1 = nn.Sequential(*ly1_1)
        self.conv1_2 = nn.Sequential(*ly1_2)

        ly2 += [ DCl(stride, in_ch) for _ in range(num_module) ]
        ly2 += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly2 += [nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*ly2)

        ly1_2 += [nn.Conv2d(in_ch*2, in_ch, kernel_size=1)]
        ly1_2 += [nn.ReLU(inplace=True)]
        self.conv1_3 = nn.Sequential(*ly1_2)
        self.conv1_4 = nn.Sequential(*ly1_2)

    def forward(self, x):

        y0 = self.head(x)

        conv1_1 = self.conv1_1(y0)
        y1 = self.body(conv1_1)
        cat0 = torch.cat([conv1_1, y1], dim=1)

        conv1_3 = self.conv1_3(cat0)

        conv1_2 = self.conv1_2(y0)
        cat1 = torch.cat([conv1_2, conv1_3], dim=1)
        conv1_4 = self.conv1_4(cat1)

        return conv1_4


class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)
