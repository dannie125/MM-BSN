import torch.nn as nn


class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH//2] = 0
        # if kH == 5:
        #     self.mask[:, :, 1:-1, 1:-1] = 0
        # else:
        #     pass
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class ColMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class RowMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, :, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class fSzMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, :, kH // 2] = 0
        self.mask[:, :, kW // 2, :] = 0


    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class SzMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        self.mask[:, :, :, kH // 2] = 1
        self.mask[:, :, kW // 2, :] = 1
        self.mask[:, :, kW // 2, kH // 2] = 0


    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class angle135MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, i, i] = 0
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class angle45MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, kW -1-i, i] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class chaMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        for i in range(kH):
            self.mask[:, :, i, i] = 1
            self.mask[:, :, kW - 1 - i, i] = 1
            self.mask[:, :, kH // 2, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class fchaMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, i, i] = 0
            self.mask[:, :, kW -1-i, i] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class huiMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, 1:-1, 1:-1] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)