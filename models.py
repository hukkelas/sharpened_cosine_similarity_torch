from torch import nn
from sharpened_cosine_similarity import SharpenedCosineSimilarity
from absolute_pooling import MaxAbsPool2d


class ResBlk(nn.Module):

    def __init__(self, in_ch, out_ch, pool, use_residual):
        super().__init__()
        self.mod1 = SharpenedCosineSimilarity(in_ch, out_ch, kernel_size=3, padding=1)
        self.mod2 = SharpenedCosineSimilarity(out_ch, out_ch, kernel_size=3, padding=1)
        if use_residual:
            self.conv1x1 = SharpenedCosineSimilarity(in_ch, out_ch, kernel_size=1, padding=0)
        if pool:
            self.pool = MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True)
    
    def forward(self, x):
        y = x
        x = self.mod1(x)
        x = self.mod2(x)
        if hasattr(self, "pool"):
            x = self.pool(x)
        if hasattr(self, "conv1x1"):
            if hasattr(self, "pool"):
                y = self.pool(y)
            y = self.conv1x1(y)
            x += y
        return x


class ResidualNetwork(nn.Module):
    def __init__(
            self,
            start_ch=32,
            num_blocks_per_level=1,
            use_residual=True
        ):
        super().__init__()

        self.backbone = nn.Sequential(
            SharpenedCosineSimilarity(3, 32, 7, padding=3),
        )
        for level in range(3):
            cur_ch = start_ch * 2**(level)
            for i in range(num_blocks_per_level-1):
                self.backbone.add_module(f"level{level}_block{i}",
                    ResBlk(cur_ch, cur_ch, pool=False, use_residual=use_residual)
                )
            self.backbone.add_module(f"level{level}",
                ResBlk(cur_ch, cur_ch*2, pool=True,use_residual=use_residual))
        
        self.out = nn.Linear(in_features=start_ch*8*4*4, out_features=10)

    def forward(self, t):
        t = self.backbone(t).flatten(start_dim=1)
        t = self.out(t)
        return t


class OriginalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.scs1 = SharpenedCosineSimilarity(
            in_channels=3, out_channels=24, kernel_size=3, padding=0)
        self.pool1 = MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.scs2 = SharpenedCosineSimilarity(
            in_channels=24, out_channels=48, kernel_size=3, padding=1)
        self.pool2 = MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.scs3 = SharpenedCosineSimilarity(
            in_channels=48, out_channels=96, kernel_size=3, padding=1)
        self.pool3 = MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.out = nn.Linear(in_features=96*4*4, out_features=10)

    def forward(self, t):
        t = self.scs1(t)
        t = self.pool1(t)

        t = self.scs2(t)
        t = self.pool2(t)

        t = self.scs3(t)
        t = self.pool3(t)

        t = t.reshape(-1, 96*4*4)
        t = self.out(t)

        return t
