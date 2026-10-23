import torch
from torch import nn


class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(CausalConv2d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            # padding=(self.padding, 0),
            padding=0,
            dilation=(dilation, 1),
        )

    def forward(self, x):
        x = nn.functional.pad(x, (0, 0, self.padding, 0), mode="replicate")
        x = self.conv(x)
        return x


class ModuleWavenet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(ModuleWavenet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.res_down = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride
        )

        self.conv_dilated = CausalConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)

        self.gate_tanh = nn.ReLU6()
        self.gate_sigmoid = nn.Hardsigmoid()

        self.conv1x1 = nn.Conv2d(
            in_channels=out_channels, out_channels=32, kernel_size=(1, 1)
        )

    def forward(self, x):
        x_old = x

        # explaination:Tom - conv_dilated and batchnorm used to be called twice on
        # the same input with shared weights, so x_a and x_b were identical. In the
        # quantized graph that gave the input transpose two fix ops, which the DPU
        # compiler rejects (XCOM_FRONTEND_GRAPH_INVALID_STRUCTURE). Now they are
        # computed once; the output is unchanged and the trained weights still load.
        x_a = self.batchnorm(self.conv_dilated(x))

        # explaination:Tom - the same x_a now feeds both gate activations (was x_a / x_b).
        # pixel gate
        x_skip = self.gate_tanh(x_a) * self.gate_sigmoid(x_a)
        x_skip = self.conv1x1(x_skip)

        # skip-connection
        # if self.in_channels != self.out_channels:
        # x_res = x_skip
        # else:
        # if x_old.size(2) < x_skip.size(2):
        # x_res = x_old + x_skip[:, :, : x_old.size(2), :]
        # else:
        # x_res = x_skip + x_old[:, :, : x_skip.size(2), :]

        # skip-connection (NEW, REPLACES ABOVE - NATE)
        if self.in_channels != self.out_channels:
            x_res = x_skip
        else:
            x_res = self.res_down(x_old) + x_skip

        return x_res, x_skip


class GWaveNet2DModified_causal_fix(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        stride,
        input_len=155648,
    ):
        super().__init__()

        self.Mod0 = ModuleWavenet(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**0,
        )
        self.Mod1 = ModuleWavenet(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**1,
        )
        self.Mod2 = ModuleWavenet(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**2,
        )
        self.Mod3 = ModuleWavenet(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**3,
        )
        self.Mod4 = ModuleWavenet(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**4,
        )
        self.Mod5 = ModuleWavenet(
            in_channels=32,
            out_channels=32,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**5,
        )
        self.Mod6 = ModuleWavenet(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation**6,
        )

        conv1x1_out_channels = 1
        self.conv_1x1 = nn.Conv2d(
            in_channels=32, out_channels=conv1x1_out_channels, kernel_size=(1, 1)
        )

        # Calculating skip lengths for alignment in the forward pass
        # explaination:Tom - input_len is now a constructor argument (default unchanged:
        # 155648) so the compile check can use a shorter input. fc1's size follows
        # target_len automatically. A checkpoint only fits the length it was trained with.
        lengths = self._skip_lengths(in_channels=in_channels, input_len=input_len)
        self.target_len = lengths[-1]  # 1216
        self.skip_starts = [n - self.target_len for n in lengths[:-1]]

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # self.fc1 = nn.Linear(1302, 50)
        self.fc1 = nn.Linear(self.target_len * conv1x1_out_channels, 50)  # 1216 * 1

        self.fc2 = nn.Linear(50, 1)

    @torch.no_grad()
    def _skip_lengths(self, in_channels, input_len):
        was_training = self.training
        self.eval()  # don't touch BatchNorm running stats
        x = torch.zeros(1, in_channels, input_len, 1)
        lengths = []
        for m in [
            self.Mod0,
            self.Mod1,
            self.Mod2,
            self.Mod3,
            self.Mod4,
            self.Mod5,
            self.Mod6,
        ]:
            x, s = m(x)
            lengths.append(s.size(2))
        self.train(was_training)
        return lengths

    def forward(self, x):
        # explaination:Tom - the view() that added the dummy width dimension is
        # replaced by this check. Preferred: feed (N, C, L, 1) directly (reshape in
        # data loading) so no view/reshape ends up in the DPU graph and can become a
        # transpose. A 3-D (N, C, L) input is still accepted for the old pipeline;
        # dim() is static when tracing, so the branch costs nothing at compile time.
        if x.dim() == 3:
            x = x.unsqueeze(-1)

        x, xs0 = self.Mod0(x)
        x, xs1 = self.Mod1(x)
        x, xs2 = self.Mod2(x)
        x, xs3 = self.Mod3(x)
        x, xs4 = self.Mod4(x)
        x, xs5 = self.Mod5(x)
        x, xs6 = self.Mod6(x)

        xs = xs6

        # for i in [xs0, xs1, xs2, xs3, xs4, xs5]:
        # if xs.size(2) < i.size(2):
        # xs = xs + i[:, :, -xs.size(2) :, :]
        # else:
        # xs = i + xs[:, :, -i.size(2) :, :]

        # skip alignment (replaces above - NATE)
        xs = xs6
        for i, start in zip([xs0, xs1, xs2, xs3, xs4, xs5], self.skip_starts):
            xs = xs + i[:, :, start : start + self.target_len, :]

        xs = self.conv_1x1(xs)
        xs = self.relu(xs)

        xs = self.flatten(xs)
        xs = self.fc1(xs)
        xs = self.relu(xs)

        xs = self.fc2(xs)

        return xs


if __name__ == "__main__":
    from globals import DETECTORS, DILATION, FILTERS, STRIDE
    from torchsummary import summary

    Lin = 155648
    kernel_size_convolution = 2

    model = GWaveNet2DModified_causal_fix(
        in_channels=len(DETECTORS),
        out_channels=FILTERS,
        kernel_size=kernel_size_convolution,
        stride=STRIDE,
        dilation=DILATION,
    )

    summary(model, (len(DETECTORS), Lin))