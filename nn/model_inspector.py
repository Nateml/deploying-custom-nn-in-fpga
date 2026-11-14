# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Inspector tutorial

# %% [markdown]
# #### What is inspector ?
#    Vai_q_pytorch provides a function called inspector to help users diagnose neural network (NN) models under different device architectures. The inspector can predict target device assignments based on hardware constraints.The generated inspection report can be used to guide  users to modify or optimize the NN model, greatly reducing the difficulty and time of deployment. It is recommended to inspect float models before quantization.

# %%
import torch
import torch.nn as nn

# %%
# Import inspector API
#
# Note:
# You can ignore warning message related with XIR.
# The inspector relies on 'vai_utf' package. In conda env vitis-ai-pytorch in Vitis-AI docker, vai_utf is ready. But if vai_q_pytorch is installed by source code, it needs to install vai_utf in advance.
from pytorch_nndct import Inspector

# %%
from models.config import build_model
from models.globals import DETECTORS
from models.GregNet2D import GregNet2D
from models.GregNet2DModified import GregNet2DModified
from models.GWaveNet2DModified import GWaveNet2DModified
from models.GWaveNet2DModified_causal_fix import GWaveNet2DModified_causal_fix


# %%
import torch
from torch import nn

# %%
# Specify a target name or fingerprint you want to deploy on
target = "DPUCZDX8G_ISA1_B4096"
# Initialize inspector with target
inspector = Inspector(target)

# %%
# Start to inspect the float model
# Note: visualization of inspection results relies on the dot engine.If you don't install dot successfully, set 'image_format = None' when inspecting.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = ModifiedGWaveNet(detectors=DETECTORS, filters=FILTERS, kernel_size=KERNEL_SIZE_CONVOLUTION, dilation=DILATION, stride=STRIDE)
# model = GregNet2D(d="pc")
# model = GregNet2DModified(d="pc")
SEQ_LEN = 155648 // 8
# architecture from models/configs/GWaveNet2DModified_causal_fix.yaml
model, _ = build_model("GWaveNet2DModified_causal_fix", input_len=SEQ_LEN)
# model = GWaveNet2D(in_channels=3, out_channels=64, kernel_size=16, dilation=2, stride=2)

dummy_input = torch.FloatTensor(1, len(DETECTORS), SEQ_LEN, 1).uniform_(-1, 1)

inspector.inspect(
    model, (dummy_input,), device=device, output_dir="inspect", image_format="png"
)

# %% [markdown]
# ##### Inspection result anaysis:
# 1. In this toy model, there are two operations will be assigned to cpu. The reasons for being allocated to the CPU are shown in the column of hardware constraints.
# 2. The target device of an input node is determined by its consumer.
# 3. Here, we found that a special permute operation("ToyModel::ToyModel/AdaptiveAvgPool2d[adaptive_avg_pool2d]/146_sink_transpose_0") is inserted by quantizer. Due to data layout difference between Pytorch('NCHW') and XIR('NHWC'), if quantizer inserts some permutes(which the node message in inspect file will inform us about), these permutes may prevent the entire model from being deployed to the target device. Sometimes, we can cancel out this automatically inserted permute by inserting a permute in the original float model, sometimes, we can't.
# 4. If you want more details about the inspection, see the inspection report under the output directory you specified.
# 5. The dot image can help you to analyze inspection result in a more intuitive way. "svg" and "png" format are both supported.
#

# %%
# Show the dot image
# from IPython.display import Image
# Image('inspect/inspect_DPUCZDX8G_ISA1_B4096.png')


# %% [raw]
# ##### How to deploy this entire model on DPU:
# 1. Find the location of CPU operations in original model. All the details of operations are shown in inspect_{target}.txt, including source range, points to a source which is a stack track and helps to find the exact location of this operation in source code. Take resize op as a example, from source range, we can find the location of resize op in #3 input block line 11.
# node name: ToyModel::ToyModel/Upsample[upsample]/input.3
# ...
# op type: resize
# ...
# source range:
# ...
# <font color=red>ipython-input-3-7ec1fc6b678f(11): forward</font>
# ...
#
#

# %% [markdown]
# 2. Modify the original model according to the hardware constrains message and try to cancel out the permute inserted by quantizer with by inserting a permute in the original float model(more details see hints in inspect_{target}.txt)
