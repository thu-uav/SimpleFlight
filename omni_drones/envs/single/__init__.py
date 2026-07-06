# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from .hover import Hover
from .track import Track
from .track_pinn import TrackPINN
from .track_pinn_0504 import TrackPINN0504
from .track_pinn_0528 import TrackPINN0528  # [0528 新增] 含6D扰动标签和18D特征
from .track_pinn_test_0518 import TrackPINNTest0518
from .track_resid_gain import TrackResidGain
from .track_residual import TrackResidual
from .track_datt import TrackDATT
from .track_datt0427 import TrackDATT0427  # 原值: 无（0427新增）
from .track_datt0615_6d import TrackDATT0615  # [0615 新增] 6D扰动版本（平动+转动 GT oracle）
