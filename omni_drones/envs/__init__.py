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


from .single import Rotate, Hover, Exchange, MultiGoto, Goto, Goto_static, Goto_return, Infeasible, Track, Track_datt, ZigZag, Star, TrackV1, Turn, Takeoff, Line
from .platform import PlatformHover, PlatformFlyThrough
from .inv_pendulum import InvPendulumHover, InvPendulumFlyThrough
from .transport import TransportHover, TransportFlyThrough, TransportTrack
from .formation import Formation
from .formation_forward import FormationForward
from .formation_dodging import FormationDodge
from .formation_ball_forward import FormationBallForward
from .formation_multi_ball_forward import FormationMultiBallForward
from .multi_catch_old import MultiCatch_old
from .multi_catch import MultiCatch
from .spread import Spread
from .forest import Forest
from .payload import PayloadTrack, PayloadFlyThrough
from .dragon import DragonHover
# from .multi_gate import MultiGate
from .rearrange import Rearrange
from .pinballV0 import PinballV0
from .pinball_multiV0 import PingPongMultiV0
from .isaac_env import IsaacEnv

try:
    from .velocity import VelocityEnv
except:
    pass
