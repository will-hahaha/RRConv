import torch
import torch.nn as nn
import scipy.io as sio
import os
import torch.nn.functional as F


class mySin(torch.nn.Module):  # sin激活函数
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.sin(x)
        return x


class RectConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, l_max=9, w_max=9, flag=False, modulation=True):
        super(RectConv2d, self).__init__()
        self.lmax = l_max
        self.wmax = w_max
        self.inc = inc
        self.outc = outc
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.flag = flag
        self.modulation = modulation

        self.i_list = [33, 35, 53, 37, 73, 55, 57, 75, 77]

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(inc, outc, kernel_size=(i // 10, i % 10), stride=(i // 10, i % 10), padding=0)
                for i in self.i_list
            ]
        )

        self.m_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
            nn.Tanh()
        )
        self.b_conv = nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride)
        self.l_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.Sigmoid()
        )
        self.w_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.Sigmoid()
        )
        self.theta_conv = nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride)

        self.hook_handles = []
        self.hook_handles.append(self.l_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.w_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.theta_conv[0].register_full_backward_hook(self._set_lr))

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = tuple(g * 0.1 if g is not None else None for g in grad_input)
        grad_output = tuple(g * 0.1 if g is not None else None for g in grad_output)
        return grad_input

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()  # 移除钩子函数
        self.hook_handles.clear()  # 清空句柄列表

    def forward(self, x, epoch, label, nx, ny):
        m = self.m_conv(x)
        bias = self.b_conv(x)
        l = self.l_conv(x) * 8 + 1
        w = self.w_conv(x) * 8 + 1
        theta = self.theta_conv(x)
        mean_l = l.mean(dim=0).mean(dim=1).mean(dim=1)
        mean_w = w.mean(dim=0).mean(dim=1).mean(dim=1)
        N_X = int(mean_l // 1)
        N_Y = int(mean_w // 1)
        if N_X % 2 == 0:
            N_X -= 1
        if N_Y % 2 == 0:
            N_Y -= 1
        if N_X < 3:
            N_X = 3
        if N_Y < 3:
            N_Y = 3
        if epoch >= 300 and self.hook_handles:
            self.remove_hooks()

        N = N_X * N_Y
        l = l.repeat([1, N, 1, 1])
        w = w.repeat([1, N, 1, 1])
        offset = torch.cat((l, w), dim=1)
        dtype = offset.data.type()
        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype, N_X, N_Y, theta)  # (b, 2*N, h, w)
        p = p.contiguous().permute(0, 2, 3, 1)  # (b, h, w, 2*N)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat(
            [
                torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_lt[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_rb = torch.cat(
            [
                torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_rb[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        # clip p
        p = torch.cat(
            [
                torch.clamp(p[..., :N], 0, x.size(2) - 1),
                torch.clamp(p[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        )

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (
                1 + (q_lt[..., N:].type_as(p) - p[..., N:])
        )
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_rb[..., N:].type_as(p) - p[..., N:])
        )
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_lb[..., N:].type_as(p) - p[..., N:])
        )
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (
                1 + (q_rt[..., N:].type_as(p) - p[..., N:])
        )
        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        # (b, c, h, w, N)
        x_offset = (
                g_lt.unsqueeze(dim=1) * x_q_lt
                + g_rb.unsqueeze(dim=1) * x_q_rb
                + g_lb.unsqueeze(dim=1) * x_q_lb
                + g_rt.unsqueeze(dim=1) * x_q_rt
        )

        x_offset = self._reshape_x_offset(x_offset, N_X, N_Y)
        x_offset = self.convs[self.i_list.index(N_X * 10 + N_Y)](x_offset)
        out = m * x_offset + bias
        return out

    def _get_p_n(self, N, dtype, n_x, n_y):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(n_x - 1) // 2, (n_x - 1) // 2 + 1),
            torch.arange(-(n_y - 1) // 2, (n_y - 1) // 2 + 1),
        )
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype, n_x, n_y, theta):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        L, W = offset.split([N, N], dim=1)
        L = L / n_x
        W = W / n_y
        offsett = torch.cat([L, W], dim=1)
        p_n = self._get_p_n(N, dtype, n_x, n_y)
        p_n = p_n.repeat([1, 1, h, w])
        p_x, p_y = p_n.split([N, N], dim=1)
        p_xx = p_x * torch.cos(theta) - p_y * torch.sin(theta)
        p_yy = p_x * torch.sin(theta) + p_y * torch.cos(theta)
        p_n = torch.cat([p_xx, p_yy], dim=1)  # b, 2N, h, w
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + offsett * p_n
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = (
            index.contiguous()
            .unsqueeze(dim=1)
            .expand(-1, c, -1, -1, -1)
            .contiguous()
            .view(b, c, -1)
        )
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, n_x, n_y):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + n_y].contiguous().view(b, c, h, w * n_y) for s in range(0, N, n_y)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * n_x, w * n_y)
        return x_offset
