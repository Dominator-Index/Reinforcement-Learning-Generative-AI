
    def sample_RK4(
        self,
        xT: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        steps: int = 100,
        w_cf: float = 0.0,
        w_cg: float = 0.0,
        verbose: bool = False,
        return_traj: bool = False,
        clip_denoised: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Flow-Matching Sampling from x_T to x_0 using RK4 integration
        """
        device = xT.device
        B = xT.shape[0]
        xt = xT
        traj = [xt] if return_traj else None

        # 构造线性时间步 t_i，从 1 -> 0（正向流）
        t_steps = torch.linspace(1.0, 0.0, steps + 1, device=device)

        for i in range(steps):
            t_cur = t_steps[i].expand(B)
            t_next = t_steps[i + 1].expand(B)
            delta_t = (t_next - t_cur).view(-1, *[1]*(xt.ndim-1))  # Δt < 0

            # 定义 vector field v(t, x)
            def v(t, x):
                return self.guided_sampling_flow(
                    xt=x,
                    t=t,
                    model=self.model,
                    condition=condition,
                    w_cf=w_cf,
                    w_cg=w_cg,
                    requires_grad=False,
                )[0]  # only return v, discard logp

            # Runge-Kutta 4 integration step
            k1 = v(t_cur, xt)
            k2 = v(t_cur + 0.5 * delta_t.view(-1), xt + 0.5 * delta_t * k1)
            k3 = v(t_cur + 0.5 * delta_t.view(-1), xt + 0.5 * delta_t * k2)
            k4 = v(t_next, xt + delta_t * k3)

            xt = xt + (delta_t / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            if clip_denoised:
                xt = self.clip_prediction(xt)

            if return_traj:
                traj.append(xt)

        return (xt, torch.stack(traj, dim=1)) if return_traj else xt
    
    def sample(
        self,
        x1: torch.Tensor,                                # 目标样本
        condition: Optional[torch.Tensor] = None,        # 条件输入
        steps: int = 100,                                # 时间离散步数（用于输出轨迹）
        w_cf: float = 0.0,                               # classifier-free guidance 权重
        w_cg: float = 0.0,                               # classifier guidance 权重
        return_traj: bool = False,                       # 是否返回整个轨迹
        requires_grad: bool = False,                     # 是否对采样过程保留梯度
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        使用 Flow Matching (NeuralODE) 的采样方法，支持 classifier guidance 与 classifier-free guidance。
        采用高阶 RK 方法（如 RK4 / dopri5）进行数值积分。

        Inputs:
        - x1:            目标样本 (B, D)
        - condition:     条件 (B, C)
        - steps:         积分轨迹步数
        - w_cf:          classifier-free guidance scale
        - w_cg:          classifier guidance scale
        - return_traj:   是否返回采样轨迹（形如 [T, B, D]）
        - requires_grad: 是否保留采样过程梯度

        Returns:
        - x0:            生成的样本 (B, D)
        - traj:          [可选] (T, B, D) 的轨迹
        """
        from torchdiffeq import odeint

        device = x1.device
        dtype = x1.dtype
        B, D = x1.shape

        # 初始状态（采样源）
        x0 = self.sample_source(x1, temperature=1.0).to(device)

        # 构造时间步（从 0 到 1 的连续时间）
        t_span = torch.linspace(0, 1, steps, device=device, dtype=dtype)

        # 定义速度场函数（兼容 odeint 接口）
        def vf(t, x):
            """
            ODE 右边的向量场 dx/dt = v(x, t)
            输入:
            - t: (scalar tensor,)
            - x: (B, D)
            输出:
            - v: (B, D)
            """
            t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)

            # Flow Matching 的 guidance 模块（支持 classifier / classifier-free guidance）
            v, _ = self.guided_sampling_flow(
                xt=x,
                t=t_batch,
                model=self.model,
                condition=condition,
                w_cf=w_cf,
                w_cg=w_cg,
                requires_grad=requires_grad
            )
            return v

        # 调用 ODE 解算器：RK4 / dopri5 等
        traj = odeint(
            vf,
            x0,
            t_span,
            method="dopri5",
            atol=1e-4,
            rtol=1e-4
        )  # 输出形状: (T, B, D)

        xT = traj[-1]  # 最后一个时间步的结果

        if return_traj:
            return xT, traj
        else:
            return xT
        
    def sample_Neurl_ODE(
        self,
        x1: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        steps: int = 100,
        w_cf: float = 0.0,
        w_cg: float = 0.0,
        return_traj: bool = False,
        requires_grad: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        用 NeuralODE 实现 flow matching 采样，支持 classifier-guidance 和 classifier-free guidance。
        """
        from torchdiffeq import odeint
        from torchdyn.core import NeuralODE

        device = x1.device
        dtype = x1.dtype
        B, D = x1.shape

        x0 = self.sample_source(x1, temperature=1.0).to(device)
        t_span = torch.linspace(0, 1, steps, device=device, dtype=dtype)

        class GuidedWrapper(nn.Module):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer

            def forward(self, t, x):
                # x: (B, D), t: scalar
                t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
                v, _ = self.outer.guided_sampling_flow(
                    xt=x,
                    t=t_batch,
                    model=self.outer.model,
                    condition=condition,
                    w_cf=w_cf,
                    w_cg=w_cg,
                    requires_grad=requires_grad
                )
                return v

        # 包装为 NeuralODE 模型
        wrapper = GuidedWrapper(self)
        node = NeuralODE(wrapper, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

        # 调用 trajectory（等价于 odeint）
        traj = node.trajectory(x0, t_span)

        xT = traj[-1]
        return (xT, traj) if return_traj else xT