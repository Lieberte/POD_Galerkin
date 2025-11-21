# 参数化稳态 Navier-Stokes 方程 POD-Galerkin 降阶建模完整推导

## 1. 问题描述与符号定义

### 1.1 物理域与参数空间

考虑空间域 $\Omega \subset \mathbb{R}^d$ (d=2或3)，边界 $\partial\Omega = \Gamma_D \cup \Gamma_N$，参数域 $\mathcal{D} \subset \mathbb{R}^p$。

### 1.2 控制方程

**稳态不可压缩 Navier-Stokes 方程的强形式：**

$$
\begin{cases}
(\mathbf{u} \cdot \nabla)\mathbf{u} - \nu \Delta \mathbf{u} + \frac{1}{\rho}\nabla p = \mathbf{f} & \text{in } \Omega \\
\nabla \cdot \mathbf{u} = 0 & \text{in } \Omega \\
\mathbf{u} = \mathbf{g}_D(\mu) & \text{on } \Gamma_D \\
\nu \frac{\partial \mathbf{u}}{\partial n} - \frac{p}{\rho}\mathbf{n} = \mathbf{g}_N & \text{on } \Gamma_N
\end{cases}
$$

其中：
- $\mathbf{u}(\mathbf{x};\mu): \Omega \times \mathcal{D} \to \mathbb{R}^d$ 为速度场
- $p(\mathbf{x};\mu): \Omega \times \mathcal{D} \to \mathbb{R}$ 为压力场
- $\nu$ 为运动粘度
- $\rho$ 为密度
- $\mathbf{f}$ 为体积力（可能包含科氏力 $\mathbf{f}_c \times \mathbf{u}$）
- $\mu \in \mathcal{D}$ 为参数向量（如雷诺数、入口速度等）

### 1.3 弱形式

引入测试函数 $\mathbf{v} \in V_0 = \{\mathbf{v} \in H^1(\Omega)^d : \mathbf{v}|_{\Gamma_D} = 0\}$ 和 $q \in Q = L^2(\Omega)$。

弱形式为：寻找 $(\mathbf{u}, p) \in V \times Q$ 使得

$$
\begin{aligned}
a(\mathbf{u}, \mathbf{v};\mu) + c(\mathbf{u}, \mathbf{u}, \mathbf{v}) + b(\mathbf{v}, p) &= \langle \mathbf{f}, \mathbf{v} \rangle \quad \forall \mathbf{v} \in V_0 \\
b(\mathbf{u}, q) &= 0 \quad \forall q \in Q
\end{aligned}
$$

其中双线性形式和三线性形式定义为：

$$
\begin{aligned}
a(\mathbf{u}, \mathbf{v};\mu) &= \nu \int_\Omega \nabla \mathbf{u} : \nabla \mathbf{v} \, d\Omega \\
c(\mathbf{w}, \mathbf{u}, \mathbf{v}) &= \int_\Omega [(\mathbf{w} \cdot \nabla)\mathbf{u}] \cdot \mathbf{v} \, d\Omega \\
b(\mathbf{v}, p) &= -\frac{1}{\rho}\int_\Omega (\nabla \cdot \mathbf{v}) p \, d\Omega
\end{aligned}
$$

## 2. 快照采集与 POD 分解

### 2.1 快照生成

在参数空间 $\mathcal{D}$ 中选取训练样本 $\{\mu^{(i)}\}_{i=1}^{N_s}$，通过高保真求解器（如有限元法）获得快照集合：

$$
\mathcal{S}_u = \{\mathbf{u}^{(1)}, \mathbf{u}^{(2)}, \ldots, \mathbf{u}^{(N_s)}\}, \quad \mathcal{S}_p = \{p^{(1)}, p^{(2)}, \ldots, p^{(N_s)}\}
$$

### 2.2 中心化处理

计算平均场：

$$
\bar{\mathbf{u}} = \frac{1}{N_s}\sum_{i=1}^{N_s} \mathbf{u}^{(i)}, \quad \bar{p} = \frac{1}{N_s}\sum_{i=1}^{N_s} p^{(i)}
$$

构造脉动场快照矩阵：

$$
\mathbf{U}' = [\mathbf{u}^{(1)} - \bar{\mathbf{u}}, \ldots, \mathbf{u}^{(N_s)} - \bar{\mathbf{u}}] \in \mathbb{R}^{N_u \times N_s}
$$

$$
\mathbf{P}' = [p^{(1)} - \bar{p}, \ldots, p^{(N_s)} - \bar{p}] \in \mathbb{R}^{N_p \times N_s}
$$

其中 $N_u$ 和 $N_p$ 分别为速度和压力的自由度数。

### 2.3 奇异值分解

对快照矩阵执行 SVD：

$$
\mathbf{U}' = \mathbf{\Phi}_u \mathbf{\Sigma}_u \mathbf{V}_u^T, \quad \mathbf{P}' = \mathbf{\Phi}_p \mathbf{\Sigma}_p \mathbf{V}_p^T
$$

### 2.4 能量准则截断

根据相对信息内容 (RIC) 确定保留模态数 $r_u$ 和 $r_p$：

$$
\text{RIC}(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{N_s} \sigma_i^2} \geq 1 - \epsilon
$$

通常取 $\epsilon = 10^{-4}$ 到 $10^{-6}$。

得到截断后的 POD 基：

$$
\boldsymbol{\Phi}_u = [\boldsymbol{\phi}_{u,1}, \ldots, \boldsymbol{\phi}_{u,r_u}], \quad \boldsymbol{\Phi}_p = [\boldsymbol{\phi}_{p,1}, \ldots, \boldsymbol{\phi}_{p,r_p}]
$$

### 2.5 正交性验证

POD 基满足 $L^2$ 正交性：

$$
\langle \boldsymbol{\phi}_{u,i}, \boldsymbol{\phi}_{u,j} \rangle_{L^2} = \delta_{ij}, \quad \langle \boldsymbol{\phi}_{p,i}, \boldsymbol{\phi}_{p,j} \rangle_{L^2} = \delta_{ij}
$$

## 3. 降阶空间展开

### 3.1 速度场展开

$$
\mathbf{u}(\mathbf{x};\mu) = \bar{\mathbf{u}}(\mathbf{x}) + \sum_{k=1}^{r_u} \alpha_k(\mu) \boldsymbol{\phi}_{u,k}(\mathbf{x}) = \bar{\mathbf{u}} + \boldsymbol{\Phi}_u \boldsymbol{\alpha}(\mu)
$$

其中 $\boldsymbol{\alpha}(\mu) = [\alpha_1(\mu), \ldots, \alpha_{r_u}(\mu)]^T \in \mathbb{R}^{r_u}$ 为模态系数向量。

### 3.2 压力场展开

$$
p(\mathbf{x};\mu) = \bar{p}(\mathbf{x}) + \sum_{j=1}^{r_p} \beta_j(\mu) \boldsymbol{\phi}_{p,j}(\mathbf{x}) = \bar{p} + \boldsymbol{\Phi}_p \boldsymbol{\beta}(\mu)
$$

其中 $\boldsymbol{\beta}(\mu) = [\beta_1(\mu), \ldots, \beta_{r_p}(\mu)]^T \in \mathbb{R}^{r_p}$。

### 3.3 边界条件处理

对于非齐次 Dirichlet 边界条件，采用提升函数方法：

$$
\mathbf{u} = \mathbf{u}_{lift}(\mu) + \tilde{\mathbf{u}}, \quad \tilde{\mathbf{u}}|_{\Gamma_D} = 0
$$

其中 $\mathbf{u}_{lift}(\mu)$ 满足边界条件但不一定满足 PDE，$\tilde{\mathbf{u}}$ 为齐次边界条件下的解。

## 4. Galerkin 投影推导

### 4.1 残差定义

将展开式代入弱形式，定义残差：

$$
\mathcal{R}(\boldsymbol{\alpha}, \boldsymbol{\beta}; \mu) = a(\bar{\mathbf{u}} + \boldsymbol{\Phi}_u\boldsymbol{\alpha}, \cdot) + c(\bar{\mathbf{u}} + \boldsymbol{\Phi}_u\boldsymbol{\alpha}, \bar{\mathbf{u}} + \boldsymbol{\Phi}_u\boldsymbol{\alpha}, \cdot) + b(\cdot, \bar{p} + \boldsymbol{\Phi}_p\boldsymbol{\beta}) - \langle \mathbf{f}, \cdot \rangle
$$

### 4.2 动量方程投影

使用 $\boldsymbol{\phi}_{u,i}$ 作为测试函数（$i=1,\ldots,r_u$），要求：

$$
\mathcal{R}_i = \langle \mathcal{R}, \boldsymbol{\phi}_{u,i} \rangle = 0
$$

**展开各项：**

#### (A) 扩散项

$$
\begin{aligned}
a(\bar{\mathbf{u}} + \boldsymbol{\Phi}_u\boldsymbol{\alpha}, \boldsymbol{\phi}_{u,i}) &= \nu \int_\Omega \nabla(\bar{\mathbf{u}} + \sum_k \alpha_k \boldsymbol{\phi}_{u,k}) : \nabla \boldsymbol{\phi}_{u,i} \, d\Omega \\
&= \underbrace{\nu \int_\Omega \nabla\bar{\mathbf{u}} : \nabla\boldsymbol{\phi}_{u,i} \, d\Omega}_{a_{i}^0} + \sum_k \alpha_k \underbrace{\nu \int_\Omega \nabla\boldsymbol{\phi}_{u,k} : \nabla\boldsymbol{\phi}_{u,i} \, d\Omega}_{A_{ik}^{diff}}
\end{aligned}
$$

定义：

$$
A_{ik}^{diff} = \nu \langle \nabla\boldsymbol{\phi}_{u,k}, \nabla\boldsymbol{\phi}_{u,i} \rangle
$$

#### (B) 对流项（非线性核心）

$$
\begin{aligned}
&c(\bar{\mathbf{u}} + \boldsymbol{\Phi}_u\boldsymbol{\alpha}, \bar{\mathbf{u}} + \boldsymbol{\Phi}_u\boldsymbol{\alpha}, \boldsymbol{\phi}_{u,i}) \\
&= \int_\Omega [(\bar{\mathbf{u}} + \sum_k \alpha_k\boldsymbol{\phi}_{u,k}) \cdot \nabla](\bar{\mathbf{u}} + \sum_\ell \alpha_\ell\boldsymbol{\phi}_{u,\ell}) \cdot \boldsymbol{\phi}_{u,i} \, d\Omega
\end{aligned}
$$

展开为四项：

1. **常数项**：
$$
c_i^0 = \int_\Omega [(\bar{\mathbf{u}} \cdot \nabla)\bar{\mathbf{u}}] \cdot \boldsymbol{\phi}_{u,i} \, d\Omega
$$

2. **线性项（类型 I）**：
$$
A_{ik}^{conv,1} = \int_\Omega [(\bar{\mathbf{u}} \cdot \nabla)\boldsymbol{\phi}_{u,k}] \cdot \boldsymbol{\phi}_{u,i} \, d\Omega
$$

3. **线性项（类型 II）**：
$$
A_{ik}^{conv,2} = \int_\Omega [(\boldsymbol{\phi}_{u,k} \cdot \nabla)\bar{\mathbf{u}}] \cdot \boldsymbol{\phi}_{u,i} \, d\Omega
$$

4. **二次非线性项**：
$$
B_{ik\ell} = \int_\Omega [(\boldsymbol{\phi}_{u,k} \cdot \nabla)\boldsymbol{\phi}_{u,\ell}] \cdot \boldsymbol{\phi}_{u,i} \, d\Omega
$$

#### (C) 压力梯度项

通过分部积分：

$$
b(\boldsymbol{\phi}_{u,i}, \bar{p} + \boldsymbol{\Phi}_p\boldsymbol{\beta}) = -\frac{1}{\rho}\int_\Omega (\nabla \cdot \boldsymbol{\phi}_{u,i})(\bar{p} + \sum_j \beta_j \boldsymbol{\phi}_{p,j}) \, d\Omega
$$

定义：

$$
\begin{aligned}
b_i^{p,0} &= -\frac{1}{\rho}\int_\Omega (\nabla \cdot \boldsymbol{\phi}_{u,i})\bar{p} \, d\Omega \\
C_{ij} &= -\frac{1}{\rho}\int_\Omega (\nabla \cdot \boldsymbol{\phi}_{u,i})\boldsymbol{\phi}_{p,j} \, d\Omega
\end{aligned}
$$

#### (D) 体积力项

$$
f_i = \int_\Omega \mathbf{f} \cdot \boldsymbol{\phi}_{u,i} \, d\Omega
$$

若包含科氏力 $\mathbf{f}_c \times \mathbf{u}$：

$$
f_i^{cor} = \int_\Omega (\mathbf{f}_c \times \bar{\mathbf{u}}) \cdot \boldsymbol{\phi}_{u,i} \, d\Omega + \sum_k \alpha_k \int_\Omega (\mathbf{f}_c \times \boldsymbol{\phi}_{u,k}) \cdot \boldsymbol{\phi}_{u,i} \, d\Omega
$$

### 4.3 动量方程组装

综合所有项，第 $i$ 个动量方程为：

$$
\underbrace{(a_i^0 + c_i^0 + b_i^{p,0} - f_i)}_{=: b_i} + \sum_k \underbrace{(A_{ik}^{diff} + A_{ik}^{conv,1} + A_{ik}^{conv,2})}_{=: A_{ik}} \alpha_k + \sum_{k,\ell} B_{ik\ell}\alpha_k\alpha_\ell + \sum_j C_{ij}\beta_j = 0
$$

矩阵形式：

$$
\mathbf{b} + \mathbf{A}\boldsymbol{\alpha} + \mathcal{B}(\boldsymbol{\alpha}, \boldsymbol{\alpha}) + \mathbf{C}\boldsymbol{\beta} = \mathbf{0}
$$

其中：
- $\mathbf{b} \in \mathbb{R}^{r_u}$：常数向量
- $\mathbf{A} \in \mathbb{R}^{r_u \times r_u}$：线性刚度矩阵
- $\mathcal{B} \in \mathbb{R}^{r_u \times r_u \times r_u}$：非线性张量
- $\mathbf{C} \in \mathbb{R}^{r_u \times r_p}$：压力耦合矩阵
- $\mathcal{B}(\boldsymbol{\alpha}, \boldsymbol{\alpha})$ 的第 $i$ 分量为 $\sum_{k=1}^{r_u}\sum_{\ell=1}^{r_u} B_{ik\ell}\alpha_k\alpha_\ell$

### 4.4 连续性方程投影

使用 $\boldsymbol{\phi}_{p,j}$ 作为测试函数（$j=1,\ldots,r_p$）：

$$
\int_\Omega \boldsymbol{\phi}_{p,j} (\nabla \cdot \mathbf{u}) \, d\Omega = 0
$$

代入展开式：

$$
\int_\Omega \boldsymbol{\phi}_{p,j} \nabla \cdot (\bar{\mathbf{u}} + \sum_k \alpha_k \boldsymbol{\phi}_{u,k}) \, d\Omega = 0
$$

假设平均场满足无散条件 $\nabla \cdot \bar{\mathbf{u}} = 0$（通常快照已满足），则：

$$
\sum_k \alpha_k \underbrace{\int_\Omega \boldsymbol{\phi}_{p,j} (\nabla \cdot \boldsymbol{\phi}_{u,k}) \, d\Omega}_{D_{jk}} = 0
$$

矩阵形式：

$$
\mathbf{D}\boldsymbol{\alpha} = \mathbf{0}
$$

其中 $\mathbf{D} \in \mathbb{R}^{r_p \times r_u}$ 为散度矩阵。

## 5. 降阶系统的完整形式

### 5.1 鞍点系统

联立动量和连续性方程，得到非线性代数系统：

$$
\boxed{
\begin{cases}
\mathbf{b} + \mathbf{A}\boldsymbol{\alpha} + \mathcal{B}(\boldsymbol{\alpha}, \boldsymbol{\alpha}) + \mathbf{C}\boldsymbol{\beta} = \mathbf{0} & \in \mathbb{R}^{r_u} \\
\mathbf{D}\boldsymbol{\alpha} = \mathbf{0} & \in \mathbb{R}^{r_p}
\end{cases}
}
$$

求解未知量 $\mathbf{x} = [\boldsymbol{\alpha}^T, \boldsymbol{\beta}^T]^T \in \mathbb{R}^{r_u + r_p}$。

### 5.2 矩阵存储与计算复杂度

**离线阶段（一次性计算）：**
- $\mathbf{b}$: $O(N_u r_u)$
- $\mathbf{A}$: $O(N_u r_u^2)$
- $\mathcal{B}$: $O(N_u r_u^3)$（最耗时）
- $\mathbf{C}$: $O(N_u r_u r_p)$
- $\mathbf{D}$: $O(N_p r_u r_p)$

**在线阶段（每个新参数）：**
求解 $(r_u + r_p)$ 维非线性系统，计算量 $O((r_u + r_p)^3)$，远小于原问题的 $O((N_u + N_p)^3)$。

## 6. 牛顿-拉夫逊求解算法

### 6.1 残差向量

定义残差函数 $\mathbf{R}: \mathbb{R}^{r_u + r_p} \to \mathbb{R}^{r_u + r_p}$：

$$
\mathbf{R}(\mathbf{x}) = \begin{bmatrix}
\mathbf{b} + \mathbf{A}\boldsymbol{\alpha} + \mathcal{B}(\boldsymbol{\alpha}, \boldsymbol{\alpha}) + \mathbf{C}\boldsymbol{\beta} \\
\mathbf{D}\boldsymbol{\alpha}
\end{bmatrix}
$$

### 6.2 雅可比矩阵

计算 Jacobian：

$$
\mathbf{J}(\mathbf{x}) = \frac{\partial \mathbf{R}}{\partial \mathbf{x}} = \begin{bmatrix}
\mathbf{A} + \frac{\partial}{\partial \boldsymbol{\alpha}}\mathcal{B}(\boldsymbol{\alpha}, \boldsymbol{\alpha}) & \mathbf{C} \\
\mathbf{D} & \mathbf{0}
\end{bmatrix}
$$

其中非线性项的导数：

$$
\left[\frac{\partial \mathcal{B}}{\partial \boldsymbol{\alpha}}\right]_{ik} = \sum_{\ell=1}^{r_u} (B_{ik\ell} + B_{i\ell k})\alpha_\ell
$$

因此：

$$
\mathbf{J}_{11} = \mathbf{A} + \sum_{\ell=1}^{r_u} \alpha_\ell (\mathcal{B}_{:,:,\ell} + \mathcal{B}_{:,\ell,:})
$$

### 6.3 迭代格式

给定初值 $\mathbf{x}^{(0)}$（通常取零向量或前次参数的解），迭代：

$$
\mathbf{x}^{(n+1)} = \mathbf{x}^{(n)} - [\mathbf{J}(\mathbf{x}^{(n)})]^{-1} \mathbf{R}(\mathbf{x}^{(n)})
$$

收敛准则：

$$
\frac{\|\mathbf{R}(\mathbf{x}^{(n+1)})\|}{\|\mathbf{R}(\mathbf{x}^{(0)})\|} < \text{tol} \quad \text{或} \quad \|\mathbf{x}^{(n+1)} - \mathbf{x}^{(n)}\| < \text{tol}
$$

通常取 $\text{tol} = 10^{-6}$ 到 $10^{-8}$。

### 6.4 鞍点系统求解

实际计算中，使用块消元或预条件 Krylov 子空间方法求解：

$$
\begin{bmatrix}
\mathbf{J}_{11} & \mathbf{C} \\
\mathbf{D} & \mathbf{0}
\end{bmatrix}
\begin{bmatrix}
\Delta\boldsymbol{\alpha} \\
\Delta\boldsymbol{\beta}
\end{bmatrix}
= -\begin{bmatrix}
\mathbf{R}_u \\
\mathbf{R}_p
\end{bmatrix}
$$

## 7. 参数依赖性处理

### 7.1 仿射参数分解

若问题具有仿射参数依赖性：

$$
\mathbf{A}(\mu) = \sum_{q=1}^{Q_A} \Theta_q^A(\mu) \mathbf{A}_q
$$

则离线阶段存储 $\{\mathbf{A}_q\}_{q=1}^{Q_A}$，在线阶段仅需计算系数 $\Theta_q^A(\mu)$ 并组装。

### 7.2 非仿射情形

对于几何参数化（如形状变化），采用经验插值方法 (EIM) 或离散经验插值 (DEIM) 实现近似仿射分解。

### 7.3 边界条件参数化

参数化入口速度 $\mathbf{u}_{in}(\mu) = \mu \cdot \mathbf{u}_{ref}$，则：

$$
\mathbf{b}(\mu) = \mathbf{b}_0 + \mu \cdot \mathbf{b}_1
$$

其中 $\mathbf{b}_0, \mathbf{b}_1$ 在离线阶段预计算。

## 8. 稳定性与精度分析

### 8.1 Inf-Sup 条件

降阶速度-压力空间需满足离散 inf-sup 条件（Ladyzhenskaya-Babuška-Brezzi 条件）：

$$
\inf_{\beta \in \mathbb{R}^{r_p}} \sup_{\alpha \in \mathbb{R}^{r_u}} \frac{\boldsymbol{\beta}^T \mathbf{D} \boldsymbol{\alpha}}{\|\boldsymbol{\alpha}\| \|\boldsymbol{\beta}\|} \geq \gamma > 0
$$

若不满足，需采用最高阶压力抑制 (supremizer enrichment) 或压力泊松方程稳定化。

### 8.2 误差估计

定义误差：

$$
\mathbf{e}_u(\mu) = \mathbf{u}_{FOM}(\mu) - \mathbf{u}_{ROM}(\mu)
$$

相对 $H^1$ 误差：

$$
\epsilon_{rel}(\mu) = \frac{\|\mathbf{e}_u(\mu)\|_{H^1}}{\|\mathbf{u}_{FOM}(\mu)\|_{H^1}}
$$

### 8.3 后验误差指示器

计算残差范数作为误差指示器：

$$
\eta(\mu) = \|\mathcal{R}(\boldsymbol{\alpha}_{ROM}, \boldsymbol{\beta}_{ROM}; \mu)\|_{V' \times Q'}
$$

用于自适应基函数增补。

## 9. 算法总结

### 9.1 离线阶段

1. 在参数空间采样 $\{\mu^{(i)}\}_{i=1}^{N_s}$
2. 求解高保真 FOM 获得快照集 $\{\mathbf{u}^{(i)}, p^{(i)}\}$
3. 计算平均场 $\bar{\mathbf{u}}, \bar{p}$
4. 对中心化快照执行 POD/SVD
5. 根据能量准则截断得到基 $\boldsymbol{\Phi}_u, \boldsymbol{\Phi}_p$
6. 预计算并存储张量 $\mathbf{b}, \mathbf{A}, \mathcal{B}, \mathbf{C}, \mathbf{D}$

### 9.2 在线阶段

给定新参数 $\mu^*$：

1. 初始化 $\mathbf{x}^{(0)} = \mathbf{0}$ 或使用插值预测
2. 牛顿迭代求解 $\mathbf{R}(\mathbf{x}) = 0$
3. 重构物理场：
$$
\mathbf{u}_{ROM}(\mu^*) = \bar{\mathbf{u}} + \boldsymbol{\Phi}_u \boldsymbol{\alpha}^*
$$
$$
p_{ROM}(\mu^*) = \bar{p} + \boldsymbol{\Phi}_p \boldsymbol{\beta}^*
$$

## 10. 数值实现要点

### 10.1 积分计算

所有积分项（如 $B_{ik\ell}$）采用与 FOM 相同的数值积分方案（Gauss 积分）。

### 10.2 张量存储优化

利用对称性减少存储：

$$
B_{ik\ell} = -B_{i\ell k} \quad \text{（反对称性）}
$$

仅存储上三角部分。

### 10.3 并行化

张量 $\mathcal{B}$ 的计算高度并行，可使用 OpenMP 或 MPI。

### 10.4 验证步骤

- 检查 POD 基正交性
- 验证 $\mathbf{D}\boldsymbol{\Phi}_u = \mathbf{0}$（若原快照无散）
- 对比 ROM 与 FOM 在训练参数上的解

---
