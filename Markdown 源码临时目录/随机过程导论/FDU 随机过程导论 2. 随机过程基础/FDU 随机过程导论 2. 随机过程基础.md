# FDU 随机过程导论 2. 随机过程基础

本文根据王老师课堂笔记整理而成，并参考了以下教材: 

  - 随机过程 (苏中根) 第 $2$ 章
  - Introduction to Probability Models: Applied Stochastic Processes (S. Ross) 
  - 应用随机过程概率论模型导论 (S. Ross) 龚光鲁译 第 $2$ & $3$ 章

欢迎批评指正!

## 2.1 随机过程简介

### 2.1.1 随机过程的基本概念

考虑一个定义良好的概率空间 $(\Omega,\mathcal E,\text{P})$  
其中 $\Omega$ 为样本空间，$\mathcal E=\{E:E\subseteq \Omega\}$ 为事件空间，$\text{P}:\mathcal E\to [0,1]$ 为概率测度函数.

**随机过程** $\{X(\cdot,t):t\in T\}$ 是一族随机变量，描述了某个过程经历的时间发展.  
也就是说，对于任意指标 $t\in T$，$X(\cdot,t):\Omega\to \mathbb R$ 都是**随机变量**  
为简洁起见，当我们讨论随机变量 $X(\cdot,t)$ 时，通常会将其写作 $X(t)$ 或 $X_t$  
我们还引入一个新的概念:    
对于任意 $\omega\in\Omega$，称 $X(\omega,\cdot):T\to \mathbb R$ 为**样本路径** (sample path) 或**样本曲线**.

- 指标 $t$ 通常解释为**时刻** (time)   
  **指标集** (index set) $T$ 通常解释为此过程的**时间参数空间** (time space): 
  - 当 $T$ 为可数集时，$\{X(t):t\in T\}$ 称为**离散时间随机过程** (discrete time stochastic process)  
    例如 $\{X_n:n=0,1,\dots\}$ 是一个以非负整数为指标的离散时间随机过程.
  - 当 $T$ 为连续区间时，$\{X(t):t\in T\}$ 称为**连续时间随机过程** (continuous time stochastic process)  
    例如 $\{X(t):t\geq 0\}$ 是一个以非负实数为指标的连续时间随机过程.
- $X(\omega,t)$ 通常解释为该随机过程沿着**样本路径** $X(\omega,\cdot)$ 在**时刻** $t$ 所达到的**状态** (state)，  
  状态的全体称为**状态空间** (state space)，记为 $S=\{X(\omega,t):\omega\in \Omega,t\in T\}\subseteq \mathbb R$.



### 2.1.2 随机过程的概率分布

如何刻画随机过程 $\{X(t):t\in T\}$ 的概率分布呢？    
我们通过其所有**有限维分布函数族**来描述:

- **一维分布: **  
  对于任意 $t\in T$，定义随机变量 $X(t)$ 的分布函数为:   
  $F_t(x) = \text{P}\{X(t)\leq x\}$  
  **一维分布函数族**为 $\{F_t:t\in T\}$ 
- **多维分布: **  
  对于任意 $t_1,\dots,t_k\in T$，定义随机变量 $X(t_1),\dots,X(t_k)$ 的联合分布函数为:   
  $F_{t_1,\dots,t_k}(x_1,\dots,x_k) = \text{P}\{X(t_1)\leq x_1,\dotsm,X(t_k)\leq x_k\}$   
  **多维分布函数族**为 $\{F_{t_1,\dots,t_k}:t_1,\dots,t_k\in T\}$ 
- 考虑同一概率空间 $(\Omega,\mathcal E,\text{P})$ 上的两个随机过程 $\begin{cases}
  \mathbf X = \{X(t):t\in T\}\\
  \mathbf Y = \{Y(t):t\in T\}
  \end{cases}$   
  $\mathbf X,\mathbf Y$ 拥有相同的概率分布，当且仅当其任意有限维分布函数族都相同，  
  即对于任意 $\begin{cases}
  k\geq 1\\
  t_1,\dots,t_k\in T\end{cases}$ 都有 $F_{t_1,\dots,t_k}^{\mathbf X}\equiv F_{t_1,\dots,t_k}^{\mathbf Y}$ 成立.

计算有限维分布函数并不是一件容易的事情，  
下面仅针对两种特殊情况给出计算方法: 

- 给定 $\begin{cases}
  k\geq 1\\
  t_1,\dots,t_k\in T\end{cases}$   
  若 $X(t_1),X(t_2)-X(t_1),\dots,X(t_k)-X(t_{k-1})$ 的联合分布 $F_{X(t_1),X(t_2)-X(t_1),\dots,X(t_k)-X(t_{k-1})}$ 已知，  
  则可根据线性变换方法计算 $X(t_1),X(t_2),\dots,X(t_k)$ 的联合分布 $F_{X(t_1),X(t_2),\dots,X(t_k)}$.  
  具体来说，我们有 $\begin{cases}
  X(t_1) = X(t_1)\\
  X(t_2) = X(t_1) + [X(t_2)-X(t_1)]\\
  \quad\vdots\\
  X(t_k) = X(t_1) + [X(t_2)-X(t_1)] + \dotsm + [X(t_k)-X(t_{k-1})]\end{cases}$  
  即 $\begin{bmatrix}
  X(t_1)\\
  X(t_2)\\
  \vdots\\
  X(t_k)\end{bmatrix}=
  \begin{bmatrix}
  1 &&&\\
  1&1&&\\
  \vdots &\ddots &\ddots\\
  1&\dots &1 & 1\end{bmatrix}\begin{bmatrix}
  X(t_1)\\
  X(t_2)-X(t_1)\\
  \vdots\\
  X(t_k)-X(t_{k-1})\end{bmatrix}:= J_k
  \begin{bmatrix}
  X(t_1)\\
  X(t_2)-X(t_1)\\
  \vdots\\
  X(t_k)-X(t_{k-1})\end{bmatrix}$    
  根据 **1. 概率论回顾 $1.3.5$** 的内容可知:
  $$
  \begin{align}
  F_{X(t_1),X(t_2),\dots,X(t_k)}(x_1,x_2,\dots,x_k)
  &=F_{X(t_1),X(t_2)-X(t_1),\dots,X(t_k)-X(t_{k-1})}(x_1,x_2-x_1,\dots,x_k-x_{k-1})\cdot |\det(J_k)|^{-1}\\
  &=F_{X(t_1),X(t_2)-X(t_1),\dots,X(t_k)-X(t_{k-1})}(x_1,x_2-x_1,\dots,x_k-x_{k-1})\end{align}
  $$
  上述结果表明:   
  **通过考虑随机过程在各个时间点的增量，我们可以更简洁地表达多时间点上随机变量的联合分布.**
  
- 定义事件 $E_k := \{X(t_k)=x_k\}$ 
  若对于任意 $\begin{cases}
  k\geq 2\\
  t_1,\dots,t_k\in T\\
  x_1,\dots,x_{k-1} \in \mathbb R\end{cases}$ ，$X(t_k)$ 在给定 $\begin{cases}
  X(t_1)=x_1\\
  X(t_2)=x_2\\
  \quad\dotsm\\
  X(t_{k-1})=x_{k-1}\end{cases}$ 时的条件分布都是已知的，  
  即对于任意 $x_k\in\mathbb R$，条件概率 $\text{P}\{X(t_k)=x_k|E_1E_2\dotsm E_{k-1}\}$ 都是已知的，  
  则根据**条件概率链式法则**可以计算 $X_1,\dots,X_n$ 的联合分布函数:  
  $$
  \begin{align}
  \text{P}\{X(t_1)=x_1,\dotsm,X(t_k)=x_k\}
  &= \text{P}\{E_1E_2\dotsm E_k\}\\
  &= \text{P}(E_1)\text{P}(E_2|E_1)\dotsm\text{P}(E_k|E_1E_2\dotsm E_{k-1})\end{align}
  $$
  上述结果表明:   
  我们可以将随机过程在多个时间点上取值的联合概率分解为一系列条件概率的乘积.  
  这一分解手段不仅在计算上是实用的，而且提供了对随机过程如何随时间演进的直观理解.  
  特别地，对于具有 **Markov 性**的随机过程 (即过程的未来状态仅依赖于当前状态，而与历史路径无关)，  
  条件概率链式法则可以极大简化整个过程的分析.



### 2.1.3 随机过程的数字特征

考虑随机过程 $\mathbf X=\{X(t):t\in T\}$ 

- **(1) 均值函数 (Mean Function)**  
  若对于任意 $t\in T$ 都有 $\text{E}[X(t)]<\infty$，  
  则称 $\mu_{\mathbf X}(t)=\text{E}[X(t)]\ \ (t\in T)$ 为 $\mathbf X$ 的均值函数.

- **(2) 自相关函数 (Autocorrelation Function)**  
  称 $r_{\mathbf X}(s,t) = \text{E}[X(s)X(t)]\ \ (s,t\in T)$ 为 $\mathbf X$ 的自相关函数.

- **(3) 自协方差函数 (Autocovariance Function)**    
  称 $\text{Cov}_{\mathbf X}(s,t)=\text{Cov}(X(s),X(t)) = r_{\mathbf X}(s,t) - \mu_{\mathbf X}(s)\mu_{\mathbf X}(t)\ \ (s,t\in T)$ 为 $\mathbf X$ 的自协方差函数.

  若对于任意 $t\in T$ 都有 $r_{\mathbf X}(t,t)=\text{E}[(X(t))^2]<\infty$，   
  则称 $\sigma_{\mathbf X}^2(t)=\text{Var}(X(t))=r_{\mathbf X}(t,t) - \mu_{\mathbf X}^2(t)\ \ (t\in T)$ 为 $\mathbf X$ 的**方差函数** (variance function).

考虑两个随机过程 $\begin{cases}
\mathbf X=\{X(t):t\in T\}\\
\mathbf Y=\{Y(t):t\in T\}\end{cases}$   

- **(4) 互相关函数 (Cross-Correlation Function)**   
  若对于任意 $t\in T$ 都有 $\begin{cases}
  \text{E}[(X(t))^2]<\infty\\
  \text{E}[(Y(t))^2]<\infty\end{cases}$    
  则称 $r_{\mathbf X,\mathbf Y}(s,t)=\text{E}[X(s)Y(t)]\ \ (s,t\in T)$ 为 $\mathbf X,\mathbf Y$ 的互相关函数
- **(5) 互协方差函数 (Cross-Covariance Function)**    
  称 $\text{Cov}_{\mathbf X,\mathbf Y}(s,t)=\text{Cov}(X(s),Y(t)) = r_{\mathbf X,\mathbf Y}(s,t) - \mu_{\mathbf X}(s)\mu_{\mathbf Y}(t)\ \ (s,t\in T)$ 为 $\mathbf X,\mathbf Y$ 的互协方差函数 

***

 **(余弦波过程)**   
给定二元函数 $X(\theta,t) = a\cos(wt+\theta)\  \ (\theta\in [0,2\pi),t\in\mathbb R)$   
其中 $a,w>0$ 为给定的常数  
我们定义余弦波过程为 $\mathbf X=\{X(t)= a\cos(wt+\Theta):t\in \mathbb R\}$  
其中相位 $\Theta\sim \text{Uniform}[0,2\pi)$ 为均匀随机变量.

我们知道这个随机过程的时间参数空间 $T=\mathbb R$，状态空间 $S=[-a,a]$   
任意给定相位 $\Theta=\theta \in [0,2\pi)$，$X(\theta,t)$ 都是一条关于 $t\in \mathbb R$ 的余弦波曲线；  
(下图给出了两条样本曲线，即余弦波曲线)  

<img src="SP苏中根 Figure 2.1.png" style="zoom:50%;" />

任意给定时刻 $t\in \mathbb R$，$X(\theta,t)$ 都是一个关于样本 $\theta \in [0,2\pi)$ 的随机变量，简记为 $X(t)$   
余弦波过程 $\mathbf X=\{X(t)= a\cos(wt+\Theta):t\in \mathbb R\}$ 的数字特征为: 

- 均值函数:
  $$
  \begin{align}
  \mu_{\mathbf X}(t)
  &= \text{E}[X(t)]\\
  &= a\text{E}[\cos(wt+\Theta)]\\
  &= a\int_0^{2\pi} \cos(wt+\theta)\cdot \frac{1}{2\pi}\mathrm{d}\theta\\
  &= \frac{a}{2\pi}\cdot \sin(wt+\theta)|_0^{2\pi}\\
  &= 0\quad (\forall\ t\in \mathbb R)\end{align}
  $$
  
- 自相关函数:
  $$
  \begin{align}
  r_{\mathbf X}(s,t)
  &= \text{E}[X(s)X(t)]\\
  &= a^2\text{E}[\cos(ws+\Theta)\cos(wt+\Theta)]\\
  &= a^2\int_0^{2\pi} \cos(ws+\theta)\cos(wt+\theta)\cdot \frac{1}{2\pi}\mathrm{d}\theta\\
  &= \frac{a^2}{2\pi}\cdot \frac12 \int_0^{2\pi} \{\cos{[w(s-t))]} + \cos{[w(s+t)+2\theta]}\}\mathrm{d}\theta\\
  &=  \frac{a^2}{4\pi}\{\cos{[w(s-t)]}\theta +\frac{1}{2}\sin{[w(s+t)+2\theta]}\}|_0^{2\pi}\\
  &= \frac{a^2}{2}\cos{[w(s-t)]}\quad (\forall\ s,t\in \mathbb R)\end{align}
  $$
  
- 自协方差函数:
  $$
  \begin{align}
  \text{Cov}_{\mathbf X}(s,t)
  &=r_{\mathbf X}(s,t) -\mu_{\mathbf X}(s)\mu_{\mathbf X}(t)\\
  &=\frac{a^2}{2}\cos{[w(s-t)]}-0\cdot 0\\
  &= \frac{a^2}{2}\cos{[w(s-t)]} \quad (\forall\ s,t\in \mathbb R)\end{align}
  $$
  
- 方差函数:
  $$
  \begin{align}
  \sigma_{\mathbf X}^2(t) 
  &=\text{Cov}_{\mathbf X}(t,t)\\
  &= r_{\mathbf X}(t,t) - \mu_{\mathbf X}^2(t)\\
  &= \frac{a^2}{2}\cos {[w(t-t)]} -0^2\\
  &= \frac{a^2}{2}\quad (\forall\ t\in \mathbb R)\end{align}
  $$

***

**(简单随机游走)**  
设 $\{\xi_n\}$ 为一列独立同分布的随机变量，满足 $\begin{cases}
\text{P}\{\xi_n = 1\}=p\\
\text{P}\{\xi_n = -1\}=1-p\\\end{cases}(0<p<1)$   
定义部分和 $\begin{cases}
S_0 = 0\\
S_n = S_{n-1} + \xi_n = S_0 +\sum_{i=1}^n \xi_i = \sum_{i=1}^n \xi_i\end{cases}$  
那么 $\mathbf S=\{S_n:n\geq 0\}$ 是一个随机过程，描述了质点在直线上的**简单随机游走** (simple random walk)  
时间参数空间 $T=\mathbb N$，状态空间为 $S=\mathbb Z$   
随机变量 $S_n$ 描述了第 $n$ 步后质点所处的位置.

- 均值函数:
  $$
  \begin{align}
  \mu_{\mathbf S}(n)
  &= \text{E}[S_n] \\
  &= \begin{cases}
  0,&\text{if }n=0\\
  n\text{E}[\xi] = n(2p-1),&\text{if }n\geq 1\end{cases}\\
  &= (2p-1)n\ \ (\forall\ n\in \mathbb N)\end{align}
  $$
  
- 自相关函数:
  $$
  \begin{align}
  r_{\mathbf S}(n,m)
  &= \text{E}(S_nS_m)\\
  &= \text{E}(\sum_{i=1}^n \xi_i \cdot\underset{j=1}{\overset{m}\sum}\xi_j)\\
  &= \underset{i=1}{\overset{\min\{n,m\}}\sum}\text{E}(\xi_i^2) +
  \sum_{i=1}^n\underset{j\neq i}{\overset{m}\sum}\text{E}[\xi_i]\text{E}[\xi_j]\\
  &= \min\{n,m\}\cdot 1 + [nm-\min\{n,m\}]\cdot (2p-1)^2\\
  &= \min\{n,m\} + [nm-\min\{n,m\}](2p-1)^2\quad(\forall\ n,m\geq 0)
  \end{align}
  $$
  
- 自协方差函数:
  $$
  \begin{align}
  \text{Cov}_{\mathbf S}(n,m)
  &= r_{\mathbf S}(n,m) -\mu_{\mathbf S}(n)\mu_{\mathbf S}(m)\\
  &= \min\{n,m\} + [nm-\min\{n,m\}](2p-1)^2 - n(2p-1)\cdot m(2p-1)\\
  &= 4p(1-p)\min\{n,m\}\quad(\forall\ n,m\geq 0)\end{align}
  $$
  
- 方差函数:
  $$
  \begin{align}
  \sigma_{\mathbf S}^2(n)
  &= \text{Cov}_{\mathbf S}(n,n)\\
  &= 4p(1-p)n\quad(\forall\ n\in \mathbb N)\end{align}
  $$



### 2.1.4 随机过程的特殊情况

#### (1) 弱平稳过程

设随机过程 $\mathbf X=\{X(t):t\in T\}$ 满足 $\forall\ t\in T, \ \text{E}[(X(t))^2]<\infty$  
如果:   

- **① 均值函数 $\mu_{\mathbf X}(t)$ 为常数: (均值的不变性)**   
  即存在 $\mu\in \mathbb R$ 使得 $\mu_{\mathbf X}(t)\equiv \mu\ \ (\forall\ t\in T)$   
- **② 自相关函数 $r_{\mathbf X}(s,t)$ 仅与时间差 $(s-t)$ 相关: (自相关函数的时间位移不变性)**  
  即存在函数 $g:\mathbb R\to\mathbb R$ 使得 $r_{\mathbf X}(s,t) \equiv g(s-t)\ \ (\forall\ s,t\in T)$   

那么我们称 $\mathbf X$ 为**弱平稳过程** (Weakly Stationary Process).  
根据性质①②我们还可以推出:

- **③ 弱平稳过程的自协方差函数 $\text{Cov}_{\mathbf X}(s,t)$ 仅与时间差 $(s-t)$ 相关: (自协方差的时间位移不变性)**  
  这是因为 $\text{Cov}_{\mathbf X}(s,t)=r_{\mathbf X}(s,t) -\mu_{\mathbf X}(s)\mu_{\mathbf X}(t) \equiv  g^2(s-t) -\mu^2\ \ (\forall\ s,t\in T)$.
- **④ 弱平稳过程的自方差函数 $\sigma_{\mathbf X}^2(t)$  为常数: (方差的一致性)**    
  这是因为  $\sigma_{\mathbf X}^2(t)=\text{Cov}_{\mathbf X}(t,t)\equiv g^2(0)-\mu^2\ \ (\forall\ t\in T)$.

**弱平稳性**只要求一、二阶矩的不变性，并不要求所有高阶矩都一致 (区别于后文的**强平稳性**).
考虑之前的两个例子:   

- **余弦波过程** $\mathbf X=\{X(t)= a\cos(wt+\Theta):t\in \mathbb R\}$ 的数字特征为  
  $$
  \begin{cases}
  \mu_{\mathbf X}(t) \equiv 0\\
  r_{\mathbf X}(s,t) = \frac{a^2}{2}\cos{[w(s-t)]}\ \ (\forall\ s,t\in \mathbb R)\\
  \text{Cov}_{\mathbf X}(s,t) = \frac{a^2}{2}\cos{[w(s-t)]}\ \ (\forall\ s,t\in \mathbb R)\\
  \sigma_{\mathbf X}^2(t)\equiv \frac{a^2}{2}\end{cases}
  $$
  
  因此对于任意给定的常数 $a,w>0$，余弦波过程都是弱平稳过程.
  
- **简单随机游动** $\mathbf S=\{S_n:n\geq 0\}$ 的数字特征为:   
  $$
  \begin{cases}
  \mu_{\mathbf S}(n) = (2p-1)n\ \ (\forall\ n\in \mathbb N)\\
  r_{\mathbf S}(n,m) = \min\{n,m\} + [nm-\min\{n,m\}](2p-1)^2\quad(\forall\ n,m\geq 0)\\
  \text{Cov}_{\mathbf S}(n,m)= 4p(1-p)\min\{n,m\}\quad(\forall\ n,m\geq 0)\\
  \sigma_{\mathbf S}^2(n)=4p(1-p)n\quad(\forall\ n\in \mathbb N)\end{cases}
  $$
  
  因此对于任意给定的常数 $0<p<1$，简单随机游动都不是弱平稳过程.



#### (2) 强平稳过程  

若随机过程 $\mathbf X=\{X(t):t\in T\}$ 满足:  

- 对于任意 $\begin{cases}
  k\geq 1\\
  t_1,\dots,t_k\in T\\
  t\in T\end{cases}$ 都有 $[X(t_1+t),\dots,X(t_k+t)]\overset{\mathrm{d}}= [X(t_1),\dots,X(t_k)]$   
  (即要求随机过程的所有统计性质对于任意时间位移 $t$ 都保持不变，这是一个非常强的条件)

则称 $\mathbf X$ 为**强平稳过程** (Strongly/Strictly Stationary Process).

- 一般来说，随机过程的强、弱平稳性之间没有直接联系，  
  但如果强平稳过程的二阶矩存在且有限 (即对于任意 $t<T$ 都有 $\text{E}[(X(t))^2]<\infty$)，  
  则它一定是弱平稳过程.
- **(强平稳过程的充分条件)**
  若 $\mathbf X=\{X_n:n\geq 0\}$ 是一个随机过程，且 $\{X_n\}$ 独立同分布，  
  则 $\mathbf X$ 是强平稳过程.



#### (3) 平稳/独立增量过程

对于任意 $s<t\in T$，我们称 $X(t)-X(s)$ 为随机过程 $\mathbf X=\{X(t):t\in T\}$ 的**增量** (increment) 

- 若任意增量 $X(t)-X(s)$ 的分布仅依赖于时间差 $t-s$，而与时刻 $s,t$ 无关，  
  则称 $\mathbf X$ 为**平稳增量过程** (stationary increment process)
- 若对于任意 $t_1<t_2<\dots<t_k\in T$，  
  增量 $X(t_1),X(t_2)-X(t_1),\dots,X(t_k)-X(t_{k-1})$ 都相互独立，  
  则称 $\mathbf X$ 为**独立增量过程** (independent increment process)

后面要学到的 Poisson 过程和 Brown 运动都是**独立平稳增量过程**  
考虑**简单随机游动** $\mathbf S=\{S_n:n\geq 0\}$，  
其中 $\begin{cases}
S_0 = 0\\
S_n = S_{n-1} + \xi_n = S_0 +\sum_{i=1}^n \xi_i = \sum_{i=1}^n \xi_i\end{cases}$   
且 $\{\xi_n\}$ 独立同分布，满足 $\begin{cases}
\text{P}\{\xi_n = 1\}=p\\
\text{P}\{\xi_n = -1\}=1-p\\\end{cases}(0<p<1)$   
这里 $\xi_n=S_n-S_{n-1}$ 扮演的就是增量的角色，  
由于 $\{\xi_n\}$ 独立同分布，故简单随机游动 $\mathbf S$ 是一个**独立平稳增量过程**.



#### (4) 正态过程

若随机过程 $\mathbf X=\{X(t):t\in T\}$ 的任意有限维联合分布都是多元正态分布，  
即对于任意  $\begin{cases}
k\geq 1\\  
t_1<t_2<\dots<t_k\in T\end{cases}$ ，$X(t_1),\dots,X(t_k)$ 都服从多元正态分布，  
则称 $X$ 为**正态过程** (Gaussian Process)，即 **Gauss 过程**.

- **(正态过程的充要条件)**     
  随机过程 $\mathbf X=\{X(t):t\in T\}$ 是正态过程，  
  当且仅当对于任意给定的 $\begin{cases}
  k\geq 1\\  
  t_1<t_2<\dots<t_k\in T\end{cases}$，  
  $X(t_1),\dots,X(t_k)$ 的任意线性组合都服从正态分布，  
  即对于任意 $a_1,\dots,a_k\in \mathbb R$，线性组合 $\sum_{i=1}^k a_i X(t_i)$ 都服从正态分布.

由于多元正态随机向量的分布由均值向量和协方差矩阵唯一确定，  
因此**弱平稳正态过程与强平稳正态过程是等价的**.



#### (5) 白噪声

若随机过程 $\mathbf X = \{X(t):t\in T\}$ 满足:   

- 零均值: 对于任意 $t\in T$ 都有 $\mu_{\mathbf X}(t) = 0$  
- 不相关性: 对于任意 $s\neq t\in T$ 都有 $\text{Cov}_{\mathbf X}(s,t)= r_{\mathbf X}(s,t) = 0$ 

则称 $\mathbf X$ 为**白噪声** (white noise).



### 2.1.5 随机过程的存在性

**(随机过程的存在性, Kolmogorov 扩展定理)**  
若分布函数族 $\mathcal F=\{F_{t_1,\dots,t_k}:k\geq 1, t_1,\dots,t_k \in \mathbb R\}$  是**一致的** (consistent)，  
即对于任意 $\begin{cases}
k\geq 1\\
t_1,\dots,t_k\in\mathbb R\end{cases}$ 满足: 

- ① 对于 $\{1,\dots,k\}$ 的任意排列 $\pi = [\pi_1,\dots,\pi_k]$ 和任意 $x_1,\dots,x_k\in \mathbb R$    
  都有 $F_{t_1,\dots,t_k}(x_1,\dots,x_k) = F_{t_{\pi_1},\dots,t_{\pi_k}}(x_{\pi_1},\dots,x_{\pi_k})$ 成立.    
  (即要求分布函数对时间点的任意排列是不变的，这反映了随机过程的时间点是可交换的)
- ② 对于任意 $x_1,\dots,x_k\in \mathbb R$ 和任意 $t_{k+1}\in\mathbb R$  
  都有 $\underset{x_{k+1}\to \infty}{\lim} F_{t_1,\dots,t_k,t_{k+1}}(x_1,\dots,x_k,x_{k+1}) = F_{t_1,\dots,t_k}(x_1,\dots,x_k)$ 成立.  
  (即要求当任意一个变量趋向于无穷大时，$k+1$ 维分布函数趋向于 $k$ 维分布函数.  
  这个性质保证了分布函数族在维度上的相容性，  
  即通过增加变量并让它趋向于无穷大，不会改变已有变量的联合分布)

则一定存在一个**概率空间** $(\Omega,\mathcal E,\text{P})$ 以及一个**随机过程** $\mathbf X=\{X(t):t\in \mathbb R\}$ 使得:    
对任意 $\begin{cases}
k\geq 1\\
t_1,\dots,t_k\in \mathbb R\\
x_1,\dots,x_k\in \mathbb R\end{cases}$ 都满足 $\text{P}\{X(t_1)\leq x_1,\dots,X(t_k)\leq x_k\}= F_{t_1,\dots,t_k}(x_1,\dots,x_k)$ 成立.

- **(独立随机过程的存在性)**  
  若分布函数族 $\mathcal F=\{F_{t_1,\dots,t_k}:k\geq 1, t_1,\dots,t_k \in \mathbb R\}$ 满足:   
  对于任意 $\begin{cases}
  k\geq 1\\
  t_1,\dots,t_k\in\mathbb R\\
  x_1,\dots,x_k\in\mathbb R\end{cases}$ 都有 $F_{t_1,\dots,t_k}(x_1,\dots,x_k) = \underset{i=1}{\overset{k}\prod}F_{t_i}(x_i)$ 成立，  
  (此条件下，上述命题的**一致性条件**显然成立)  
  则一定存在一个**概率空间** $(\Omega,\mathcal E,\text{P})$ 以及一个**独立随机过程** $\mathbf X=\{X(t):t\in \mathbb R\}$ 使得:   
  对任意 $\begin{cases}
  t\in \mathbb R\\
  x\in \mathbb R\end{cases}$ 都有 $\text{P}\{X(t)\leq x\} = F_t(x)$ 成立.

  这个推论表明:   
  如果一族分布函数在任意有限维度上的联合分布函数可以分解为各个维度对应分布函数的乘积，  
  那么这族分布函数定义了一个独立随机过程.  



## 2.2 一些例子

### 2.2.1 队列模型 (List Model)

考虑一个队列 (有序的列表) $e_1,\dots,e_n$  
在每个单位时间中，对于其中一个元素 $e_i$ 有需求的概率 $P_i$ 独立于过去的情形 (即队列目前的排列情况).  
在元素 $e_i$ 被需求后，它就移至队列的第一个位置.    
例如一摞书本，在每个单位时间随机抽取一本书，然后放回这摞书的最上面.  
我们希望知道，上述过程长时间运作后，**被需求元素的位置的期望**是多少.  

- **解答: **
  对选取的元素取条件，可得:   
  $\begin{align}
  \text{E}[被需求的元素的位置]
  &= \sum_{i=1}^n\text{E}[被需求的元素的位置\mid 选取到\ e_i\  ]\cdot P_i\\
  &= \sum_{i=1}^n\text{E}[e_i\ 的位置\mid 选取到\ e_i]\cdot P_i\quad (e_i\ 的位置与\ e_i\ 被需求这个事件相互独立)\\ 
  &= \sum_{i=1}^n\text{E}[e_i\ 的位置]\cdot P_i\\
  \end{align}$   
  对于任意给定的 $i=1,\dots,n$，记 $I_j = \begin{cases}
  1,&若\ e_j\ 在\ e_i\ 前面\\
  0,&\text{otherwise}\end{cases}$   
  则 $e_i\ 的位置 = 1 + \underset{j\neq i}\sum I_j$   
  因此 $\text{E}[e_i\ 的位置] = 1 + \underset{j\neq i}{\sum} \text{E}[I_j] = 1 + \underset{j\neq i}{\sum} \text{P}[e_j\ 在\ e_i\ 前面]$   
  注意到: 对于任意给定的 $i\neq j= 1,\dots,n$，  
  若 $e_i,e_j$ 两者中最近需求的是 $e_j$，则 $e_j$ 在 $e_i$ 前面，易知 $\text{P}\{e_j\ 在\ e_i\ 前面\} = \frac{P_j}{P_i+P_j}$    
  综上所述，我们有:   
  $\begin{align}
  \text{E}[被需求的元素的位置]
  &= \sum_{i=1}^n\text{E}[e_i\ 的位置]\cdot P_i\\
  &= \sum_{i=1}^n\{1 + \underset{j\neq i}{\sum} \text{P}[e_j\ 在\ e_i\ 前面]\}\cdot P_i\\
  &= \sum_{i=1}^n\{1+\underset{j\neq i}{\sum}\frac{P_j}{P_i+P_j} \}\cdot P_i\\
  &=1 + \sum_{i=1}^n\{P_i\underset{j\neq i}{\sum}\frac{P_j}{P_i+P_j}\}\end{align}$



### 2.2.2 随机游走

 <img src="Ross Figure 2.3.png" alt="Ross Figure 2.3" style="zoom:50%;" />
考虑一个质点沿以 $0,1,\dots,m$ 标识的 $m+1$ 个顶点的一个集合上移动，  
每一步等可能地沿顺时针方向或逆时针方向移动一个位置.  
记 $X_n$ 为质点在第 $n$ 步后的位置，  
则有 $\text{P}\{X_{n+1}=i+1|X_n=i\}= \text{P}\{X_{n+1}=i-1|X_n=i\} = \frac12$ (这称为**转移概率**)   
其中当 $i=m$ 时 $i+1\equiv 0$，而当 $i=0$ 时 $i-1\equiv m$    

假设质点从顶点 $0$ 出发，不断进行上述过程，直到遍历了所有顶点. 
问顶点 $i\ (i=1,\dots,m)$ 是最后访问的概率为多少？   

- **解答: **  
  不失一般性地，假设质点已经首次到达顶点 $i-1$，而顶点 $i,i+1$ 都还没访问到.   
  易知，顶点 $i$ 是最后访问到的当且仅当 $i+1$ 在 $i$ 之前访问到.     
  这意味着，质点必须从顶点 $i-1$ 沿逆时针方向净前进 $m-1$ 步，(途中净前进步数不能为负值)   
  经过其余未访问的顶点，最后从顶点 $i+1$ 到达 $i$.   
  由于状态空间的对称性和随机游走的性质，  
  我们可以推断出顶点 $i$ 是最后访问的概率对于所有 $i=1,\dots,m$ 都是相同的， 
  于是我们得到 $\text{P}\{i \ 是最后访问的顶点\} = \frac1m\quad(i=1,\dots,m)$ 
- **让我们从另一视角描述这个问题: **  
  考虑一个开始时拥有一个单位的赌徒，  
  当每次赌博等可能地赢或输时 (每次赢、输一个单位)，  
  在他破产 (净亏损 $1$ 单位) 之前的某时刻财富净增加到 $m-1$ 个单位的概率.  
  上面的解答表明，这个概率是 $\frac1m$   
  于是 $\text{P}\{赌徒在净增加到\ m-1\ 个单位之前的某时刻实现净亏损\ 1\ 单位\} = \frac{m-1}m$   
  或者等价地，$\text{P}\{赌徒在净减少到\ m-1\ 个单位之前的某时刻实现净盈利\ 1\ 单位\} = \frac{m-1}{m}$  
  即 $\text{P}\{赌徒在净减少到\ n\ 个单位之前的某时刻实现净盈利\ 1\ 单位\} = \frac{n}{n+1}$
- **推广形式: **  
  假设现在我们考虑赌徒在净减少到 $n$ 个单位之前的某时刻实现净盈利 $2$ 单位的概率.  
  取条件于他在净减少到 $n$ 个单位之前是否有某时刻实现了净盈利 $1$ 单位，可得:   
  $\begin{align}
  &\text{P}\{净减少到\ n\ 个单位之前的某时刻实现净盈利\ 2\ 单位\}\\
  &= \text{P}\{净减少到\ n\ 个单位之前的某时刻实现净盈利\ 2\ 单位\mid 净减少到\ n\ 个单位之前的某时刻实现净盈利\ 1\ 单位\}\cdot \frac{n}{n+1}\\
  &= \text{P}\{净减少到\ n+1\ 个单位之前的某时刻实现净盈利\ 1\ 单位\}\frac{n}{n+1}\\
  &= \frac{n+1}{n+2}\frac{n}{n+1}\\
  &= \frac{n}{n+2}\end{align}$  
  重复上述推理可知:   
  $\text{P}\{净减少到\ n\ 个单位之前的某时刻实现净盈利\ k\ 单位\} = \frac{n}{n+k}$ 



### 2.2.3 随机图 (Random Graph)

一个**无向图** (undirected graph) 由**顶点集** (vertex set) $V= \{1,2,\dots,n\}$ 和**弧集** (arc set) $A \subseteq V\times V$ 构成.  
(弧集 $A$ 中的弧是形如 $(i,j)$ 的无序对，其中 $i,j$ 是顶点集 $V$ 中的顶点)  
我们说顶点 $i\neq j$ 之间**存在一条路径**，指的是存在一列顶点 $i,i_1,\dots,i_k,j$ 使 $(i,i_1),\dots,(i_k,j)$ 都是弧.  
若 $V$ 中 $\binom{n}{2}$ 对不同顶点中的每一对都存在一条路径，则称这个无向图是**连通的** (connected)

现在考虑这样一个的无向图: $\begin{cases}
V=\{1,2,\dots,n\}\\
A=\{(i,X_i),i=1,2,\dots,n\}\subseteq V\times V\end{cases}$   
其中 $\{X_i\}_{i=1}^n$ 是 $n$ 个相互独立的随机变量，  
对于任意 $\begin{cases}
i=1,\dots,n\\
j=1,\dots,n\end{cases}$ 都有 $\text{P}\{X_i=j\} = \frac1n$ 成立.  
我们称这种图为**随机图** (random graph)，我们希望确定**随机图是连通图的概率**.  

**(待补充)**

**The End**
