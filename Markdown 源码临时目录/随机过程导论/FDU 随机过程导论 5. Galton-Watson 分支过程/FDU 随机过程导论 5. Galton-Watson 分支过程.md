# FDU 随机过程导论 5. Galton-Watson 分支过程

本文根据王老师课堂笔记整理而成，并参考了以下教材：

  - 随机过程 (苏中根) 第 $5$ 章
  - Introduction to Probability Models: Applied Stochastic Processes (S. Ross) 
  - 应用随机过程概率论模型导论 (S. Ross) 龚光鲁译 第 $4.7$ 节

欢迎批评指正！

## 5.1 Introduction

考察一个总体，它由能产生同类型后代的个体构成.  
假设每个个体在其生命结束时，以概率 $P_i(i\geq 0)$ 产生 $i$ 个后代，且独立于其他个体产生的后代个数.  
假设对于一切 $i\geq0$ 都有 $P_i<1$ 

记最初的个体数为 $X_0$，记第 $n$ 代的个体数为 $X_n$，  
显然**分支过程** $\{X_n:n\geq 0\}$ 是以 $\N$ 为状态空间的 **Markov 链**.  

- 我们有**递推关系** $X_n = \underset{i=1}{\overset{X_{n-1}}\sum}\xi_i$  
  其中 $\xi_i$ 表示第 $n-1$ 代中的第 $i$ 个个体产生的后代数.  
  因此对于任意 $i\geq 1$，向状态 $j$ 的**转移概率** $P_{ij} = \text{P}\{X_n=j|X_{n-1}=i\} = \text{P}\{\underset{k=1}{\overset{i}\sum}\xi_k = j\}$   
  一般情况下，很难计算出转移概率 $P_{ij}$ 

显然状态 $0$ 是**吸收态** (自然是正常返态)，有 $P_{00}=1$ 成立.  
此外，若个体产生 $0$ 个后代的概率 $P_0>0$，则除 $0$ 以外的其他状态都是**瞬时态**.  
这是因为：对于任意状态 $i>0$，它的所有个体都不产生后代的概率 $P_{i0} = (P_0)^i>0$ 是正值.  
此时，由于瞬时态的任意有限集 $\{1,2,\dots,n\}$ 都只能有限次地被访问，  
这就导出一个重要的结论：  

- **若个体产生 $0$ 个后代的概率 $P_0>0$，则该总体或者灭绝，或者趋于无穷.**

我们记单个个体产生后代数的均值和方差为 $\begin{cases}
\mu = \text{E}[\xi] = \underset{i=0}{\overset{\infty}\sum} i P_i\\
\sigma^2 = \text{Var}[\xi] = \underset{i=0}{\overset{\infty}\sum} (i-\mu)^2 P_i\end{cases}$  

假设 $X_0 = 1$，我们尝试计算第 $n$ 代个体数的期望 $\text{E}[X_n]$ 和方差 $\text{Var}[X_n]$：  

- 取条件于 $X_{n-1}$，我们得到期望：  
  $\begin{align}
  \text{E}[X_n]
  &= \text{E}[\text{E}[X_n|X_{n-1}]]\\
  &= \text{E}[\text{E}[ \underset{i=1}{\overset{X_{n-1}}\sum}\xi_i|X_{n-1}]]\\
  &= \text{E}[X_{n-1}\text{E}[\xi]]\\
  &= \text{E}[X_{n-1}\mu]\\
  &= \mu \text{E}[X_{n-1}]\end{align}$ 

  根据 $X_0 = 1$ (自然有 $\text{E}[X_0]=1$) 可得 $\text{E}[X_n]=\mu^n$   
  它表明：

  - 若 $\mu>1$，则 $\text{E}[X_n]=\mu^n\to \infty\ \ (n\to \infty)$ (几何级数增长)
  - 若 $\mu= 1$，则 $\text{E}[X_n]=\mu^n\equiv 1$ 
  - 若 $0<\mu<1$，则 $\text{E}[X_n]=\mu^n\to 0\ \ (n\to \infty)$ (几何级数衰减)

- 类似地，取条件于 $X_{n-1}$，我们得到方差：  
  $\begin{align}
  \text{Var}[X_n]
  &= \text{E}[\text{Var}(X_n|X_{n-1})] + \text{Var}(\text{E}[X_n|X_{n-1}])\\
  &= \text{E}[\text{Var}(\underset{i=1}{\overset{X_{n-1}}\sum}\xi_i|X_{n-1})] + \text{Var}(\text{E}[\underset{i=1}{\overset{X_{n-1}}\sum}\xi_i|X_{n-1}])\\
  &= \text{E}[X_{n-1}\sigma^2] + \text{Var}(X_{n-1}\mu)\\
  &= \sigma^2 \text{E}[X_{n-1}] + \mu^2 \text{Var}[X_{n-1}]\\
  &= \sigma^2\mu^{n-1} + \mu^2(\sigma^2\mu^{n-2} +\mu^2\text{Var}[X_{n-2}])\\
  &= \sigma^2(\mu^{n-1} + \mu^n) + \mu^4 \text{Var}[X_{n-2}]\\
  &=\dotsm\\
  &= \sigma^2(\mu^{n-1} + \mu^n + \dots + \mu^{2n-2})
  +\mu^{2n}\text{Var}[X_0]\\
  &=\sigma^2(\mu^{n-1} + \mu^n + \dots + \mu^{2n-2})
  +\mu^{2n}\cdot 0\\
  &= \sigma^2(\mu^{n-1} + \mu^n + \dots + \mu^{2n-2})\\
  &= \begin{cases}
  n\sigma^2 & \text{if }\mu = 1\\
  \sigma^2\mu^{n-1}(\frac{1-\mu^n}{1-\mu})
  &\text{if }\mu\neq 1\end{cases}\end{align}$

在假定 $X_0 = 1$ 的前提下，我们定义总体最终灭绝的概率为：  
$\tau =\underset{n\to\infty}{\lim} \text{P}\{X_n=0|X_0=1\}$ (可以简记为 $\underset{n\to\infty}{\lim} \text{P}\{X_n=0\}$)

- 我们首先注意如果单个个体产生后代数的均值 $\mu<1$，则灭绝概率 $\tau =1$  
  这是因为 (下面我们省略 $X_0=1$ 的条件, 单纯简化记号)：   
  $\begin{align}
  \text{P}\{X_n\geq 1\}
  &= 
  \sum_{i=1}^\infty \text{P}\{X_n = i\}\\
  &\leq 
  \sum_{i=1}^\infty i\cdot\text{P}\{X_n = i\}\\
  &= 
  \text{E}[X_n]\\
  &=
  \mu^n\to 0\ \ (n\to\infty)\end{align}$   
  表明灭绝概率 $\tau = \underset{n\to\infty}{\lim} \text{P}\{X_n=0\}=1$
- 事实上，即使单个个体产生后代数的均值 $\mu=1$，我们也可以证明灭绝概率 $\tau =1$ 
- 现在考虑 $\mu>1$ 的情况：  
  给定 $X_1 = i$，总体最终灭绝当且仅当从第一代成员开始的 $i$ 个家族都最终灭绝.  
  由于这 $i$ 个家族的分支过程相互独立，灭绝概率均为 $\tau$  
  因此我们有：  
  $\begin{align}
  \tau 
  &=
  \text{P}\{\text{extinction}\}\\
  &=
  \sum_{i=0}^\infty \text{P}\{\text{extinction}|X_1=i\}\text{P}\{X_1=i\}\\
  &=
  \sum_{i=0}^\infty (\tau)^i\cdot P_i\end{align}$   
  从而灭绝概率 $\tau$ 满足方程 $\tau = \underset{i=0}{\overset{\infty}\sum}\tau^iP_i$ (我们可以看出这个方程有一个平凡解 $\tau = 1$)  
  事实上，我们可以证明 $\tau$ 是上述方程的最小正解.

***

**(S. Ross 例 4.32)**  
设 $\begin{cases}
P_0 = \frac12\\
P_1 = \frac14\\
P_2 = \frac14\end{cases}$，可计算得单个个体产生后代数的均值 $\mu = 0\cdot \frac12 + 1\cdot \frac14 + 2\cdot \frac14 = \frac{3}{4}$   
由于 $\mu = \frac34\leq 1$，故灭绝概率 $\tau = 1$

**(S. Ross 例 4.33)**  
设 $\begin{cases}
P_0 = \frac14\\
P_1 = \frac14\\
P_2 = \frac12\end{cases}$，可计算得单个个体产生后代数的均值 $\mu = 0\cdot \frac14 + 1\cdot \frac14 + 2\cdot \frac12 = \frac{5}{4}$   
由于 $\mu = \frac54> 1$ 且 $P_0 = \frac14$，故可能灭绝也可能无穷繁衍下去.  

我们知道 $\tau$ 满足方程 $\tau = \tau^0 \cdot \frac14 + \tau^1 \cdot \frac14 + \tau^2\cdot \frac12$   
即 $2\tau^2 - 3\tau + 1=0$   
解得 $\tau = \frac12$ 或 $\tau = 1$ (平凡解)   
我们取最小正解 $\tau = \frac12$，表明这个分支过程的灭绝概率为 $\tau = \frac12$

**(S. Ross 例 4.34)**  
对于上述两个例子，若 $X_0 = m$ (即第 $0$ 代由 $m$ 个个体构成)  
则总体灭绝的概率是 $\tau^m$，  
对于**例 4.32 **是 $\tau^m = 1^m = 1$，对**例 4.33** 是 $\tau^m = (\frac12)^m=\frac{1}{2^m}$ 



## 5.2 生成函数

假设 $\xi$ 为非负整数值随机变量，概率分布由 $P_k = \text{P}\{\xi=k\}\ (k\in \N)$ 给定.  
对 $0\leq z\leq 1$ 定义 $G(z) = \text{E}[z^\xi]  = \underset{k=0}{\overset{\infty}\sum} z^kP_k$  
我们称 $G(z)$ 为随机变量 $\xi$ 的**生成函数**.  

- 等价地，我们也可以取 $z= e^{-t}\ (t\geq 0)$，  
  以 $G(t) = \text{E}[e^{-t\xi}]  = \underset{k=0}{\overset{\infty}\sum} e^{-kt}P_k$ 为随机变量 $\xi$ 的**生成函数**.

生成函数具有很多良好的性质：  

- $\begin{cases}
  G(0)=0\\
  G(1)=1
  \end{cases}$ 且 $0\leq G(z)\leq 1\ (\forall\ 0\leq z\leq 1)$ 
- $G(z)$ 在 $[0,1]$ 上一致连续，且是严格单调递增的凸函数
- $G(z)$ 在 $z=0$ 处无穷次可微，并且概率 $P_k = \text{P}\{\xi=k\}= \frac{1}{k!}G^{(k)}(0)$ 
- 若 $\xi$ 的 $k$ 阶矩 $\text{E}[\xi^k]<\infty$，则 $G(z)$ 在 $[0,1]$ 上 $k$ 次可微.  
  特别地，当 $\text{E}[\xi^2]<\infty$ 时，我们有 $\begin{cases}
  G'(1) = \text{E}[\xi]\\
  G''(1) = \text{E}[\xi^2] - \text{E}[\xi]\end{cases}$ 
- 假设 $\xi,\eta$ 是两个相互独立的非负整数值随机变量，  
  那么 $\xi+\eta$ 的生成函数等于各自生成函数的乘积，即 $G_{\xi+\eta}(z) = G_\xi(z)G_\eta(z)$   
  这个性质可以推广至任意有限个独立随机变量的和.

下面讨论分支过程 $\mathbf X=\{X_n:n\geq 0\}$ 的生成函数：  
记 $X_n$ 的生成函数为 $G_n(z)$ 

- $X_1$ 与 $\xi$ 同分布：$G_1(z) = G(z)$
- 考虑 $n\geq 2$ 的情况：  
  $\begin{align}
  G_n(z)
  &= \text{E}[z^{X_n}]\\
  &= \text{E}[z^{\underset{i=1}{\overset{X_{n-1}}\sum}\xi_i}]\\
  &= \text{E}[\text{E}[z^{\underset{i=1}{\overset{X_{n-1}}\sum}\xi_i}|X_{n-1}]]\\
  &= \text{E}[\prod_{i=1}^{X_{n-1}}\text{E}[z^{\xi_i}]]\\
  &= \text{E}[(G(z))^{X_{n-1}}]\\
  &= G_{n-1}(G(z))\end{align}$

归纳可得 $G_n(z) = G_{n-1}(G(z)) = G(G_{n-1}(z)) = \underset{n}{\underbrace{G\circ G\circ\dotsm\circ G}}(z)$ 

***

**(苏中根 例 5.3)**  
设 $0<p<1$，随机变量 $\xi \sim \text{Geo}(p) -1$，即 $\text{P}\{\xi=k\}=p(1-p)^k\ \ (k\geq 0)$   
我们计算从状态 $i\geq 1$ 到状态 $j$ 的**转移概率** $P_{ij} = P\{\underset{k=1}{\overset{i}\sum}\xi_k = j\}$：    
(一般情况下，很难计算出转移概率 $P_{ij}$)

- 随机变量 $\xi$ 的生成函数：   
  $\begin{align}
  G_\xi(z) 
  &= \text{E}[z^\xi] \\
  & = \sum_{k=0}^\infty z^k\cdot p(1-p)^k\\ 
  &= p\cdot\sum_{k=0}^\infty [z(1-p)]^k\\
  &= p\cdot \frac{1}{1-z(1-p)}\end{align}$

- 随机变量和 $S_i = \underset{k=1}{\overset{i}\sum}\xi_k$ 的生成函数：  
  $\begin{align}
  G_{S_i}(z) 
  &= \text{E}[z^{S_i}]\\
  &= \text{E}[z^{\underset{k=1}{\overset{i}\sum}\xi_k}]\\
  &= \prod_{k=1}^i \text{E}[z^{\xi_k}]\\
  &= (G_\xi(z))^i\\
  &= p^i\cdot (\frac{1}{1-z(1-p)})^i\\
  &= p^i\cdot \sum_{j=0}^\infty \binom{i+j-1}{i-1}[z(1-p)]^j\end{align}$   

  从而我们知道 $P_{ij} = P\{\underset{k=1}{\overset{i}\sum}\xi_k = j\}=\text{P}\{S_i = j\} = \binom{i+j-1}{i-1}p^i(1-p)^j$ 

****

**(王勤文补充示例, 苏中根 例 5.5)**  
设随机变量 $\xi$ 的分布由 $P_k = \text{P}\{\xi=k\}=\frac{1}{2^{k+1}}\ \ (k\geq 0)$ 确定.  
(也就是说 $\xi \sim \text{Geo}(\frac12) - 1$，即**苏中根 例 5.3** 取 $p=\frac12$ 的情况)

- 利用**苏中根 例 5.3** 的结论，随机变量 $\xi$ 的生成函数为：  
  $\begin{align}
  G(z) 
  &= \frac{p}{1-z(1-p)}\\
  &= \frac{\frac12}{1-z(1-\frac12)}\\
  &= \frac{1}{2-z}\end{align}$
  
- $X_1$ 的生成函数 $G_1(z) = G(z)$  
  $X_2$ 的生成函数 $G_2(z)=G_1(G(z)) = G(G(z)) = \frac{1}{2-\frac{1}{2-z}} = \frac{2-z}{3-2z}$   
  $X_3$ 的生成函数 $G_3(z)=G_2(G(z)) =\frac{2-\frac{1}{2-z}}{3-2\frac{1}{2-z}} = \frac{3-2z}{4-3z}$  
  归纳可得 $X_n$ 生成函数的通式为 $G_n(z) = \frac{n-(n-1)z}{n+1-nz}$
  
- 为避免对 $G_n(z)$ 求高阶导数，我们可以对 $G_n(z)$ 进行 Taylor 展开：  
  $\begin{align}
  G_n(z) - \frac{n}{n+1} 
  &= \frac{n-(n-1)z}{n+1-nz} - \frac{n}{n+1}\\
  &= \frac{z}{(n+1)(n+1-nz)}\\
  &= \frac{z}{(n+1)^2}\cdot \frac{1}{1-\frac{n}{n+1}z}\end{align}$   
  
  我们知道，当 $|x|<1$ 时，函数 $\frac{1}{1-x}$ 的 Taylor 展开式为 $\underset{k=0}{\overset{\infty}\sum}x^k$   
  因此我们有 $\frac{1}{1-\frac{n}{n+1}z} = \underset{k=0}{\overset{\infty}\sum} (\frac{n}{n+1}z)^k$     
  于是我们有：  
  $\begin{align}
  G_n(z) 
  &= \frac{n}{n+1} + \frac{z}{(n+1)^2}\cdot \frac{1}{1-\frac{n}{n+1}z}\\
  &= \frac{n}{n+1} + \frac{z}{(n+1)^2}\underset{k=0}{\overset{\infty}\sum} (\frac{n}{n+1}z)^k \\
  &=  \frac{n}{n+1} + \frac{1}{(n+1)^2}\sum_{k=0}^\infty (\frac{n}{n+1})^k z^{k+1}\\
  &= \frac{n}{n+1} + \frac{1}{(n+1)^2}\sum_{k=1}^\infty (\frac{n}{n+1})^{k-1} z^{k}\end{align}$   
  
  因此我们有 $\begin{cases}
  \text{P}\{X_n = 0\} = G_n(0) = \frac{n}{n+1}\\
  \text{P}\{X_n = k\} = \frac1{k!}G_n^{(k)}(0) = \frac{n^{k-1}}{(n+1)^{k+1}}\ \ (k\geq 1)\end{cases}$



## 5.3 灭绝概率

我们现在重新审视灭绝概率：  
在假定 $X_0 = 1$ 的前提下，我们定义总体最终灭绝的概率为：  
$\tau =\underset{n\to\infty}{\lim} \text{P}\{X_n=0|X_0=1\}$ (可以简记为 $\underset{n\to\infty}{\lim} \text{P}\{X_n=0\}$)

利用 5.2 节生成函数的结论，我们有 $\text{P}\{X_n = 0\}=G_n(0)$ 成立.  
记 $\alpha_n = \text{P}\{X_n = 0\}$ 即得：  
$\alpha_n =\text{P}\{X_n = 0\}=G_n(0) = G(G_{n-1}(0)) = G(\alpha_{n-1})$   
令 $n\to\infty$ 可得：  
$\tau = \underset{n\to\infty}{\lim} \alpha_n = \underset{n\to\infty}{\lim} G(\alpha_{n-1}) = G(\underset{n\to\infty}{\lim} \alpha_{n-1}) = G(\tau)$   
(其中倒数第二步使用了生成函数 $G(\cdot)$ 的连续性)

**定理 5.3.1：(苏中根 定理 5.3)**  
在假定第 $0$ 代个体数 $X_0 = 1$ 且单个个体产生 $0$ 个后代的概率 $P_0>0$ 的前提下，  
设单个个体产生后代数 $\xi$ 的均值是 $\mu=\text{E}[\xi]$   
设 $\xi$ 的生成函数为 $G(z) = \text{E}[z^\xi]\ (z\in [0,1])$  
定义灭绝概率为 $\tau =\underset{n\to\infty}{\lim} \text{P}\{X_n=0|X_0=1\}$ (可以简记为 $\underset{n\to\infty}{\lim} \text{P}\{X_n=0\}$)   

- 若 $0<\mu\leq 1$，则 $\tau = 1$ 
- 若 $\mu>1$，则 $0<\tau<1$ 为方程 $\tau = G(\tau)$

**证明：**

- $0<\mu<1$ 的情况我们已经在 5.1 节证明过了.
- 若 $\mu=1$，则根据生成函数的性质 (参考 5.2 节) 可知 $G'(1) = \mu=1$   
  又由于生成函数在 $[0,1]$ 上是严格单调递增的凸函数，  
  故我们可以绘制 $y=G(z)$ 和 $y=z$ 的图像，  
  发现方程 $z = G(z)$ 仅有 $z = 1$ 一个解，  
  因此灭绝概率 $\tau = 1$  
  <img src="SZG 图 5.2.jpeg" style="zoom:50%;" />
- 若 $\mu>1$，类似地我们有 $G'(1) = \mu>1$   
  而我们又知道 $G(0) = P_0 >0$  
  因此我们知道 $y=G(z)$ 和 $y=z$ 的图像在 $(0,1)$ 上必有一个交点.  
  这个交点就是我们要求的灭绝概率 $\tau \in (0,1)$  
  <img src="SZG 图 5.2(2).jpeg" style="zoom:50%;" />

***

**(苏中根 例 5.7-续例 5.3, 5.5)**  
设 $0<p<1$，随机变量 $\xi \sim \text{Geo}(p) -1$，即 $\text{P}\{\xi=k\}=p(1-p)^k\ \ (k\geq 0)$   
我们知道 $\mu = \frac{1}{p}-1$   
根据**定理 5.3.1** 可以知道：

- 当 $0<\mu \leq 1$，即 $\frac12\leq p<1$ 时，灭绝概率 $\tau = 1$
- 当 $\mu>1$，即 $0<p<\frac12$ 时，灭绝概率 $\tau \in (0,1)$ 且是方程 $\tau = G(\tau)$ 的最小正解.  
  根据**苏中根 例 5.3** 可知 $\xi$ 的生成函数为 $G(z) = \text{E}[z^\xi] = \frac{p}{1-z(1-p)}$   
  求解方程 $\tau = G(\tau)=\frac{p}{1-\tau (1-p)}$ 解得 $\tau = \frac{p}{1-p}\in (0,1)$ 或 $\tau =1$ (平凡解)  
  因此灭绝概率为 $\tau = \frac{p}{1-p}\in (0,1)$
