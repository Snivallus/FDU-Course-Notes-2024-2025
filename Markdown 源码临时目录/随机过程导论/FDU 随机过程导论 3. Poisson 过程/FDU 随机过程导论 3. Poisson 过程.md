# FDU 随机过程导论 3. Poisson 过程

本文根据王老师课堂笔记整理而成，并参考了以下教材: 

  - 随机过程 (苏中根) 第 $3$ 章
  - Introduction to Probability Models: Applied Stochastic Processes (S. Ross) 
  - 应用随机过程概率论模型导论 (S. Ross) 龚光鲁译 第 $5$ 章

欢迎批评指正!

## 3.1 指数分布 (The Exponential Distribution)

在实际问题的建模中，我们常做的一个简化假设是假定某些随机变量是服从指数分布的.  
这是因为指数分布不仅处理起来相对容易，而且它常常是实际分布的一个良好的近似.  
我们将说明指数分布具有**无记忆性 **(Memoryless Property):   
它的行为不随时间而改变，  
即如果一个零件的寿命服从指数分布，  
那么不论该零件已经使用了多长时间，其预期的剩余使用寿命仍与全新零件一样长.  
(也就是说，这个零件并不记住它已经使用了多长时间)  
我们将证明**指数分布是具有无记忆性的唯一的连续型分布**.   
(另一种具有无记忆性的分布是几何分布 $\text{Geo}(p)$，它是离散型分布)  
我们随后会研究一类称为 Poisson  过程的计数过程，并揭示它与指数分布的紧密联系.

### 3.1.1 指数分布的定义

**定义3.1: (指数分布)**   
我们称一个连续随机变量 $X$ 服从具有参数 $\lambda\ (\lambda>0)$ 的**指数分布** $\exp(\lambda)$，  
当且仅当它的**概率密度函数**为 $f(x)=\begin{cases}
\lambda e^{-\lambda x}, &\text{if } x\geq 0\\
0,&\text{if }x<0\end{cases}$    
或者等价地说，它的**累计分布函数**为 $F(x) = \int_{-\infty}^{x}f(t)\mathrm{d}t = \begin{cases}
1-e^{-\lambda x},&\text{if }x\geq 0\\
0, &\text{if }x<0\end{cases}$  
或者等价地说，它的**矩母函数**为 $M(t) = \text{E}[e^{tX}] = \frac{\lambda}{\lambda-t}\ \  (t<\lambda)$  
或者等价地说，它的**特征函数**为 $\phi(t) = \text{E}[e^{itX}] = \frac{\lambda}{\lambda-it}\ \ (t\in \mathbb R)$ 

**定理3.1: (指数随机变量的任意阶矩)**
我们可以利用矩母函数计算 $X\sim\exp(\lambda)$ 的任意阶矩:   
对于任意 $k\geq 1$ 都有 $\text{E}[X^k]=\text{E}[X^k e^{0\cdot X}] = M_X^{(k)}(0) = \lambda\cdot \frac{k!}{(\lambda-t)^{k+1}}|_{t=0} = \frac{k!}{\lambda^k}$ 成立    
因此 $\begin{cases}
\text{E}[X]= \frac{1}{\lambda}\\
\text{E}[X^2]= \frac{2}{\lambda}\end{cases}\Rightarrow \ \text{Var}[X]=\text{E}[X^2]-(\text{E}[X])^2 = \frac{1}{\lambda^2}$ 

- 一个具体的例子:   
  设我们连续地以随机变化的速率接受报酬，记 $t$ 时刻的速率为随机变量 $R(t)$  
  定义**折扣后的总报酬**为 $R=\int_0^{\infty} e^{-\lambda t} R(t)\mathrm{d}t$，其中**折扣率** $\lambda\geq 0$  
  则**折扣后的平均总报酬**为 $\text{E}[R]= \text{E}[\int_0^{\infty} e^{-\lambda t} R(t)\mathrm{d}t]=\int_0^{\infty} e^{-\lambda t} \text{E}[R(t)]\mathrm{d}t$   
  定义表示时间的随机变量 $T\sim \exp(\lambda)$，满足 $T\ \bot \ R(t)\ \ (\forall\ t\geq 0)$    
  对于任意 $t \geq 0$，定义 $I_{[0,T]}(t)=\begin{cases}
  1, &\text{if }\ 0 \leq t\leq T \\
  0, &\text{if }\ t > T \end{cases}$ 

  我们希望证明 $\text{E}[R]=\int_0^{\infty} e^{-\lambda t} \text{E}[R(t)]\mathrm{d}t= \text{E}[\int_{0}^{T}R(t)\mathrm{d}t]$:      
  $\begin{align}
  \text{E}[\int_0^{T}R(t)\mathrm{d}t]
  &= \text{E}[\int_0^{\infty}R(t)I_{[0,T]}(t)\mathrm{d}t]\\
  &= \int_0^{\infty}\text{E}[R(t)I_{[0,T]}(t)]\mathrm{d}t\quad(T\bot\ R(t)\ \text{for all }t\geq 0)\\
  &= \int_0^{\infty}\text{E}[R(t)]\cdot \text{E}[I_{[0,T]}(t)]\mathrm{d}t\\
  &= \int_0^{\infty}\text{E}[R(t)]\text{P}\{T\geq t\}\mathrm{d}t\\
  &= \int_0^{\infty}[1-(1-e^{-\lambda t})]\text{E}[R(t)]\mathrm{d}t\\
  &= \int_0^{\infty}e^{-\lambda t}\text{E}[R(t)]\mathrm{d}t\end{align}$   
  得证 $\text{E}[R]= \text{E}[\int_0^{\infty} e^{-\lambda t} R(t)\mathrm{d}t]=\int_0^{\infty} e^{-\lambda t} \text{E}[R(t)]\mathrm{d}t= \text{E}[\int_{0}^{T}R(t)\mathrm{d}t]$   
  表明**折扣率**为 $\lambda$ 的平均总报酬 $=$ 服从 $\exp(\lambda)$分布的随机时间 $T$ 里**无折扣**的平均总报酬.



### 3.1.2 指数分布的性质

#### (1) 无记忆性

我们称随机变量 $X$ 具有**无记忆性 **(memoryless property)，  
当且仅当对于一切 $s,t\geq 0$ 都有 $\text{P}\{X>s+t|X>t\} = \text{P}\{X>s\}$ 成立，  
或者等价地，都有 $\frac{\text{P}\{X>s+t,X>t\}}{\text{P}\{X>t\}} =\frac{\text{P}\{X>s+t\}}{\text{P}\{X>t\}}= \text{P}\{X>s\}$ 成立，  
或者等价地，都有 $\text{P}\{X>s+t\} = \text{P}\{X>s\}\text{P}\{X>t\}$ 成立.  

**定理3.2: **
指数分布是**唯一**具有**无记忆性**的连续型分布:   
它的行为不随时间而改变，  
即如果一个零件的寿命服从指数分布，  
那么不论该零件已经使用了多长时间，其预期的剩余使用寿命仍与全新零件一样长.  
(也就是说，这个零件并不记住它已经使用了多长时间)

**证明: **  

- **① 指数分布具有无记忆性: **   
  对于任意给定的 $\lambda> 0$，设随机变量 $X\sim \exp(\lambda)$，则有:   
  $$
  \begin{align}
  \text{P}\{X>s+t\} - \text{P}\{X>s\}\text{P}\{X>t\}
  &= e^{-\lambda (s+t)} - e^{-\lambda s}\cdot e^{-\lambda t}\\
  &= 0\end{align}
  $$
  因此有 $\text{P}\{X>s+t\} = \text{P}\{X>s\}\text{P}\{X>t\}$ 成立，  
  说明指数分布 $\exp(\lambda)$ 具有无记忆性.
  
- **② 具有无记忆性的分布一定是指数分布: **  
  假定 $X$ 的分布具有无记忆性，记 $\bar F(x) = 1-F(x) = \text{P}\{X>x\}\ \ (x\geq 0)$  
  则对于任意 $s,t\geq 0$ 都有 $\text{P}\{X>s+t\} = \text{P}\{X>s\}\text{P}\{X>t\}$ 成立，  
  即 $\bar F(s+t)= \bar F(s)\cdot \bar F(t)$ 成立，  
  即函数 $\bar F(x)$ 满足函数方程 $g(s+t)=g(s)g(t)$: 
  - 显然 $g(x)$ 在 $[0,\infty)$ 上是非增函数，取值范围为 $[0,1]$ 
  - 根据 $\forall\ s,t\geq 0,\ g(s+t)=g(s)g(t)$ 可知 $g(\frac{2}{n})=g(\frac1n+\frac1n) = g^2(\frac1n)$   
    重复上述过程可得 $\forall\ m,n\geq 1,\ g(\frac{m}{n})=g^m(\frac1n)$  
    取 $m=n$ 则有 $g(1)=g^n(\frac1n)$  
    从而有 $g(\frac1n)=(g(1))^{\frac1n}$  
    因此有 $\forall\ m,n\geq 1,\ g(\frac{m}{n})=(g(1))^{\frac{m}{n}}$  
  - 根据有理数在实数域上的稠密性，我们知道:        
    对于任意实数 $x\geq 0$，都存在有理数序列 $\{x_n\}$ 满足 $\begin{cases}
    \forall\ n\geq 1,\ x_n> x\\
    \underset{n\to\infty}{\lim} x_n = x
    \end{cases}$   
    假设 $g$ 是右连续函数 (这是因为**分布函数 $F(x)$ 总是右连续的**，故 $\bar F(x)$ 也右连续)，  
    则我们有 $g(x) =\underset{n\to\infty}{\lim} g(x_n) =\underset{n\to\infty}{\lim} (g(1))^{x_n} = (g(1))^x$   
    - 若 $g(1)=0$，则 $\forall\ x>0,\ g(x)=0$，  
      根据右连续性可知 $g(0)= \underset{x\to 0_+}{\lim} g(x) = 0$   
      说明 $\text{P}\{X=0\} = 1$，$X$ 为 $0$ 处的**退化分布**，这一情况通常不在我们考虑范围内.
    - 若 $g(1)\in (0,1]$，则可取 $\lambda = -\log(g(1))\geq 0$  
      于是 $\bar F(x)=g (x) = e^{-\lambda x}\ \ (x\geq 0)$     
      说明 $F(x) = 1-e^{-\lambda x}\ \ (x>0)$，即 $X$ 服从**指数分布** $\exp(\lambda)$   

综上所述，命题得证.



## 3.2 Poisson 过程

### 3.2.1 计数过程 (Counting Processes)

**定义3.2: (计数过程)**
若 $N(t)$ 表示到时刻 $t\geq 0$ 为止发生的事件总数，则称随机过程 $\{N(t):t\geq 0\}$ 为**计数过程**.  
从定义我们看到，一个计数过程 $\{N(t):t\geq 0\}$ 必须满足:   

- 对于任意 $t\geq 0$，$N(t)$ 都取非负整数值；
- 对于任意 $0<s<t$ 都有 $N(s) \leq N(t)$ 成立；
- 对于任意 $0<s<t$，增量 $N(t)-N(s)$ 都表示区间 $(s,t]$ 中发生的事件数；

为方便起见，记 $\begin{cases}
N(t)= N(0,t]\\
N(s,t) = N(t)-N(s)\end{cases}$ 

### 3.2.2 Poisson 过程的定义

考虑某服务系统为顾客提供服务: 

- 该系统服务设施、服务内容是确定的，不具有随机性.   
- 每位顾客是否需要服务是随机的，并且顾客之间的到达是相互独立的；
- 在不同时间段内寻求服务的顾客量基本相同，意味着顾客到达遵循一个恒定的平均速率；
- 系统提供服务不占用时间 (非占用性)，但不能同时为两名以上顾客提供服务 (排他性)；
- 系统遵循 “先来后到“ 原则，顾客依序排队等待服务；

上述顾客流即为一个 **Poisson 流**，   
其计数过程即为一个 **Poisson 过程**，它是最重要的计数过程之一.

**定义3.3: (高阶无穷小量)**    
我们称函数 $f(\cdot)$ 为 $x$ 的高阶无穷小量 (记为 $o(x)$)，如果它满足 $\underset{x\to 0}{\lim} \frac{f(x)}{x}=0$     
记号 $o(x)$ 的使用可以使命题的描述更加简洁.

**定理3.3: (高阶无穷小量的运算性质)**  

- 若函数 $f(\cdot),g(\cdot)$ 都是 $o(x)$，则函数 $f(\cdot)\pm g(\cdot)$ 也是 $o(x)$  
- 若函数 $f(\cdot)$ 是 $o(x)$，则对于任意常数 $c\in \mathbb R$，函数 $c f(\cdot)$ 也是 $o(x)$  
- 综合上述两点可知，若一列函数都是 $o(x)$，则它们的任意有限线性组合也是 $o(x)$ 

**定义3.4: (Poisson 过程)**
我们称计数过程 $\{N(t):t\geq 0\}$ 为 **Poisson 过程**，它满足:    

- **初始条件: **$N(0) = 0$
- **独立增量: **对于任意 $\begin{cases}
  k\geq 1\\
  0<t_1<\dotsm<t_k\end{cases}$ 都有 $N(t_1),N(t_2)-N(t_1),\dots,N(t_k)-N(t_{k-1})$ 相互独立
- **平稳增量: **对于任意 $0<s<t$ 都有 $N(t)-N(s) \overset{\mathrm{d}}= N(t-s)$  
- **稀有性: **存在常数 $\lambda > 0$ 使得对于任意 $t>0$ 都有 $\begin{cases}
  \text{P}\{N(t,t+\text{d} t)=0\} = (1-\lambda) \text{d}t + o(\text{d}t)\\
  \text{P}\{N(t,t+\text{d} t)=1\} = \lambda \text{d}t + o(\text{d}t)\\
  \text{P}\{N(t,t+\text{d}t)\geq 2\} = o(\text{d} t)\end{cases}$   
  其中 $\text{d}t$ 是一个相对小的时间增量，   
  而 $o(\text{d}t)$ 代表比 $\text{d}t$ 高阶的无穷小，即有 $\underset{\text{d}t\to 0}{\lim} \frac{o(\text{d}t)}{\text{d}t} = 0$   
  我们称常数 $\lambda>0$ 为该 Poisson 过程的**速率** (rate)   
  它刻画系统的平均繁忙程度: $\lambda$ 越大，平均客流量越大，系统越繁忙.   
  (根据后面的结论我们知道 $\lambda = \frac{\text{E}[N(t)]}{t} = \text{E}[N(1)]$)

**定理3.4 (Poisson 过程的分布规律, Introduction to Probability Models 定理5.1)**  
若 $\{N(t):t\geq 0\}$ 是一个参数为 $\lambda>0$ 的 Poisson 过程，   
则对于任意 $s,t>0$ (**存疑**: 原来写的是 $0<s<t$) 都有 $N(s+t)-N(s) \sim \text{Poisson}(\lambda t)$ 成立，
即对于任意 $t>0$ 都有 $N(t)\sim \text{Poisson}(\lambda t)$ 成立，   
即对于任意 $\begin{cases}
t>0\\
k \in \mathbb N \end{cases}$ 都有 $\text{P}\{N(t) = k\} = \frac{(\lambda t)^k}{k!}e^{-\lambda t}$  成立.     

- **证明: **    
  定义 $\{N(t):t\geq 0\}$ 的 Laplace 变换为 $\text{E}[e^{-u N(t)}]$   
  任意给定 $u>0$，记 $g(t) = \text{E}[e^{-u N(t)}]$，我们有:   
  $\begin{align}
  g(t+h)
  &= \text{E}[e^{-u N(t+h)}]\\
  &= \text{E}[e^{-uN(t)}e^{-u (N(t+h)-N(t))}]\quad (\bold{增量独立性})\\
  &= \text{E}[e^{-uN(t)}]\cdot \text{E}[e^{-u(N(t+h)-N(t))}]\\
  &= g(t)\text{E}[e^{-uN(t,t+h)}]\\
  &= g(t)\text{E}[e^{-uN(h)}]\end{align}$    
  由**稀有性**可知 $\begin{cases}
  \text{P}\{N(h)=0\}=\text{P}\{N(t,t+h)=0\} = (1-\lambda) h + o(h)\\
  \text{P}\{N(h)=1\}=\text{P}\{N(t,t+h)=1\} = \lambda h + o(h)\\
  \text{P}\{N(h)\geq 2\}=\text{P}\{N(t,t+h)\geq 2\} = o(h)\end{cases}$    
  因此对 $N(h)=0,\ N(h) = 1,\ N(h)\geq 2$ 取条件，根据**全期望公式**可得:   
  $$
  \begin{align}
  \text{E}[e^{-u N(h)}]
  &= e^{-u\cdot 0}(1-\lambda)h + e^{-u\cdot 1}\lambda h + o(h)\\
  &= 1-\lambda h + e^{-u}(\lambda h) + o(h)\\
  &= 1 + \lambda h(e^{-u}-1) + o(h)\end{align}
  $$
  于是我们有 $g(t+h) = g(t)\text{E}[e^{-u N(h)}] = g(t)[1+\lambda h(e^{-u}-1)] + o(h)$   
  由此推出:    
  $$
  \begin{align}
  g'(t) 
  &= \underset{h\to 0}{\lim} \frac{g(t+h)-g(t)}{h}\\
  &= \underset{h\to 0}{\lim} [g(t)\lambda(e^{-u}-1) + \frac{o(h)}{h}]\\
  &= g(t)\lambda (e^{-u}-1)\end{align}
  $$
  于是有 $\frac{d}{\mathrm{d}t}\log(g(t)) = \frac{g'(t)}{g(t)}= \lambda (e^{-u}-1)$   
  积分后得到 $\log(g(t)) = \lambda (e^{-u}-1)t + C$  
  根据 $g(0) = \text{E}[e^{-uN(0)}] = \text{E}[e^{-u\cdot 0}] = 1$ 可知常数 $C = 0$   
  所以 $g(t) = e^{\lambda t (e^{-u}-1)}$    
  因此对于任意 $\begin{cases}
  u>0\\
  t\geq 0\end{cases}$ 都有 $\text{E}[e^{-uN(t)}] = g_u(t) = e^{\lambda t (e^{-u}-1)}$ (其中 $g_u$ 即固定 $u$ 时的函数 $g$)
  
  **考虑 $X\sim \text{Poisson}(\lambda t)$ 的 Laplace 变换: **   
  $\begin{align}
  \text{E}[e^{-uX}]
  &= \underset{i=0}{\overset{\infty}\sum} e^{-ui} \cdot e^{-\lambda t}\frac{(\lambda t)^i}{i!}\\
  &= e^{-\lambda t}\underset{i=0}{\overset{\infty}\sum} \frac{(\lambda t e^{-u})^i}{i!}\\
  &= e^{-\lambda t} e^{\lambda t e^{-u}}\\
  &= e^{\lambda t(e^{-u}-1)}\end{align}$   
  由于 Laplace 变换唯一地确定了分布，因此我们得出结论: $N(t)\sim \text{Poisson}(\lambda t)$  

**定义3.5 (Poisson 过程的等价定义)**  
若计数过程 $\{N(t):t\geq 0\}$ 满足 $N(0)=0$，具有**独立平稳增量**，  
并且对于任意 $t\geq 0$ 都有 $N(t) \sim \text{Poisson}(\lambda t)$ (自然有 $\lambda >0$)，   
则称 $\{N(t):t\geq 0\}$ 是参数为 $\lambda$ 的 **Poisson 过程**.  
(这也是我们日后最常用的 Poisson 过程的定义)



### 3.2.3 Poisson 过程的基本性质

#### (1) 数字特征

考虑参数为 $\lambda >0$ 的 Poisson 过程 $\mathbf N =\{N(t):t\geq 0\}$.  
对于任意 $t> 0$，都有 $N(t)\sim \text{Poisson}(\lambda t)$ 成立.   
因此对于任意 $t>0$ 都有 $\begin{cases}
M(u,t) = \text{E}[e^{uN(t)}] = e^{\lambda t(e^u-1)}\quad (u\in \mathbb R)\\
\varphi(u,t) = \text{E}[e^{iu N(t)}] = e^{\lambda t(e^{iu}-1)}\quad(u\in \mathbb R)\\
\mu_{\mathbf N}(t) = \text{E}[N(t)]=\lambda t\\
\sigma^2_{\mathbf N}(t) = \text{Var}[N(t)] = \lambda t\\
\text{E}[N(t)[N(t)-1]\dotsm\text{E}[N(t)-k+1]] = (\lambda t)^k\end{cases}$   
**下面计算 Poisson 过程 $\{N(t):t\geq 0\}$ 的自相关函数和自协方差函数: **   
对于任意 $0<s<t$:   

- 自相关函数:   
  $\begin{align}
  r_{\mathbf N}(s,t)
  &=\text{E}[N(s)N(t)]\\
  &= \text{E}[N(s)(N(s)+N(t) - N(s))]\\
  &= \text{E}[(N(s))^2] - \text{E}[N(s)(N(t)-N(s))]\\
  &= \text{E}[(N(s))^2] - \text{E}[N(s)]\cdot \text{E}[N(t-s)]\\
  &= [(\lambda s)^2 + (\lambda s)] + \lambda s\cdot \lambda(t-s)\\
  &= \lambda^2 st + \lambda s\end{align}$   
- 自协方差函数:   
  $\begin{align}
  \text{Cov}_{\mathbf N}(s,t) 
  &= \text{Cov}(N(s),N(t))\\
  &= \text{E}[N(s)N(t)] - \text{E}[N(s)]\text{E}[N(t)]\\
  &= \lambda^2 st + \lambda s - \lambda s \cdot \lambda t\\
  &= \lambda s\end{align}$ 

**下面计算 Poisson 过程的联合分布: **  
对于任意 $\begin{cases}
m\geq 1\\
0<t_1<t_2<\dotsm<t_m\\
0\leq k_1\leq k_2\leq \dotsm \leq k_m\end{cases}$ 都有:
$$
\begin{align}
\text{P}\{N(t_1)=k_1,\dots,N(t_m)=k_m\}
&= \text{P}\{N(t_1)=k_1,N(t_2)-N(t_1)=k_2-k_1,\dots,N(t_m)-N(t_{m-1})=k_m-k_{m-1}\}\\
&= \text{P}\{N(t_1) = k_1\}
\text{P}\{N(t_2-t_1) = k_2-k_1\}\dotsm \text{P}\{N(t_m-t_{m-1})=k_m-k_{m-1}\}\\
&= \frac{(\lambda t_1)^{k_1}}{k_1!}e^{-\lambda t_1} \frac{[\lambda(t_2-t_1)]^{k_2-k_1}}{(k_2-k_1)!}e^{-\lambda (t_2-t_1)}\dotsm \frac{[\lambda (t_m-t_{m-1})]^{k_m-k_{m-1}}}{(k_m-k_{m-1})!}e^{-\lambda (t_m-t_{m-1})}\\
&= \frac{(\lambda t_1)^{k_1}}{k_1!} \frac{[\lambda(t_2-t_1)]^{k_2-k_1}}{(k_2-k_1)!}\dotsm \frac{[\lambda (t_m-t_{m-1})]^{k_m-k_{m-1}}}{(k_m-k_{m-1})!}e^{-\lambda t_m}\end{align}
$$
**有了联合分布，就可以计算条件分布: **    
假设 $0<s<t$ 

- **给定总时间段的某个子时间段发生的事件数，考虑剩余事件段内发生的事件数: **
  在给定 $N(s)=n$ 的条件下，对于任意 $k\geq n$ 都有:   
  $$
  \begin{align}
  \text{P}\{N(t)=k|N(s)=n\}
  &= \frac{\text{P}\{N(t)=k,N(s)=n\}}{\text{P}\{N(s)=n\}}\\  
  &= \frac{\text{P}\{N(s)=n\}\text{P}\{N(t-s)=k-n\}}{\text{P}\{N(s)=n\}}\\
  &= \text{P}\{N(t-s)=k-n\}\\
  &= \frac{[\lambda(t-s)]^{k-n}}{(k-n)!}e^{-\lambda (t-s)}\end{align}
  $$
  等式 $\text{P}\{N(t)=k|N(s)=n\} = \text{P}\{N(t-s)=k-n\}$ 说明了 Poisson 过程的**无后效性**，  
  即在给定时间段内已知某个子时间段发生的事件数量后，   
  剩余时间段内发生的事件数量仅取决于该剩余时间段的长度，与之前的事件发生时间无关.   
  这也直接反映了 Poisson 过程**增量的平稳性**.
  
- **给定总时间段内发生的总事件数，考虑某个子时间段内发生的事件数: **
  在给定 $N(t) = n$ 的条件下，对于任意 $0\leq k\leq n$ 都有:   
  $$
  \begin{align}
  \text{P}\{N(s) = k|N(t)=n\}
  &= \frac{\text{P}\{N(s)=k,N(t)=n\}}{\text{P}\{N(t)=n\}}\\
  &= \frac{(\lambda s)^k}{k!}\frac{[\lambda(t-s)]^{n-k}}{(n-k)!}e^{-\lambda t}{\Large{/}} \frac{(\lambda t)^{n}}{n!}e^{-\lambda t}\\
  &= \frac{n!}{k!(n-k)!}(\frac{s}{t})^k(1-\frac{s}{t})^{n-k}\\
  &= \text{P}\{B(n,\frac{s}{t})=k\}\end{align}
  $$
  等式 $\text{P}\{N(s) = k|N(t)=n\} = \text{P}\{B(n,\frac{s}{t})=k\}$ 说明了在一个 Poisson 过程中，   
  给定总时间段内事件的总数后，在该时间段的一个子区间内事件数量的分布，可以通过二项分布来描述，   
  这体现了 Poisson 过程的事件在时间上的均匀分布性质 (**增量的平稳性**).

总结得到以下定理: 
**定理3.5: (Poisson 过程的条件分布, 随机过程 (苏中根) 3.1节)**   
对于任意给定的 $0<s<t$ 和 $n\geq 0$ 都有:   

- 给定 $N(s)=n$ 时，$N(t)$ 的条件分布为 $\text{Poisson}(\lambda(t-s))+n$    
  即对于任意 $k\geq n$ 都有:   
  $$
  \text{P}\{N(t)=k|N(s)=n\} = \text{P}\{N(t-s)=k-n\}=\frac{[\lambda(t-s)]^{k-n}}{(k-n)!}e^{-\lambda (t-s)}
  $$
  
- 给定 $N(t)=n$ 时，$N(s)$ 的条件分布为 $B(n,\frac{s}{t})$   
  即对于任意 $0\leq k\leq n$ 都有:   
  $$
  \text{P}\{N(s) = k|N(t)=n\} = \text{P}\{B(n,\frac{s}{t})=k\} = \frac{n!}{k!(n-k)!}(\frac{s}{t})^k(1-\frac{s}{t})^{n-k}
  $$

**最后分析 Poisson 过程的样本曲线: **   
记 $T_1,T_2,\dots$ 为事件发生的时刻 (或者称为**到达时刻**, arrival time)，令 $T_0=0$   
这些事件还未发生时，$T_1,T_2,\dots$ 都是随机的，  
然而一旦这些事件发生，$T_1,T_2,\dots$ 就变为固定的值 $t_1,t_2,\dots$，   
即变成了一系列样本点，也就是说，成为了**随机变量的实现** (Realization of Random Variable).  
这一系列样本点的整体 $\{t_0,t_1,t_2,\dots\}$ 构成了**随机过程 $\{T_0,T_1,T_2,\dots\}$ 的实现** (Realization of Stochastic Process).  
此时，样本曲线便完全确定了:

<img src="SP SZG Figure 3.1.png" style="zoom:67%;" />



#### (2) 到达时刻 & 等待时间的分布

刚刚我们定义了 Poisson 过程的**到达时刻 **(Arrival Time) 为 $T_1,T_2,\dots$，并且令 $T_0 = 0$    
我们现在定义 Poisson 过程的第 $i$ 个**等待时间** (Waiting Time) 为 $X_i = T_i-T_{i-1}\ \ (\forall\ i=1,2,\dots)$   
**现在我们来确定等待时间 $X_n$ 的分布: **  

- 首先我们注意到事件 $\{X_1>t\}$ 发生，当且仅当区间 $(0,t]$ 内没有事件发生，  
  也就等价于事件 $\{N(t)=0\}$ 发生.  
  因此 $\text{P}\{X_1>t\} = \text{P}\{N(t)=0\} = e^{-\lambda t} = \text{P}\{\exp(\lambda)>t\}$   
  说明 $X_1\sim\exp(\lambda)$ 
  
- 现在考虑事件 $\{X_2>t\}$:   
  对 $X_1$ 取条件有: $\text{P}\{X_2>t\}=\text{E}[\text{P}\{X_2>t|X_1\}]$   
  对于任意 $s\geq 0$，都有:
  $$
  \begin{align}
  \text{P}\{X_2>t|X_1=s\}
  &= \text{P}\{(s,s+t]内没有事件发生\mid X_1=s\}\\
  &= \text{P}\{N(s+t)-N(s)=0\mid X_1=s\}\quad(\text{Poisson}\ 过程增量的独立平稳性)\\
  &= \text{P}\{N(t) = 0\}\\
  &= e^{-\lambda t}\end{align}
  $$
  说明 $\text{P}\{X_2>t|X_1=s\}$ 的值与 $X_1$ 的取值 $s$ 无关  
  因此 $\text{P}\{X_2>t\}=\text{E}[\text{P}\{X_2>t|X_1\}] = e^{-\lambda t} = \text{P}\{\exp(\lambda)>t\}$  
  说明 $X_2\sim \exp(\lambda)$ 且 $X_2\ \bot\ X_1$    

不断重复上述论证，我们就得到下面的定理:   
**定理3.6: (Poisson 过程等待时间的分布, Introduction to Probability Models 命题5.2)**  
Poisson 过程等待时间 $X_n(n=1,2,\dots)$ 是独立同分布的指数随机变量   
即 $\{X_n\}\overset{\text{i.i.d.}}\sim \exp(\lambda) = \text{Gamma}(1,\lambda)$  
这个命题并不使我们感到惊奇:   
**独立平稳增量**的假设本质上等价于限定在任意时间点，过程概率意义下重新开始.  
即过程从任意时间点往后，独立于它以前发生的一切 (由独立增量性)，  
而且也有与原过程相同的分布 (由平稳增量性).  
换句话说，**Poisson 过程没有记忆**，因而等待时间服从指数分布是相当自然的.

**至于到达时间 $T_n(n=1,2,\dots)$ 的分布: **  
对于任意 $n=1,2,\dots$，我们有:   
$T_n = \sum_{i=1}^n(T_i-T_{i-1}) = \sum_{i=1}^nX_i$   
因此 $T_n$ 作为 $n$ 个独立指数随机变量的和，它服从 $\text{Gamma}(n,\lambda)$ 分布  
(这个结论我们在 FDU 随机过程导论 1. 概率论回顾 1.4.2 已经说明过了: **Gamma 分布的可再生性**)  
其概率密度 $f_{T_n}(t) = \text{P}\{T_n = t\} = \text{P}\{\text{Gamma}(n,\lambda)=t\} = \lambda e^{-\lambda t}\frac{(\lambda t)^{n-1}}{(n-1)!}\ \ \ (t\geq 0)$   
实际上，我们还有另外两种方法证明 $T_n\sim \text{Gamma}(n,\lambda)$:   

- **法二: **  
  任意给定 $t\geq 0$  
  注意到第 $n$ 个事件在时刻 $t$ 发生 (即事件 $\{T_n\leq t\}$ 发生)，  
  当且仅当直到 $t$ 为止发生事件的个数至少是 $n$ (即事件 $\{N(t)\geq n\}$ 发生)  
  因此有:   
  $\begin{align}
  F_{T_n}(t)
  &= \text{P}\{T_n\leq t\}\\
  &= \text{P}\{N(t)\geq n\}\\
  &= \underset{k=n}{\overset{\infty}\sum}\text{P}\{N(t)=k\}\\
  &= \underset{k=n}{\overset{\infty}\sum}e^{-\lambda t}\frac{(\lambda t)^k}{k!}\end{align}$  
  对上式求微分导出:   
  $\begin{align}
  f_{T_n}(t) 
  &= \underset{k=n}{\overset{\infty}\sum}[-\lambda e^{-\lambda t}\frac{(\lambda t)^{k}}{k!} + e^{-\lambda t}\lambda \frac{(\lambda t)^{k-1}}{(k-1)!}]\\
  &= \lambda e^{-\lambda t}[\frac{(\lambda t)^{n-1}}{(n-1)!} - \underset{k=n}{\overset{\infty}\sum}\frac{(\lambda t)^{k}}{k!} + \underset{k=n+1}{\overset{\infty}\sum}\frac{(\lambda t)^{k-1}}{(k-1)!}]\\
  &=  \lambda e^{-\lambda t}\frac{(\lambda t)^{n-1}}{(n-1)!}\\
  &= \text{P}\{\text{Gamma}(n,\lambda)= t\}\end{align}$     
  说明 $T_n\sim \text{Gamma}(n,\lambda)$.  
- **法三: **  
  注意到 $T_n$ 是第 $n$ 个事件的到达时刻，因此对于任意 $t\geq 0$ 都有:   
  $\begin{align}
  \text{P}\{t<T_n<t+h\}
  &= \text{P}\{N(t)=n-1,在\ (t,t+h)\ 内有一个事件\} + o(h)\\
  &= \text{P}\{N(t)=n-1\}\text{P}\{N(t+h)-N(h)=1\} + o(h)\\
  &= \text{P}\{N(t)=n-1\}\text{P}\{N(t,t+h)=1\} + o(h)\quad(根据\text{ Poisson } 过程的稀有性)\\
  &= e^{-\lambda t}\frac{(\lambda t)^{n-1}}{(n-1)!}\cdot [\lambda h + o(h)]+o(h)\\
  &= \lambda e^{-\lambda t}\frac{(\lambda t)^{n-1}}{(n-1)!} h +o(h)\end{align}$ 
  令 $h\to 0$ 即有 $f_{T_n}(t) = \text{P}\{T_n = t\} = \lambda e^{-\lambda t}\frac{(\lambda t)^{n-1}}{(n-1)!}=\text{P}\{\text{Gamma}(n,\lambda)=t\}$   
  说明 $T_n\sim \text{Gamma}(n,\lambda)$.

**定理3.6 为我们提供了 Poisson 过程的另一个等价定义: **  
**定义3.6: (Poisson 过程的等价定义, Introduction to Probability Models 5.3.3节)**  
设 $\{X_n:n\geq 1\}$ 是独立同分布的指数随机变量序列，每一项都服从 $\exp(\lambda)$ 分布.  
定义 $T_n = X_1+X_2+\dots+X_n\ \ (\forall\ n=1,2,\dots)$   
现在我们定义一个**计数过程** $\{N(t):t\geq 0\}$，其中 $N(t)\equiv \max\{n:T_n\leq t\}$，  
则这个计数过程是速率为 $\lambda$ 的 **Poisson 过程**.

**证明定义3.6和定义3.5的等价性: **   
我们只需证明定义3.6中计数过程 $\{N(t):t\geq 0\}$ 满足定义3.5的前提条件: 

- **(1) 证明对于任意 $t\geq 0$ 都有 $N(t)\sim \text{Poisson}(\lambda t)$**  
  若固定 $t\geq 0$，则等价于证明对于任意 $k\geq 0$ 都有 $\text{P}\{N(t)=k\}=e^{-\lambda t}\frac{(\lambda t)^k}{k!}$ 成立:   
  $\begin{align}
  \text{P}\{N(t)=k\} 
  &= \text{P}\{T_k\leq t< T_{k+1}\}\\
  &= \text{P}\{T_k\leq t< T_{k}+X_{k+1}\}\\
  &= \text{E}[\text{P}\{T_k\leq t < T_k + X_{k+1}|T_k\}]\\
  &= \text{E}[\text{P}\{X_{k+1}>t-T_k|T_k\}]\\
  &= \int_0^t \text{P}\{X_{k+1}> t-s|T_k = s\}\text{P}\{T_k = s\}\mathrm{d}s\quad(X_{k+1}\ \bot\ T_k)\\
  &= \int_0^t \text{P}\{X_{k+1}> t-s\}\text{P}\{T_k = s\}\mathrm{d}s\\
  &= \int_0^t e^{-\lambda(t-s)}\cdot\lambda e^{-\lambda s}\frac{(\lambda s)^{k-1}}{(k-1)!}\mathrm{d}s\\
  &= e^{-\lambda t}\int_0^t \lambda \frac{(\lambda s)^{k-1}}{(k-1)!}\mathrm{d}s\\
  &= e^{-\lambda t}\frac{(\lambda t)^k}{k!}   \end{align}$  
  于是得证对于任意 $t\geq 0$ 都有 $N(t)\sim \text{Poisson}(\lambda t)$
- **(2) 证明计数过程 $\{N(t):t\geq 0\}$ 的增量具有独立性、平稳性**  
  即要证明对于任意 $0<s<t$，都有 $\begin{cases}
  N(t)-N(s)\ \bot\ N(s)\\
  N(t)-N(s)\overset{\mathrm{d}}= N(t-s)\end{cases}$ 成立:    
  不失一般性，我们固定 $0<s<t$  
  - **① 证明 $N(t)-N(s)\ \bot\ N(s)$: **  
    即要证明对于任意 $\begin{cases}
    k\geq 0\\
    m\geq 0\end{cases}$ 都有 $\text{P}\{N(t)-N(s)=k|N(s)=m\}$ 的值不依赖于 $m$  
    我们从 $\text{P}\{N(t)-N(s)=k|N(s)=m\}
    = \frac{\text{P}\{N(t)-N(s) = k,N(s)=m\}}{\text{P}\{N(s)=m\}}$ 出发:      
    - **(i) 当 $k=0$ 时: **   
      $\begin{align}
      &\text{P}\{N(t)-N(s)=0,N(s)=m\}\\
      &= \text{P}\{T_m\leq s<t<T_{m+1}\}\\
      &= \text{P}\{T_m\leq s<t<T_m + X_{m+1}\}\\
      &= \text{E}[\text{P}\{T_m\leq s< t < T_m + X_{m+1}|T_{m}\}]\\
      &= \text{E}[\text{P}\{X_{m+1}>t-T_m|T_m\}]\\
      &= \int_0^s \text{P}\{X_{m+1}>t-u|T_m=u\}\text{P}\{T_m=u\}du\quad (X_{m+1}\ \bot\ T_m)\\
      &= \int_0^s \text{P}\{X_{m+1}>t-u\}\text{P}\{T_m=u\}du\\
      &= \int_0^s \text{P}\{\exp(\lambda)> t-u\}\text{P}\{\text{Gamma}(m,\lambda) = u\}du\\
      &= \int_0^s e^{-\lambda (t-u)} \lambda e^{-\lambda u}\frac{(\lambda u)^{m-1}}{(m-1)!}du\\
      &= e^{-\lambda t} \int_0^s \lambda \frac{(\lambda u)^{m-1}}{(m-1)!}du\\
      &= e^{-\lambda t} \frac{(\lambda u)^{m}}{m!}du
      \end{align}$   
      因此:   
      $\begin{align}
      \text{P}\{N(t)-N(s)=0|N(s)=m\}
      &= \frac{\text{P}\{N(t)-N(s) = 0,N(s)=m\}}{\text{P}\{N(s)=m\}}\\
      &= e^{-\lambda t}\frac{(\lambda s)^m}{m!}{\Large /} e^{-\lambda s}\frac{(\lambda s)^m}{m!}\\
      &= e^{-\lambda (t-s)}\\
      &= \text{P}\{\text{Poisson}(\lambda (t-s))=0\}\end{align}$    
      表明当 $k=0$ 时，条件概率  $\text{P}\{N(t)-N(s)=k|N(s)=m\}$ 的值不依赖于 $m$ 
      
    - **(ii) 当 $k\geq 1$ 时: **  
      
      $\begin{align}
      &\text{P}\{N(t)-N(s) = k,N(s)=m\}\\
      &= \text{P}\{T_{m+k}\leq t<T_{m+k+1},T_{m}\leq s<T_{m+1}\}\\
      &= \text{P}\{T_m\leq s<T_{m+1}<T_{m+k}\leq t<T_{m+k+1}\}\\
      &= \text{P}\{T_m\leq s<T_m+X_{m+1}<T_{m+k} \leq t<T_{m+k}+X_{m+k+1}\}\\
      &= \int_0^s \int_{s-u}^{t-u}  
      \text{P}\{X_{m+1}>s-u,X_{m+k+1}>t-u-v|T_m = u, T_{m+k}-T_m=v\}\text{P}\{T_m = u, T_{m+k}-T_m=v\}dvdu\\
      &= \int_0^s \int_{s-u}^{t-u}  
      \text{P}\{X_{m+1}>s-u\}\text{P}\{X_{m+k+1}>t-u-v\}\text{P}\{T_m = u\}\text{P}\{T_{m+k}-T_m = v\}dvdu\\
      &= \int_0^s \int_{s-u}^{t-u}  
      \text{P}\{\exp(\lambda)>s-u\}\text{P}\{\exp(\lambda)>t-u-v\}\text{P}\{\text{Gamma}(m,\lambda) = u\}\text{P}\{\text{Gamma}(k,\lambda) = v\}dvdu\\
      &= \int_0^s \int_{s-u}^{t-u}  
      e^{-\lambda(s-u)}
      e^{-\lambda(t-u-v)}
      \lambda e^{-\lambda u}\frac{(\lambda u)^{m-1}}{(m-1)!}  
      \lambda e^{-\lambda v}\frac{(\lambda v)^{k-1}}{(k-1)!}dvdu\\
      &= \end{align}$
      
      
      
      
      
      $\begin{align}
      &\text{P}\{N(t)-N(s) = k,N(s)=m\}\\
      &= \text{P}\{T_{m+k}\leq t<T_{m+k+1},T_{m}\leq s<T_{m+1}\}\\
      &= \text{P}\{T_m\leq s<T_{m+1}<T_{m+k}\leq t<T_{m+k+1}\}\\
      &= \text{P}\{T_m\leq s<T_m+X_{m+1}<T_{m+k} \leq t<T_{m+k}+X_{m+k+1}\}\\
      &= \int_0^s\int_u^t \text{P}\{X_{m+1}>s-u,X_{m+k+1}>t-v|T_m=u,T_{m+k}=v\}\text{P}\{T_m=u,T_{m+k}=v\}dvdu\\
      &= \int_0^s\int_u^t \text{P}\{X_{m+1}>s-u\}\text{P}\{X_{m+k+1}>t-v\}\text{P}\{T_m=u\}\text{P}\{T_{m+k}-T_m = v-u\}dvdu\\ 
      &= \int_0^s\int_u^t \text{P}\{\exp(\lambda)>s-u\}\text{P}\{\exp(\lambda)>t-v\}\text{P}\{\text{Gamma}(m,\lambda)=u\}\text{P}\{\text{Gamma}(k,\lambda) = v - u\}dvdu\\
      &= \int_0^s\int_u^t e^{-\lambda (s-u)} e^{-\lambda(t-v)} \lambda e^{-\lambda u}\frac{(\lambda u)^{m-1}}{(m-1)!} \lambda e^{-\lambda (v-u)}\frac{[\lambda(v-u) ]^{k-1}}{(k-1)!} dvdu\\
      &= e^{-\lambda t} \int_0^s \lambda e^{-\lambda (s-u)} \frac{(\lambda u)^{m-1}}{(m-1)!} \int_u^t \lambda \frac{[\lambda (v-u)]^{k-1}}{(k-1)!}dvdu\\
      &= e^{-\lambda t} \int_0^s \lambda e^{-\lambda (s-u)}\frac{(\lambda u)^{m-1}}{(m-1)!} \frac{[\lambda (t-u)]^k}{k!}du\\
      &= e^{-\lambda t}[e^{-\lambda(s-u)}\frac{(\lambda u)^m}{m!}\frac{[\lambda (t-u)]^k}{k!}]{\Large{|}}_0^s\\
      &= e^{-\lambda t}\frac{(\lambda s)^m}{m!}\frac{[\lambda (t-s)]^k}{k!}\end{align}$
      
      因此:   
      $\begin{align}
      \text{P}\{N(t)-N(s)=k|N(s)=m\}
      &= \frac{\text{P}\{N(t)-N(s) = k,N(s)=m\}}{\text{P}\{N(s)=m\}}\\
      &= e^{-\lambda t}\frac{(\lambda s)^m}{m!}\frac{[\lambda (t-s)]^k}{k!}{\Large /} e^{-\lambda s}\frac{(\lambda s)^m}{m!}\\
      &= e^{-\lambda (t-s)} \frac{[\lambda (t-s)]^k}{k!}\\
      &= \text{P}\{\text{Poisson}(\lambda (t-s))=k\}\end{align}$​    
      表明当 $k\geq 0$ 时，条件概率  $\text{P}\{N(t)-N(s)=k|N(s)=m\}$ 的值不依赖于 $m$ 
      
    - **综合 (i)(ii) 可知: **      
      任意给定 $0<s<t$  
      对于任意 $\begin{cases}
      k\geq 0\\
      m\geq 0\end{cases}$ 都有 $\text{P}\{N(t)-N(s)=k|N(s)=m\}$ 的值不依赖于 $m$  
      说明对于任意 $0<s<t$ 都有 $N(t)-N(s)\ \bot\ N(s)$   
      **表明计数过程 $\{N(t):t\geq 0\}$ 的增量具有独立性.**  
      因此对于任意 $k\geq 0$ 都有:   
      $\begin{align}
      &\text{P}\{N(t)-N(s) = k\}\quad (根据增量独立性)\\
      &\quad= \text{P}\{N(t)-N(s) = k|N(s) = m\}\quad (\forall\ m\geq 0)\\
      &\quad=  e^{-\lambda (t-s)} \frac{[\lambda (t-s)]^k}{k!}\\
      &\quad= \text{P}\{\text{Poisson}(\lambda (t-s))=k\}\\\end{align}$
  - **② 证明 $N(t)-N(s)\overset{\mathrm{d}}= N(t-s)$: **  
    根据 (2)①(i)(ii) 中的结论可知:   
    任意给定 $0<s<t$  
    对于任意 $k\geq 0$ 都有 $\text{P}\{N(t)-N(s) = k\} = \text{P}\{\text{Poisson}(\lambda (t-s))=k\}$   
    说明 $N(t)-N(s)\sim \text{Poisson}(\lambda(t-s))$   
    再结合 (1) 中的结论 $N(t-s) \sim \text{Poisson}(\lambda(t-s))$，  
    我们就证明了 $N(t)-N(s)\overset{\mathrm{d}}= N(t-s)$   
    **表明计数过程 $\{N(t):t\geq 0\}$ 的增量具有平稳性.**
- 综合(1)(2)可知，计数过程 $\{N(t):t\geq 0\}$ 满足**定义3.5**的前提条件，因而是一个 **Poisson 过程**.  
  表明**定义3.6**是 Poisson 过程的一个等价定义.



#### (3) 到达时刻的条件分布

**定理 3.7: (到达时刻的联合条件分布, Introduction to Probability Models 定理5.2)**  
假设 $\mathbf N = \{N(t):t\geq 0\}$ 是参数为 $\lambda$ 的 Poisson 过程，记 $T_1,T_2,\dots$ 为到达时刻.  
给定 $t>0$，则有 $(T_1,T_2,\dots,T_n|N(t)=n)\overset{\mathrm{d}}= (U_{(1)},U_{(2)},\dots,U_{(n)})$   
其中 $U_{(1)},U_{(2)},\dots,U_{(n)}$ 为对应于 $U_1,U_2,\dots,U_n\overset{\text{i.i.d.}}\sim \text{Uniform}(0,t)$ 的次序统计量.  
**上面的结论通常可表述为: **  
在 $(0,t)$ 中已经发生了 $n$ 个事件的条件下，  
事件发生的时间 $T_1,T_2,\dots,T_n$ (考虑为无次序的随机变量时) 是在 $(0,t)$ 上独立均匀地分布的.  

- **引理 3.8: (次序统计量的性质1)**  
  令 $Y_1,Y_2,\dots,Y_n$ 为 $n$ 个随机变量.  
  我们记 $Y_{(1)},Y_{(2)},\dots,Y_{(n)}$ 为对应于 $Y_1,Y_2,\dots,Y_n$ 的次序统计量，   
  其中 $Y_{(k)}$ 是在 $Y_1,Y_2,\dots,Y_n$ 中第 $k$ 小的值.  
  若 $Y_1,Y_2,\dots,Y_n$ 为具有概率密度函数 $f_Y(\cdot)$ 的**独立同分布**的连续随机变量，  
  则**次序统计量** $Y_{(1)},Y_{(2)},\dots,Y_{(n)}$ 的联合密度为:   
  $f_{Y_{(1)},\dots,Y_{(n)}}(y_1,\dots,y_n) = n!\underset{i=1}{\overset{n}\prod} f_Y(y_i)\quad (\forall\ y_1<\dots<y_n)$   
  上式的得到是由于:   

  - ① 次序统计量 $Y_{(1)},Y_{(2)},\dots,Y_{(n)}$ 的取值是 $y_1,y_2,\dots,y_n$，  
    当且仅当 $Y_1,Y_2,\dots,Y_n$ 的取值是 $y_1,y_2,\dots,y_n$ 的 $n!$ 个排列中的任意一个.

  - ② 对于 $1,2,\dots,n$ 的任意排列 $i_1,i_2,\dots,i_n$，  
    $Y_1,Y_2,\dots,Y_n$ 的取值为 $y_{i_1},y_{i_2},\dots,y_{i_n}$ 的概率密度都是 $\underset{j=1}{\overset{n}\prod} f_Y(y_{i_j}) = \underset{j=1}{\overset{n}\prod} f_Y(y_j)$   

  特殊地，如果 $Y_1,Y_2,\dots,Y_n\overset{\text{i.i.d.}}\sim \text{Uniform}(0,t)$，  
  则次序统计量 $Y_{(1)},Y_{(2)},\dots,Y_{(n)}$ 的联合概率密度为:  
  $$
  f_{Y_{(1)},\dots,Y_{(n)}}(y_1,\dots,y_n) = n!\underset{i=1}{\overset{n}\prod} \frac{1}{t} = \frac{n!}{t^n}\quad (\forall\ 0<y_1<\dots<y_n<t)
  $$

**定理 3.7 的证明:   **  
给定条件 $N(t)=n$   
注意到对于 $0<t_1<\dots<t_n<t$，  
事件 $\{T_1=t_1,\dots,T_n=t_n,N(t)=n\}$ 发生等价于  
事件 $\{X_1=t_1,X_2=t_2-t_1,\dots,X_n=t_n-t_{n-1},X_{n+1}>t-t_n\}$ 发生.  
利用**定理3.6**我们知道:   
$\begin{align}
\text{P}\{T_1=t_1,\dots,T_n=t_n|N(t)=n\}
&= \frac{\text{P}\{T_1=t_1,\dots,T_n=t_n,N(t)=n\}}{\text{P}\{N(t)=n\}}\\
&= \frac{\text{P}\{X_1=t_1,X_2=t_2-t_1,\dots,X_n=t_n-t_{n-1},X_{n+1}>t-t_n\}}{\text{P}\{N(t)=n\}}\\
(定理3.6)&= \frac{\text{P}\{X_1=t_1\}\text{P}\{X_2=t_2-t_1\}\dotsm \text{P}\{X_n=t_n-t_{n-1}\}\text{P}\{X_{n+1}>t-t_n\}}{\text{P}\{N(t)=n\}}\\
&= \frac{\lambda e^{-\lambda t_1}\lambda e^{-\lambda(t_2-t_1)}\dotsm \lambda e^{-\lambda(t_n-t_{n-1})}e^{-\lambda (t-t_n)}}{e^{-\lambda t}(\lambda t)^n/n!}\\
&= \frac{n!}{t^n}\\
(引理3.8)&= \text{P}\{U_{(1)} = t_1,\dots,U_{(n)}=t_n\}\quad (\forall\ 0<t_1<\dotsm<t_n<t)\end{align}$ 

这就证明了结论: $(T_1,T_2,\dots,T_n|N(t)=n)\overset{\mathrm{d}}= (U_{(1)},U_{(2)},\dots,U_{(n)})$   

***

**定理 3.9: (到达时刻的边缘条件分布)**      
假设 $\mathbf N = \{N(t):t\geq 0\}$ 是参数为 $\lambda$ 的 Poisson 过程，记 $T_1,T_2,\dots$ 为到达时刻.  
给定 $t>0$，则对于任意 $1\leq k\leq n$ 都有 $(T_k|N(t)=n)\overset{\mathrm{d}}= U_{(k)}$   
其中 $U_{(1)},U_{(2)},\dots,U_{(n)}$ 为对应于 $U_1,U_2,\dots,U_n\overset{\text{i.i.d.}}\sim \text{Uniform}(0,t)$ 的次序统计量.

- **引理 3.10: (次序统计量的性质2)**    
  若 $Y_1,Y_2,\dots,Y_n$ 为具有概率密度函数 $f_Y(\cdot)$ 的**独立同分布**的连续随机变量，  
  记累计分布函数为 $F_Y(\cdot)$，  
  对应的**次序统计量**为 $Y_{(1)},Y_{(2)},\dots,Y_{(n)}$     
  下面我们计算 $Y_{(k)}$ 的概率密度函数:   
  $\begin{align}
  &\text{P}\{Y_{(k)}\in [x,x+\mathrm{d}x]\}\\
  &= \binom{n}{k-1}\binom{n-k+1}{1} \text{P}\{Y_1\leq x,\dots,Y_{k-1}\leq  x,Y_k \in [x,x+\mathrm{d}x],Y_{k+1}>x+\mathrm{d}x,Y_{n}>x+\mathrm{d}x\}\\
  &= \binom{n}{k-1}\binom{n-k+1}{1} \underset{i=1}{\overset{k-1}\prod} \text{P}\{Y_i\leq x\} \cdot \text{P}\{Y_k \in [x,x+\mathrm{d}x]\}\cdot  \underset{i=k+1}{\overset{n}\prod} \text{P}\{Y_i >x+\mathrm{d}x\}\\
  &= \binom{n}{k-1}\binom{n-k+1}{1} 
  F_Y(x)^{k-1} \text{P}\{Y_k \in [x,x+\mathrm{d}x]\} 
  [1-F_Y(x+\mathrm{d}x)]^{n-k}\end{align}$ 
  于是我们左右同除 $\mathrm{d}x$，并令 $\mathrm{d}x\to 0$，则有:   
  $\begin{align}
  f_{Y_{(k)}}(x) 
  &=\text{P}\{Y_{(k)}=x\}\\ 
  &= \binom{n}{k-1}\binom{n-k+1}{1} 
  F(x)^{k-1} \text{P}\{Y_k =x\} 
  [1-F_Y(x)]^{n-k}\\
  &= \binom{n}{k-1}\binom{n-k+1}{1} 
  F_Y(x)^{k-1} f_Y(x)
  [1-F_Y(x)]^{n-k}\\
  \end{align}$  
  我们就得到了第 $k$ 个次序统计量 $Y_{(k)}$​ 的概率密度函数.  
  - 特殊地，如果 $Y_1,Y_2,\dots,Y_n\overset{\text{i.i.d.}}\sim \text{Uniform}(0,t)$，  
    则第 $k$ 个次序统计量 $Y_{(k)}$ 的概率密度函数为:   
    $\begin{align}
    f_{Y_{(k)}}(x) 
    &=\text{P}\{Y_{(k)}=x\}\\ 
    &= \binom{n}{k-1}\binom{n-k+1}{1} 
    F_Y(x)^{k-1} f_Y(x)
    [1-F_Y(x)]^{n-k}\\
    &= \frac{n!}{(k-1)!(n-k+1)!}(n-k+1) (\frac{x}{t})^{k-1} \frac 1t (1-\frac{x}{t})^{n-k} \\ 
    &= \frac{n!}{(k-1)!(n-k)!}(\frac{x}{t})^{k-1} \frac 1t (1-\frac{x}{t})^{n-k} \end{align}$ 

**定理 3.9 的证明: **  
我们首先复述**定理3.5**的第二条结论:     

> 给定 $N(t)=n$ 时，$N(s)$ 的条件分布为 $B(n,\frac{s}{t})$   
> 即对于任意 $0\leq k\leq n$ 都有:   
> $\text{P}\{N(s) = k|N(t)=n\} = \text{P}\{B(n,\frac{s}{t})=k\} = \frac{n!}{k!(n-k)!}(\frac{s}{t})^k(1-\frac{s}{t})^{n-k}$ 

- **① $T_k$ 累积分布函数为: **(其中 $1\leq k\leq n$)  
  $\begin{align}
  \text{P}\{T_k\leq s|N(t)=n\}  
  &= \underset{i=k}{\overset{n}\sum}\text{P}\{N(s)=i|N(t)=n\}\\
  (定理3.5)&= \underset{i=k}{\overset{n}\sum}\text{P}\{B(n,\frac{s}{t})=i\}\\
  &= \underset{i=k}{\overset{n}\sum} \binom{n}{i}(\frac{s}{t})^i(1-\frac{s}{t})^{n-i}\end{align}$ 

- **② $T_k$ 的条件概率密度函数为: **(其中 $1\leq k\leq n$)  
  $\begin{align}
  \text{P}\{T_k= s|N(t)=n\}
  &= \frac{d}{\mathrm{d}s}\text{P}\{T_k\leq s|N(t)=n\}\\
  &= \underset{i=k}{\overset{n}\sum}
  \{\binom{n}{i}i(\frac{s}{t})^{i-1}\frac{1}{t}(1-\frac{s}{t})^{n-i} + \binom{n}{i}(\frac{s}{t})^i(n-i)(1-\frac{s}{t})^{n-i-1}(-\frac1t)\}\\
  &= \underset{i=k}{\overset{n}\sum}  
  \frac{n!}{(i-1)!(n-i)!}(\frac st)^{i-1}\frac 1t (1-\frac {s}{t})^{n-i} - \underset{i=k}{\overset{n-1}\sum}\frac{n!}{i!(n-i-1)!}
  (\frac{s}{t})^{i}\frac1t (1-\frac{s}{t})^{n-i-1}\\
  (第二项中令\ l = i+1 )&= \underset{i=k}{\overset{n}\sum}  
  \frac{n!}{(i-1)!(n-i)!}(\frac st)^{i-1}\frac 1t (1-\frac {s}{t})^{n-i} - \underset{l=k+1}{\overset{n}\sum}  
  \frac{n!}{(l-1)!(n-l)!}(\frac st)^{l-1}\frac 1t (1-\frac {s}{t})^{n-l}\\
  &= \frac{n!}{(k-1)!(n-k)!}(\frac{s}{t})^{k-1} \frac 1t (1-\frac{s}{t})^{n-k}\\
  (引理3.10)&= \text{P}\{U_{(k)}=s\}\end{align}$ 

  这就证明了结论: $(T_k|N(t)=n)\overset{\mathrm{d}}= U_{(k)}$ 

***

**下面我们看两个具体的例子: **  

- ① 假设 $\mathbf N=\{N(t):t\geq 0\}$ 是参数为 $\lambda$ 的 Poisson 过程.  
  记 $T_1,T_2$ 分别为第 $1,2$ 个事件发生的时刻，  
  计算 $\text{E}[T_1T_2|N(t)=2]$:   
  $\begin{align}
  \text{E}[T_1T_2|N(t)=2]
  &= \text{E}[U_{(1)}U_{(2)}]\\
  &= \text{E}[U_1U_2]\\
  &= \text{E}[U_1]\cdot \text{E}[U_2]\\
  &= (\frac t2)^2\\
  &= \frac{t^2}{4}\end{align}$   
  其中 $U_{(1)},U_{(2)}$ 为对应于 $U_1,U_2\overset{\text{i.i.d.}}\sim\text{Uniform}(0,t)$ 的次序统计量.
- ② 假设 $\mathbf N=\{N(t):t\geq 0\}$​ 是参数为 $\lambda$​ 的 Poisson 过程，代表一个车站的乘客计数过程.  
  一辆公交车将在 $t$​ 时刻到达，  
  请计算能够乘坐这辆公交车的所有乘客的总等待时间的期望:   
  这个问题可以翻译成:   
  $\begin{align}
  \text{E}[\sum_{i=1}^{N(t)}(t-T_i)]
  &= \text{E}\{\text{E}[\sum_{i=1}^{N(t)}(t-T_i)|N(t)]\}\end{align}$​   
  考虑条件分布 $[\sum_{i=1}^{N(t)}(t-T_i)|N(t)=n] = [\sum_{i=1}^n(t-T_i)|N(t)=n]$:   
  根据**定理3.7**可知在给定 $N(t)=n$​ 的条件下，  
  我们有 $(T_1,T_2,\dots,T_n|N(t)=n)\overset{\mathrm{d}}= (U_{(1)},U_{(2)},\dots,U_{(n)})$​ 成立.  
  其中 $U_{(1)},U_{(2)},\dots,U_{(n)}$​ 为对应于 $U_1,U_2,\dots,U_n\overset{\text{i.i.d.}}\sim \text{Uniform}(0,t)$​ 的次序统计量.  
  因此在给定 $N(t)=n$​ 的条件下:   
  $\begin{align}
  \text{E}[\sum_{i=1}^{N(t)}(t-T_i)|N(t)=n]
  &= \text{E}[\sum_{i=1}^n(t-T_i)|N(t)=n]\\
  (定理3.7)&= \text{E}[\sum_{i=1}^n(t-U_{(i)})]\\
  &= \text{E}[\sum_{i=1}^n(t-U_{i})]\\
  &= nt - \sum_{i=1}^n\text{E}[U_i]\\
  &= nt - n\frac{t}{2}\\
  &= \frac{1}{2}nt\end{align}$​  
  因此我们有:    
  $\begin{align}
  \text{E}[\sum_{i=1}^{N(t)}(t-T_i)]
  &= \text{E}\{\text{E}[\sum_{i=1}^{N(t)}(t-T_i)|N(t)]\}\\
  &= \text{E}[\frac 12 N(t)t]\\
  &= \frac{1}{2}t\text{E}[N(t)]\\
  &= \frac12 t \text{E}[\text{Poisson}(\lambda t)]\\
  &= \frac12 t \cdot \lambda t\\
  &= \frac{1}{2} \lambda t^2\end{align}$​



#### (4) Poisson 过程的拆分

我们复述 3.2.3 (3) 中**定理 3.7** 的结论:   

> **定理 3.7: (到达时刻的联合条件分布)**  
> 假设 $\mathbf N = \{N(t):t\geq 0\}$ 是参数为 $\lambda$ 的 Poisson 过程，记 $T_1,T_2,\dots$ 为到达时刻.  
> 给定 $t>0$，则有 $(T_1,T_2,\dots,T_n|N(t)=n)\overset{\mathrm{d}}= (U_{(1)},U_{(2)},\dots,U_{(n)})$   
> 其中 $U_{(1)},U_{(2)},\dots,U_{(n)}$ 为对应于 $U_1,U_2,\dots,U_n\overset{\text{i.i.d.}}\sim \text{Uniform}(0,t)$ 的次序统计量.  

**定理 3.7 的应用: (Poisson 过程的拆分, 又称时间抽样, Introduction to Probability Models 命题5.4)**    
考虑一个参数为 $\lambda>0$ 的 Poisson 过程 $\mathbf N = \{N(t):t\geq 0\}$:   
假设有 $k$ 种可能类型的事件，  
而一个事件被分类为类型 $i\ (i=1,\dots,k)$ 事件的概率依赖于事件发生的时间.  
特别地，我们假设若一个事件在时刻 $t$ 发生，  
则它将以概率 $p_i(t)$ 被分类到类型 $i\ (i=1,\dots,k)$ (独立于以前发生的任何事件)，  
其中 $\{p_i(t)\}_{i=1}^k$ 满足 $\sum_{i=1}^kp_i(t) = 1\ (\forall\ t>0)$   
利用定理 3.7，我们可以证明以下有用的命题:   
**定理 3.10: (Poisson 过程的拆分)**  
在之前的假设下，记 $N_i(t)\ (i=1,\dots,k)$ 为到时刻 $t$ 为止类型 $i$ 事件发生的个数，  
则 $N_i(t)\ (i=1,\dots,k)$ 是具有均值 $\text{E}[N_i(t)] = \lambda \int_0^t p_i(t) \mathrm{d}s$ 的**独立 Poisson 随机变量**，  
即 $N_i(t) \sim \text{Poisson}(\lambda \int_0^t p_i(t) \mathrm{d}s)$ 且它们相互独立.  
这样我们就拆分得到了一系列非齐次 Poisson 过程 $\{N_i(t):t\geq 0\}$.

- **推论: **  
  特殊地，取 $k=2$ 并令 $\begin{cases}
  p_1(s)\equiv p\\
  p_2(s)\equiv 1-p\end{cases}$   
  也就是说，只有两种可能类型的事件 (例如顾客性别为男/女)，  
  并且任意时刻 $s$ 时发生的事件分成两类的概率恒为 $p$ 和 $1-p$，  
  则我们有 $\begin{cases}
  N_1(t)\sim \text{Poisson}(\lambda p t)\\
  N_2(t)\sim \text{Poisson}(\lambda (1-p) t)\end{cases}$ 且它们相互独立，  
  即 $\{N_1(t):t\geq 0\}$ 和 $\{N_2(t):t\geq 0\}$ 分别是参数为 $\lambda p$ 和 $\lambda(1-p)$ 的独立 Poisson 过程.
- **进一步推论: **  
  特殊地，若 $p_1(\cdot),p_2(\cdot),\dots,p_k(\cdot)$ 都是常数函数，  
  则对于任意 $i=1,2,\dots,k$ 我们有 $N_i(t) = \text{Poisson}(\lambda p_i)$ 且它们相互独立，  
  即 $\{N_i(t):t\geq 0\}$ 是参数为 $\lambda p_i$ 的独立 Poisson 过程.

**定理 3.10 的证明: **
我们想要计算 $N_1(t),N_2(t),\dots,N_k(t)$ 的联合概率密度 $\text{P}\{N_i(t)=n_i,i=1,\dots,k\}$.  
为此首先注意，我们需要取条件 $N(t)=\sum_{i=1}^kn_i$ 以保证可行性:   
$\text{P}\{N_i(t)=n_i,i=1,\dots,k\} = \text{P}\{N_i(t)=n_i,i=1,\dots,k|N(t)=\sum_{i=1}^kn_i\}\times \text{P}\{N(t)=\sum_{i=1}^kn_i\}$   
现在考虑发生在区间 $[0,t]$ 中的一个任意的事件，  
如果它在时刻 $s$ 发生，则根据定理 3.10 的假设，它被分入类型 $i$ 的概率将是 $p_i(s)$.  
根据**定理 3.7** 可知，这个事件的发生时刻在 $[0,t]$ **均匀分布** (并且**独立**于其他事件)，  
因此它是类型 $i$ 事件的概率为 $P_i \overset{\Delta}=\frac{1}{t}\int_0^tp_i(s)\mathrm{d}s$.  
故条件分布 $(N_1(t),N_2(t),\dots,N_k(t)|N(t))$ 是一个 **$k$ 维多项分布** $M(k,P_1,P_2,\dots,P_k)$，  
有 $\text{P}\{N_i(t)=n_i,i=1,\dots,k|N(t)=\sum_{i=1}^kn_i\} = \frac{(\sum_{i=1}^kn_i)!}{n_1!\dots n_k!}P_1^{n_1}\dots P_k^{n_k}$   
从而有:    
$\begin{align}
\text{P}\{N_i(t)=n_i,i=1,\dots,k\} 
&= \text{P}\{N_i(t)=n_i,i=1,\dots,k|N(t)=\sum_{i=1}^kn_i\}\times \text{P}\{N(t)=\sum_{i=1}^kn_i\}\\
&= \frac{(\sum_{i=1}^kn_i)!}{n_1!\dots n_k!}P_1^{n_1}\dots P_k^{n_k} \times e^{-\lambda t} \frac{(\lambda t)^{\sum_{i=1}^kn_i}}{(\sum_{i=1}^kn_i)!}\\
&= \underset{i=1}{\overset{k}\prod} e^{-\lambda P_i t}\frac{(\lambda P_i t)^{n_i}}{n_i!}\\
&= \underset{i=1}{\overset{k}\prod} \text{P}\{\text{Poisson}(\lambda P_i t) = n_i\}\\
&= \underset{i=1}{\overset{k}\prod} \text{P}\{\text{Poisson}(\lambda\int_0^tp_i(s)\mathrm{d}s) = n_i\}\end{align}$   
这样我们就证明了 $N_i(t)\sim \text{Poisson}(\lambda\int_0^tp_i(s)\mathrm{d}s)$ 且它们相互独立.

**一个具体的例子 (奖券收集问题, Introduction to Probability Models 例5.17)**  
有 $m$ 种类型的奖券，某人每次以概率 $p_i\ (\sum_{i=1}^mp_i=1)$ 收集一张类型 $i$ 的奖券 (独立于过去收集的奖券)，    
记 $N$ 为这个人为了全套收藏 (每种类型至少一张) 所需要收藏的奖券的张数，请计算 $\text{E}[N]$:

- **Solution:**  
  假设奖券的收集过程可由 $\lambda = 1$ **(待推广至一般情况)** 的 Poisson 过程 $\{N(t):t\geq 0\}$ 表征.  
  记 $N_i(t)$ 为直到时刻 $t$ 为止收集到的类型 $i$ 的张数.  
  根据**定理 3.10**可知 $\{N_i(t):t\geq 0\}\ (i=1,\dots,m)$ 是参数为 $\lambda p_i = p_i$ 的独立 Poisson 过程.  
  记 $T_i$ 为 $\{N_i(t):t\geq 0\}$ 的**首次到达时刻**，  
  记 $T=\underset{1\leq i\leq m}\max T_i$ 为集齐全套收藏的时刻.  
  我们知道 $T_i \sim \exp(p_i)\ \ (i=1,\dots,m)$ 且相互独立，因此有:   
  $\begin{align}
  \text{P}\{T\leq t\}
  &= \text{P}\{\underset{1\leq i\leq m}\max T_i\leq t\}\\
  &= \text{P}\{T_i\leq t\text{ for all }i=1,\dots,m\}\\
  &= \underset{i=1}{\overset{m}\prod}\text{P}\{T_i\leq t\}\\
  &= \underset{i=1}{\overset{m}\prod}\text{P}\{\exp(p_i)\leq t\}\\
  &=\underset{i=1}{\overset{m}\prod} (1-e^{-p_it}) \end{align}$    
  于是 $\text{E}[T] = \int_0^{\infty}\text{P}\{T>t\}\mathrm{d}t = \int_0^{\infty}\{1-\underset{i=1}{\overset{m}\prod} (1-e^{-p_it})\}\mathrm{d}t$   
  我们记 $X_i$ 为 Poisson 过程 $\{N(t):t\geq 0\}$ 的第 $i$ 个等待时间，则我们有:   
  $\begin{align}
  \text{E}[T] 
  &= \text{E}[\sum_{i=1}^NX_i]\\
  &= \text{E}[\text{E}[\sum_{i=1}^NX_i|N]]\\
  &= \text{E}[N\text{E}[X]]\\
  &= \text{E}[N]\text{E}[X]\quad (\text{E}[X] = \text{E}[\exp(1)] = 1)\\
  &= \text{E}[N]\end{align}$   
  因此 $\text{E}[N] = \text{E}[T] = \int_0^{\infty}\{1-\underset{i=1}{\overset{m}\prod} (1-e^{-p_it})\}\mathrm{d}t$ 

**我们现在计算在最后的全套收藏中只出现一张的类型的数量的期望.**  

- **Solution:**  
  定义随机变量 $I_i$，若在最后的全套收藏中类型 $i$ 有且仅有一张，则 $I_i = 1$；否则 $I_i=0$.  
  我们的目标是计算 $\text{E}[\sum_{i=1}^mI_i] = \sum_{i=1}^m\text{E}[I_i] = \sum_{i=1}^m\text{P}\{I_i=1\}$   
  记 $S_i$ 为 $\{N_i(t):t\geq 0\}$ 的**第二个到达时刻** (首次到达时刻记为 $T_i$)   
  我们知道 $\text{P}\{I_i=1\} = \text{P}\{T_j < S_i\ \text{for all }j\neq i\}$   
  ($T_i<S_i$ 是平凡的，因此我们规定 $j\neq i$)  
  也就是说，如果每一种类型在类型 $i$ 第二次出现之前都已经出现，  
  那么最终的全套收藏中有且仅有一张类型 $i$.  

  我们知道 $S_i\sim \text{Gamma}(2,p_i)\ \ (i=1,\dots,m)$，可以推出:   
  $\begin{align}
  \text{P}\{I_i=1\}
  &= \int_0^{\infty}\text{P}\{T_j<S_i\ \text{for all }j\neq i|S_i = x\}\text{P}\{S_i=x\}\mathrm{d}x\\
  &= \int_0^{\infty}\text{P}\{T_j<x\ \text{for all }j\neq i\}\frac{p_i^2}{1!}xe^{-p_i x}\mathrm{d}x\\
  &=
  \int_0^{\infty} \underset{j\neq i}{\prod}(1-e^{-p_i x})p_i^2 xe^{-p_i x}\mathrm{d}x\end{align}$    
  因此我们有结果:   
  $\begin{align}
  \text{E}[\sum_{i=1}^mI_i] 
  &= \sum_{i=1}^m\text{E}[I_i]\\
  &= \sum_{i=1}^m\text{P}\{I_i=1\}\\
  &=  \sum_{i=1}^m\{\int_0^{\infty} \underset{j\neq i}{\prod}(1-e^{-p_i x})p_i^2 xe^{-p_i x}\mathrm{d}x \}\\
  &= \int_0^{\infty} \sum_{i=1}^m\{\underset{j\neq i}{\prod}(1-e^{-p_i x})p_i^2 xe^{-p_i x}\}\mathrm{d}x\\
  &= \int_0^{\infty} x\{\underset{j=1}{\overset{m}\prod}(1-e^{-p_i x})\}\cdot \sum_{i=1}^m p_i^2 \frac{e^{-p_ix}}{1-e^{-p_ix}}\mathrm{d}x\end{align}$

***

**有趣的探索: (Introduction to Probability Models 5.3.4 节)**  
我们想要确定一个 Poisson 过程中 $n$ 个事件的发生  
先于另一个与之独立的 Poisson 过程中 $m$ 个事件的发生的概率.  
具体来说，设 $\{N_1(t):t\geq 0\}$ 和 $\{N_1(t):t\geq 0\}$ 是分别具有速率 $\lambda_1$ 和 $\lambda_2$ 的独立 Poisson 过程.  
记 $T_{n}^{(1)}$ 和 $T_m^{(2)}$ 分别为第一个过程的第 $n$ 个事件发生的时刻、第二个过程的第 $m$ 个事件发生的时刻.  
我们想要确定 $\text{P}\{T_n^{(1)}<T_m^{(2)}\}$   

- **首先考虑 $n=m=1$ 的特殊情形: **  
  我们知道 $T_1^{(1)}\sim \exp(\lambda_1),\ T_1^{(2)} \sim \exp(\lambda_2)$，对 $T_1^{(1)}$ 取条件可知:   
  $\begin{align}
  \text{P}\{T_1^{(1)}<T_1^{(2)}\}
  &=
  \int_0^{\infty} \text{P}\{T^{(1)}_1<T_1^{(2)}|T_1^{(1)}=x\}\lambda_1 e^{-\lambda_1x}\mathrm{d}x\\
  &=
  \int_0^{\infty} \text{P}\{x<T_1^{(2)}\}\lambda_1e^{-\lambda_1 x}\mathrm{d}x\\
  &=
  \int_0^{\infty}e^{-\lambda_2 x}\lambda_1 e^{-\lambda_1 x}\mathrm{d}x\\
  &=
  \lambda_1 \int_0^{\infty}e^{-(\lambda_1+\lambda_2) x}\mathrm{d}x\\
  &=
  \frac{\lambda_1}{\lambda_1 + \lambda_2}\end{align}$ 
- **其次考虑 $n=2,\ m=1$ 的情况: **   
  (待补充)



#### (5) Poisson 过程的加和

**定理 3.11: **  
假设 $\mathbf N_i = \{N_i(t):t\geq 0\}\ (i=1,\dots,k)$ 是 $k$ 个独立的 Poisson 过程，参数分别为 $\lambda_i\ (i=1,\dots,k)$  
令 $N(t) = N_1(t)+N_2(t) + \dots + N_k(t)$   
则 $\mathbf N = \{N(t):t\geq 0\}$ 也是 Poisson 过程，参数为 $\sum_{i=1}^k\lambda_i$     

- **推论: **  
  在定理 3.11 假设下，  
  **条件分布** $(N_i(t)|N(t)=n) \overset{\mathrm{d}} = B(n,\frac{\lambda_i}{\sum_{i=1}^k\lambda_i})\ (i=1,\dots,k)$ 

**这个定理可以这样理解 (以 $k=2$ 的情况为例): ** 
两个独立的 Poisson 流在某处汇合，形成一个新的 Poisson 流 (如图所示)  
<img src="SP SZG Figure 3.2.png" style="zoom:50%;" />  
**证明: **  

- **① 证明 $N(t)\sim \text{Poisson}((\lambda_1+\lambda_2)t)$: **  
  给定 $t>0$，对于任意整数 $k\geq 0$ 都有:   
  $\begin{align}
  \text{P}\{N(t)=k\}
  &= \text{P}\{N_1(t)+N_2(t) = k\}\\
  &= \underset{i=0}{\overset{k}\sum} 
  \text{P}\{N_1(t)=i,N_2(t)=k-i\}\\
  &= \underset{i=0}{\overset{k}\sum} 
  \frac{}{i!}\\
  \end{align}$
  
  **(待补充)**



## 3.3 Poisson 过程的推广

### 3.3.1 复合 Poisson 过程

Poisson 过程是一个计数过程，用于统计一段时间内稀有事件发生的次数.  
但是在很多实际问题中，仅仅统计次数并不是我们的目的，  
例如对于表征顾客人次的 Poisson 过程 $\{N(t):t\geq 0\}$，记第 $i$ 名顾客的消费为 $\xi_i$   
我们可能更关心的是 $[0,t]$ 时间内所有顾客的消费总额 $Z(t)=\sum_{i=1}^{N(t)}\xi_i$   

**定义 3.7: (复合 Poisson 过程)**  
一个随机过程 $\mathbf Z= \{Z(t):t\geq 0\}$ 称为**复合 Poisson 过程**，  
如果它的每一项可以表示为 $Z(t)=\sum_{i=1}^{N(t)}\xi_i$，   
其中 $\mathbf N = \{N(t):t\geq 0\}$ 是一个 Poisson 过程，  
而 $\{\xi_n\}$ 是独立于 $\mathbf N$ 的一列独立同分布的随机变量.  
我们称 $Z(t)=\sum_{i=1}^{N(t)}\xi_i$ 为**复合 Poisson 随机变量**.

- 特殊地，若 $\xi$ 的分布为在点 $1$ 处的**退化分布**，  
  则复合 Poisson 过程 $\mathbf Z= \{Z(t):t\geq 0\}$ 退化为 Poisson 过程 $\mathbf N = \{N(t):t\geq 0\}$ 
- 特殊地，若 $\xi$ 的分布为 **Bernoulli 分布** $B(1,p)$，  
  则复合 Poisson 过程 $\mathbf Z= \{Z(t):t\geq 0\}$ 退化为参数为 $\lambda p$ 的 Poisson 过程 $\mathbf N_1 = \{N_1(t):t\geq 0\}$   
  (回忆 3.2.3 (4) Poisson 过程的拆分)

由于 $Z(t)$ 的分布依赖于随机变量 $\xi_1,\xi_2,\dots$，  
故一般情况下很难显式地计算出 $Z(t)$ 的分布.  
但在一定条件下，可以计算出 $Z(t)$ 的**数学期望**和**方差**.

***

**引理 3.12: (复合随机变量的期望和方差, 这个第一章出现过, 可省略证明)** 
设 $X_1,X_2,\dots$ 是一列独立同分布的随机变量，具有均值 $\text{E}[X]=\mu$ 和方差 $\text{Var}(X)=\sigma^2$   
假设它们与取非负整数值的随机变量 $N$ 独立.  
我们称 $S= \sum_{i=1}^N X_i$ 为**复合随机变量**，有 $\begin{cases}
\text{E}[S] = \mu \text{E}[N]\\
\text{Var}[S] = \sigma^2 \text{E}[N] + \mu^2\text{Var}(N)\end{cases}$   成立.

- 特殊地，如果 $N \sim \text{Possion}(\lambda)$  
  则我们称 $S= \sum_{i=1}^N X_i$ 为**复合 Poisson 随机变量** 
  根据 $\begin{cases}
  \text{E}[S] = \mu\text{E}[N]\\
  \text{Var}(S) = \sigma^2 \text{E}[N] + \mu^2\text{Var}(N)\\
  \text{E}[N] = \text{Var}[N]=\lambda\\
  \text{E}[X^2] = \text{Var}(X) + (\text{E}[X])^2 = \sigma^2 + \mu^2\end{cases}$  
  可知 $\begin{cases}
  \text{E}[S] = \mu\lambda = \lambda \text{E}[X]\\
  \text{Var}(S) = (\sigma^2+ \mu^2)\lambda =\lambda \text{E}[X^2]\end{cases}$ 

**证明: **

- ① 计算 $S= \sum_{i=1}^N X_i$ 的**期望**:     
  注意到:   
  $\begin{align}
  \text{E}[S|N=n]&=\text{E}[\sum_{i=1}^N X_i|N=n]\\
  &= \text{E}[\sum_{i=1}^n X_i | N=n]\\
  &= \text{E}[\sum_{i=1}^n X_i]\quad (N\ \bot\ X_i,\forall \ i=1,2,\dots)\\
  &= \sum_{i=1}^n\text{E}[X_i]\\
  &= n\mu  
  \end{align}$    
  因此有:    
  $\begin{align}
  \text{E}[S] 
  &= \text{E}[\sum_{i=1}^N X_i]\\
  &= \text{E}[\text{E}[\sum_{i=1}^N X_i|N]]\\
  &= \text{E}[N\mu]\\
  &= \mu \text{E}[N]\end{align}$ 
- ② 计算 $S= \sum_{i=1}^N X_i$ 的**方差**:    
  注意到:   
  $\begin{align}
  \text{Var}(S|N=n)
  &= \text{Var}(\sum_{i=1}^NX_i |N=n)\\
  &= \text{Var}(\sum_{i=1}^nX_i |N=n)\\
  &= \text{Var}(\sum_{i=1}^nX_i)\quad (N\ \bot\ X_i,\forall \ i=1,2,\dots)\\
  &= \sum_{i=1}^n \text{Var}(X_i)\\
  &= n\sigma^2\end{align}$  
  因此有:   
  $\begin{align}
  \text{Var}(S) 
  &= \text{E}(\text{Var}(S|N)) + \text{Var}(\text{E}(S|N))\\
  &= \text{E}[N\sigma^2] + \text{Var}(N\mu)\\
  &= \sigma^2 \text{E}[N] + \mu^2\text{Var}(N)\end{align}$   

****

**定理 3.13: (复合 Poisson 过程的性质)**  
假设 $\mathbf N=\{N(t):t\geq 0\}$ 是参数为 $\lambda>0$ 的 Poisson 过程，  
$\xi_1,\xi_2,\dots$ 是一列独立同分布的随机变量，均值为 $\mu$，方差为 $\sigma^2$，且独立于 $\mathbf N$.  
定义复合 Poisson 过程 $\mathbf Z=\{Z(t)=\sum_{i=1}^{N(t)}\xi_i:t\geq 0\}$，则有: 

- ① **均值函数**和**方差函数**: $\begin{cases}
  \text{E}[Z(t)] = \mu \lambda t\\
  \text{Var}[Z(t)] = (\mu^2 + \sigma^2)  \lambda t\end{cases}$
- ② $\mathbf Z=\{Z(t)=\sum_{i=1}^{N(t)}\xi_i:t\geq 0\}$ 具有**独立平稳增量性**   
  (我们将其与 Poisson 过程的性质比较发现，复合 Poisson 过程失去了**稀有性**)
  - **独立增量: **  
    对于任意 $0<s<t$ 都有 $Z(t)-Z(s)\ \bot\ Z(s)$ 相互独立.
  - **平稳增量:  ** 
    对于任意 $0<s<t$ 都有 $Z(t)-Z(s) \overset{\mathrm{d}}= Z(t-s)$ 成立.  

**证明: **  

- **① 可由引理 3.12 直接得到: **  
  $\begin{cases}
  \text{E}[Z(t)] = \text{E}[N(t)]\text{E}[\xi]\\
  \qquad\ \ \ \ =
  \lambda t \cdot \mu\\
  \qquad\ \ \ \ = \mu\lambda t\\
  \text{Var}[Z(t)] = \sigma^2 \text{E}[N] + \mu^2\text{Var}(N)\\
  \qquad\ \ \ \ \ \ \ \ =
  \sigma^2\cdot \lambda t+ \mu^2 \cdot\lambda t\\
  \qquad\ \ \ \ \ \ \ \ = (\mu^2+\sigma^2)\lambda t\end{cases}$  
- **② 增量独立性: **  
  记 $\xi$ 的特征函数为 $\varphi(u) = \text{E}[e^{iu\xi}]\ (u\in\mathbb R)$  
  对于任意 $0<s<t$ 和 $u,v\in \mathbb R$，  
  利用 $\mathbf N = \{N(t):t\geq 0\}$ 和 $\{\xi_n\}$ 的独立性，我们有:   
  $\begin{align}
  \text{E}[e^{iu[Z(t)-Z(s)]+iv\text{Z}(s)}]
  &= \text{E}[e^{iu\underset{i=N(s)+1}{\overset{N(t)}\sum}\xi_i+iv\underset{i=1}{\overset{N(s)}\sum}\xi_i}]\\
  &= \text{E}\{\text{E}[e^{iu\underset{i=N(s)+1}{\overset{N(t)}\sum}\xi_i+iv\underset{i=1}{\overset{N(s)}\sum}\xi_i}|\mathbf N]\}\\
  &= \text{E}[\varphi(u)^{N(t)-N(s)}\cdot \varphi(v)^{N(s)}]\qquad (N(t)-N(s)\ \bot\ N(s))\\
  &= \text{E}[\varphi(u)^{N(t)-N(s)}]\cdot \text{E}[\varphi(v)^{N(s)}]\\
  &= \text{E}[e^{iu[Z(t)-Z(s)]}]\cdot \text{E}[e^{ivZ(s)}]\end{align}$    
  因此 $Z(t)-Z(s)\ \bot\ Z(s)$
- **③ 增量平稳性: **  
  和 ② 类似地，记 $\xi$ 的特征函数为 $\varphi(u) = \text{E}[e^{iu\xi}]\ (u\in\mathbb R)$   
  对于任意 $0<s<t$ 和 $u\in \mathbb R$，我们有:   
  $\begin{align}
  \text{E}[e^{iu[Z(t)-Z(s)]}]
  &= \text{E}[e^{iu\underset{i=N(s)+1}{\overset{N(t)}\sum}\xi_i}]\\
  &= \text{E}\{\text{E}[e^{iu\underset{i=N(s)+1}{\overset{N(t)}\sum}\xi_i}|\mathbf N]\}\\
  &= \text{E}[\varphi(u)^{N(t)-N(s)}]\qquad (N(t)-N(s)\overset{\mathrm{d}}= N(t-s))\\
  &= \text{E}[\varphi(u)^{N(t-s)}]\\
  &= \text{E}[e^{iuZ(t-s)}]\end{align}$  

**计算 $Z(t)$ 的分布: **   

- **① 累积分布函数: **  
  $\begin{align}
  \text{P}\{Z(t)> x\}
  &=
  \text{P}\{\sum_{i=1}^{N(t)}\xi_i > x\}\\
  &=
  \underset{n=0}{\overset{\infty}\sum}
  \text{P}\{\sum_{i=1}^{N(t)}\xi_i > x|N(t)=n\}
  \text{P}\{N(t)=n\}\\
  &=\underset{n=0}{\overset{\infty}\sum}
  \text{P}\{\sum_{i=1}^n\xi_i > x\}
  \text{P}\{N(t)=n\}
  \end{align}$   
  我们知道 $\text{P}\{N(t)=n\}=\text{P}\{\text{Poisson}(\lambda t) = n\} = e^{-\lambda t}\frac{(\lambda t)^n}{n!}$   
  如果我们还知道 $\sum_{i=1}^n\xi_i$ 的分布，就能确定 $Z(t)$ 的累计分布函数.  
  例如 $\{\xi_i\}\overset{\text{i.i.d.}}\sim B(1,p)$ 时 $\sum_{i=1}^n\xi_i\sim B(n,p)$   
  例如 $\{\xi_i\}\overset{\text{i.i.d.}}\sim \text{exp}(\beta)$ 时 $\sum_{i=1}^n\xi_i\sim \text{Gamma}(n,\beta)$
- **② 特征函数 (或矩母函数): **  
  $\begin{align}
  \varphi_{Z(t)}(a)
  &=
  \text{E}[e^{iaZ(t)}]\\
  &=
  \text{E}[e^{ia\sum_{i=1}^{N(t)}\xi_i}]\\
  &= 
  \text{E}[\text{E}[e^{ia\sum_{i=1}^{N(t)}\xi_i}]|\mathbf N]\\
  &= 
  \text{E}[\varphi_\xi(a)^{N(t)}]\\
  &=
  \text{E}[e^{\log(\varphi_\xi(a))N(t)}]\\
  &=
  M_{N(t)}(\log(\varphi_\xi (a)))\qquad(M_{\text{Poisson}(\lambda)}(a) = e^{\lambda(e^a-1)})\\
  &=
  e^{\lambda t(e^{\log {(\varphi_\xi(a))}}-1)}\\
  &=
  e^{\lambda t(\varphi_\xi(a)-1)}\end{align}$  
  类似地，对于矩母函数，我们有:   
  $M_{Z(t)}(a) = e^{\lambda t(M_\xi(a)-1)}$   
  如果我们确定了 $\xi$ 的分布，就能确定 $Z(t)$ 的特征函数，等价于确定了 $Z(t)$ 的分布.

****

**一个具体的例子: **  
某零件在运行过程中受到撞击，用 $N(t)$ 表示 $[0,t]$ 时间段内该零件受到的撞击次数.  
设 $\mathbf N = \{N(t):t\geq 0\}$ 是参数为 $\lambda>0$ 的 Poisson 过程.  
每次撞击会给零件带来一定的磨损，磨损量是参数为 $\beta>0$ 的指数随机变量.  
设各次磨损量与撞击次数相互独立.  
假设当磨损总量 $\geq\alpha>0$ 时需要更换零件，求该零件的平均寿命？  

- 在求解这个问题之前，我们先给出两个引理:     

  - **引理 3.14: (Tonelli 定理, Fubini 定理的一个特殊情况)**  
    设 $(X,\mathcal A,\mu)$ 和 $(Y,\mathcal B,\nu)$ 是两个测度空间，  
    $f:X\times Y\to [0,\infty]$ 是一个**非负** $\mathcal A\times \mathcal B$-可测函数.  
    则有:   
    - 对于任意给定的 $x\in X$，函数 $y\mapsto f(x,y)$ 都是 $\mathcal B$-可测的.
    - 对于任意给定的 $y\in Y$，函数 $x\mapsto f(x,y)$ 都是 $\mathcal A$-可测的.
    - 函数 $x\mapsto \int_Yf(x,y)d\nu(y)$ 是 $\mathcal A$-可测的，函数 $y\mapsto \int_Xf(x,y)d\mu(x)$ 是 $\mathcal B$-可测的.
    - 两个迭代积分是相等的 (即积分可交换顺序):   
      $\int_X(\int_Y f(x,y)d\nu(y))d\mu(x) = \int_Y (\int_Xf(x,y)d\mu(x))d\nu(y)$ 

  - **引理 3.15: **  
    对于任意**非负**随机向量 $X$ (无论是离散的、连续的，还是这两者的混合形式)  
    我们都有 $X=\int_0^X 1 \mathrm{d}t = \int_0^\infty \mathbf 1_{\{X>t\}}(t)\mathrm{d}t$  
    应用**引理 3.14 (Tonelli 定理)** 得到:   
    $\begin{align}
    \text{E}[X] 
    &= \text{E}[\int_0^\infty \mathbf 1_{\{X>t\}}(t)\mathrm{d}t]\\
    &= \int_0^{\infty} \text{E}[\mathbf 1_{\{X>t\}}(t)]\mathrm{d}t\\
    &= \int_0^{\infty} \text{P}\{X>t\}\mathrm{d}t
    \end{align}$    
    (由于指示函数 $\mathbf 1_{\{X>t\}}(t)$ 是非负的，Tonelli 定理允许我们交换期望和积分的顺序)
  

**Solution: **  
记零件寿命为 $T$.  
对于任意 $t>0$，我们有 $T>t\ \Leftrightarrow\ Z(t)<\alpha$   
因此有 $\text{P}\{T>t\} = \text{P}\{Z(t)<\alpha\}$  
根据**引理 3.15** 有:   
$\begin{align}
\text{E}[T]
&= \int_0^{\infty}\text{P}\{T>t\}\mathrm{d}t\\
&= \int_0^{\infty} \text{P}\{Z(t)<\alpha\}\mathrm{d}t\\
&= \int_0^{\infty} \text{P}\{\sum_{i=1}^{N(t)}\xi_i<\alpha\}\mathrm{d}t\end{align}$   
根据 $\mathbf N = \{N(t):t\geq 0\}$ 和 $\{\xi_n\}$ 相互独立，我们有: (全概率公式)  
$\begin{align}
\text{P}\{\sum_{i=1}^{N(t)}\xi_i<\alpha\}
&= 
\underset{n=0}{\overset{\infty}\sum}\text{P}\{\sum_{i=1}^{N(t)}\xi_i<\alpha|N(t)=n\}\text{P}\{N(t)=n\}\\
&=
\underset{n=0}{\overset{\infty}\sum}\text{P}\{\sum_{i=1}^n\xi_i<\alpha\}\text{P}\{N(t)=n\}\end{align}$  
记 $\tilde N(t) = \max \{n\geq 0:\sum_{i=1}^n\xi_i \leq t\}$ (注意到 $\{\xi_n\}\overset{\text{i.i.d.}}\sim \exp(\beta)$)  
根据 **Poisson 过程的第 $3$ 个等价定义** (定义 3.6)   
可知 $\mathbf {\tilde N} = \{\tilde N(t):t\geq 0\}$ 是参数为 $\beta>0$ 的 Poisson 过程.  
于是有:   
$\begin{align}
\text{P}\{\sum_{i=1}^n\xi_i <\alpha\}
&= \text{P}\{\tilde N(\alpha)\geq n\}\\
&= \underset{k=n}{\overset{\infty}\sum}\text{P}\{\text{Poisson}(\beta \alpha) = k\}\\
&= \underset{k=n}{\overset{\infty}\sum} e^{-\alpha\beta}\frac{(\alpha\beta)^k}{k!}\end{align}$   
因此我们有:   
$\begin{align}
\text{E}[T]
&= \int_0^{\infty} \text{P}\{\sum_{i=1}^{N(t)}\xi_i<\alpha\}\mathrm{d}t\\
&= \int_0^{\infty} \underset{n=0}{\overset{\infty}\sum}\text{P}\{\sum_{i=1}^n\xi_i<\alpha\}\text{P}\{N(t)=n\}\mathrm{d}t\\
&= \int_0^{\infty} \underset{n=0}{\overset{\infty}\sum} \{\underset{k=n}{\overset{\infty}\sum} \{e^{-\alpha\beta}\frac{(\alpha\beta)^k}{k!}\}\cdot e^{-\lambda t}\frac{(\lambda t)^n}{n!}\}\mathrm{d}t\\
&= \frac{1}{\lambda}\underset{n=0}{\overset{\infty}\sum} \underset{k=n}{\overset{\infty}\sum} \frac{\alpha^k\beta^k}{k!}e^{-\alpha\beta} 
\int_0^{\infty} \frac{\lambda^{n+1}}{n!}t^ne^{-\lambda t}\mathrm{d}t\\
&= \frac{1}{\lambda}\underset{n=0}{\overset{\infty}\sum} \underset{k=n}{\overset{\infty}\sum} \frac{\alpha^k\beta^k}{k!}e^{-\alpha\beta}
\int_0^{\infty} \text{P}\{\text{Gamma}(n+1,\lambda) = t\}\mathrm{d}t\\
&= \frac{1}{\lambda}\underset{n=0}{\overset{\infty}\sum} \underset{k=n}{\overset{\infty}\sum} \frac{\alpha^k\beta^k}{k!}e^{-\alpha\beta}\\
&= \frac{1}{\lambda}[\underset{n=0}{\overset{\infty}\sum}\frac{\alpha^n\beta^n}{n!}e^{-\alpha\beta}+ \underset{n=0}{\overset{\infty}\sum} \underset{k=n+1}{\overset{\infty}\sum} \frac{\alpha^k\beta^k}{k!}e^{-\alpha\beta}]\\
&= \frac{1}{\lambda}[\underset{n=0}{\overset{\infty}\sum}  
\text{P}\{\text{Poisson}(\alpha\beta)=n\}  + 
\underset{n=0}{\overset{\infty}\sum} \text{P}\{\text{Poisson}(\alpha\beta)\geq n+1\}]\qquad(引理3.15)\\
&= \frac{1}{\lambda} [1 + \text{E}[\text{Poisson}(\alpha\beta)]]\\
&= \frac{1}{\lambda} (1+ \alpha\beta)\end{align}$



### 3.3.2 非齐次 Poisson 过程

**(非齐次 Poisson 过程, Nonhomogeneous Poisson Process)**  
计数过程 $\mathbf N= \{N(t):t\geq 0\}$ 称为具有**强度函数** $\lambda (t)\ (t\geq 0)$ 的非齐次 Poisson 过程，  
如果 $\mathbf N= \{N(t):t\geq 0\}$ 满足:   

- **初始条件: **$N(0)=0$
- **增量独立性: **对于任意 $0<s<t$ 都有 $N(t)-N(s)\ \bot\ N(s)$
- **稀有性: **  
  对于任意 $t>0$ 都有 $\begin{cases}
  \text{P}\{N(t,t+\text{d} t)=0\} = (1-\lambda(t)) \text{d}t + o(\text{d}t)\\
  \text{P}\{N(t,t+\text{d} t)=1\} = \lambda(t) \text{d}t + o(\text{d}t)\\
  \text{P}\{N(t,t+\text{d}t)\geq 2\} = o(\text{d} t)\end{cases}$   
  其中 $\text{d}t$ 是一个相对小的时间增量，   
  而 $o(\text{d}t)$ 代表比 $\text{d}t$ 高阶的无穷小，即有 $\underset{\text{d}t\to 0}{\lim} \frac{o(\text{d}t)}{\text{d}t} = 0$   

我们定义非齐次 Poisson 过程的**均值函数**为 $m(t) = \int_0^t \lambda(v)dv$  
给出这样一个定义的理由表述在以下的重要定理中.  

**定理 $3.16$ (非齐次 Poisson 过程增量的分布, Introduction to Probability Models 定理 $5.3$)**   
若 $\{N(t):t\geq 0\}$ 是强度函数为 $\lambda(t)\ (t\geq 0)$ 的非齐次 Poisson 过程，  
则对于任意 $s,t\geq 0$，$N(t+s)-N(s)$ 都是均值为 $m(t+s)-m(s) = \int_s^{t+s}\lambda(u)du$ 的 Poisson 随机变量. 

- **证明: **     
  定义 $\{N(t):t\geq 0\}$ 的 Laplace 变换为 $\text{E}[e^{-u N(t)}]$   
  任意给定 $u>0$，记 $g(t) = \text{E}[e^{-u N(t)}]$，我们有:   
  $\begin{align}
  g(t+h)
  &= \text{E}[e^{-u N(t+h)}]\\
  &= \text{E}[e^{-uN(t)}e^{-u (N(t+h)-N(t))}]\quad (\bold{增量独立性})\\
  &= \text{E}[e^{-uN(t)}]\cdot \text{E}[e^{-u(N(t+h)-N(t))}]\\
  &= g(t)\text{E}[e^{-uN_t(h)}]\end{align}$    
  其中 $N_t(h) = N(t+h)-N(t) = N(t,t+h)$  
  由**稀有性**可知 $\begin{cases}
  \text{P}\{N_t(h)=0\}=\text{P}\{N(t,t+h)=0\} = (1-\lambda(t)) h + o(h)\\
  \text{P}\{N_t(h)=1\}=\text{P}\{N(t,t+h)=1\} = \lambda(t) h + o(h)\\
  \text{P}\{N_t(h)\geq 2\}=\text{P}\{N(t,t+h)\geq 2\} = o(h)\end{cases}$    
  因此对 $N_t(h)=0,\ N_t(h) = 1,\ N_t(h)\geq 2$ 取条件，根据**全期望公式**可得:   
  $\begin{align}
  \text{E}[e^{-u N_t(h)}]
  &= e^{-u\cdot 0}(1-\lambda(t))h + e^{-u\cdot 1}\lambda(t) h + o(h)\\
  &= 1-\lambda(t) h + e^{-u}(\lambda(t) h) + o(h)\\
  &= 1 + \lambda(t) h(e^{-u}-1)\end{align}$   
  于是我们有 $g(t+h) = g(t)\text{E}[e^{-u N_t(h)}] = g(t)[1+\lambda(t) h(e^{-u}-1)] + o(h)$   
  由此推出:    
  $\begin{align}
  g'(t) 
  &= \underset{h\to 0}{\lim} \frac{g(t+h)-g(t)}{h}\\
  &= \underset{h\to 0}{\lim} [g(t)\lambda(t)(e^{-u}-1) + \frac{o(h)}{h}]\\
  &= g(t)\lambda(t) (e^{-u}-1)\end{align}$  
  于是有 $\frac{d}{\mathrm{d}t}\log(g(t)) = \frac{g'(t)}{g(t)}= \lambda(t) (e^{-u}-1)$   
  两边从 $0$ 到 $t$ 求积分后得到 $\log(g(t))- \log(g(0)) = (e^{-u}-1)\int_0^t \lambda (v)dv$   
  根据 $g(0) = \text{E}[e^{-uN_t(0)}] = \text{E}[e^{-u\cdot 0}] = 1$ 和 $m(t) = \int_0^t \lambda(v)dv$，  
  由上式得出: $g(t) = e^{m(t) (e^{-u}-1)}$    
  因此对于任意 $\begin{cases}
  u>0\\
  t\geq 0\end{cases}$ 都有 $\text{E}[e^{-uN(t)}] = g_u(t) = e^{m(t) (e^{-u}-1)}$ (其中 $g_u$ 即固定 $u$ 时的函数 $g$)

  **考虑 $X\sim \text{Poisson}(m(t))$ 的 Laplace 变换: **   
  $\begin{align}
  \text{E}[e^{-uX}]
  &= \underset{i=0}{\overset{\infty}\sum} e^{-ui} \cdot e^{-m(t)}\frac{(m(t))^i}{i!}\\
  &= e^{-m(t)}\underset{i=0}{\overset{\infty}\sum} \frac{(m(t)e^{-u})^i}{i!}\\
  &= e^{-m(t)} e^{m(t) e^{-u}}\\
  &= e^{m(t)(e^{-u}-1)}\end{align}$   
  由于 Laplace 变换唯一地确定了分布，因此我们得出结论: $N(t)\sim \text{Poisson}(m(t))$     
  (老师通过生成函数 $\text{E}[z^{N(t)}]=e^{m(t)(z-1)}$ 证明 $N(t)\sim \text{Poisson}(m(t))$，殊途同归)
  
  **进一步，对于任意 $0<s<t$，**  
  记 $N_s(t) = N(s+t)-N(s)$，   
  注意到计数过程 $\{N_s(t),t\geq 0\}$ 是强度函数为 $\lambda_s(t) = \lambda(s+t)\ (t>0)$ 的非齐次 Poisson 过程.  
  因此 $N_s(t)$ 是具有均值 $\int_0^t\lambda_s(v)dv = \int_0^t \lambda(s+v)dv = \int_s^{s+t} \lambda(v)dv = m(s+t)-m(s)$   
  的 Poission 随机变量.  
  这就证明了结论 $N(s+t)-N(s) \sim \text{Poisson}(m(s+t)-m(s))$.  
  (老师通过生成函数 $\text{E}[z^{N(s+t)}] = \text{E}[z^{(N(s)+N(s+t)-N(s))}] = \text{E}[z^{N(s)}]\text{E}[z^{(N(s+t)-N(s))}]$   
   得到 $\text{E}[z^{N(s+t)-N(s)}] = e^{[m(s+t)-m(s)](z-1)}$   
   证明 $N(s+t)-N(s) \sim \text{Poisson}(m(s+t)-m(s))$，殊途同归)

<img src="S.Moss Theorem 5.3.png" style="zoom:67%;" />



**我们可以从一个通常意义的 (齐次) Poisson 过程 (通过时间抽样) 生成一个非齐次的 Poisson 过程.**  
设 $\{N(t):t\geq 0\}$ 是一个速率为 $\lambda$ 的齐次 Poisson 过程，  
并且假设在时刻 $t$ 发生的一个事件，以独立于先前事件的概率 $p(t)$ 计数，  
记这个计数过程为 $\{N_c(t):t\geq 0\}$   
则它是一个强度函数为 $\lambda(t) = \lambda p(t)$ 的非齐次 Poisson 过程.  
我们来验证 $\{N_c(t):t\geq 0\}$ 满足非齐次 Poisson 过程的定义: 

- ① 初始条件: $N_c(0) = 0$

- ② 增量独立性:   
  在 $(s,s+t)$ 时间中被计数的事件数 $N_c(t+s)-N_c(s)$   
  只依赖于 $(s,s+t)$ 时间中的事件发生数 $N(t+s)-N(s)$    
  因此独立于早于 $s$ 发生的事件 (数量为 $N(s)$)，  
  于是也独立于早于 $s$ 发生且被计数的事件 (数量为 $N_c(s)$)，  
  从而建立了增量的独立性.

- ③ 稀有性:   

  - 定义 $N_c(t,t+h) = N_c(t+h)-N(t)$    
    $\text{P}\{N_c(t,t+h)\geq 2\} \leq \text{P}\{N(t,t+h)\geq 2\} = o(h)$
  - 为计算 $\text{P}\{N_c(t,t+h) = 1\}$，我们取条件于 $N(t,t+h)$:   
    $\begin{align}
    &\text{P}\{N_c(t,t+h) = 1\}\\
    &=
    \text{P}\{N_c(t,t+h)=1|N(t,t+h)=1\} \text{P}\{N(t,t+h)=1\} \\
    &\quad+ 
    \text{P}\{N_c(t,t+h)=1|N(t,t+h)\geq 2\} \text{P}\{N(t,t+h)\geq 2\}\\
    &= 
    \text{P}\{N_c(t,t+h)=1|N(t,t+h)=1\}\lambda h + o(h)\\
    &=
    p(t)\lambda h + o(h)\end{align}$

  因此稀有性也是满足的.

非齐次 Poisson 过程的重要性在于:   
**我们不再需要增量平稳性这个条件**  
现在我们可以认为事件在某些时间比在其他时间更可能发生.

***

**一个具体的例子:  (Introduction to Probability Models 例 5.24)**  
一家热狗店从上午 $8$ 点开始营业，  
从每小时 $5$ 个顾客的速率稳定增长到上午 $11$ 点的每小时 $20$ 个顾客的速率，  
并维持该速率到下午 $1$ 点，之后稳定下降到下午 $5$ 点的每小时 $12$ 个顾客的速率.  
假设不相交的时间段的顾客数是独立的，  
请你建立合适的模型，  
并计算上午 $8:30$ 至上午 $9:30$ **没有顾客的概率**和**平均顾客数**.  

- **解答: **  
  我们构建一个非齐次的 Poisson 过程 $\{N(t):t\geq 0\}$，  
  假定这个过程开始于午夜零时，则其强度函数为:
  $$
  \lambda(t) = \begin{cases}
  0,&0\leq t<8\\
  5+5(t-8),&8\leq t\leq 11\\
  20,&11\leq t\leq 13\\
  20-2(t-13), &13\leq t\leq 17\\
  0,&17<t<24\\
  \lambda(t-24),&t\geq 24\end{cases}
  $$
  上午 $8:30$ 至上午 $9:30$ 之间到达的顾客数 $N(9.5)-N(8.5)$ 是 Poisson 随机变量，  
  其参数为 $m(9.5)-m(8.5) = \int_{8.5}^{9.5}\lambda(t)\mathrm{d}t = \int_{8.5}^{9.5} [5+5(t-8)]\mathrm{d}t = 10$   
  因此我们有: 
  
  - 没有顾客的概率 $\text{P}\{N(9.5)-N(8.5)=0\} = e^{-10}\frac{(10)^0}{0!}\approx 0.000045$ 
  
  - 平均顾客数 $\text{E}[N(9.5)-N(8.5)] = 10$   

****

**现在我们从另一个角度了解齐次 Poisson 过程如何 (通过时间抽样) 生成非齐次 Poisson 过程: **  
设 $\{N(t):t\geq 0\}$ 是一个速率为 $\lambda$ 的齐次 Poisson 过程，  
并且假设在时刻 $t$ 发生的一个事件，  
分别以概率 $p_1(t),p_2(t),\dots,p_m(t)$ 分类为类型 $1,2,\dots,m$ (满足 $\sum_{i=1}^mp_i(t) = 1\ (\forall\ t>0)$)
我们记 $\{N_i(t):t\geq 0\}$ 为类型 $i$ 事件的计数过程，  
则容易验证 $\{N_i(t):t\geq 0\}\ (i=1,\dots,m)$ 分别是具有强度函数 $\lambda_i(t) = \lambda p_i(t)\ (i=1,\dots,m)$   
的相互独立的非齐次 Poisson 过程.

**(Introduction to Probability Models 例 $5.25$)**  
我们记一个具有无穷条服务线、Poisson 到达、服务时间分布 $G$ 的排队系统为 $M/G/\infty$ 系统.  
我们想要证明其**顾客离开的计数过程** $\{N(t):t\geq 0\}$为具有强度函数 $\lambda(t)=\lambda G(t)$ 的非齐次 Poisson 过程.  

- **证明: ** 

  - **首先论证离开过程具有增量独立性: **  
    考虑不相交的区间 $O_1,\dots,O_k$，  
    我们称一个到达是类型 $i$ 的，如果这个到达在区间 $O_i$ 离开.  
    根据**定理 3.10** 可知在这些区间中离开的个数是相互独立的，这就建立了增量独立性.

    > **定理 3.10: (Poisson 过程的拆分)**
    > 考虑一个参数为 $\lambda>0$ 的 Poisson 过程 $\mathbf N = \{N(t):t\geq 0\}$:   
    > 假设有 $k$ 种可能类型的事件，  
    > 而一个事件被分类为类型 $i\ (i=1,\dots,k)$ 事件的概率依赖于事件发生的时间.  
    > 特别地，我们假设若一个事件在时刻 $t$ 发生，  
    > 则它将以概率 $p_i(t)$ 被分类到类型 $i\ (i=1,\dots,k)$ (独立于以前发生的任何事件)，  
    > 其中 $\{p_i(t)\}_{i=1}^k$ 满足 $\sum_{i=1}^kp_i(t) = 1\ (\forall\ t>0)$   
    > 记 $N_i(t)\ (i=1,\dots,k)$ 为到时刻 $t$ 为止类型 $i$ 事件发生的个数，  
    > 则 $N_i(t)\ (i=1,\dots,k)$ 是具有均值 $\text{E}[N_i(t)] = \lambda \int_0^t p_i(t) \mathrm{d}s$ 的**独立 Poisson 随机变量**，  
    > 即 $N_i(t) \sim \text{Poisson}(\lambda \int_0^t p_i(t) \mathrm{d}s)$ 且它们相互独立，  
    > 即 $\{N_i(t):t\geq 0\}$ 是参数为 $\lambda \int_0^t p_i(t) \mathrm{d}s$ 的**独立 Poisson 过程**.

  - **现在论证离开过程的计数是 Poisson 随机变量，且过程具有稀有性: **  
    考虑区间 $(t,t+h)$ 和在时间 $s<t+h$ 的一个到达.  
    这个到达被计数的概率是 $p(s)=\begin{cases}
    G(t+h-s), & \text{if}\ \ t<s<t+h\\
    G(t+h-s) - G(t-s), &\text{if}\ \ s\leq t\end{cases}$   
    因此由**定理 3.10** 可知:   
    在 $(t,t+h)$ 中离开的个数 $N(t+h)-N(t)$ 是 Poisson 随机变量，且具有均值:   
    $\begin{align}
    \lambda \int_0^{t+h}p(s)\mathrm{d}s 
    &= \lambda \int_0^{t+h} G(t+h-s)\mathrm{d}s - \lambda \int_0^t G(t-s)\mathrm{d}s\\
    &= \lambda \int_{0}^{t+h}G(u)du  - \lambda \int_0^t G(u)du\\  
    &= \lambda \int_t^{t+h} G(u)du\\
    &= \lambda G(t)h + o(h)\end{align}$   
    因此我们有:   
    
    - 计算 $\text{P}\{N(t+h)-N(t) = 1\}$:
      $\begin{align}
      \text{P}\{N(t+h)-N(t) = 1\} 
      &= e^{-\lambda G(t)h +o(h)} \frac{(\lambda G(t)h +o(h))^1}{1!}\\ 
      &= \lambda G(t)h\cdot e^{-\lambda G(t)h}+ o(h)\\
      &= \lambda G(t)h + o(h)\end{align}$
    - 计算 $\text{P}\{N(t+h)-N(t) = 0\}$:  
      $\begin{align}
      \text{P}\{N(t+h)-N(t) \geq 2\} 
      &= e^{-\lambda G(t)h +o(h)} \underset{k=2}{\overset{\infty}\sum}\frac{(\lambda G(t)h +o(h))^k}{k!}\\ 
      &= o(h)\end{align}$
  
  综上所述，**顾客离开的计数过程** $\{N(t):t\geq 0\}$为具有强度函数 $\lambda(t)=\lambda G(t)$ 的非齐次 Poisson 过程.
  

****

**非齐次 Poisson 过程的到达时刻: **  
考虑一个均值函数为 $m(t) = \int_0^t\lambda(u)du$ 的非齐次 Poisson 过程 $\{N(t):t\geq 0\}$  
记第 $n$ 个事件的到达时刻记为 $T_n$，  
我们可以得到它的概率密度函数如下:   
$\begin{align}
\text{P}\{t<T_n<t+h\}
&= 
\text{P}\{N(t) = n-1,N(t+h)-N(t) = 1\} + o(h)\qquad (增量独立性)\\
&=
\text{P}\{N(t) = n-1\}\text{P}\{N(t+h)-N(t) = 1\} + o(h)\\
&=
e^{-m(t)}\frac{[m(t)]^{n-1}}{(n-1)!}[\lambda(t)h + o(h)] + o(h)\\
&=
\lambda(t)e^{-m(t)}\frac{[m(t)]^{n-1}}{(n-1)!}h + o(h)\end{align}$   
于是有 $f_{T_n}(t) =\underset{h\to 0}{\lim} \text{P}\{t<T_n<t+h\} =\lambda(t)e^{-m(t)}\frac{[m(t)]^{n-1}}{(n-1)!}$ 

- 特殊地，如果 $\begin{cases}
  \lambda(t)\equiv \lambda\\
  \text{E}[N(t)] = m(t) =\int_0^t\lambda du= \lambda t\end{cases}\ (\forall\ t>0)$   
  则 $f_{T_n}(t) = \lambda e^{-\lambda t}\frac{(\lambda t)^{n-1}}{(n-1)!} = 
  \frac{\lambda ^n}{\Gamma(n)}t^{n-1}e^{-\lambda t} = \text{P}\{\text{Gamma}(n,\lambda)=t\}$    
  此时 $T_n\sim \text{Gamma}(n,\lambda)$ 

**非齐次 Poisson 过程的等待时间: **  
记第 $n$ 个等待时间为 $X_n$，  
我们可以得到它的概率密度函数如下:    
$\begin{align}
\text{P}\{X_n>t\}
&= \text{P}\{T_n-T_{n-1}>t\}\\
&= \int_0^{\infty}\text{P}\{T_n-T_{n-1}>t|T_{n-1}=s\}\text{P}\{T_{n-1}=s\}\mathrm{d}s\\
&= \int_0^{\infty}\text{P}\{N(s,s+t)=0\}\text{P}\{T_{n-1}=s\}\mathrm{d}s\\
&= \int_0^{\infty}\text{P}\{\text{Poisson}(m(s,s+t))=0\}f_{T_{n-1}}(s)\mathrm{d}s\\
&= \int_0^{\infty} e^{-m(s,s+t)}\cdot \lambda(s)e^{-m(s)}\frac{[m(s)]^{n-2}}{(n-2)!}\mathrm{d}s\\
&= \int_0^{\infty} e^{-m(s+t)}\cdot \lambda(s)\frac{[m(s)]^{n-2}}{(n-2)!}\mathrm{d}s\end{align}$

(待补充)





**定理3.17: (非齐次 Poisson 过程的到达时刻的条件分布，定理3.7的推广，[苏中根] 定理3.9)**    
假设 $\mathbf N = \{N(t):t\geq 0\}$ 是均值函数为 $m(t) = \int_0^t\lambda(u)du$ 的非齐次 Poisson 过程，  
记 $T_1,T_2,\dots$ 为到达时刻.   
定义独立同分布的随机变量 $V_1,V_2,\dots,V_n$，其密度函数为 $f_V(v) = \frac{\lambda(v)}{m(t)}\ \ (0\leq v\leq t)$  
给定 $t>0$，则有 $(T_1,T_2,\dots,T_n|N(t)=n)\overset{\mathrm{d}}= (V_{(1)},V_{(2)},\dots,V_{(n)})$   
其中 $V_{(1)},V_{_{(2)}},\dots,V_{(n)}$ 为对应于 $V_1,V_2,\dots,V_n$ 的次序统计量.  
(证明与定理 3.7 类似，只需要证明二者密度函数相等即可证明同分布)

- 特殊地，如果 $\begin{cases}
  \lambda(t)\equiv \lambda\\
  \text{E}[N(t)] = m(t) =\int_0^t\lambda du= \lambda t\end{cases}\ (\forall\ t>0)$，  
  则 $V_1,V_2,\dots,V_n$ 的密度函数退化为 $f_V(v)\equiv \frac{\lambda}{\lambda t} = \frac{1}{t}\ \ (0\leq v\leq t)$，  
  也就是说，$V_1,V_2,\dots,V_n\overset{\text{i.i.d.}}\sim \text{Uniform}(0,t)$  
  此时定理 3.17 退化为定理 3.7: 

  > **定理 3.7: (到达时刻的联合条件分布, Introduction to Probability Models 定理5.2)**  
  > 假设 $\mathbf N = \{N(t):t\geq 0\}$ 是参数为 $\lambda$ 的 Poisson 过程，记 $T_1,T_2,\dots$ 为到达时刻.  
  > 给定 $t>0$，则有 $(T_1,T_2,\dots,T_n|N(t)=n)\overset{\mathrm{d}}= (U_{(1)},U_{(2)},\dots,U_{(n)})$   
  > 其中 $U_{(1)},U_{(2)},\dots,U_{(n)}$ 为对应于 $U_1,U_2,\dots,U_n\overset{\text{i.i.d.}}\sim \text{Uniform}(0,t)$ 的次序统计量.  
  > **上面的结论通常可表述为: **  
  > 在 $(0,t)$ 中已经发生了 $n$ 个事件的条件下，  
  > 事件发生的时间 $T_1,T_2,\dots,T_n$ (考虑为无次序的随机变量时) 是在 $(0,t)$ 上独立均匀地分布的.  

***

**一些具体的例子: **

- **([苏中根] 例3.7): **  
  考虑强度函数为 $\lambda(t) = \begin{cases}
  2t,&0\leq t<1\\
  2,&1\leq t<2\\
  4-t,&2\leq t\leq 4\end{cases}$ 的非齐次 Poisson 过程 $\mathbf N=\{N(t):t\geq 0\}$     
  计算 $\text{P}\{N(2)=2,N(2,4] = 2\}$:  
  根据 $\begin{cases}
  m(2) = \int_0^2 \lambda(t)\mathrm{d}t = \int_0^1 2t\mathrm{d}t + \int_1^2 2\mathrm{d}t = 3\\
  m(4)-m(2) = \int_2^4\lambda (t)\mathrm{d}t = \int_2^4 (4-t)\mathrm{d}t = 2\end{cases}$   
  可知 $\begin{cases}
  N(2)\sim\text{Poisson}(3)\\
  N(2,4] = N(4)-N(2) \sim \text{Poisson}(2)\end{cases}$   
  根据**增量独立性**可知:   
  $\begin{align}
  \text{P}\{N(2)=2,N(2,4] = 2\}
  &=\text{P}\{N(2)=2\}\text{P}\{N(2,4]=2\}\\
  &=e^{-3}\frac{3^2}{2!}\cdot e^{-2}\frac{2^2}{2!}\\
  &=9e^{-5}\end{align}$
- **([苏中根] 例3.8): **  
- **([苏中根] 例3.9): **  
  去年期中考试 Problem 3 (Introduction to Probability Models 例 5.18)

***

**通过非齐次 Poisson 过程构造齐次 Poisson 过程: **  
假设 $\tilde {\mathbf N}= \{\tilde N(t):t\geq 0\}$ 是均值函数为 $m(t) = \int_0^t\lambda(u)du$ 的非齐次 Poisson 过程，  
由于强度函数 $\lambda(t)$ 是恒正的，故均值函数 $m(t)$ 是严格单调递增的.  
因此 $m(t)$ 存在反函数 $m^{-1}$，  
我们定义 $\begin{cases}
m(0)=0\\
m^{-1}(0)=0\\
t_0 = m^{-1}(t)\\
s_0 = m^{-1}(s)\end{cases}$ (若有 $0<s<t$，则有 $0<s_0<t_0$)   
定义随机过程 $\mathbf N=\{N(t) = N(m^{-1}(t)):t\geq 0\}$  
我们可以证明 $\mathbf N$ 是一个速率 $\lambda = 1$ 的齐次 Poisson 过程.

- **① 初始条件: **  
  $N(0) = \tilde N(m^{-1}(0))=\tilde N(0)= 0$ 
- **② 证明 $N(t)\overset{\mathrm{d}}= \text{Poisson}(t)$: **   
  对于任意 $k=0,1,\dots$，我们都有:   
  $\begin{align}
  \text{P}\{N(t)=k\}
  &= \text{P}\{\tilde N(m^{-1}(t))=k\}\\
  &= \text{P}\{\tilde N(t_0)=k\}\\
  &= \text{P}\{\text{Poisson}(m(t_0))=k\}\\
  &= \text{P}\{\text{Poisson}(t)=k\}\end{align}$   
  因此 $N(t)\overset{\mathrm{d}}= \text{Poisson}(t)$ 成立.
- **③ 证明 $N(t) -N(s)\ \bot\ N(t-s):$**   
  $\begin{align}
  \text{E}[e^{a[N(t)-N(s)]+bN(s)}]
  &=
  \text{E}[e^{a[\tilde N(m^{-1}(t))-\tilde N(m^{-1}(s))]+b\tilde N(m^{-1}(s))}]\\
  &=
  \text{E}[e^{a[\tilde N(t_0)-\tilde N(s_0)]+b\tilde N(s_0)}]\quad (\tilde N(t_0)-\tilde N(s_0)\ \bot\ \tilde N(s_0))\\
  &=
  \text{E}[e^{a[\tilde N(t_0)-\tilde N(s_0)]}]\text{E}[e^{b\tilde N(s_0)}]\\
  &=
  \text{E}[e^{a[N(t)-N(s)]}]\text{E}[e^{bN(s)}]\end{align}$ 
- **④ 证明 $N(t)-N(s)\overset{\mathrm{d}}= N(t-s) \overset{\mathrm{d}}= \text{Poisson}(t-s)$:**  
  对于任意 $k=0,1,\dots$，我们都有:   
  $\begin{align}
  \text{P}\{N(t)-N(s)=k\}
  &=
  \text{P}\{\tilde N(m^{-1}(t))-\tilde N(m^{-1}(s))=k\}\\
  &=
  \text{P}\{\tilde N(t_0)-\tilde N(s_0)=k\}\quad (\tilde N(t_0)-\tilde N(s_0)\overset{\mathrm{d}}= \tilde N(t_0-s_0) \overset{\mathrm{d}}= \text{Poisson}(m(s_0,t_0)))\\
  &=
  \text{P}\{\text{Poisson}(m(s_0,t_0))=k\}\\
  &=
  \text{P}\{\text{Poisson}(m(t_0)-m(s_0))=k\}\\
  &=\text{P}\{\text{Poisson}(t-s)=k\}\end{align}$



### 3.3.3 高维 Poisson 过程

苏中根 $3.6$ 节



### 3.3.4 条件 Poisson 过程

Introduction to Probability Models $5.4.3$ 节







