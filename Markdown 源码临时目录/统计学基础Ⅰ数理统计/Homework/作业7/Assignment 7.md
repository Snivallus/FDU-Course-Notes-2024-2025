## 统计学基础Ⅰ: 数理统计 Assignment 7

**姓名: ** 雍崔扬  
**学号: ** 21307140051      
**习题: ** E2.25, E2.27(2)(3), E2.30(2)(3), E2.33, 补充题 5

### Problem 1 (习题 2.25)

设 $X=(X_1,\dots,X_n)$ 是取自 Gamma 分布族 $\{\text{Gamma}(\alpha,\lambda):\lambda>0\}$ 的简单随机样本，  
其中参数 $\alpha>0$ 已知.  
试证明 $\hat g(\lambda) = \frac{1}{\alpha}\overline X$ 是 $g(\lambda)= \frac1\lambda$ 的一致最小方差无偏估计量.

**Solution:**    
首先我们说明 $\hat g(\lambda) = \frac{1}{\alpha}\overline X$ 是 $g(\lambda)= \frac1\lambda$ 的无偏估计量:   
考虑总体 $\xi \sim \text{Gamma}(\alpha,\lambda)$  
我们知道总体均值 $\text{E}[\xi] = \frac{\alpha}{\lambda}$ 且样本均值 $\overline X$ 是总体均值的无偏估计量，  
因此 $\hat g(\lambda) = \frac{1}{\alpha}\overline X$ 是 $g(\lambda)= \frac1\lambda$ 的无偏估计量.

而证明 $\hat g(\lambda) = \frac{1}{\alpha}\overline X$ 是 $g(\lambda)= \frac1\lambda$ 的一致最小方差无偏估计量有两种方法.

- **(法一)** 一种思路是这样的:   
  在 $\alpha$ 已知的情况下，我们可以证明 $\overline X$ 是参数 $\lambda$ 的**充分完备统计量** (证明从略)，  
  而我们又知道 $\hat g(\lambda) = \frac{1}{\alpha}\overline X$ 是 $g(\lambda)= \frac1\lambda$ 的无偏估计量，  
  那么根据 **Lehmann-Scheffé 定理**可知 $\hat g(\lambda) = \frac{1}{\alpha}\overline X$ 是 $g(\lambda)= \frac1\lambda$ 的一致最小方差无偏估计量.

- **(法二) **另一种思路，我们尝试说明 $\hat g(\lambda) = \frac{1}{\alpha}\overline X$ 达到 $g(\lambda)= \frac1\lambda$ 的 **C-R 下界**:   
  总体 $\xi$ 的对数似然函数为:
  $$
  \begin{align}
  l(\lambda;x) 
  &= \log(\text{P}\{\text{Gamma}(\alpha,\lambda) = x\})\\
  &= \log(\frac{\lambda^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\lambda x})\\
  &= \alpha \log(\lambda) - \log(\Gamma(\alpha)) + (\alpha-1)\log(x) -\lambda x\end{align}
  $$
  我们有 $\begin{cases}
  \frac{\partial }{\partial \lambda}l(\lambda;x) = \frac{\alpha}{\lambda}-x\\
  \frac{\partial^2}{\partial \lambda^2}l(\lambda;x) =
  -\frac{\alpha}{\lambda^2}\end{cases}$

  Fisher 信息量为 (第二步的转化基于 "$\int p(x;\alpha,\lambda)\mathrm{d}x=1$ 关于 $\lambda$ 可在积分号下微分两次" 的条件):   
  $$
  \begin{align}
  I_\xi(\lambda) 
  &=\text{E}_\lambda \left[\left(\frac{\partial }{\partial \lambda}l(\lambda; x)\right)^2\right]\\
  &= -\text{E}_\lambda \left[\frac{\partial^2 }{\partial \lambda^2}l(\lambda; x)\right]\\
  &= -\text{E}_\lambda \left[-\frac{\alpha}{\lambda^2}\right]\\
  &= \frac{\alpha}{\lambda^2}\end{align}
  $$
  因此 C-R 下界为 $\frac{(\frac{\mathrm{d}}{\mathrm{d}\lambda}g(\lambda))^2}{I_X(\lambda)} = \frac{(-\frac{1}{\lambda^2})^2}{n I_\xi(\lambda)} = \frac{\frac{1}{\lambda^4}}{n\cdot \frac{\alpha}{\lambda^2}} = \frac{1}{n\alpha \lambda^2}$   
  而 $\hat g(\lambda) = \frac{1}{\alpha}\overline X$ 的方差为:   
  $$
  \begin{align}
  \text{Var}_\lambda(\hat g(\lambda))
  &= \text{Var}_\lambda \left(\frac{1}{\alpha}\overline X\right)\\
  &= \frac{1}{\alpha^2}\cdot \frac{1}{n}\cdot\text{Var}_\lambda(\xi)\\
  &= \frac{1}{\alpha^2}\cdot \frac{1}{n}\cdot \frac{\alpha}{\lambda^2}\\
  &= \frac{1}{n\alpha \lambda^2}\end{align}
  $$

  达到了 C-R 下界，因而 $\hat g(\lambda) = \frac{1}{\alpha}\overline X$ 是 $g(\lambda)= \frac1\lambda$ 的一致最小方差无偏估计量.



### Problem 2 (习题 2.27)

设 $X=(X_1,\dots,X_n)$ 是取自正态分布族 $\{\text{N}(\mu,\sigma^2):\mu\in \mathbb R,\sigma^2>0\}$ 的简单随机样本.  

**Lemma $1$ (正态总体的样本均值与样本方差的联合分布, S. Ross 命题 $2.5$)**   
若 $X=(X_1,\dots,X_n)$ 为取自 $\text{N}(\mu,\sigma^2)$ 的简单随机样本，样本量为 $n$，  
定义样本均值 $\overline{X}=\frac{1}{n}\sum_{i=1}^nx_i$ 和已修偏样本方差 ${S_n^*}^2 = \frac{1}{n-1}\sum_{i=1}^n(x_i-\overline{X})^2$，  
则有 $\begin{cases}
\overline{X} \ \bot\ {S_n^*}^2\\
\overline{X} \sim \text{N}(\mu,\frac{\sigma^2}{n})\\
{S_n^*}^2 \sim \sigma^2 \frac{\chi^2(n-1)}{n-1}
\end{cases}$ 成立.  

**Lemma $2$: (助教您好，请问这个引理考试时需要证明吗?)**  
我们证明 $(\overline X,S_n^2)$ 是该正态分布族参数 $(\mu,\sigma^2)$ 的**充分完备统计量**:   

- 首先我们利用因子化定理证明统计量 $(\overline X,S_n^2)$ 的**充分性**:   
  记 $x=(x_1,\dots,x_n)$，则我们有:   
  $$
  \begin{align}
  \text{P}\{X=x\}
  &=\text{P}\{X_1=x_1,\dots,X_n=x_n\}\\
  &=\prod_{i=1}^n \text{P}\{N(\mu,\sigma^2)=x_i\}\\
  &=\prod_{i=1}^n (2\pi\sigma^2)^{-\frac12}\exp\left\{-\frac{1}{2\sigma^2}(x_i-\mu)^2\right\}\\
  &=(2\pi\sigma^2)^{-n/2}\exp\left\{-\frac{1}{2\sigma^2}
  \sum_{i=1}^n(x_i-\mu)^2\right\}\\
  &=(2\pi\sigma^2)^{-n/2}\exp\left\{-\frac{1}{2\sigma^2}
  \left[\sum_{i=1}^n(x_i-\bar x)^2+n(\bar x-\mu)^2\right]\right\}\\
  &=(2\pi\sigma^2)^{-n/2}\exp\left\{-\frac{n}{2\sigma^2}
  s^2\right\}\exp\left\{-\frac{n}{2\sigma^2}(\bar x-\mu)^2\right\}\end{align}
  $$
  
  其中 $\begin{cases}
  s^2= \frac{1}{n}\sum_{i=1}^n(x_i-\bar x)^2\\
  \bar x = \frac1n \sum_{i=1}^nx_i\end{cases}$  
  
  考虑统计量 $T=(\overline X,S_n^2)$，  
  记 $\begin{cases}
  g(T(x);\mu,\sigma^2) = g(\bar x,s^2;\mu,\sigma^2) = (2\pi\sigma^2)^{-n/2}\exp\{-\frac{n-1}{2\sigma^2}
  s^2\}\exp\{-\frac{n}{2\sigma^2}(\bar x-\mu)^2\}\\
  h(x) \equiv 1\end{cases}$  
  根据**因子化定理**我们知道，$T=(\overline X,S_n^2)$ 是参数 $(\mu,\sigma^2)$的充分统计量.
  
- 下面我们证明统计量 $(\overline X,S_n^2)$ 的**完备性**:   
  根据 **Lemma $1$** 我们知道 $\begin{cases}
  \overline X \ \bot\ S_n^2\\
  \overline X \sim N(\mu,\frac{\sigma^2}{n})\\
  S_n^2 \sim \sigma^2 \frac{\chi^2(n-1)}{n}
  \end{cases}$     
  因此 $(\overline X,S_n^2)$ 的联合分布 $p_{(\overline X,S_n^2)}(\bar x,s^2;\mu,\sigma^2) = p_{\overline X}(\bar x;\mu,\sigma^2)\cdot p_{S_n^2}(s^2;\mu,\sigma^2)$  

  任意给定 $(\overline X,S_n^2)$ 的函数 $\phi(\overline X,S_n^2)$，我们有:     
  $$
  \begin{align}
  \text{E}[\phi(\overline X,S_n^2)]
  &=
  \text{E}[\text{E}[\phi(\overline X,S_n^2)|S_n^2]]\\ 
  &=
  \int_0^\infty \text{E}[\phi(\overline X,S_n^2)|S_n^2=s^2]\cdot 
  p_{S_n^2}(s^2;\mu,\sigma^2) \mathrm{d}s^2\\
  &=
  \int_0^\infty \text{E}[\phi(\overline X,s^2)]\cdot 
  p_{S_n^2}(s^2;\mu,\sigma^2) \mathrm{d}s^2\end{align}
  $$
  **假设**对于任意 $(\mu,\sigma^2)$ 我们都有 $\text{E}[\phi(\overline X,S_n^2)]=0$ 成立，  
  则我们知道 $\text{E}[\phi(\overline X,s^2)]$ 作为 $s^2$ 的函数，在 $s^2\in(0,\infty)$ 上几乎处处为 $0$ 
  
  这个结论表明，对于任意给定 $s^2>0$，  
  我们都有 $\text{E}[\phi(\overline X,s^2)]
  = \int_{-\infty}^{\infty} \phi(\bar x,s^2)p_{\overline X}(\bar x;\mu,\sigma^2) \mathrm{d}\bar x = 0\ \ (\forall\ u\in\mathbb R,\sigma^2>0)$ 成立，  
  这说明对于任意 $\begin{cases}
  \bar x \in \mathbb R\\
  s^2 >0\end{cases}$ 我们都有 $\phi(\bar x,s^2)=0$ 成立.
  
  上述推理表明统计量 $(\overline X,S_n^2)$ 是**完备的**.

综上所述， $(\overline X,S_n^2)$ 是该正态分布族参数 $(\mu,\sigma^2)$ 的**充分完备统计量**.

****

**(1)** 试证明 $\hat \sigma = \sqrt{\frac{n}{2}} \frac{\Gamma(\frac{n-1}{2})}{\Gamma(\frac{n}{2})} S_n$ 是 $\sigma$ 的一致最小方差无偏估计量，  
其中 $S_n^2 = \frac1n\sum_{i=1}^n(X_i-\overline X)^2$ 为未修偏的样本方差.

**Solution:**  
我们知道 $(\overline X,S_n^2)$ 是该正态分布族参数 $(\mu,\sigma^2)$ 的**充分完备统计量**，  
而 $\hat \sigma = \sqrt{\frac{n}{2}} \frac{\Gamma(\frac{n-1}{2})}{\Gamma(\frac{n}{2})} S_n$ 是 $(\overline X,S_n^2)$ 的函数.  
我们只要证明 $\hat\sigma$ 是 $\sigma$ 的无偏估计量，  
即可根据 **Lehmann-Scheffé 定理**说明 $\hat \sigma = \sqrt{\frac{n}{2}} \frac{\Gamma(\frac{n-1}{2})}{\Gamma(\frac{n}{2})} S_n$ 是 $\sigma$ 的一致最小方差无偏估计量.

- 下面我们证明 $\hat\sigma$ 是 $\sigma$ 的**无偏估计量**:     
  我们有 $S_n^2 \sim \frac{\chi^2_{n-1}}{n}\sigma^2 = \frac{\text{Gamma}(\frac{n-1}{2},\frac12)}{n}\sigma^2$ 成立.  
  因此 $S_n\sim \frac{\sigma}{\sqrt{n}}\sqrt{\text{Gamma}(\frac{n-1}{2},\frac12)}$ ，我们有:       
  $$
  \begin{align}
  \text{E}[S_n]
  &= \text{E}\left[\frac{\sigma}{\sqrt{n}}\sqrt{\text{Gamma}\left(\frac{n-1}2,\frac12\right)}\right]\\
  &= \frac{\sigma}{\sqrt{n}} \int_0^\infty \sqrt t 
  \cdot\text{P}\left\{\text{Gamma}\left(\frac{n-1}{2},\frac12\right)=t\right\}\mathrm{d}t\\
  &= \frac{\sigma}{\sqrt{n}} 
  \int_0^\infty \sqrt t \frac{(\frac12)^{\frac{n-1}{2}}}{\Gamma(\frac{n-1}{2})}t^{\frac{n-1}{2}-1}e^{-\frac{1}2 t} \mathrm{d}t\\
  &= 
  \sqrt{\frac{2}{n}}\frac{1}{\Gamma(\frac{n-1}{2})}\sigma \cdot 
  \int_0^\infty t^{\frac{n}{2}-1} e^{-\frac12t} \left(\frac12\right)^{\frac{n}2} \mathrm{d}t\\
  &= 
  \sqrt{\frac{2}{n}}\frac{1}{\Gamma(\frac{n-1}{2})} \sigma \cdot 
  \int_0^\infty u^{\frac{n}{2}-1} e^{-u} \mathrm{d}u\qquad(u:= \frac12 t)\\
  &= \sqrt{\frac{2}{n}}\frac{1}{\Gamma(\frac{n-1}{2})}\sigma \cdot  
  \Gamma\left(\frac{n}{2}\right)\\
  &= \sqrt{\frac{2}{n}}\frac{\Gamma(\frac{n}{2})}{\Gamma(\frac{n-1}{2})}\sigma\end{align}
  $$
  于是我们有:   
  $$
  \begin{align}
  \text{E}_\sigma[\hat \sigma]
  &= \sqrt{\frac{n}{2}} \frac{\Gamma(\frac{n-1}{2})}{\Gamma(\frac{n}{2})}\cdot \text{E}[S_n]\\
  &= \sqrt{\frac{n}{2}} \frac{\Gamma(\frac{n-1}{2})}{\Gamma(\frac{n}{2})}\cdot \sqrt{\frac{2}{n}}\frac{\Gamma(\frac{n}{2})}{\Gamma(\frac{n-1}{2})}\sigma\\
  &= \sigma\end{align}
  $$
  因此 $\hat \sigma = \sqrt{\frac{n}{2}} \frac{\Gamma(\frac{n-1}{2})}{\Gamma(\frac{n}{2})} S_n$ 是 $\sigma$ 的**无偏估计量**. 

根据我们之前的讨论 (**Lehmann-Scheffé 定理**)，  
进而可知 $\hat \sigma = \sqrt{\frac{n}{2}} \frac{\Gamma(\frac{n-1}{2})}{\Gamma(\frac{n}{2})} S_n$ 是 $\sigma$ 的**一致最小方差无偏估计量**.

**(2)** 求 $3\mu + 4\sigma$ 的一致最小方差无偏估计量.

- **Solution:**  
  我们知道样本均值 $\hat \mu = \overline X$ 是充分完备统计量 $(\overline X,S_n^2)$ 的函数，  
  而 $\hat \mu = \overline X$ 又是总体均值 $\mu$ 的无偏估计量，  
  因而根据 **Lehmann-Scheffé 定理**可知 $\hat \mu = \overline X$ 是 $\mu$ 的**一致最小方差无偏估计量**.

  根据 $(1)$ 的结论， $\hat \sigma = \sqrt{\frac{n}{2}} \frac{\Gamma(\frac{n-1}{2})}{\Gamma(\frac{n}{2})} S_n$ 是 $\sigma$ 的**一致最小方差无偏估计量**.  
  因此 $3\mu + 4\sigma$ 的一致最小方差无偏估计量是 $3\hat \mu + 4\hat\sigma$，其展开形式为:   
  $$
  3\hat\mu + 4\hat\sigma = 3\overline X + 4\sqrt{\frac{n}{2}} \frac{\Gamma(\frac{n-1}{2})}{\Gamma(\frac{n}{2})} S_n
  $$

**(3)** 求 $\mu^2$ 的一致最小方差无偏估计量.

- **Solution:**  
  根据 **Lehmann-Scheffé 定理**的指导，  
  我们需要使用 $(\overline X,S_n^2)$ 构造 $\mu^2$ 的无偏估计量.

  根据 **Lemma $1$** 我们有 $\begin{cases}
  \overline X \ \bot\ S_n^2\\
  \overline X \sim N(\mu,\frac{\sigma^2}{n})\\
  S_n^2 \sim \sigma^2 \frac{\chi^2(n-1)}{n}
  \end{cases}$ 

  因此 $\begin{cases}
  \text{E}[\overline X^2] 
  = \text{Var}[\overline X] + (\text{E}[\overline X])^2
  = \frac{\sigma^2}{n} + \mu^2\\
  \text{E}[S_n^2] = \sigma^2 \cdot \frac{n-1}{n}\end{cases}$ 

  于是我们构造 $\overline X^2 - \frac{S_n^2}{n-1}$ 作为 $\mu^2$ 的无偏估计量.

  根据 **Lehmann-Scheffé 定理**可知 $\overline X^2 - \frac{S_n^2}{n-1}$ 是 $\mu^2$ 的一致最小方差无偏估计量.



### Problem 3 (习题 2.30) 

设 $X=(X_1,\dots,X_n)$ 是取自 Poisson 分布族 $\{\text{Poisson}(\lambda):\lambda>0\}$ 的简单随机样本.  

- **Lemma:**  
  我们证明 $\sum_{i=1}^n X_i$ 是 Poisson 分布族 $\{\text{Poisson}(\lambda):\lambda>0\}$ 的**充分完备统计量**.

  - 首先利用**因子化定理**证明 $\sum_{i=1}^n X_i$ 的**充分性**:     
    记 $x=(x_1,\dots,x_n)$，则我们有:   
    $$
    \begin{align}
    \text{P}\{X=x\}
    &=\text{P}\{X_1=x_1,\dots,X_n=x_n\}\\
    &=\prod_{i=1}^n \text{P}\{\text{Poisson}(\lambda)=x_i\}\\
    &= \prod_{i=1}^n
    e^{-\lambda}\frac{\lambda^{x_i}}{x_i!}\\
    &=
    \lambda^{\sum_{i=1}^nx_i}e^{-n\lambda}\cdot \frac{1}{x_1!\dotsm x_n!}\end{align}
    $$
    考虑统计量 $T(X) = \sum_{i=1}^n X_i$        
    记 $\begin{cases}
    g(T(x);\lambda) = \lambda^{T(x)}e^{-n\lambda}=\lambda^{\sum_{i=1}^nx_i}e^{-n\lambda}\\
    h(x) = \frac{1}{x_1!\dotsm x_n!}\end{cases}$  
    则有 $\text{P}\{X=x\} = g(T(x);\lambda) h(x)$  
    根据因子化定理我们知道，$T(X) = \sum_{i=1}^n X_i$ 是参数 $\lambda$ 的充分统计量.
    
  - 其次证明统计量 $\sum_{i=1}^n X_i$ 的**完备性**:   
    根据 Poisson 分布的再生性我们知道 $T=\sum_{i=1}^n X_i\sim \text{Poisson}(n\lambda)$     
    给定 $T=\sum_{i=1}^n X_i$ 的函数 $\phi(T)$  
    **假设**对于任意 $\lambda>0$ 都有 $\text{E}[\phi(T)]=0$ 成立，这个条件可展开为:   
    $$
    \text{E}[\phi(T)]
    = \sum_{k=0}^\infty \phi(k)\cdot e^{-n\lambda}\frac{(n\lambda)^k}{k!}= 0\ \ (\forall\ \lambda>0)
    $$
    即有 $\sum_{k=0}^\infty \phi(k)\frac{(n\lambda)^k}{k!}=0\ \ (\forall\ \lambda>0)$
    
    注意到 $\sum_{k=0}^\infty \phi(k)\frac{(n\lambda)^k}{k!}$ 是一个关于 $\lambda$ 的幂级数.  
    而对于幂级数，如果对于其收敛半径内所有 $\lambda$ 都为 $0$，则幂级数的系数必须全为零.  
    也就是说，对于任意 $k\in \N$ 都有 $\phi(k)=0$ 成立.
    
    这表明统计量 $\sum_{i=1}^n X_i$ 是**完备的**.
  
  综上所述， $\sum_{i=1}^n X_i$ 是 Poisson 分布族 $\{\text{Poisson}(\lambda):\lambda>0\}$ 的**充分完备统计量**.

****

**(1)** 试求 $\lambda$ 和 $\lambda^2$ 的一致最小方差无偏估计量.  

- **Solution:**    
  根据 $\text{E}[\overline X] = \text{E}[\xi] = \lambda$ 可知 $\overline X$ 是 $\lambda$ 的无偏估计量.     
  根据 **Lehmann-Scheffé 定理**可知 $\overline X$ 是 $\lambda$ 的一致最小方差无偏估计量.
  
  注意到:    
  $$
  \begin{align}
  \text{E}[\overline X^2] 
  &= \text{Var}[\overline X] + (\text{E}[\overline X])^2 \\
  &= \frac1n\text{Var}[\xi] + (\text{E}[\xi])^2\\
  &= \frac1n \lambda + \lambda^2\end{align}
  $$
  因此 $\text{E}[\overline X^2 - \frac{\overline X}{n}] = \lambda^2$   
  说明 $\overline X^2 - \frac{\overline X}{n}$ 是 $\lambda^2$ 的无偏估计量.  
  根据 **Lehmann-Scheffé 定理**可知 $\overline X^2 - \frac{\overline X}{n}$ 是 $\lambda^2$ 的一致最小方差无偏估计量.

**(2)** 证明 $\varphi (X_1) = \begin{cases}
1 & \text{if }X_1 \text{ is even}\\
-1 & \text{if }X_1 \text{ is odd}\end{cases}$ 是 $e^{-2\lambda}$ 的无偏估计量.

- **Solution:**
  $$
  \begin{align}
  \text{E}[\varphi(X_1)]
  &=
  \sum_{k=0}^{\infty}1\cdot\text{P}\{\text{Poisson}(\lambda)=2k\}
  +\sum_{k=0}^{\infty}(-1)\cdot \text{P}\{\text{Poisson}(\lambda)=2k+1\}\\
  &=
  \sum_{k=0}^{\infty} e^{-\lambda}\frac{\lambda^{2k}}{(2k)!}
  -\sum_{k=0}^{\infty} e^{-\lambda}\frac{\lambda^{2k+1}}{(2k+1)!}\\
  &=
  e^{-\lambda}\sum_{m=0}^\infty \frac{(-\lambda)^m}{m!}\\
  &=
  e^{-\lambda}\cdot e^{-\lambda}\\
  &=
  e^{-2\lambda}\end{align}
  $$
  
  命题得证.

**(3)** 对 $T= \sum_{i=1}^n X_i$ 计算条件期望 $\text{E}_\lambda[\varphi(X_1)|T=t]$，并由此得到 $e^{-2\lambda}$ 的一致最小方差无偏估计量.  

- **Solution:**  
  $$
  \begin{align}
  \text{P}_\lambda\{X_1=k|T=t\}
  &= \frac{\text{P}_\lambda \{X_1=k,T=t\}}{\text{P}_\lambda\{T=t\}}\\
  &= \frac{\text{P}_\lambda\{X_1=k\}\text{P}_\lambda\{\sum_{i=2}^n X_i = t-k\}}{\text{P}_\lambda\{T=t\}}\\
  &= \frac{\text{P}\{\text{Poisson}(\lambda)=k\}\text{P}\{\text{Poisson}((n-1)\lambda)=t-k\}}{\text{P}\{\text{Poisson}(n\lambda)=t\}}\\
  &=
  \frac{e^{-\lambda}\frac{\lambda^k}{k!}\cdot e^{-(n-1)\lambda}\frac{[(n-1)\lambda]^{t-k}}{(t-k)!}}{e^{-n\lambda}\frac{(n\lambda)^t}{t!}}\\
  &=
  \binom{t}{k}\left(\frac{1}{n}\right)^k\left(1-\frac{1}{n}\right)^{t-k}\\
  &=
  \text{P}\left\{B\left(t,\frac{1}{n}\right)=k\right\}\end{align}
  $$
  因此条件分布 $(X_1|T=t)\overset{\mathrm{d}}= B(t,\frac1n)$ 是二项分布.  
  
  条件期望 (实际上与 $\lambda$ 无关，因此下标 $\lambda$ 省略):   
  $$
  \begin{align}
  \text{E}[\varphi(X_1)|T=t]
  &= 
  \sum_{k \text{ even}}1\cdot \text{P}\{X_1=k|T=t\} + \sum_{k \text{ odd}}(-1)\cdot \text{P}\{X_1=k|T=t\}\\
  &=
  \sum_{k \text{ even}}1\cdot \text{P}\left\{B\left(t,\frac{1}{n}\right)=k\right\} + \sum_{k \text{ odd}}(-1)\cdot \text{P}\left\{B\left(t,\frac{1}{n}\right)=k\right\}\\
  &=
  \sum_{k \text{ even}}1\cdot \text{P}\left\{B\left(t,\frac{1}{n}\right)=k\right\} + \sum_{k \text{ odd}}(-1)\cdot \text{P}\left\{B\left(t,\frac{1}{n}\right)=k\right\}\\
  &=
  \sum_{k \text{ even}}1\cdot \binom{t}{k}\left(\frac{1}{n}\right)^k\left(1-\frac{1}{n}\right)^{t-k} + \sum_{k \text{ odd}}(-1)\cdot \binom{t}{k}\left(\frac{1}{n}\right)^k\left(1-\frac{1}{n}\right)^{t-k}\\
  &=
  \sum_{k=0}^{t}\binom{t}{k}\left(-\frac1n\right)^k \left(1-\frac1n\right)^{t-k}\\
  &=
  \left(-\frac1n + 1-\frac1n\right)^{t}\\
  &=
  \left(1-\frac{2}{n}\right)^t\end{align}
  $$
  因此 $\text{E}[\varphi(X_1)|T] = (1-\frac2n)^T$   
  根据全期望公式可知 $\text{E}[\varphi(X_1)] = \text{E}[\text{E}[\varphi(X_1)|T]] = \text{E}[(1-\frac2n)^T]$   
  而根据 $(2)$ 的结论可知 $\text{E}[\varphi(X_1)]=e^{-2\lambda}$   
  联立两式可知 $\text{E}[(1-\frac2n)^T]= e^{-2\lambda}$  
  说明 $(1-\frac2n)^T$ 是 $e^{-2\lambda}$ 的无偏估计量.
  
  根据 **Lemma** 可知 $T= \sum_{i=1}^n X_i$ 是 Poisson 分布族 $\{\text{Poisson}(\lambda):\lambda>0\}$ 的**充分完备统计量**，  
  因此 $(1-\frac2n)^T$ 是 $e^{-2\lambda}$ 的**一致最小方差估计量**.



###  Problem 4 (习题 2.33)

(零件寿命的抽样调查)  
设 $X=(X_1,\dots,X_n)$ 是取自指数分布族 $\{\exp(\lambda):\lambda>0\}$ 的简单随机样本.  
给定 $\tau >0$，试求零件总体可靠度 $\text{P}_\lambda\{\xi \geq \tau\}$ 基于样本 $X$ 的一致最小方差无偏估计量.

**Lemma:**  
我们证明 $\sum_{i=1}^n X_i$ 是指数分布族 $\{\exp(\lambda):\lambda>0\}$ 的**充分完备统计量**.

- 首先利用**因子化定理**证明 $\sum_{i=1}^n X_i$ 的**充分性**:     
  记 $x=(x_1,\dots,x_n)$，则我们有:   
  $$
  \begin{align}
  \text{P}\{X=x\}
  &=\text{P}\{X_1=x_1,\dots,X_n=x_n\}\\
  &=\prod_{i=1}^n \text{P}\{\exp(\lambda)=x_i\}\\
  &= \prod_{i=1}^n
  \lambda e^{-\lambda x_i}I(x_i>0)\\
  &=
  \lambda^ne^{-\lambda \sum_{i=1}^nx_i}I(\min_{i=1,\dots,n} x_i > 0)\end{align}
  $$
  考虑统计量 $T(X) = \sum_{i=1}^n X_i$  
  记 $\begin{cases}
  g(T(x);\lambda) = \lambda^n e^{-\lambda T(x)}=\lambda^ne^{-\lambda \sum_{i=1}^nx_i}\\
  h(x) = I(\min_{i=1,\dots,n} x_i > 0) \end{cases}$  
  则有 $\text{P}\{X=x\} = g(T(x);\lambda) h(x)$  
  根据因子化定理我们知道，$T(X) = \sum_{i=1}^n X_i$ 是参数 $\lambda$ 的充分统计量.
  
- 其次证明统计量 $\sum_{i=1}^n X_i$ 的**完备性**:   
  根据 Gamma 分布的再生性我们知道 $T=\sum_{i=1}^n X_i\sim \text{Gamma}(n,\lambda)$     
  给定 $T=\sum_{i=1}^n X_i$ 的函数 $\phi(T)$  
  **假设**对于任意 $\lambda>0$ 都有 $\text{E}[\phi(T)]=0$ 成立，这个条件可展开为:   
  $$
  \text{E}[\phi(T)]
  = \int_{0}^\infty \phi(t) \frac{\lambda^n}{\Gamma(n)}t^{n-1}e^{-\lambda t}\mathrm{d}t = 0\ \ (\forall\ \lambda>0)
  $$
  即 $\phi(T)T^{n-1}$ 的 Laplace 变换 $g(\lambda) = \int_{0}^\infty \phi(t)t^{n-1}e^{-\lambda t}\mathrm{d}t = 0\ \ \ \ (\forall\ \lambda>0)$    
  Laplace 变换的唯一性定理表明 $\phi(T)T^{n-1}$ 几乎处处为 $0$，即 $\phi(T)$ 几乎处处为 $0$.  
  这表明统计量 $\sum_{i=1}^n X_i$ 是**完备的**.

综上所述，$T=\sum_{i=1}^n X_i$ 是指数分布族 $\{\exp(\lambda):\lambda>0\}$ 的**充分完备统计量**.

***

**Solution:**  

- 我们计算 $\text{P}_\lambda\{\xi \geq \tau\}$ 的解析表示:   
  $$
  \begin{align}
  \text{P}_\lambda\{\xi \geq \tau\} 
  &= \int_\tau^\infty \lambda e^{-\lambda x}\mathrm{d}x\\
  &= \int_{\lambda\tau}^{\infty} e^{-u}\mathrm{d}u\quad (u=\lambda x)\\
  &= -e^{-u}|_{\lambda\tau}^{\infty}\\
  &= e^{-\lambda \tau}\end{align}
  $$
  **Lehmann-Scheffé 定理**表明，  
  如果能找到一个关于**充分完备统计量** $T=\sum_{i=1}^n X_i$ 的函数，其期望为 $e^{-\lambda \tau}$，  
  那么它就是 $\text{P}_\lambda\{\xi \geq \tau\}=e^{-\lambda \tau}$ 的**一致最小方差无偏估计量**.
  
- 显然 $\varphi (X_1) = {\Large\mathbb 1}(X_1\geq \tau) = \begin{cases}
  1 & \text{if }X_1 \geq \tau\\
  0& \text{otherwise}\end{cases}$ 是 $\text{P}_\lambda\{\xi \geq \tau\}=e^{-\lambda \tau}$ 的无偏估计量.    

  $\varphi(X_1)$ 对 $T=\sum_{i=1}^n X_i$ 取期望，有:
  $$
  \begin{align}
  \text{E}[\varphi(X_1)|T=t]
  &= 
  \int_0^t \varphi(x)\cdot \frac{\text{P}\{X_1=x,T=t\}}{\text{P}\{T=t\}}\mathrm{d}x\\
  &=
  \int_0^t {\Large \mathbb 1}(x\geq \tau) \cdot \frac{\text{P}\{X_1=x\}\cdot \text{P}\{T-X_1 = t-x\}}{\text{P}\{T=t\}}\mathrm{d}x\\
  &=
  \int_\tau^t 1 \cdot \frac{\text{P}\{\text{exp}(\lambda)=x\}\cdot \text{P}\{\text{Gamma}(n-1,\lambda) = t-x\}}{\text{P}\{\text{Gamma}(n,\lambda)=t\}}\mathrm{d}x\\
  &=
  \int_\tau^t \frac{\lambda e^{-\lambda x}\cdot \frac{\lambda^{n-1}}{\Gamma(n-1)}(t-x)^{n-2}e^{-\lambda(t-x)}}{\frac{\lambda^n}{\Gamma(n)}t^{n-1}e^{-\lambda t}}\mathrm{d}x\\
  &=
  \frac{1}{(n-1)t^{n-1}}\int_{\tau}^t (t-x)^{n-2}\mathrm{d}x\\
  &=
  \frac{1}{(n-1)t^{n-1}}\cdot \left(-\int_{t-\tau}^0 u^{n-2}\mathrm{d}u\right)\qquad (u:=t-x)\\
  &=
  \frac{1}{t^{n-1}}\int_{0}^{t-\tau} \frac{1}{n-1}u^{n-2}\mathrm{d}u\\
  &=
  \frac{1}{t^{n-1}}\cdot u^{n-1}\mathrm{d}u|_{0}^{t-\tau}\\
  &=
  \left(\frac{t-\tau}{t}\right)^{n-1}\\
  &=
  \left(1 - \frac{\tau}{t}\right)^{n-1}\end{align}
  $$
  因此 $\text{E}[\varphi(X_1)|T] = (1-\frac{\tau}{T})^{n-1}$   
  
  根据全期望公式可知 $\text{E}[\varphi(X_1)] = \text{E}[\text{E}[\varphi(X_1)|T]] = \text{E}[(1-\frac{\tau}{T})^{n-1}]$   
  而我们知道 $\text{E}[\varphi(X_1)]=e^{-\lambda \tau}$   
  联立两式可知 $\text{E}[(1-\frac{\tau}{T})^{n-1}]= e^{-\lambda \tau}$  
  说明 $(1-\frac{\tau}{T})^{n-1}$ 是 $e^{-\lambda \tau}$ 的无偏估计量.

***

> **助教评语: **  
> $2.33$ 可以通过类似 $2.30$ 的方法找到一个 $e^{-\lambda\tau}$ 的无偏估计量，  
> 然后对充分完备统计量 $T$ 取条件期望得到 $\text{UVMUE}$，  
> 你的解有些奇怪，照道理应该不会有 $\exp$.
>
> **Faulty Solution:**
>
> - 我们计算 $\text{P}_\lambda\{\xi \geq \tau\}$ 的解析表示:   
>   $$
>   \begin{align}
>   \text{P}_\lambda\{\xi \geq \tau\} 
>   &= \int_\tau^\infty \lambda e^{-\lambda x}\mathrm{d}x\\
>   &= \int_{\lambda\tau}^{\infty} e^{-u}\mathrm{d}u\quad (u=\lambda x)\\
>   &= -e^{-u}|_{\lambda\tau}^{\infty}\\
>   &= e^{-\lambda \tau}\end{align}
>   $$
>   **Lehmann-Scheffé 定理**表明，  
>   如果能找到一个关于**充分完备统计量** $T=\sum_{i=1}^n X_i$ 的函数，其期望为 $e^{-\lambda \tau}$，  
>   那么它就是 $\text{P}_\lambda\{\xi \geq \tau\}=e^{-\lambda \tau}$ 的**一致最小方差无偏估计量**.
>   
> - 我们计算 $T$ 的 Laplace 变换:   
>   $$
>   \begin{align}
>   \text{E}[e^{-zT}]
>   &= \int_0^{\infty} e^{-zt}\cdot\frac{\lambda^n t^{n-1}e^{-\lambda t}}{\Gamma(n)}\mathrm{d}t\\
>   &=\frac{\lambda^n}{\Gamma(n)} \int_0^{\infty} e^{-(z+\lambda)t} t^{n-1}\mathrm{d}x\\
>   &=\frac{\lambda^n}{\Gamma(n)(\lambda+z)^n} 
>   \int_0^{\infty}e^{-(\lambda+z)t}((\lambda+z)t)^{n-1}d(\lambda+z)t\\
>   &=\frac{\lambda^n}{\Gamma(n)(\lambda+z)^n}\cdot
>   \Gamma(n)\\
>   &=\left(\frac{\lambda}{\lambda + z}\right)^n \end{align}
>   $$
>   解方程 $(\frac{\lambda}{\lambda + z})^n = e^{-\lambda \tau}$ 可得 $z=\frac{1-e^{-\frac{\lambda\tau}{n}}}{e^{-\frac{\lambda\tau}{n}}}\lambda = \lambda (e^{\frac{\lambda \tau}{n}}-1)$   
>   因此 $\exp\{-\lambda (e^{\frac{\lambda \tau}{n}}-1)T\}$ 是 $e^{-\lambda \tau}$ 的**无偏估计量**.
>
> 综上所述，$\exp\{-\lambda (e^{\frac{\lambda \tau}{n}}-1)T\}$ 是 $\text{P}_\lambda\{\xi \geq \tau\}=e^{-\lambda \tau}$ 的**一致最小方差无偏估计量**.
>
> ***
>
> 这个解答错误的原因很简单:   
> $\lambda$ 是未知参数，因此 $\exp\{-\lambda (e^{\frac{\lambda \tau}{n}}-1)T\}$ 并不是一个统计量!



### Problem 5 (补充题 5)

设 $X=(X_1,\dots,X_n)$ 是取自分布族 $\{p(x;\theta) = \frac{1+\theta x}{2}I_{[-1,1]}(x):|\theta|<1\}$ 的简单随机样本.  
求 $\theta$ 的一个相合估计量.

**Lemma: (Khinchin 弱大数定律)**     
设 $X_1,X_2,\dots$ 是一列独立同分布的随机变量，  
记部分和为 $S_n = \sum_{i=1}^nX_i$，记样本均值 $\overline {X}_n = \frac{S_n}{n}$，  
则当且仅当对于任意 $i=1,2,\dots$ 都有 $\text{E}[X_i]=\mu<\infty$ 时，  
有 $\overline{X}_n=\frac{S_n}{n}\overset{\mathrm{p}}\to \mu = \text{E}[\frac{S_n}{n}]$ 成立.  

- **证明: **  
  要证明 $\frac{S_n}{n}\overset{\mathrm{p}}\to \mu$，等价于证明 $\frac{S_n}{n}\overset{\mathrm{d}}\to \mu$ 成立，  
  即等价于证明当 $n\to\infty$ 时，  
  $\frac{S_n}{n}$ 的**特征函数** $\varphi_n(t)=\text{E}[e^{\mathrm{i}t\frac{S_n}{n}}]$ **弱收敛**于单点分布 $\mu$ 的**特征函数** $\text{E}[e^{\mathrm{i}t\mu}]=e^{\mathrm{i}t\mu}$   
  即要说明对于任意 $t\in \mathbb R$ (特征函数在整个 $\mathbb R$ 上连续) 都有 $\underset{n\to\infty}{\lim}\varphi_n(t)= e^{i\mu t}$ 成立.   
  $$
  \begin{align}
  \varphi_n(t)
  &= \text{E}[e^{\mathrm{i}t\frac{S_n}{n}}]\\
  &= \text{E}\left[\exp\left\{\mathrm{i}\frac{t}{n}\underset{j=1}{\overset{n}\sum}X_j\right\}\right]\quad(X_i\ \bot\ X_j\text{ for all }i\neq j=1,2,\dots)\\
  &= \underset{j=1}{\overset{n}\prod}
  \text{E}\left[\exp\left\{\mathrm{i}\frac{t}{n}X_j\right\}\right]\end{align}
  $$
  由于 $\{X_n\}$ 独立同分布，故它们拥有相同的特征函数，记为 $\varphi(t)$，于是有 $\varphi_n(t)=[\varphi(\frac{t}{n})]^n$   
  
  下面计算 $\varphi(t) = \text{E}[e^{\mathrm{i}tX}]$ 在 $t=0$ 处的**一阶 Taylor 展开式**:   
  根据 $\begin{cases}
  \varphi(0) = \text{E}[e^{\mathrm{i}tX}]|_{t=0} = \text{E}[0]=0\\
  \varphi'(0) = \text{E}[e^{\mathrm{i}tX}iX]|_{t=0} = \mathrm{i}\text{E}[X]=i\mu\\
  \end{cases}$  可知 $\varphi(t) = 1 + \mathrm{i}\mu t+o(t)$   
  因此对于任意 $t\in \mathbb R$ 都有:   
  $$
  \begin{align}
  \underset{n\to\infty}{\lim}\varphi_n(t)
  &=\underset{n\to\infty}{\lim}\left[\varphi\left(\frac{t}{n}\right)\right]^n\\
  &=\underset{n\to\infty}{\lim}\left[1+\mathrm{i}\mu \frac{t}{n} + o\left(\frac{t}{n}\right)\right]^n\\
  &=\underset{n\to\infty}{\lim}\left(1+\frac{\mathrm{i}\mu t}{n}\right)^n\\
  &= e^{\mathrm{i}\mu t}\end{align}
  $$
  
  命题得证.

**Solution:**  
计算总体均值:   
$$
\begin{align}
\text{E}[\xi]
&= \int_{-1}^1 x\cdot \frac{1+\theta x}{2}\mathrm{d}x\\
&= \left(\frac16\theta x^3 + \frac14x^2\right)|_{-1}^1\\
&= \frac\theta3 \end{align}
$$
计算总体二阶矩:   
$$
\begin{align}
\text{E}[\xi^2]
&= \int_{-1}^1 x^2\cdot \frac{1+\theta x}{2}\mathrm{d}x\\
&= \left(\frac18\theta x^4 + \frac16x^3\right)|_{-1}^1\\
&= \frac13 \end{align}
$$
(总体二阶矩有限，因而符合 **Khinchin 弱大数定律**的条件)

根据 **Khinchin 弱大数定律**，$\overline X$ 是 $\text{E}[\xi]=\frac{\theta}{3}$ 的**相合估计量** (即 $\overline X \overset{\mathrm{p}}\to \frac{\theta}{3}$)  
因此 $\hat \theta = 3\overline X$ 是 $\theta$ 的**相合估计量**.

****

> **助教评语:**   
> 最后一题可以考虑用 Chebyshev 不等式证明，会更简单一点.  
> 你这里用了更强的结论在证明更简单的东西.

- **(Chebyshev 不等式)**  
  若 $X$ 是具有均值 $\mu$ 和 $k\geq 2$ 阶中心距 $\text{E}[|X-\mu|^k]<\infty$ 的随机变量，  
  则对于任意 $a>0$ 都有 $\text{P}\{|X-\mu|\geq a\}\leq \frac{\text{E}[|X-\mu|^k]}{a^k}$ 成立.  

  **证明: **  
  由于 $|X-\mu|^k$ 是非负随机变量，因此对其应用 Markov 不等式可得:   
  $$
  \begin{align}
  \text{P}\{|X-\mu|\geq a\} 
  &= \text{P}\{|X-\mu|^k\geq a^k\}\\
  &\leq \frac{\text{E}[|X-\mu|^k]}{a^k}\qquad(\text{Markov Inequality})
  \end{align}
  $$

**使用 Chebyshev 不等式的证明: **    
我们首先计算这个分布的期望和方差:   

- 计算总体均值:   
  $$
  \begin{align}
  \text{E}[\xi]
  &= \int_{-1}^1 x\cdot \frac{1+\theta x}{2}\mathrm{d}x\\
  &= \left(\frac16\theta x^3 + \frac14x^2\right)|_{-1}^1\\
  &= \frac\theta3 \end{align}
  $$
  
- 计算总体二阶矩:   
  $$
  \begin{align}
  \text{E}[\xi^2]
  &= \int_{-1}^1 x^2\cdot \frac{1+\theta x}{2}\mathrm{d}x\\
  &= \left(\frac18\theta x^4 + \frac16x^3\right)|_{-1}^1\\
  &= \frac13 \end{align}
  $$
  
- 计算总体方差:   
  $$
  \begin{align}
  \text{Var}[\xi^2] 
  &= \text{E}[\xi^2] - (\text{E}[\xi])^2\\
  &= \frac{1}{3} - \left(\frac{\theta}{3}\right)^2\\
  &= \frac{3-\theta^2}{9}\end{align}
  $$
  (总体方差有限，因而满足使用 **Chebyshev 不等式**的条件)

因此根据 **Chebyshev 不等式**，对于任意 $\begin{cases}
\theta \in [-1,1]\\
\varepsilon>0
\end{cases}$ 我们都有:   
$$
\begin{align}
\text{P}\{|\overline X-\text{E}[\overline X]|>\varepsilon\}
&\leq 
\frac{\text{E}[|\overline X - \text{E}[\overline X]|^2]}{\varepsilon^2}\quad (\text{Chebyshev inequality})\\
&=\frac{\text{Var}[\overline X]}{\varepsilon^2}\\
&=
\frac{\frac1n\text{Var}[\xi]}{\varepsilon^2}\\
&=
\frac{3-\theta^2}{9n\varepsilon^2}\to 0\ (n\to \infty)\end{align}
$$
因此 $\text{P}\{|\overline X-\text{E}[\overline X]|>\varepsilon\} = \text{P}\{|\overline X - \frac{\theta}{3}|>\varepsilon\} \to 0\ (n\to\infty)$   
表明 $\overline X$ 是 $\frac{\theta}{3}$ 的相合估计量，  
因此 $\hat \theta=3\overline X$ 是 $\theta$ 的相合估计量.

**The End**
