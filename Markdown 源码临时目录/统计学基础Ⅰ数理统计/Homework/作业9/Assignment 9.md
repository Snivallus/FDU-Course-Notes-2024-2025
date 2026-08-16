## 统计学基础Ⅰ: 数理统计 Assignment 9

**姓名: ** 雍崔扬  
**学号: ** 21307140051   
**习题: ** E3.2, E3.5, E3.6

### Problem 1 (习题 3.2)

为了验证投币正面出现的概率 $p$ 是否为 $\frac12$，独立地投币 $10$ 次检验如下假设: 
$$
H_0:p=\frac12 \quad\leftrightarrow\quad H_1:p\neq \frac12
$$
$10$ 次投币全为正面或全为反面时拒绝原假设.  
试问这一检验法的**实际显著性水平** (即第一类错误概率) 是多少?   
若正面出现的概率为 $0.1$ 时，检验法的功效是多少?

**Solution:**  

$T(X) = \sum_{i=1}^{10} X_i$ 是取自 Bernoulli 分布族的简单随机样本 $X=(X_1,\dots,X_n)$ 的充分统计量.    
当 $p$ 给定时，我们知道 $T(X) = \sum_{i=1}^{10} X_i\overset{\mathrm{d}}= B(10,p)$   
检验法可表述为:
$$
\phi(x) = \begin{cases}
 1, & T(x)=\sum_{i=1}^{10} x_i = 0\ \text{or}\ 10\\
0, &\text{otherwise}\end{cases}
$$
其功效函数为: 
$$
\begin{align}
\gamma_\phi(p) 
&= \text{E}_p[\phi(X)]\\
&= \text{P}_p\{T(X) = 0\} + \text{P}_p\{T(X)=10\}\\
&= \text{P}\{B(10,p) = 0\} + \text{P}\{B(10,p)=10\}\\
&= \binom{10}{0}p^0(1-p)^{10} + \binom{10}{10}p^{10} (1-p)^0\\
&= (1-p)^{10} + p^{10}\end{align}
$$

- 检验法的**实际水平** (即第一类错误概率) 为:   
  $$
  \gamma_\phi\left(\frac12\right) = \left(1-\frac12\right)^{10} + \left(\frac12\right)^{10} = \frac{1}{2^9}\approx 0.00195
  $$

- 当 $p=0.1$ 时，检验法的功效为:   
  $$
  \gamma_\phi(0.1) = (1-0.1)^{10} + (0.1)^{10} = (0.9)^{10} + (0.1)^{10}\approx 0.349
  $$

 

### Problem 2 (习题 3.5)

设一个观测值 $x$ 取自密度函数为 $p(x)$ 的总体.  
求下列检验问题的水平 $\alpha$ 的概率比检验:
$$
H_0:p(x) = p_0(x) = \begin{cases}
4x, & 0\leq x<\frac12\\
4-4x,&\frac12\leq x\leq 1\end{cases}\ \ \leftrightarrow\ \  H_1:p(x) =p_1(x)=I_{[0,1]}(x)
$$

> **定理 $3.2.1$: (Neyman-Pearson 引理, 数理统计讲义 命题 $3.2.2$)**    
> 设参数空间为 $\Theta = \{\theta_0,\theta_1\}$   
> 样本 $X$ 的分布具有分布密度 (或离散的概率) $p_0(\cdot):=p_{\theta_0}(\cdot)$ 和 $p_1(\cdot):=p_{\theta_1}(\cdot)$   
> 对于假设检验问题 $H_0:\theta = \theta_0\leftrightarrow H_1:\theta = \theta_1$ 和显著水平 $\alpha\in(0,1)$ 
>
> - **存在性: **  
>   存在非负常数 $k$​ 以及**概率比检验** $\phi_0(x) = \begin{cases}
>   1, &\text{if } \frac{p_1(x)}{p_0(x)}> k\\
>   0, &\text{if } \frac{p_1(x)}{p_0(x)}< k\end{cases}$   
>   满足 $\gamma_{\phi_0}(\theta_0) =\mathbb{E}_{\theta_0}[\phi_0(X)] = \int \phi_0(x)p_{\theta_0}(x)\mathrm{d}x = \alpha$ 
>
> - **充分性:**  
>   存在性中描述的 $\phi_0$ 是显著水平 $\alpha$ 的**最有效检验法** (MP, most powerful)   
>   即对于显著水平 $\alpha$ 的任意检验函数 $\phi\in \mathcal\Phi_\alpha$ 都有:    
>   $$
>   \gamma_{\phi_0}(\theta_1) = \mathbb{E}_{\theta_1}[\phi_0(X)] \geq \mathbb{E}_{\theta_1}[\phi(X)] = \gamma_{\phi}(\theta_1)
>   $$
>
> - **必要性:**  
>   若 $\phi^\star$ 为显著水平 $\alpha$ 的**最有效检验法**，则 $\phi^\star$ 必定是概率比检验. 

**Solution:**    
为方便起见，我们考虑概率比统计量 $\frac{p_0(X)}{p_1(X)}$ (其分母恒不为 $0$) 而不是 $\frac{p_1(X)}{p_0(X)}$:
$$
\frac{p_0(x)}{p_1(x)}=\begin{cases}
4x, & 0\leq x<\frac12\\
4-4x,&\frac12\leq x\leq 1\end{cases}
$$
它的取值范围为 $[0,2]$   
**Neyman-Pearson 引理**所描述的检验法形如:
$$
\phi(x) = \begin{cases}
1, &\text{if } \frac{p_0(x)}{p_1(x)}< k\\
0, &\text{if } \frac{p_0(x)}{p_1(x)}> k\end{cases}
$$
其中 $k\in (0,2)$
为确定阈值 $k$，考虑计算 $\text{P}_{H_0}\{\frac{p_0(X)}{p_1(X)}<k\}$:  
$$
\begin{align}
\text{P}_{H_0}\left\{\frac{p_0(X)}{p_1(X)} <k\right\}
&=
\int_{-\infty}^{\infty}\text{P}_{H_0} \left\{\frac{p_0(X)}{p_1(X)}<k \ {\Large |}\  X=x\right\}\cdot \text{P}_{H_0}\{X=x\}\mathrm{d}x\\
&=
\int_{0}^{1} {\Large\mathbb 1} \left\{\frac{p_0(x)}{p_1(x)}<k\right\} \cdot p_0(x)\mathrm{d}x\\
&=
\int_{0}^1 {\Large\mathbb 1}\left\{0\leq x<\frac{k}{4}\ \text{or}\ 1-\frac{k}{4} <x\leq 1\right\} \cdot p_0(x)\mathrm{d}x\\
&=
\int_0^{\frac{k}{4}} 1\cdot 4x\mathrm{d}x + \int_{1-\frac{k}{4}}^1 1\cdot (4-4x)\mathrm{d}x\\
&=
2x^2|_{0}^{\frac{k}{4}} + (4x-2x^2)|_{1-\frac{k}{4}}^1\\
&=
\frac{k^2}{8} + \frac{k^2}{8}\\
&= 
\frac{k^2}{4}\end{align}
$$
令 $\text{P}_{H_0}\{\frac{p_0(X)}{p_1(X)}<k\} = \frac{k^2}{4} = \alpha$ 解得 $k = 2\sqrt{\alpha}$ (舍弃负值)  
因此概率比检验为: 
$$
\phi(x) = \begin{cases}
1, &\text{if } \frac{p_0(x)}{p_1(x)}< 2\sqrt{\alpha}\\
0, &\text{if } \frac{p_0(x)}{p_1(x)}> 2\sqrt{\alpha}\end{cases}
$$



### Problem 3 (习题 3.6)

设样本 $X = (X_1,\dots,X_n)$ 取自指数分布 $\exp(\frac1\theta)$       
试求下列检验问题的显著水平 $\alpha = 0.05$ 的 $\text{MP}$ (Most Powerful) 检验: 

- 考虑指数分布族的充分统计量 $T(X)=\sum_{i=1}^nX_i$  
  样本分布为 $p(x)=\frac{1}{\theta^n} \exp\{-\frac{1}{\theta} \sum_{i=1}^nx_i\} = \frac{1}{\theta^n}\exp\{-\frac{1}{\theta}T(x)\}$   
  $$
  \frac{p_1(x)}{p_0(x)} = \frac{\frac{1}{\theta_1^n} \exp\{-\frac{1}{\theta_1} T(x)\}}{\frac{1}{\theta_0^n} \exp\{-\frac{1}{\theta_0} T(x)\}} = \frac{\theta_0^n}{\theta_1^n} \exp\left\{-\left(\frac{1}{\theta_1}-\frac{1}{\theta_0}\right)T(x)\right\}
  $$
  

**(1)** $H_0:\theta = 2 \leftrightarrow H_1:\theta = 4$  
**Solution:**  
$$
\frac{p_1(x)}{p_0(x)} = \frac{\theta_0^n}{\theta_1^n} \exp\left\{-\left(\frac{1}{\theta_1}-\frac{1}{\theta_0}\right)T(x)\right\} = \frac1{2^n}\exp\left\{\frac{1}{4}T(x)\right\}
$$
注意到 $\frac{p_1(x)}{p_0(x)}$ 关于 $T(x)$ 是严格单调递增的.  
**Neyman-Pearson 引理**所描述的检验法形如:
$$
\phi(x) = \begin{cases}
1, &\text{if } \frac{p_1(x)}{p_0(x)}> k'\\
0, &\text{if } \frac{p_1(x)}{p_0(x)}< k'\end{cases} = \begin{cases}
1,& \text{if }T(x)>k\\
0,&\text{if }T(x)<k\end{cases}
$$
为确定阈值 $k$，我们求解 $\text{P}_{H_0}\{T(x)>k\} = \alpha$:   
$$
\begin{align}
\text{P}_{H_0}\{T(x)>k\}
&=
\text{P}\left\{\text{Gamma}\left(n,\frac12\right)>k\right\}\\
&=
1-F(k)\end{align}
$$
其中 $F(t)$ 为 $T(X)$ 的累计分布函数，即:    
$$
F(t) = \int_0^t \frac{(\frac12)^n}{\Gamma(n)}s^{n-1}e^{-\frac12s}\mathrm{d}s
$$
令 $\text{P}_{H_0}\{T(x)>k\} =1-F(k)= \alpha$ 解得 $k = F^{-1}(1-\alpha)$   
因此概率比检验为: 
$$
\phi(x) = \begin{cases}
1, &T(x) = \sum_{i=1}^nx_i>k= F^{-1}(1-\alpha)\\
0, &T(x) = \sum_{i=1}^nx_i<k= F^{-1}(1-\alpha)\end{cases}
$$

***

**(2)** $H_0 : \theta = 2 \leftrightarrow H_1:\theta = 1$  
**Solution:**  
$$
\frac{p_1(x)}{p_0(x)} = \frac{\theta_0^n}{\theta_1^n} \exp\left\{-\left(\frac{1}{\theta_1}-\frac{1}{\theta_0}\right)T(x)\right\} = 2^n\exp\left\{-\frac{1}{2}T(x)\right\}
$$

注意到 $\frac{p_1(x)}{p_0(x)}$ 关于 $T(x)$ 是严格单调递减的.   
**Neyman-Pearson 引理**所描述的检验法形如:
$$
\phi(x) = \begin{cases}
1, &\text{if } \frac{p_1(x)}{p_0(x)}> k'\\
0, &\text{if } \frac{p_1(x)}{p_0(x)}< k'\end{cases} = \begin{cases}
1,& \text{if }T(x)<k\\
0,&\text{if }T(x)>k\end{cases}
$$
为确定阈值 $k$，我们求解 $\text{P}_{H_0}\{T(x)<k\} = \alpha$:   
$$
\begin{align}
\text{P}_{H_0}\{T(x)<k\}
&=
\text{P}\left\{\text{Gamma}\left(n,\frac{1}{2}\right)<k\right\}\\
&=
F(k)\end{align}
$$
其中 $F(t)$ 为 $T(X)$ 的累计分布函数，即:    
$$
F(t) = \int_0^t \frac{(\frac12)^n}{\Gamma(n)}s^{n-1}e^{-\frac12s}\mathrm{d}s
$$
令 $\text{P}_{H_0}\{T(x)<k\} =F(k)= \alpha$ 解得 $k = F^{-1}(\alpha)$   
因此概率比检验为: 
$$
\phi(x) = \begin{cases}
1, &T(x) = \sum_{i=1}^nx_i<k= F^{-1}(\alpha)\\
0, &T(x) = \sum_{i=1}^nx_i>k= F^{-1}(\alpha)\end{cases}
$$
**The End**
