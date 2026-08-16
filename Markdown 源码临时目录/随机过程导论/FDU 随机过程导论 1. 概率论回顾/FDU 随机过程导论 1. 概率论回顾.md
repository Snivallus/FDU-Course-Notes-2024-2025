# FDU 随机过程导论 1. 概率论回顾

本文根据王老师课堂笔记整理而成，并参考了以下教材: 

  - Introduction to Probability Models: Applied Stochastic Processes (S. Ross)
  - 应用随机过程概率论模型导论 (S. Ross) 龚光鲁译 第 $1,2,3$ 章
  - 随机过程 (方兆本 & 缪柏其) 第 $1$ 章
  - 随机过程 (苏中根) 第 $1$ 章

欢迎批评指正! 

  # 1.1 概率论基本概念

  ## 1.1.1 样本空间 (Sample Space) & 事件 (Event)

一个试验的所有可能结果的集合称为该试验的**样本空间** (sample space)，记为 $\Omega$  
样本空间 $\Omega$ 的任意子集 $E$ 称为一个事件 (event)，即部分可能结果的集合.  
事件 $E_1, E_2$ 的**并** (union) 记为 $E_1\cup E_2$  
事件 $E_1, E_2$ 的**交** (intersection) 记为 $E_1 E_2$ (也记为 $E_1\cap E_2$)
有 $\begin{cases} 
  (E_1 \cup E_2) E_3 = (E_1E_3) \cup (E_2E_3)\\ 
  (E_1E_2)\cup E_3 = (E_1\cup E_3)(E_2\cup E_3) 
  \end{cases}$ 成立.

**不可能事件** (null event) 记为 $\emptyset$  
若 $E_1E_2=\emptyset$，则我们称事件 $E_1,E_2$ **互不相容** (mutually exclusive)  
事件 $E$ 的**对立事件** (complement) 记为 $E^c$，满足 $E\cup E^c=\Omega$  
有 $\begin{cases} (E_1 \cup E_2)^c= E_1^c \cap E_2^c\\ (E_1 \cap E_2)^c= E_1^c \cup E_2^c\\ \end{cases}$ 成立.  

事件全体构成的集合称为**事件空间** (event space)，记为 $\mathcal E=\{E:E\subseteq \Omega\}$   
它是样本空间 $\Omega$ 的一个 $\sigma$-代数，满足以下性质:      

- $\emptyset,\Omega \in \mathcal E$  
- 对于任意 $E\in \mathcal E$，都有 $E^c \in \mathcal E$  成立
- 对于 $\mathcal E$ 中的任意序列 $E_1,E_2,\dots$ 都有 $\bigcup_{i=1}^{\infty} E_i\in \mathcal E$    

$\sigma$-代数的上述性质确保了事件的交、并、补、差仍然留在事件空间 $\mathcal E$ 内.



  ## 1.1.2 定义在事件上的概率 (Probability)

考虑一个样本空间为 $\Omega$ 的**试验** (trial)   
对于任意事件 $E\subseteq \Omega$，记其**概率**为 $\text{P}(E)$，它是 $\mathcal E\to [0,1]$ 的函数，满足: 

  - (**规范性**) $\begin{cases}
    \text{P}(\emptyset)=0\\
    \text{P}(\Omega)=1\\
    0\leq \text{P}(E) \leq 1\end{cases}$
  - (**可列可加性**) 对于任意互不相容的事件序列 $\{E_i\}$ (即 $\forall\ i\neq j,\ E_iE_j = \emptyset$)   
    都有 $\text P(\bigcup_{i=1}^{\infty}E_i) = \sum_{i=1}^{\infty} \text P(E_i)$ 成立


我们还可以推出以下结果: 

  - 对于任意 $E\subseteq \Omega$ 都有 $\begin{cases} E\cup E^c = \Omega\\ \text P(\Omega) = 1 \end{cases}$ 成立，因此有 $\text{P}(E^c) = 1-\text P(E)$ 成立
  - **容斥原理 (Inclusion-Exclusion Principle): **  
    对于任意 $n$ 个事件 $E_1,E_2,\dots ,E_n\subseteq \Omega$，   
    都有 $\text{P}(\bigcup_{i=1}^nE_i) = \sum_{k=1}^n \{ (-1)^{k+1}\underset{1\leq j_1<j_2<\dots <j_k \leq n}{\sum}\text P(E_{j_1}E_{j_2}\dotsm E_{j_k})\}$ 成立.  
    特殊地，考虑 $n=2$ 的情况:   
    对于任意 $E_1,E_2\subseteq \Omega$ 都有 $\text{P}(E_1\cup E_2) = \text P(E_1) + \text P(E_2) - \text P(E_1E_2)$ 成立.  
    当且仅当 $E_1,E_2$ 互不相容时 $(\text{i.e.}\ E_1E_2=\emptyset)$，
    有 $\text{P}(E_1\cup E_2) = \text P(E_1) + \text P(E_2)$ 成立.
  - **次可加性 (Subadditivity): **  
    对于任意事件序列 $\{E_i\}\subseteq \Omega$ 都有 $\text{P}(\bigcup_{i=1}^{\infty}E_i) \leq \sum_{i=1}^{\infty}\text{P}(E_i)$
  - **连续性 (Continuity) : **  
    连续性保证了概率函数作为一个测度函数的合理性.  
    此外，连续性在证明各种极限定理时是不可或缺的，例如大数定律和中心极限定理.
      - **上连续性: **  
        若有一个**单调递减**的事件序列 $\{E_n\}\subseteq \Omega$，即 $E_{n+1} \subseteq E_n\ (\forall n\in \mathbb N_+)$，  
        并且其**交集**为 $E$  (即 $\bigcap_{n=1}^{\infty} E_n = E$，于是有 $\underset{n\rightarrow \infty}{\lim}E_n=E$)，  
        则这个序列的概率趋于 $E$ 的概率，即有 $\underset{n\rightarrow \infty}{\lim} \text{P}(E_n) = \text{P}(\underset{n\rightarrow \infty}{\lim}E_n) = \text{P}(E)$
      - **下连续性: **  
        若有一个**单调递增**的事件序列 $\{E_n\}\subseteq \Omega$，即 $E_{n+1} \supseteq E_n\ (\forall n\in \mathbb N_+)$，  
        并且其并集为 $E$ (即 $\bigcup_{n=1}^{\infty} E_n = E$，于是有 $\underset{n\rightarrow \infty}{\lim}E_n=E$)，  
        则这个序列的概率趋于 $E$ 的概率，即有 $\underset{n\rightarrow \infty}{\lim} \text{P}(E_n) = \text{P}(\underset{n\rightarrow \infty}{\lim}E_n) = \text{P}(E)$

这样我们就构造了一个**概率空间** $(\Omega,\mathcal E,\text{P})$:   
它由**样本空间** $\Omega$、**事件空间** $\mathcal E=\{E:E\subseteq \Omega\}$、**概率函数** $\text{P}:\mathcal E\to[0,1]$ 构成.



  ## 1.1.3 条件概率 (Conditional Probability) & 事件的独立性

**(Ⅰ) 链式法则 (Chain Rule):**  
当且仅当 $\text{P}(E_2)>0$ 时，我们可以定义已知 $E_2$ 发生的条件下 $E_1$ 发生的**条件概率**，  
记为 $\text P(E_1|E_2) = \frac{\text{P}(E_1E_2)}{\text{P}(E_2)}$  
由定义我们可以导出**链式法则**:    
$$
\begin{align}\text P(E_1E_2) 
&= \text P(E_1) \text P(E_2|E_1)\\ 
&=\text P(E_2)\text P(E_1|E_2)
\end{align}
$$
其中 $\text{P}(E_1),\text{P}(E_2)$ 均大于 $0$.

**链式法则的推广形式: **  
给定 $n$ 个事件 $E_1,\dots,E_n \subseteq \Omega$，若 $\text{P}(E_1),\dots,\text{P}(E_n)$ 均大于 $0$，  
则有 $\text{P}(E_1E_2\dotsm E_n) = \text{P}(E_1)\text{P}(E_2|E_1)\dotsm \text{P}(E_n|E_1E_2\dotsm E_{n-1})$  

一个简单的例子:   
假定盒中有 $7$ 个黑球和 $5$ 个白球，不放回地摸取两个球，这两个球都是黑球的概率是多少？  
我们可以用 $E_1,E_2$ 分别代表第 $1,2$ 次摸到的球是黑球的事件，易知 
$$
\begin{cases} \text{P}(E_1) = \frac{7}{7+5} = \frac{7}{12}\\ \text{P}(E_2|E_1) = \frac{6}{6+5} = \frac{6}{11}\\ \end{cases}
$$
因此两次都摸到黑球的概率为
$$
\text{P}(E_1E_2) = \text{P}(E_1) \text P(E_2|E_1) = \frac{7}{12} \cdot \frac{6}{11} = \frac{7}{22}
$$
****

**(Ⅱ) 全概率公式: (Law of Total Probability)**   
给定事件 $E_1,E_2\subseteq \Omega$，我们可以将 $E_1$ 表示为:    
$$
\begin{align}E_1 &= E_1\Omega \\
&= E_1(E_2\cup E_2^c)\\
&= (E_1E_2) \cup (E_1E_2^c)
\end{align}
$$
而事件 $E_1E_2$ 和 $E_1E_2^c$ 是不相容的，因此有:   
$$
\begin{align}
\text P (E_1) &= \text P(E_1E_2) + \text P(E_1E_2^c)\\
&= \text{P}(E_1|E_2) \text P(E_2) + \text P(E_1|E_2^c) \text P(E_2^c)
\end{align}
$$
**全概率公式的推广形式: **  
若 $E_1,E_2,\dots,E_n\subseteq \Omega$ 是一个**完备事件组** (即这些事件互斥，且并集为整个样本空间 $\text{ i.e.}\begin{cases} E_iE_j = \emptyset,\ i\neq j\\ \bigcup_{i=1}^nE_i = \Omega  \end{cases}$)，  
则任意事件 $A\subseteq \Omega$ 的概率可以表示为 $\text P(A) = \sum_{i=1}^n \text P(A|E_i)\cdot \text{P}(E_i)$

****

**(Ⅲ) Bayes 公式: **  
结合**链式法则**和**全概率公式**可知:   
若 $E_1,E_2,\dots,E_n\subseteq \Omega$ 是一个完备事件组，  
则对于任意事件 $A\subseteq \Omega$ 都有:
$$
\begin{align}\text{P}
(E_i|A) 
&= \frac{\text{P}(AE_i)}{\text P(A)}\\
&= \frac{\text{P}(A|E_i)\text P(E_i)}{\sum_{j=1}^n\text{P}(A|E_j)\text{P}(E_j)}\ \ \ \ \ \ \ \ (\forall\ i=1,\dots,n)
\end{align}
$$

****

**(Ⅳ) 事件的独立性: **  
若 $\text{P}(E_1E_2) = \text P(E_1) \text P(E_2)$ (即有 $\begin{cases} \text P(E_1|E_2) = \text P(E_1)\\ \text P(E_2|E_1) =\text P(E_2) \end{cases}$)，  
则我们称事件 $E_1,E_2$ **相互独立** (independent)  
也就是说，事件 $E_1$ 的发生独立于 $E_2$ 是否发生，反之亦然.  

**两两独立** (Pairwise Independent) $\not\Rightarrow$ **相互独立** (Mutually Independent) (又称联合独立, Jointly Independent)  
即对于 $n$ 个事件 $E_1,E_2,\dots,E_n \subseteq \Omega$，  
$$
\text P(E_iE_j) = \text{P}(E_i) \text P(E_j)\ \ (\forall \ i\neq j)\ \ \not\Rightarrow\ \ 
  \begin{cases}
  \text{P}(E_i E_j) = \text{P}(E_i)\text{P}(E_j)\ \ \ \ (\forall\ i < j)\\
  \text{P}(E_i E_j E_j) = \text{P}(E_i)\text{P}(E_j)\text{P}(E_k)\ \ \ \ (\forall\ i < j < k)\\
  \qquad \qquad \dotsm\\
  \text{P}(E_{i_1}E_{i_2}\dotsm E_{i_{n-1}})= \text{P}(E_{i_1})\text{P}(E_{i_2})\dotsm \text{P}(E_{i_{n-1}})\ \ \ \ (\forall\ i_1<i_2<\dots<i_{n-1})\\
  \text P(\bigcap_{i=1}^n E_i) = \prod_{i=1}^n\text P(E_i)
  \end{cases}
$$
**这样的例子很好构造: **  
考虑事件 $\begin{cases}  E_1 = \{1,2\}\\ E_2 = \{1,3\}\\ E_3 = \{1,4\}\\ \end{cases}$ 则有 $\begin{cases} \text{P}(E_1E_2) =\frac14= \text{P}(E_1)\text{P}(E_2) \\ \text{P}(E_1E_3) =\frac14= \text{P}(E_1)\text{P}(E_3) \\ \text{P}(E_2E_3) =\frac14= \text{P}(E_2)\text{P}(E_3)  \end{cases}$  
然而 $\frac14 = \text P(E_1E_2E_3) \neq \text P(E_1) \text{P}(E_2) \text P(E_3) = \frac18$，说明联合不独立.



  # 1.2 随机变量

在进行统计试验时，相较于试验结果本身，我们其实对试验结果的某些函数更感兴趣.  
例如，在掷骰子时，我们可能更关心两颗骰子的点数和，而不关心其实际结果.  
我们所关注的这些量，或者更确切地，这些定义在样本空间上的实值函数，称为**随机变量** (random variable, r.v.).  
由于随机变量的值由试验结果确定，因此我们可以给随机变量的可能取值指定**概率**.

若随机变量可能取值的个数是**可数的** (countable)，则我们称之为**离散随机变量** (discrete r.v.)    
若随机变量的取值范围由**若干个连续区间**构成，则我们称之为**连续随机变量** (continuous r.v.)  
理论上还存在混合随机变量，我们不作讨论.  

随机变量 $X$ 的**累计分布函数** (CDF, Cumulative Distribution Function) 记为 $F(\cdot)$，  
定义为 $\begin{cases} F(x) = \text P(X\leq x)\\ \text{dom}(F) = \mathbb R \end{cases}$  
具有以下性质: 

  - ① 在 $\mathbb R$ 上非严格单调递增；
  - ② $\begin{cases} \underset{x\in \mathbb R} \sup F(x) =1\\ \underset{x\in \mathbb R} \inf F(x) =0\\ \end{cases}$ (上确界、下确界分别为 $0,1$)

  ## 1.2.1 离散随机变量 (Discrete r.v.) 

离散随机变量 $X$ 的**概率质量函数** (PMF, Probability Mass Function) 记为 $\text{pmf}(\cdot)$，  
定义为 $\begin{cases} \text{pmf}(x) = \text{P}(X=x)\\ \text{dom}(\text {pmf}) = \mathbb R \end{cases}$  
若记 $X$ 的可数个取值为 $x_1,x_2,\dots$，  
则 $\text{pmf}(x)  \begin{cases} >0,\ \ \text{if }\ x=x_i\ \text{for some }i\\ =0,\ \ \text{otherwise} \end{cases}$，且有 $\sum_{i=1}^{\infty}\text{pmf}(x_i)=1$ 成立.  
而累积分布函数也可以表示为 $F(x) = \underset{\text{for all }x_i \leq x}\sum \text{pmf}(x_i)$  
容易想象到累计分布函数 $F(\cdot)$ 的图像是呈**阶梯状**的.  

**离散随机变量通常依据概率质量函数分类: **

  - **(Ⅰ) Bernoulli 随机变量 (二项随机变量的特例)**  
    $X= \begin{cases} 1,\ \text{if succeeded}\\ 0,\ \text{if failed} \end{cases} \sim \text{B}(1,p)$ 代表单次试验的成功次数，成功概率为 $p$  
    其概率质量函数的唯二有效取值为 $\begin{cases} \text{pmf}(1) = p\\ \text{pmf}(0) = 1-p \end{cases}$  
    
  - **(Ⅱ) 二项随机变量 (Binomial r.v.)**  
    若 $X$ 代表 $n$ 次独立试验的成功次数，每次试验的成功概率均为 $p$  
    则称 $X$ 为具有参数 $(n,p)$ 的二项随机变量，记为 $X\sim \text{B}(n,p)$  
    其**概率质量函数**的 $n+1$ 个取值为 $\text{pmf}(i) =\binom{n}{i}p^i(1-p)^{n-i}\ \ (i=0,1,\dots,n)$  
    其中**组合数** $\binom{n}{i} = \frac{n!}{(n-i)!i!}$  
    根据**二项式定理** (Binomial Theorem) $(a + b)^n =\sum_{i=0}^n \binom{n}{i} a^{i}b^{n-i}$ 可知  
    $$
    \begin{align}\sum_{i=0}^n \text{pmf}(i) 
    &= \sum_{i=0}^n \binom{n}{i}p^i(1-p)^{n-i}\\
    &= [p+(1-p)]^n\\ 
    &= 1\end{align}
    $$
    这与概率质量函数的性质相符.
    
  - **(Ⅲ) 几何随机变量 (Geometric r.v.)**  
    若 $X$ 代表一系列独立试验首次成功所需的试验次数，每次试验的成功概率均为 $p$，  
    则称 $X$ 为具有参数 $p$ 的几何随机变量，记为 $X\sim \text{Geo}(p)$  
    其**概率质量函数**的取值为 $\text{pmf}(i) = (1-p)^{i-1}p\ \ (i=1,2,\dots)$  
    (前 $i-1$ 次失败，第 $i$ 次成功)  
    注意到:
    $$
    \begin{align}\sum_{i=1}^{\infty} \text{pmf}(i) 
    &= \sum_{i=1}^{\infty} (1-p)^{i-1}p\\
    &= p\sum_{i=1}^{\infty} (1-p)^{i-1}\\
    &= p\cdot \frac{1}{1-(1-p)}\\
    &= 1
    \end{align}
    $$
    这与概率质量函数的性质相符.
    
  - **(Ⅳ) Poisson 随机变量**  
    对于取值于自然数集 $\mathbb N$ 的离散随机变量 $X$，  
    若对于某个 $\lambda>0$ 有 $\text{pmf}(i) = \text P(X=i) = e^{-\lambda} \frac{\lambda^i}{i!}\ \ (i=0,1,\dots)$ 成立，  
    则称 $X$ 为具有参数 $\lambda$ 的 Poisson 随机变量，记为 $X\sim \text{Poisson}(\lambda)$  
    注意到:
    $$
    \begin{align}\sum_{i=1}^{\infty} \text{pmf}(i) 
    &= \sum_{i=1}^{\infty} e^{-\lambda} \frac{\lambda^i}{i!} \\
    &= e^{-\lambda}\sum_{i=1}^{\infty} \frac{\lambda^i}{i!}\\
    &= e^{-\lambda} e^\lambda \\
    &= 1
    \end{align}
    $$
    这与概率质量函数的性质相符.

**Poisson 随机变量可用于近似 $n$ 很大而 $p$ 很小的二项随机变量.**  
设 $X\sim \text B(n,p)$，定义参数 $\lambda := np$，则有:
$$
\begin{align}\text{pmf(i)} 
&= \text P(X=i)\\
&= \binom{n}{i}p^i(1-p)^{n-i}\\
&=  \frac{n!}{i!(n-i)!} \left(\frac{\lambda}n\right)^i \left(1-\frac{\lambda}n\right)^{n-i}\\
&=  \frac{n(n-1)\dotsm(n-i+1)}{n^i}\cdot \frac{\lambda^i}{i!} \frac{(1-\frac{\lambda}n)^n}{(1-\frac{\lambda}n)^i}
\end{align}
$$
假设 $n$ 很大而 $p=\lambda/n$ 很小，则有:
$$
\begin{cases} \frac{n(n-1)\dotsm(n-i+1)}{n^i} \approx 1\\ (1-\frac{\lambda}{n})^n \approx e^{-\lambda}\\ (1-\frac{\lambda}{n})^i \approx 1\\  \end{cases}
$$
于是有 $\text{pmf}(i) \approx e^{-\lambda} \frac{\lambda^i}{i!}$  
说明在 $n$ 很大而 $p$ 很小时，二项分布 $\text B(n,p) \approx \text{Poisson}(np)$，这是一个非常重要的结论.  

**一个简单的例子: **  
假定 $1$ 克放射性物质平均每秒裂变产生 $3.2$ 个 $\alpha$ 粒子，  
则任意一秒中这 $1$ 克放射性物质产生 $\alpha$ 粒子的个数不超过 $2$ 个的近似概率是多少？  
应用刚才的结论，我们可以认为这 $1$ 克放射性物质每秒产生 $\alpha$ 粒子的数量 $X \sim \text{Poisson}(3.2)$   
因此我们有:
$$
\begin{align}\text{P}(X\leq 2) 
&= \text{P}(X=0) + \text{P}(X=1)+\text{P}(X=2)\\
&= e^{-3.2}\left(1+3.2 + \frac{(3.2)^2}{2}\right)\\
&\approx 0.380
\end{align}
$$
说明任意一秒释放的 $\alpha$ 粒子的个数不超过 $2$ 个的近似概率是 $38.0\%$.



  ## 1.2.2 连续随机变量 (Continuous r.v.)

连续随机变量 $X$ 的**概率密度函数** (PDF, Probability Density Function) 记为 $f(\cdot)$，  
定义为 $\begin{cases} f(x) = \frac{\mathrm{d}}{\mathrm{d}x}F(x)\\ \text{dom}(f) = \mathbb R \end{cases}$，  
它满足:   

  - ① **非负性** (Non-negativity): $\forall\ x\in \mathbb R,\ f(x)\geq 0$  
  - ② **归一性** (Normalized): $\int_{-\infty}^{+\infty} f(x)\mathrm{d}x = \underset{x\rightarrow +\infty}\lim F(x) = 1$  

关于连续随机变量 $X$ 的所有概率陈述都能被 $f(\cdot)$ 回答，  
具体来说，对于任意区间 $I \subseteq \mathbb R$，都有 $\text{P}(X\in I) = \int_{x\in I}f(x)\mathrm{d}x$  
特别地，对于足够小的 $\varepsilon>0$ 和任意 $x_0 \in \mathbb R$，  
都有 $\text P(X\in [x_0-\frac{\varepsilon}{2},x_0+\frac{\varepsilon}{2}]) =  \int_{x_0-\frac{\varepsilon}{2}}^{x_0+\frac{\varepsilon}{2}} f(x)\mathrm{d}x\approx \varepsilon f(x_0)$  
从中我们可以看出 $f(x_0)$ 是随机变量 $X$ 在 $x_0$ 附近可能性大小的某种 "密度".

**连续随机变量通常依据概率密度函数分类: **  

  - **(Ⅰ) 均匀随机变量 (Uniform r.v.)**  
    若 $X$ 的概率密度函数给定为 $f(x) =\begin{cases} \frac{1}{b-a}, \ \ \text{if}\ \ a<x<b\\ 0,\ \ \ \ \ \ \text{otherwise} \end{cases}$  
    则称 $X$ 的区间 $(a,b)$ 上的均匀随机变量，记为 $X\sim \text{Uniform}(a,b)$  
    其累计分布函数为 $F(x) = \begin{cases} 
    0,\ \ \ \ \ \ \ x\leq a\\ 
    \frac{x-a}{b-a},\ \ a<x<b\\ 
    1,\ \ \ \ \ \ \ x\geq b \end{cases}$  
  - **(Ⅱ) Gamma 随机变量**  
    定义 **Gamma 函数**为 $\begin{cases}
    \Gamma (\alpha) = \int_0^{+\infty} e^{-t}t^{\alpha-1}\mathrm{d}t\\ 
    \text{dom}(\Gamma) = \{\alpha\in \mathbb C:\text{Re}(\alpha)>0\}
    \end{cases}$​  
    (特殊地，对于任意整数 $n$​ 有 $\Gamma(n) = (n-1)!$​ 成立，实际上 $\Gamma$​ 函数是阶乘函数在实数域和复数域上的推广)  
    若对于某对 $\alpha,\lambda>0$ 有 $\begin{cases} 
    \frac{\lambda e^{-\lambda x}(\lambda x)^{\alpha-1}}{\Gamma(\alpha)},&x\geq 0\\ 0,&\text{otherwise}  \end{cases}$ 成立  
    则称 $X$ 为具有参数 $(\alpha,\lambda)$ 的 Gamma 随机变量，记为 $X\sim \text{Gamma}(\alpha,\lambda)$​  
    它没有解析形式的累计分布函数.
  - **(Ⅲ) 指数随机变量 (Exponential r.v.) (Gamma 随机变量的特例)**  
    若对于某个 $\lambda>0$ 有 $f(x) = \begin{cases} \lambda e^{-\lambda x},&x\geq 0\\ 0,&\text{otherwise} \end{cases}$ 成立  
    则称 $X$ 为具有参数 $\lambda$ 的指数随机变量，记为 $X\sim \text{exp}(\lambda) = \text{Gamma}(1,\lambda)$  
    其累计分布函数为 $F(x) =\int_{-\infty}^xf(t)\mathrm{d}t= \begin{cases} 1-e^{-\lambda x},\ \ \ x\geq 0\\ 0,\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \text{otherwise} \end{cases}$  
  - **(Ⅳ) 正态随机变量 (Normal r.v.)**  
    若对于某组 $\begin{cases} \mu \in \mathbb R\\ \sigma^2 >0 \end{cases}$ 有 $f(x)  = \frac{1}{\sqrt{2\pi} \sigma} \exp\{-\frac{(x-\mu)^2}{2\sigma^2}\}\ \ (\forall\ x\in \mathbb R)$ 成立，  
    则称 $X$ 为具有参数 $(\mu,\sigma^2)$ 的正态随机变量，记为 $X\sim \text N(\mu,\sigma^2)$  
    它没有解析形式的累计分布函数.  
    **一个非常重要的性质是: **  
    $X$ 的线性变换 $\alpha X+\beta$ 也是正态随机变量，满足 $(\alpha X+\beta)\sim \text N(\alpha\mu+\beta,\alpha^2 \sigma^2)$  
    后面我们还会介绍这一结论的推广: **多元正态随机变量的线性变换也是多元正态的.**  
  - **(Ⅴ) 卡方随机变量 (Chi-Squared r.v.) (Gamma 随机变量的特例)**  
    若对于某一 $k\in \mathbb N_+$ 有 $f(x) = \begin{cases} \frac{e^{-x/2 }x^{k/2-1}}{2^{k/2}\Gamma(k/2)},&x\geq 0\\ 0,&\text{otherwise}  \end{cases}$ 成立，   
    则称 $X$ 为具有自由度 $k$ 的卡方随机变量，记为 $X\sim \chi^2(k) = \text{Gamma}(\frac k2,\frac12)$  
    它没有解析形式的累计分布函数.  
    **一个非常重要的性质是: **  
    自由度为 $k$ 的卡方随机变量是 $k$ 个相互独立的标准正态随机变量的平方和.

## 1.2.3 期望 (Expectation)

### (Ⅰ) 离散情形

  若 $X$ 是具有概率质量函数 $\text{pmf}(\cdot)$ 的离散随机变量，  
  则 $X$ 的期望为 $\text{E}(X) = \underset{x:\text{pmf}(x)>0}\sum x\cdot\text{pmf(x)}$  

  **让我们来计算典型的离散随机变量的期望: **   

  - **① Bernoulli 随机变量 (二项随机变量的特例)**  
    $X\sim \text B(1,p)$ 有 $\begin{cases} \text{pmf}(1) = p\\ \text{pmf}(0) = 1-p \end{cases}$    
    于是 $\text{E} (X) =  0(1-p)+1(p) = p$    

  - **② 二项随机变量 (Binomial r.v.)**  
    $X\sim \text B(n,p)$ 有 $\text{pmf}(i) =\binom{n}{i}p^i(1-p)^{n-i}\ \ (i=0,1,\dots,n)$  
    于是:   
    $$
    \begin{align}\text{E}(X) 
    &= \sum_{i=0}^n i\cdot \text{pmf}(i)\\
    &=  \sum_{i=1}^n i\cdot\binom{n}{i}p^i(1-p)^{n-i}\\
    &=  \sum_{i=1}^n i\cdot \frac{n!}{i!(n-i)!} p^i(1-p)^{n-i} \\
    &=  \sum_{i=1}^n  \frac{n!}{(i-1)!(n-i)!} p^i(1-p)^{n-i} \\
    &=  np\cdot\sum_{i=1}^n  \frac{(n-1)!}{(i-1)!(n-i)!} p^{i-1}(1-p)^{n-i} \\
    &= np\cdot\underset{k=0}{\overset{n-1}\sum}  \frac{(n-1)!}{k!(n-1-k)!} p^{k}(1-p)^{n-1-k}\\
    &=  np\cdot\underset{k=0}{\overset{n-1}\sum}  \binom{n-1}{k} p^{k}(1-p)^{n-1-k}\\ 
    &=  np\cdot [p+(1-p)]^{n-1}\\
    &=  np
    \end{align}
    $$
    
  - **③ 几何随机变量 (Geometric r.v.)**  
    $X\sim \text{Geo}(p)$ 有 $\text{pmf}(i) = (1-p)^{i-1}p\ \ (i=1,2,\dots)$  
    于是:
    $$
    \begin{align}
    \text{E}(X) &= \sum_{i=1}^{\infty} i\cdot \text{pmf}(i) \\
    &=  \sum_{i=1}^{\infty} i\cdot (1-p)^{i-1}p \\
    &= p\sum_{i=1}^{\infty} i\cdot q^{i-1}\ \ \ \ \ \ \ (q:= 1-p) \\
    &= p\sum_{i=1}^{\infty} \frac{\mathrm{d}}{\mathrm{d}q}(q^i) \\
    &= p \frac{\mathrm{d}}{\mathrm{d}q}\left(\sum_{i=1}^{\infty}q^i\right)\\
    &= p\frac{\mathrm{d}}{\mathrm{d}q}\left(\frac{q}{1-q}\right) \\
    &= \frac{p}{(1-q)^2} \\
    &= \frac1p
    \end{align}
    $$
    
  - **④ Poisson 随机变量**  
    $X\sim \text{Poisson}(\lambda)$ 有 $\text{pmf}(i)= e^{-\lambda} \frac{\lambda^i}{i!}\ \ (i=0,1,\dots)$  
    于是:   
    $$
    \begin{align}
    \text{E}(X) &= \sum_{i=1}^{\infty} i\cdot \text{pmf}(i) \\
    &=  \sum_{i=1}^{\infty} i\cdot e^{-\lambda} \frac{\lambda^i}{i!} \\
    &=  \sum_{i=1}^{\infty} e^{-\lambda} \frac{\lambda^i}{(i-1)!} \\
    &=  \lambda e^{-\lambda}\sum_{i=1}^{\infty} \frac{\lambda^{i-1}}{(i-1)!} \\
    &=  \lambda e^{-\lambda}\underset{k=0}{\overset{\infty}\sum} \frac{\lambda^k}{k!} \\
    &=  \lambda e^{-\lambda} e^\lambda \\
    &= \lambda
    \end{align}
    $$



### (Ⅱ) 连续情形

若 $X$ 是具有概率密度函数 $f(x)$ 的连续随机变量，  
则 $X$ 的期望为 $\text{E}(X) = \int_{-\infty}^{\infty} xf(x)\mathrm{d}x$  

**让我们来计算典型的连续随机变量的期望: **  

  - **① 均匀随机变量 (Uniform r.v.)**  
    $X\sim \text{Uniform}(a,b)$ 有 $f(x) =\begin{cases} \frac{1}{b-a}, \ \ \text{if}\ \ a<x<b\\ 0,\ \ \ \ \ \ \text{otherwise} \end{cases}$  
    于是:
    $$
    \begin{align}
    \text{E}(X) &=\int_{-\infty}^{\infty} xf(x)\mathrm{d}x \\
    &= \int_{a}^b x\cdot\frac{1}{b-a} \mathrm{d}x \\
    &= \frac{1}{b-a} \cdot \frac{b^2-a^2}{2}\\
    &= \frac{a+b}{2}
    \end{align}
    $$
    
  - **② 指数随机变量 (Exponential r.v.) (Gamma 随机变量的特例)**  
    $X\sim \text{exp}(\lambda) = \text{Gamma} (1,\lambda)$ 有 $f(x) = \begin{cases} \lambda e^{-\lambda x},\ \ x\geq 0\\ 0,\ \ \ \ \ \ \ \ \ \ \text{otherwise} \end{cases}$  
    于是:   
    $$
    \begin{align}
    \text{E}(X) &=\int_{-\infty}^{\infty} xf(x)\mathrm{d}x\\ 
    &= \int_{0}^{\infty} x\cdot\lambda e^{-\lambda x} \mathrm{d}x \\
    &=  -xe^{-\lambda x}|_0^\infty - \int_{0}^{\infty} (-e^{-\lambda x})\mathrm{d}x \\
    &=  0- \frac{e^{-\lambda x}}{\lambda}|_0^\infty \\
    &=  \frac{1}{\lambda}
    \end{align}
    $$
    
    (一般形式 Gamma 随机变量的期望公式参见 ④)
    
  - **③ 正态随机变量 (Normal r.v.)**  
    $X\sim \text N(\mu,\sigma^2)$ 有 $f(x)  = \frac{1}{\sqrt{2\pi} \sigma} \exp\{-\frac{(x-\mu)^2}{2\sigma^2}\}\ \ (\forall\ x\in \mathbb R)$  
    于是:
    $$
    \begin{align}
    \text{E}(X) &=\int_{-\infty}^{\infty} xf(x)\mathrm{d}x\\
    &=  \int_{-\infty}^{\infty} x\cdot\frac{1}{\sqrt{2\pi} \sigma} \exp\{-\frac{(x-\mu)^2}{2\sigma^2}\} \mathrm{d}x \\
    &=  \frac{1}{\sqrt{2\pi} \sigma} \int_{-\infty}^\infty (x-\mu)\exp\{-\frac{(x-\mu)^2}{2\sigma^2}\} \mathrm{d}x + \mu \int_{-\infty}^\infty\frac{1}{\sqrt{2\pi} \sigma}\exp\{-\frac{(x-\mu)^2}{2\sigma^2}\} \mathrm{d}x \\
    &=  \frac{1}{\sqrt{2\pi} \sigma}\int_{-\infty}^\infty y\exp\{-\frac{y^2}{2\sigma^2}\} \mathrm{d}y + \mu \int_{-\infty}^{\infty} f(x)\mathrm{d}x\ \ \ \ \ \ \ \ \ (y:= x-\mu) \\
    &=  0 + \mu\cdot 1 \\
    &=  \mu
    \end{align}
    $$
    
  - **④ Gamma 随机变量**   
    $X\sim \text{Gamma}(\alpha,\lambda)$ 有 $f(x) = \begin{cases} \frac{\lambda e^{-\lambda x}(\lambda x)^{\alpha-1}}{\Gamma(\alpha)},\ \ x\geq 0\\ 0,\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \text{otherwise}  \end{cases}$  
    于是:
    $$
    \begin{align}
    \text{E}(X) &=\int_{-\infty}^{\infty} xf(x)\mathrm{d}x\\
    &= \int_{0}^{\infty} x\cdot \frac{\lambda e^{-\lambda x}(\lambda x)^{\alpha-1}}{\Gamma(\alpha)}\mathrm{d}x \\
    &= \frac{\Gamma(\alpha+1)}{\lambda \Gamma(\alpha)} \int_{0}^{\infty} \frac{\lambda e^{-\lambda x}(\lambda x)^\alpha}{\Gamma(\alpha+1)}\mathrm{d}x \\
    &= \frac{\Gamma(\alpha+1)}{\lambda \Gamma(\alpha)} \int_{0}^{\infty} \text{P}\{\text{Gamma}(\alpha+1,\lambda) = x\}\mathrm{d}x \\
    &= \frac{\alpha \Gamma (\alpha)}{\lambda \Gamma(\alpha)}\cdot 1 \\
    &= \frac{\alpha}{\lambda}
    \end{align}
    $$
    
  - **⑤ 卡方随机变量 (Chi-Squared r.v.) (Gamma 随机变量的特例)**   
    $X\sim \chi^2(k) = \text{Gamma}(\frac k 2, \frac 12)$ 有 $f(x) = \begin{cases} \frac{e^{-x/2 }x^{k/2-1}}{2^{k/2}\Gamma(k/2)},\ \ x\geq 0\\ 0,\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \text{otherwise}  \end{cases}$  
    根据 ④ 的结论可知 $\mathbb{E}(X) = \frac{k/2}{1/2} = k$  

  ### (Ⅲ) 推广形式

我们不禁要问: 随机变量 $X$ 的函数 $g(X)$ 的期望如何计算呢？  
注意到 $g(X)$ 本身也是一个随机变量，它必然服从一个概率分布，  
如果我们能根据 $X$ 的分布导出 $g(X)$ 的分布，我们就能根据定义计算 $\mathbb{E}[g(X)]$  
这种方法我们就不赘述了，因为我们有一个更容易的方法，无需确定 $g(X)$ 的分布就能计算期望.  

  - **① 离散情况: **  
    若 $X$ 是具有概率质量函数 $\text{pmf}(x)$ 的离散随机变量，  
    则对于任意实值函数 $g$ 都有 $\mathbb{E}[g(X)] = \underset{x:p(x)>0}\sum g(x) \text{pmf}(x)$ 成立.  

  - **② 连续情况: **  
    若 $X$ 是具有概率密度函数 $f(x)$ 的连续随机变量，  
    则对于任意实值函数 $g$ 都有 $\mathbb{E}[g(X)] = \int_{-\infty}^{\infty} g(x) f(x)\mathrm{d}x$ 成立.  

**推论:** $\mathbb{E}(\alpha X+\beta) = \alpha \mathbb{E}(X) + \beta$ 



## 1.2.4 方差 (Variance)

随机变量 $X$ 的方差定义为:  
$$
\begin{align}\text{Var}(X) 
&:= \text{E}[(X-\text{E}(X))^2] \\
&= \mathbb{E}[X^2 - 2\mathbb{E}(X) X + (\text{E}(X))^2] \\
&= \text{E}(X^2)  - 2(\text{E}(X))^2 + (\text{E}(X))^2 \\
&= \text{E}(X^2)  - (\text{E}(X))^2
\end{align}
$$

### (Ⅰ) 离散情形 

  - **① Bernoulli 随机变量 (二项随机变量的特例)**  
    $X\sim \text B(1,p)$ 有 $\begin{cases} \text{pmf}(1) = p\\ \text{pmf}(0) = 1-p \end{cases}$ 且 $\text{E} (X)  = p$  
    而 $\mathbb{E}(X^2) = 0^2 (1-p)+ 1^2 (p)=p$  
    因此:   
    $$
    \begin{align}
    \text{Var}(X) &= \mathbb{E}(X^2) - (\text{E}(X))^2\\
    &= p-p^2 \\
    &= p(1-p)
    \end{align}
    $$
    
  - **② 二项随机变量 (Binomial r.v.)**  
    $X\sim \text B(n,p)$ 有 $\text{pmf}(i) =\binom{n}{i}p^i(1-p)^{n-i}\ \ (i=0,1,\dots,n)$ 且 $\text{E(X)} = np$  
    注意到:   
    $$
    \begin{align}
    \text{E}(X^2) &= \sum_{i=0}^n i^2\cdot \text{pmf}(i) \\
    &=  \sum_{i=1}^n i^2\cdot\binom{n}{i}p^i(1-p)^{n-i} \\
    &=  \sum_{i=2}^{n-2} i(i-1)\cdot \frac{n!}{i!(n-i)!} p^i(1-p)^{n-i} +\sum_{i=1}^n i\cdot \frac{n!}{i!(n-i)!} p^i(1-p)^{n-i} \\
    &=\sum_{i=2}^{n-2}  \frac{n!}{(i-2)!(n-i)!} p^i(1-p)^{n-i}  + \mathbb{E}(X)  \\
    &= n(n-1)p^2\cdot\sum_{i=2}^{n-2}  \frac{(n-2)!}{(i-2)!(n-i)!} p^{i-2}(1-p)^{n-i}  + np \\
    &= n(n-1)p^2\cdot\sum_{k=2}^{n-2}  \frac{(n-2)!}{k!(n-2-k)!} p^k(1-p)^{n-2-k}  + np  \\
    &= n(n-1)p^2\cdot (p + (1-p))^{n-2}  + np  \\
    &=  n(n-1)p^2+np
    \end{align}
    $$
    因此:
    $$
    \begin{align}
    \text{Var}(X) &= \mathbb{E}(X^2) - (\text{E}(X))^2 \\
    &= n(n-1)p^2 +np- (np)^2 \\
    &= np(1-p)
    \end{align}
    $$
    
  - **③ 几何随机变量 (Geometric r.v.)**  
    $X\sim \text{Geo}(p)$ 有 $\text{pmf}(i) = (1-p)^{i-1}p\ \ (i=1,2,\dots)$ 而 $\text{E}(X) = \frac1p$  
    注意到:   
    $$
    \begin{align}
    \text{E}(X^2) 
    &= \sum_{i=1}^{\infty} i^2\cdot \text{pmf}(i)\\
    &=  \sum_{i=1}^{\infty} i^2\cdot (1-p)^{i-1}p\\
    &= p\sum_{i=1}^{\infty} [(i+1)i-i]\cdot q^{i-1}\ \ \ \ \ \ \ \  (q:= 1-p)\\
    &= p\sum_{i=1}^{\infty} \frac{\mathrm{d}^2}{\mathrm{d}q^2}(q^{i+1})  - p\sum_{i=1}^\infty i q^{i-1}\\
    &= p\frac{\mathrm{d}^2}{\mathrm{d}q^2}\left(\sum_{i=1}^\infty q^{i+1}\right) - \sum_{i=1}^\infty i(1-p)^{i-1}p \\
    &= p\frac{\mathrm{d}^2}{\mathrm{d}q^2}\left(\frac{q}{1-q}\right) - \text{E}(X) \\
    &= \frac{2p}{(1-q)^3} -\frac1p \\
    &= \frac{2}{p^2}-\frac1p
    \end{align}
    $$
    因此:   
    $$
    \begin{align}
    \text{Var}(X) 
    &= \mathbb{E}(X^2) - (\text{E}(X))^2 \\
    &= \frac 2{p^2}- \frac1p - \left(\frac1p\right)^2 \\
    &= \frac{1}{p^2}-\frac1p \\
    &= \frac{1-p}{p^2}
    \end{align}
    $$


  - **④ Poisson 随机变量**  
    $X\sim \text{Poisson}(\lambda)$ 有 $\text{pmf}(i)= e^{-\lambda} \frac{\lambda^i}{i!}\ \ (i=0,1,\dots)$ 且 $\mathbb{E}(X)=\lambda$   
    注意到:    
    $$
    \begin{align}
    \text{E}(X^2) &= \sum_{i=1}^{\infty} i^2\cdot \text{pmf}(i)   \\
    &=  \sum_{i=1}^{\infty} i^2\cdot e^{-\lambda} \frac{\lambda^i}{i!}   \\
    &=  \underset{i=2}{\overset{\infty}\sum} i(i-1)\cdot e^{-\lambda} \frac{\lambda^i}{i!} +  \sum_{i=1}^{\infty} i\cdot e^{-\lambda} \frac{\lambda^i}{i!}   \\
    &=  \lambda^2 e^{-\lambda}\underset{i=2}{\overset{\infty}\sum} \frac{\lambda^{i-2}}{(i-2)!} + \mathbb{E}(X)\\
    &=  \lambda^2 e^{-\lambda}\underset{k=0}{\overset{\infty}\sum} \frac{\lambda^k}{k!}  + \lambda \\
    &=  \lambda^2 e^{-\lambda} e^\lambda + \lambda \\
    &= \lambda^2+\lambda
    \end{align}
    $$
    因此:   
    $$
    \begin{align}
    \text{Var}(X) &= \mathbb{E}(X^2) - (\text{E}(X))^2 \\
    &= \lambda^2+\lambda-\lambda^2 \\
    &= \lambda
    \end{align}
    $$



  ### (Ⅱ) 连续情形

  - **① 均匀随机变量 (Uniform r.v.)**  
    $X\sim \text{Uniform}(a,b)$ 有 $f(x) =\begin{cases} \frac{1}{b-a}, \ \ \text{if}\ \ a<x<b\\ 0,\ \ \ \ \ \ \text{otherwise} \end{cases}$ 且 $\text{E}(X) =\frac{a+b}2$  
    注意到:    
    $$
    \begin{align}
    \text{E}(X^2) &=\int_{-\infty}^{\infty} x^2f(x)\mathrm{d}x \\
    &= \int_{a}^b x^2\cdot\frac{1}{b-a} \mathrm{d}x\\
    &= \frac{1}{b-a} \cdot \frac{b^3-a^3}{3} \\
    &= \frac{a^2+ab+b^2}{3}
    \end{align}
    $$
    因此:
    $$
    \begin{align}
    \text{Var}(X) &= \mathbb{E}(X^2) - (\text{E}(X))^2 \\
    &= \frac{a^2+ab+b^2}{3} - \left(\frac{a+b}{2}\right)^2 \\
    &= \frac{(a-b)^2}{12}
    \end{align}
    $$
    
  - **② 指数随机变量 (Exponential r.v.) (Gamma 随机变量的特例)**  
    $X\sim \text{exp}(\lambda) = \text{Gamma} (1,\lambda)$ 有 $f(x) = \begin{cases} \lambda e^{-\lambda x},\ \ x\geq 0\\ 0,\ \ \ \ \ \ \ \ \ \ \text{otherwise} \end{cases}$ 且 $\mathbb{E}(X) = \frac1\lambda$  
    注意到:    
    $$
    \begin{align}
    \text{E}(X^2) 
    &=\int_{-\infty}^{\infty} x^2f(x)\mathrm{d}x\\
    &= \int_{0}^{\infty} x^2\cdot\lambda e^{-\lambda x} \mathrm{d}x \\
    &=  -x^2e^{-\lambda x}|_0^\infty - \int_{0}^{\infty} (-2xe^{-\lambda x})\mathrm{d}x \\
    &=  0 + \frac{2}{\lambda}\cdot \int_0^\infty x\cdot\lambda e^{-\lambda x} \mathrm{d}x\\
    &= \frac{2}{\lambda}\cdot \mathbb{E}(X)\\
    &= \frac{2}{\lambda}\cdot \frac{1}{\lambda}\\
    &=  \frac{2}{\lambda^2}
    \end{align}
    $$
    因此:
    $$
    \begin{align}
    \text{Var}(X) 
    &= \mathbb{E}(X^2) - (\text{E}(X))^2 \\
    &= \frac2{\lambda^2} - \left(\frac1\lambda\right)^2 \\
    &= \frac1{\lambda^2}
    \end{align}
    $$
    
    (一般形式 Gamma 随机变量的方差公式参见 ④)
    
  - **③ 正态随机变量 (Normal r.v.)**  
    $X\sim \text N(\mu,\sigma^2)$ 有 $f(x)  = \frac{1}{\sqrt{2\pi} \sigma} \exp\{-\frac{(x-\mu)^2}{2\sigma^2}\}\ \ (\forall\ x\in \mathbb R)$ 且 $\mathbb{E}(X) = \mu$  
    注意到:    
    $$
    \begin{align}
    \text{Var}(X) &= \mathbb{E}[(X-\text{E}(X))^2] \\
    &= \mathbb{E}[(X-\mu)^2] \\
    &=  \int_{-\infty}^{\infty} (x-\mu)^2 \cdot \frac{1}{\sqrt{2\pi} \sigma} \exp\{-\frac{(x-\mu)^2}{2\sigma^2}\} \mathrm{d}x \\
    &=  \frac{\sigma^2}{\sqrt{2\pi}}\int_{-\infty}^{\infty} y^2 \exp\{-\frac{y^2}{2}\} \mathrm{d}y\ \ \ \ \ \ \ \ \ (y:=\frac{x-\mu}{\sigma} ) \\
    &=  \frac{\sigma^2}{\sqrt{2\pi}}\{-y\exp\{-\frac{y^2}{2}\}|_{-\infty}^{\infty}-\int_{-\infty}^{\infty} (-\exp\{-\frac{y^2}{2}\}) \mathrm{d}y\} \\
    &=  \frac{\sigma^2}{\sqrt{2\pi}}\{0 + \sqrt{2\pi}\cdot \int_{-\infty}^\infty \frac{1}{\sqrt{2\pi}} \exp\{-\frac{y^2}{2}\} \mathrm{d}y\} \\
    &=  \frac{\sigma^2}{\sqrt{2\pi}} \cdot \sqrt{2\pi}\\
    &= \sigma^2
    \end{align}
    $$
    于是我们反过来可以知道:   
    $$
    \begin{align}
    \mathbb{E}(X^2) &= \text{Var}(X) + (\mathbb{E}(X))^2 \\
    &= \sigma^2 + \mu^2
    \end{align}
    $$
    
  - **④ Gamma 随机变量**  
    $X\sim \text{Gamma}(\alpha,\lambda)$ 有 $f(x) = \begin{cases} \frac{\lambda e^{-\lambda x}(\lambda x)^{\alpha-1}}{\Gamma(\alpha)},\ \ x\geq 0\\ 0,\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \text{otherwise}  \end{cases}$ 且 $\mathbb{E}(X) = \frac{\alpha}{\lambda}$  
    注意到:    
    $$
    \begin{align}
    \text{E}(X^2) 
    &=\int_{-\infty}^{\infty} x^2f(x)\mathrm{d}x   \\
    &= \int_{0}^{\infty} x^2\cdot \frac{\lambda e^{-\lambda x}(\lambda x)^{\alpha-1}}{\Gamma(\alpha)}\mathrm{d}x  \\
    &= \frac{\Gamma(\alpha+2)}{\lambda^2 \Gamma(\alpha)} \int_{0}^{\infty} \frac{\lambda e^{-\lambda x}(\lambda x)^{\alpha+1}}{\Gamma(\alpha+2)}\mathrm{d}x\\
    &= \frac{\Gamma(\alpha+2)}{\lambda^2 \Gamma(\alpha)} \int_{0}^{\infty} \text{P}\{\text{Gamma}(\alpha+2,\lambda) = x\}\mathrm{d}x\\
    &= \frac{\Gamma(\alpha+2)}{\lambda^2 \Gamma(\alpha)} \cdot 1\\
    &= \frac{(\alpha+1)\alpha \Gamma (\alpha)}{\lambda^2 \Gamma(\alpha)}\\
    &= \frac{(\alpha+1)\alpha}{\lambda^2}
    \end{align}
    $$
    因此:   
    $$
    \begin{align}
    \text{Var}(X) 
    &= \mathbb{E}(X^2) - (\text{E}(X))^2\\
    &= \frac{(\alpha+1)\alpha}{\lambda^2} - \left(\frac{\alpha}{\lambda}\right)^2 \\
    &= \frac{\alpha}{\lambda^2}
    \end{align}
    $$
    
  - **⑤ 卡方随机变量 (Chi-Squared r.v.) (Gamma 随机变量的特例)**  
    $X\sim \chi^2(k) = \text{Gamma}(\frac k 2, \frac 12)$ 有 $f(x) = \begin{cases} \frac{e^{-x/2 }x^{k/2-1}}{2^{k/2}\Gamma(k/2)},\ \ x\geq 0\\ 0,\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \text{otherwise}  \end{cases}$  
    根据 ④ 的结论可知 $\begin{cases} \mathbb{E}(X) = \frac{k/2}{1/2} = k\\ \mathbb{E}(X^2) = \frac{(k/2+1)(k/2)}{(1/2)^2} = k(k+2)\\ \text{Var}(X) = \frac{k/2}{(1/2)^2} = 2k \end{cases}$



  # 1.3 随机变量的联合分布

  ## 1.3.1 联合分布函数  

对于任意两个随机变量 $X,Y$，  
我们定义其**联合累计分布函数** (Joint Cumulative Distribution Function) 为 $\begin{cases} F(x,y) = \text P(\begin{cases} X\leq x\\ Y\leq y \end{cases})\\ \text{dom}(F) = \mathbb R\times \mathbb R \end{cases}$  
$X,Y$ 的**边缘分布** (Marginal Distribution) 可由 $X,Y$ 的**联合分布** (Joint Distribution) 得到:    
$$
\begin{cases}
  F_X(x) = \text{P}(X\leq x) = \text{P}(\begin{cases}X\leq x\\ Y\leq \infty\end{cases} ) = F(x,\infty)\\
  F_Y(y) = \text{P}(Y\leq y) = \text{P}(\begin{cases}X\leq \infty\\ Y\leq y\end{cases} ) = F(\infty,y)
  \end{cases}
$$

  - ① 在 $X,Y$ 都是离散随机变量的情况下，  
    我们定义 $X,Y$ 的**联合概率质量函数**为 $\begin{cases}
    p(x,y) = \text{P}(\begin{cases}X=x\\Y=y\end{cases})\\
    \text{dom}(p) = \mathbb R\times \mathbb R
    \end{cases}$   
    $X,Y$ 的**概率质量函数**可由 $X,Y$ 的**联合概率质量函数**得到 $\begin{cases}
    p_X(x) = \underset{y:p(x,y)>0}{\sum} p(x,y)\\
    p_Y(y) = \underset{x:p(x,y)>0}{\sum} p(x,y)
    \end{cases}$    
    
  - ② 若 $X,Y$ **联合地连续** (jointly continuous)，  
    即存在一个定义在 $\mathbb R \times \mathbb R$ 上的函数使得:   
    对于任意 $A,B\subseteq \mathbb R$ 都有 $\text{P}(X\in A, Y\in B) = \int_B\int_A f(x,y)\mathrm{d}x\mathrm{d}y$ 成立，  
    则称 $f(x,y)$ 为 $X,Y$ 的**联合概率密度函数**.   
    $X,Y$ 的**概率质量函数**可由 $X,Y$ 的**联合概率质量函数**得到 $\begin{cases}
    f_X(x) = \int_{-\infty}^{\infty} f(x,y)\mathrm{d}y\\
    f_Y(y) = \int_{-\infty}^{\infty} f(x,y)\mathrm{d}x
    \end{cases}$    
    同单变量情形类似地，有 $F(a,b) = \text{P}(X\leq a,Y\leq b) = \int_{-\infty}^b \int_{-\infty}^a f(x,y)\mathrm{d}x\mathrm{d}y$ 成立，  
    因此 $f(x,y) = \frac{\mathrm{d}^2}{\mathrm{d}x\mathrm{d}y} F(x,y)$  

  - **③ 随机变量多元函数的期望: **   
    作为本文 $1.2.3$ (Ⅲ) 中命题的引申叙述:   
    若 $X,Y$ 都是随机变量，则对于任意双变量函数 $g$ 都有:   
    $$
    \mathbb{E}[g(x,y)]=\begin{cases}
    \underset{y}{\sum}\underset{x}{\sum} g(x,y)p(x,y)\\
    \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} g(x,y) f(x,y)\mathrm{d}x\mathrm{d}y
    \end{cases}
    $$
    **推论: ** 若 $X_1,\dots,X_n$ 是 $n$ 个随机变量，  
    则对于任意 $n$ 个常数 $c_1,\dots,c_n$ 都有 $\mathbb{E}(\sum_{i=1}^nc_iX_i) = \sum_{i=1}^n c_i\mathbb{E}(X_i)$ 成立.  

  ## 1.3.2 相互独立的随机变量

若对于任意 $a,b\in \mathbb R$ 都有 $\text{P}(X\leq a,Y\leq b) = \text{P}(X\leq a)\cdot \text{P}(Y\leq b)$，  
则我们称随机变量 $X,Y$ 是相互独立的，  
也就是说，对于任意 $a,b\in \mathbb R$，事件 $\begin{cases}E_a=\{X\leq a\}\\ E_b = \{Y\leq b\}\end{cases}$ 相互独立；   
也就是说，对于任意 $a,b\in \mathbb R$，都有 $F(a,b) = F_X(a)\cdot F_Y(b)$ 成立；

  - ① 当 $X,Y$ 均为离散随机变量时，独立性简化为 $p(x,y) \equiv p_X(x)p_Y(y)$，论据如下:    
    $$
    \begin{align}
    \text{P}(X\leq a, Y\leq b) &= \underset{y\leq b}{\sum}\underset{x\leq a}{\sum} p(x,y)\\
    &= \underset{y\leq b}{\sum}\underset{x\leq a}{\sum} p_X(x)p_Y(y)\ \ \ \ \ \ \ (\star)\\
    &= \underset{y\leq b}{\sum}p_Y(y) \cdot \underset{x\leq a}{\sum} p_X(x)\\
    &= \text{P}(Y\leq b)\cdot \text{P}(X\leq a)
    \end{align}
    $$
    
  - ② 当 $X,Y$ 联合地连续时，独立性简化为 $f(x,y) \equiv f_X(x)f_Y(y)$，论据如下:   
    $$
    \begin{align}
    \text{P}(X\leq a, Y\leq b) &= \int_{-\infty}^b \int_{-\infty}^a f(x,y)\mathrm{d}x\mathrm{d}y
    \\
    &= \int_{-\infty}^b \int_{-\infty}^a f_X(x)f_Y(y)\mathrm{d}x\mathrm{d}y\ \ \ \ \ \ \ (\star)
    \\
    &= \int_{-\infty}^b f_Y(y)\mathrm{d}y \cdot \int_{-\infty}^a f_X(x)\mathrm{d}x\\
    &= \text{P}(Y\leq b)\cdot \text{P}(X\leq a)
    \end{align}
    $$
    
  - **③ 独立随机变量多元函数的期望: **   
    若随机变量 $X,Y$ 相互独立，则对于任意函数 $g,h$ 都有 $\mathbb{E}[g(X)h(Y)] = \mathbb{E}[g(X)]\cdot \mathbb{E}[h(Y)]$  
    (证明过程与 ①② 类似，就是多元求和、多元积分的拆分)  
    特殊地，取 $g,h$ 为恒等映射，则我们有 $\mathbb{E}[XY] = \mathbb{E}(X)\cdot \mathbb{E}(Y)$ 

  ## 1.3.3 协方差 (Covariance)

随机变量 $X,Y$ 的协方差记为 $\text{Cov}(X,Y)$，定义为:   
$$
\begin{align}
\text{Cov}(X,Y) 
&= \mathbb{E}[(X-\mathbb{E}(X))(Y-\mathbb{E}(Y))]\\
&= \mathbb{E}[XY-Y\mathbb{E}(X)-X\mathbb{E}(Y)+\mathbb{E}(X)\mathbb{E}(Y)]\\
&= \mathbb{E}(XY) - \mathbb{E}(Y)\mathbb{E}(X) -\mathbb{E}(X)\mathbb{E}(Y) + \mathbb{E}(X)\mathbb{E}(Y)\\
&= \mathbb{E}(XY)- \mathbb{E}(X)\mathbb{E}(Y)
\end{align}
$$
易知，当 $X,Y$ 相互独立时，协方差 $\text{Cov}(X,Y)=0$  
一般地，可以证明 $\text{Cov}(X,Y)>0$ 是表明 $X$ 在增加时，$Y$ 倾向于增加；  
而 $\text{Cov}(X,Y)<0$ 是表明 $X$ 在增加时，$Y$ 倾向于减少；(反之亦然)

**(应用随机过程概率论模型导论, 例 $2.33$)**   
设 $X,Y$ 的联合概率密度函数为 $f(x,y) = \frac{1}{y} e^{-(y+\frac{x}{y})}\ \ (0<x,y<\infty)$  
我们使用 $\text{Cov}(X,Y)=\mathbb{E}(XY)- \mathbb{E}(X)\mathbb{E}(Y)$ 来计算 $\text{Cov}(X,Y)$.  

  - ① 注意到:   
    $$
    \begin{align}
    f_Y(y) &= \int_{-\infty}^{\infty}f(x,y)\mathrm{d}x \\
    &= e^{-y}\int_{0}^{\infty} \frac{1}{y} e^{-\frac{x}{y}}\mathrm{d}x\\
    &= e^{-y}\cdot -e^{-\frac{x}{y}}|_{x=0}^{x=\infty}\\
    &=e^{-y}
    \end{align}
    $$
    
    表明 $Y \sim \text{exp}(1)$，从而 $\mathbb{E}(Y) = 1/1 = 1$   
    
  - ② 注意到:
    $$
    \begin{align}
    \mathbb{E}(X)
    &= \int_{-\infty}^{\infty} x \left(\int_{-\infty}^{\infty} f(x,y) \mathrm{d}y \right) \mathrm{d}x\\
    &= \int_{-\infty}^{\infty} x \int_{-\infty}^{\infty} \frac{1}{y} e^{-(y+\frac{x}{y})} \mathrm{d}y\mathrm{d}x\\
    &= \int_{0}^{\infty} e^{-y} \left(\int_{0}^{\infty} \frac{x}{y} e^{-\frac{x}{y}}\mathrm{d}x\right)\mathrm{d}y\\
    &= \int_{0}^{\infty} e^{-y}\cdot \mathbb{E}\left[\text{exp}\left(\frac{1}{y}\right)\right] \mathrm{d}y\ \ \ \ \ \ (\star)\\
    &= \int_{0}^{\infty} e^{-y}\cdot \frac{1}{1/y} \mathrm{d}y\\
    &= \int_{0}^{\infty} ye^{-y} \mathrm{d}y\\
    &= -(y+1)e^{-y}|_0^{\infty}\\
    &= 1
    \end{align}
    $$
    
  - ③ 注意到:   
    $$
    \begin{align}
    \mathbb{E}(XY) 
    &= \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} xyf(x,y)\mathrm{d}x\mathrm{d}y\\
    &= \int_0^{\infty}\int_{0}^{\infty} xy\cdot\frac{1}{y}e^{-(y+\frac{x}{y})}\mathrm{d}x\mathrm{d}y\\
    &= \int_0^{\infty}ye^{-y}\left(\int_0^{\infty}\frac{x}{y}e^{-\frac{x}{y}}\mathrm{d}x\right)\mathrm{d}y\\
    &= \int_0^{\infty}ye^{-y}\cdot \mathbb{E}\left[\text{exp}\left(\frac{1}{y}\right)\right]\mathrm{d}y\\
    &= \int_0^{\infty}ye^{-y}\cdot \frac{1}{1/y}\mathrm{d}y\\
    &= \int_0^{\infty}y^2e^{-y}\mathrm{d}y\\
    &= \{y^2\cdot (-e^{-y})\}|_0^{\infty} - \int_0^{\infty}2y\cdot (-e^{-y})\mathrm{d}y\\
    &= 0 - 2(y+1)e^{-y}|_0^{\infty}\\
    &= 2
    \end{align}
    $$
    
  - ④ 综上所述:   
    $$
    \begin{align}
    \text{Cov}(X,Y) &= \mathbb{E}(XY)- \mathbb{E}(X)\mathbb{E}(Y)\\
    &= 2-1\cdot 1\\
    &=1
    \end{align}
    $$

***

**协方差的重要性质: **   
对于任意随机变量 $X,Y,Z$ 和常数 $c$

  - $\text{Cov}(X,X)=\text{Var}(X)$

  - $\text{Cov}(X,Y)=\text{Cov}(Y,X)$

  - $\text{Cov}(cX,Y)=c\cdot \text{Cov}(X,Y)$

  - $\text{Cov}(X,Y+Z)=\text{Cov}(X,Y) + \text{Cov}(X,Z)$  
    第四点的证明:   
    $$
    \begin{align}
    \text{Cov}(X,Y+Z) &= \mathbb{E}[X(Y+Z)] - \mathbb{E}(X)\cdot \mathbb{E}(Y+Z)\\
    &= \mathbb{E}(XY) - \mathbb{E}(X)\mathbb{E}(Y) + \mathbb{E}(XZ) - \mathbb{E}(X)\mathbb{E}(Z)\\
    &=\text{Cov}(X,Y) + \text{Cov}(X,Z)
    \end{align}
    $$
    容易将第四点推广:   
    $$
    \text{Cov}\left(\sum_{i=1}^nX_i,\sum_{j=1}^m Y_j\right) = \sum_{i=1}^n \sum_{j=1}^m \text{Cov}(X_i,Y_j)
    $$
    进而得到一个有用的公式:   
    $$
    \begin{align}
    \text{Var}\left(\sum_{i=1}^nX_i\right) &= \text{Cov}\left(\sum_{i=1}^nX_i, \sum_{j=1}^n X_j\right)\\
    &= \sum_{i=1}^n\underset{j=1}{\overset{n}\sum} \text{Cov}(X_i,X_j)\\
    &= \sum_{i=1}^n\text{Cov}(X_i,X_i) + \sum_{i=1}^n \sum_{j\neq i} \text{Cov}(X_i,X_j)\\
    &= \sum_{i=1}^n\text{Var}(X_i) + 2\sum_{i=1}^n \sum_{j< i}\text{Cov}(X_i,X_j)
    \end{align}
    $$
    若 $X_1,\dots,X_n$ 相互独立，则公式化简为:   
    $$
    \text{Var}\left(\sum_{i=1}^nX_i\right) =\sum_{i=1}^n\text{Var}(X_i)
    $$

***

若 $X_1,\dots,X_n$ 独立同分布，则称随机变量 $\overline X=\frac1{n}\sum_{i=1}^nX_i$ 为**样本均值** (sample mean)    
下面的命题说明了样本均值的一些性质:   
若 $X_1,\dots,X_n$ 独立同分布，且具有期望 $\mu$ 和方差 $\sigma^2$，则有:

  - ① $\mathbb{E}(\overline X)=\mu$
  - ② $\text{Var}(\overline X)=\frac{\sigma^2}{n}$
  - ③ $\text{Cov}(\overline X,X_i-\overline X)=0\ \ (\forall\ i=1,2,\dots,n)$

**证明: **

  - ① 证明 $\mathbb{E}(\overline X)=\mu$:  
    $$
    \begin{align}  
    \mathbb{E}[\overline X] 
    &=\frac1{n}\sum_{i=1}^n\mathbb{E}[X_i]\\
    &=\frac1{n}\cdot n\mu\\
    &=\mu
    \end{align}
    $$
    
  - ② 证明 $\text{Var}(\overline X)=\frac{\sigma^2}{n}$:  
    $$
    \begin{align}
    \text{Var}(\overline X)
    &=\frac{1}{n^2}\cdot\text{Var}\left(\sum_{i=1}^nX_i\right)\\
    &=\frac{1}{n^2}\cdot\sum_{i=1}^n\text{Var}(X_i)\\
    &=\frac{1}{n^2}\cdot n\sigma^2\\
    &=\frac{\sigma^2}{n}
    \end{align}
    $$
    
  - ③ 证明 $\text{Cov}(\overline X,X_i-\overline X)=0\ \ (\forall\ i=1,2,\dots,n)$:  
    对于任意 $i=1,2,\dots,n$，我们都有:
    $$
    \begin{align}
    \text{Cov}(\overline X,X_i-\overline X) 
    &= \text{Cov}(\overline X,X_i)-\text{Cov}(\overline{X},\overline X)\\
    &= \frac{1}{n}\text{Cov}\left(X_i+ \sum_{j\neq i} X_j,X_i\right) - \text{Var}(\overline X)\\
    &= \frac{1}{n}\text{Cov}(X_i,X_i) + \frac{1}{n}\text{Cov}\left(\sum_{j\neq i} X_j,X_i\right) - \frac{\sigma^2}{n}\\
    &= \frac{\sigma^2}{n}+0-\frac{\sigma^2}{n}\\
    &= 0
    \end{align}
    $$
    这说明样本均值与任一样本偏差之间的协方差为零，即二者之间不存在线性依赖.  
    这也暗示着样本均值是理论均值的**无偏估计量** (unbiased estimator).  

## 1.3.4 随机变量之和的方差

在 $1.3.3$ 节中我们得到了一个有用的公式:   
$$
\begin{align}
\text{Var}\left(\sum_{i=1}^nX_i\right) &= \text{Cov}\left(\sum_{i=1}^nX_i, \sum_{j=1}^n X_j\right)\\
&= \sum_{i=1}^n\underset{j=1}{\overset{n}\sum} \text{Cov}(X_i,X_j)\\
&= \sum_{i=1}^n\text{Cov}(X_i,X_i) + \sum_{i=1}^n \sum_{j\neq i} \text{Cov}(X_i,X_j)\\
&= \sum_{i=1}^n\text{Var}(X_i) + 2\sum_{i=1}^n \sum_{j< i}\text{Cov}(X_i,X_j)
\end{align}
$$
若 $X_1,\dots,X_n$ 相互独立，则公式化简为:   
$$
\text{Var}\left(\sum_{i=1}^nX_i\right) =\sum_{i=1}^n\text{Var}(X_i)
$$
上述公式在计算随机变量之和的方差时常常很有用.

  - **(1) 二项随机变量的方差**  
    二项随机变量 $X \sim \text{B}(n,p)$ 可以看作 $n$ 个相互独立的 Bernoulli 随机变量 $X_1,X_2,\dots,X_n \sim \text{B}(1,p)$ 之和.  
    而我们知道 Bernoulli 随机变量的方差 $\text{Var}(\text{B}(1,p))=p(1-p)$  
    因此有:   
    $$
    \begin{align}
    \text{Var}(X) 
    &= \text{Var}(\sum_{i=1}^nX_i)\\
    &= \sum_{i=1}^n\text{Var}(X_i)\\
    &= np(1-p)
    \end{align}
    $$
    
  - **(2) 从有限的总体中不放回地抽样: 超几何分布 (Hyper-geometric Distribution)**  
    超几何随机变量 $X =\sum_{i=1}^n X_i$ 可以设想为从含有 $Np$ 个白球和 $N(1-p)$ 个黑球的总体中，  
    随机地抽取 $n$ 个球所得到的白球数.  
    其中 $X_i = \begin{cases}
    1,\ \ \ 如果第\ i\ 个球是白球\\
    0,\ \ \ \text{otherwise} 
    \end{cases} \sim \text{B}(1,p)$   
    我们下面计算它的均值和方差:   

      - 由于抽取的第 $i$ 个球等可能地为总体中 $N$ 个球的任意一个，  
        因此有 $\mathbb{E}[\sum_{i=1}^n X_i] = \sum_{i=1}^n \mathbb{E}[X_i] = np$  

      - $\text{Var}(\sum_{i=1}^nX_i) =\sum_{i=1}^n\text{Var}(X_i) + 2\sum_{i=1}^n \sum_{j<i} \text{Cov}(X_i,X_j)$  
        其中 $\text{Var}(X_i) = p(1-p)$  
        而对于 $j\neq i$，有:   
        $$
        \begin{align}
        \text{Cov}(X_i,X_j) 
        &= \mathbb{E}(X_iX_j) - \mathbb{E}(X_i)\mathbb{E}(X_j)\\
        &=\text{P}\{X_i=1,X_j=1\} -p^2\\
        &=\text{P}\{X_i=1\}\text{P}\{X_j=1|X_i=1\}-p^2\\
        &=\frac{Np}{N}\frac{Np-1}{N-1} -p^2\\
        &=\frac{p(Np-1)}{N-1}-p^2
        \end{align}
        $$
        因此有:
        $$
        \begin{align}
        \text{Var}(\sum_{i=1}^nX_i) 
        &=\sum_{i=1}^n\text{Var}(X_i) + 2\sum_{i=1}^n \sum_{j<i} \text{Cov}(X_i,X_j)\\
        &=np(1-p) + 2\binom{n}{2}\left[\frac{p(Np-1)}{N-1} -p^2\right]\\
        &=np(1-p) -\frac{n(n-1)p(1-p)}{N-1}\\
        &=np(1-p)\cdot \frac{N-n}{N-1}
        \end{align}
        $$
        因此 $p$ 的**估计量** $\overline{X} =\frac1n\sum_{i=1}^n X_i$ 的期望和方差为:
        $$
        \begin{cases}
          \mathbb{E}(\overline{X}) = \mathbb{E}[\frac1n\sum_{i=1}^n X_i] = p\\
          \text{Var}(\overline{X}) = \text{Var}[\frac1n\sum_{i=1}^n X_i] = \frac{p(1-p)}{n}\cdot \frac{N-n}{N-1}
          \end{cases}
        $$
        当 $N$ 增大时，估计量的方差 $\text{Var}(\overline{X})$ 也增大，极限值为 $p(1-p)/n$.  
        这并不令人惊讶，因为当 $N$ 足够大时，每一个 $X_i \sim \text{B}(1,p)$ 都将近似地是**独立随机变量**，  
        从而 $\sum_{i=1}^n X_i\sim \text{B}(n,p)$ 近似为**二项分布**.  
        容易知道超几何随机变量的概率质量函数为:
        $$
        \text{P} \left\{\sum_{i=1}^n X_i=k\right\} = \frac{\binom{Np}{k}\binom{N-Np}{n-k}}{\binom{N}{n}}
        $$

****

**当随机变量 $X,Y$ 相互独立时，我们能从 $X$ 和 $Y$ 的分布计算出 $Z=X+Y$ 的分布，这一点非常重要.**  

  - (Ⅰ) 首先假定 $X$ 和 $Y$ 都是**连续的**，则有:   
    $$
    \begin{align}
    F_{X+Y}(a) 
    &= \text{P}\{X+Y\leq a\}\\
    &= \int_{-\infty}^{a} F_{X+Y}(z)\mathrm{d}z\qquad (z:=x+y)\\
    &= \iint_{x+y\leq a} f_X(x)f_Y(y) \mathrm{d}x\mathrm{d}y\qquad (X\ \bot\ Y)\\
    &= \int_{-\infty}^{\infty}\int_{-\infty}^{a-y}f_X(x)f_Y(y)\mathrm{d}x\mathrm{d}y\\
    &= \int_{-\infty}^{\infty}\left(\int_{-\infty}^{a-y}f_X(x)\mathrm{d}x\right)f_Y(y)\mathrm{d}y\\
    &= \int_{-\infty}^{\infty}F_X(a-y)f_Y(y)\mathrm{d}y
    \end{align}
    $$
    总之我们有:
    $$
    \begin{align}
    F_{X+Y}(a) 
    &= \int_{-\infty}^{\infty}F_X(a-y)f_Y(y)\mathrm{d}y\\
    &= \int_{-\infty}^{\infty}f_X(x)F_Y(a-x)\mathrm{d}x
    \end{align}
    $$
    对此式求微分可知:
    $$
    \begin{align}
    f_{X+Y}(a) 
    &= \frac{\mathrm{d}}{\mathrm{d}a}\int_{-\infty}^{\infty} F_X(a-y)f_Y(y)\mathrm{d}y\\
    &= \int_{-\infty}^{\infty} \frac{\mathrm{d}}{\mathrm{d}a}(F_X(a-y))f_Y(y)\mathrm{d}y\\
    &= \int_{-\infty}^{\infty}f_X(a-y)f_Y(y)\mathrm{d}y
    \end{align}
    $$
    我们称 $f_{X+Y}$ 为 $f_X$ 和 $f_Y$ 的**卷积** (convolution).
    
    ***
    
    **一个简单的例子: **  
    设 $\begin{cases}
    X,Y \sim \text{Uniform}(0,1)\\
    X\ \bot\ Y\end{cases}$ ，考虑 $X+Y$ 的概率密度函数:   
    $$
    \begin{align}
    f_{X+Y}(a) 
    &= \int_{-\infty}^{\infty}f_X(a-y)f_Y(y)\mathrm{d}y\\
    &= \int_{0}^{1}f_X(a-y)\cdot 1\mathrm{d}y\\
    &= \int_{0}^{1}f_X(a-y)\mathrm{d}y\\
    \end{align}
    $$
    当 $0\leq a\leq 1$ 时，$f_{X+Y}(a) = \int_0^1 1\cdot \mathrm{d}y = a$  
    当 $1<a<2$ 时，$f_{X+Y}(a) = \int_{a-1}^1 1\cdot \mathrm{d}y = 2-a$    
    因此 $f_{X+Y}(a) = \begin{cases}
    a,&0\leq a\leq 1\\
    2-a,&1<a<2\\
    0,&\text{otherwise}\end{cases}$  
    
  - (Ⅱ) 我们不继续推导在**离散情形**下 $X+Y$ 分布的一般表达式，而是考察一个例子:    
    设 $\begin{cases}
    X \sim \text{Poisson}(\lambda_1)\\
    Y \sim \text{Poisson}(\lambda_2)\\
    X\ \bot\ Y\end{cases}$ ，考虑 $X+Y$ 的分布:   
    因为事件 $\{X+Y=n\}$ 可以写成一系列不相交事件 $\{X=k,Y=n-k\}\ (0\leq k\leq n)$ 的并，故有:   
    $$
    \begin{align}
    \text{P}(X+Y=n) 
    &= \sum_{k=0}^n\text{P}\{X=k,Y=n-k\}\\
    &= \sum_{k=0}^n\text{P}\{X=k\}\cdot\text{P}\{Y=n-k\}\\
    &= \sum_{k=0}^n e^{-\lambda_1}\frac{\lambda_1^k}{k!}\cdot e^{-\lambda_2}\frac{\lambda_2^{(n-k)}}{(n-k)!}\\
    &= e^{-(\lambda_1+\lambda_2)}\sum_{k=0}^n
    \frac{\lambda_1^k\lambda_2^{n-k}}{k!(n-k)!}\\
    &= \frac{e^{-(\lambda_1+\lambda_2)}}{n!}\sum_{k=0}^n\binom{n}{k}\lambda_1^k\lambda_2^{n-k}\\
    &= \frac{e^{-(\lambda_1+\lambda_2)}}{n!}(\lambda_1+\lambda_2)^n
    \end{align}
    $$
    也就是说，$X+Y$ 有均值为 $\lambda_1+\lambda_2$ 的 Poisson 分布.    
    这称为 **Poisson 分布的再生性** (reproductive property).
    
  - (Ⅲ) 考虑 $n$ 个相互独立的随机变量 $X_1,\dots,X_n$,    
    对于所有的值 $a_1,\dots,a_n$ 都有:   
    $$
    P\{X_1\leq a_1,\dots,X_n\leq a_n\} = \text{P}\{X_1\leq a_1\}\dotsm \text{P}\{X_n\leq a_n\}
    $$
    **一个重要的例子: **  
    假设 $X_1,\dots,X_n$ 是独立同分布的连续随机变量，  
    若记 $X_{(i)}$ 为这些随机变量中第 $i$ 小的值，则称 $X_{(1)},\dots,X_{(n)}$ 为**次序统计量** (order statistics).  
    我们注意到 $X_{(i)}\leq x$  当且仅当 $X_1,\dots,X_n$ 至少有 $i$ 个 $\leq x$  
    因此 $\text{P}\{X_{(i)}\leq x\} = \sum_{k=i}^n\binom{n}{k} 
    (F(x))^k(1-F(x))^{n-k}$  
    微分可得 $X_{(i)}$ 的密度函数如下:   
    $$
    \begin{align}
    f_{X_{(i)}}(x) 
    &= f(x) \sum_{k=i}^n \binom{n}{k} k(F(x))^{k-1}(1 - F(x))^{n-k} -f(x) \sum_{k=i}^n \binom{n}{k} (n-k)(F(x))^k(1 - F(x))^{n-k-1} \\
    &= f(x) \sum_{k=i}^n \frac{n!}{(n-k)!(k-1)!}(F(x))^{k-1}(1 - F(x))^{n-k} 
    -f(x) \sum_{k=i}^{n-1} \frac{n!}{(n-k-1)!k!}(F(x))^k(1 - F(x))^{n-k-1} \\
    &= f(x) \sum_{k=i}^n \frac{n!}{(n-k)!(k-1)!}(F(x))^{k-1}(1 - F(x))^{n-k} 
    -f(x) \sum_{j=i+1}^n \frac{n!}{(n-j)!(j-1)!}(F(x))^{j-1}(1 - F(x))^{n-j} \\
    &= \frac{n!}{(n-i)!(i-1)!} f(x)(F(x))^{i-1}(1 - F(x))^{n-i}\\
    &= nf(x)\cdot\binom{n-1}{i-1}(F(x))^{i-1}(1 - F(x))^{n-i}\\
    &= \binom{n}{1}f(x)\cdot\binom{n-1}{i-1}(F(x))^{i-1}(1 - F(x))^{n-i}
    \end{align}
    $$
    **上述结果非常直观: **  
    为了使 $X_{(i)}=x$，$X_1,\dots,X_n$ 必然有 $i-1$ 个 $\leq x$，有 $n-i$ 个 $\geq x$，且有一个 $= x$，     
    我们将 $X_1,\dots, X_n$ 划分为以上三组，其组合数为 $\binom{n}{1}\cdot \binom{n-1}{i-1}$  
    再乘上对应的值 $f(x)(F(x))^{i-1}(1-F(x))^{n-i}$，就得到 $X_{(i)}$ 的概率密度函数.  



##  1.3.5 随机变量的函数的联合概率分布

设 $X_1,X_2$ 是**联合地连续**的随机变量，具有联合概率密度函数 $f_{X_1,}(x_1,x_2)$  
记 $Y_1 = g_1(X_1,X_2),\ \ Y_2 = g_2(X_1,X_2)$，我们想得到 $Y_1,Y_2$ 的联合分布.  
假定函数 $g_1,g_2$ 满足以下条件:   

- 由方程组 $\begin{cases}y_1 = g_1(x_1,x_2)\\
  y_2 = g_2(x_1,x_2)\end{cases}$ 可以唯一地解出 $x_1,x_2$，记为 $\begin{cases}x_1 = h_1(y_1,y_2)\\
  x_2 = h_2(y_1,y_2)\end{cases}$  
  
- 函数 $g_1,g_2$ 在所有的点 $(x_1,x_2)$ 上有连续的偏导数，且对于任意 $(x_1,x_2)$ 都有:
  $$
  J(x_1,x_2) = \begin{vmatrix} \frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} \\ \frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} \end{vmatrix} = \frac{\partial y_1}{\partial x_1}\frac{\partial y_2}{\partial x_2} - \frac{\partial y_1}{\partial x_2}\frac{\partial y_2}{\partial x_1}\neq 0
  $$

在这两个条件下，可以证明 $Y_1,Y_2$ **联合地连续**，且联合密度函数为:   
$$
f_{Y_1,Y_2}(y_1,y_2) = f_{X_1,X_2}(x_1,x_2)|J(x_1,x_2)|^{-1}
$$
其中 $\begin{cases}x_1 = h_1(y_1,y_2)\\
x_2 = h_2(y_1,y_2)\end{cases}$  

它实际上是通过下式对 $y_1,y_2$ 求微分得到的:
$$
\text{P}\{Y_1\leq y_1, Y_2\leq y_2\} = \underset{(x_1,x_2):\begin{cases}g_1(x_1,x_2)\leq y_1\\ g_2(x_1,x_2)\leq y_2\end{cases}}\iint f_{X_1,X_2}(x_1,x_2)\mathrm{d}x_1\mathrm{d}x_2
$$
**一个具体的例子: (待补充)**

**推广: **    
假设 $X$ 是 $k$ 维连续随机变量，具有概率密度函数 $f_X(\cdot)$   
给定变换 $g:\mathbb R^k \to \mathbb R^k$，记 $Y=g(X)$   
若 $g$ 满足:   

- $g$ 存在逆变换 $h=g^{-1}$ 

- $g$ 一阶连续可求偏导，即在所有 $x$ 上有连续的偏导数，且对于任意 $x$ 都有:
  $$
  J(x) = \begin{vmatrix}\frac{\partial y_1}{\partial x_1}&\dots & \frac{\partial y_1}{\partial x_k}\\
  \vdots & &\vdots\\
  \frac{\partial y_k}{\partial x_1} & \dots & \frac{\partial y_k}{\partial x_k}\end{vmatrix}\neq 0
  $$

在这两个条件下，可以证明 $Y$ **联合地连续**，且联合密度函数为:   
$$
f_{Y}(y) = f_{X}(x)|J(x)|^{-1} = f_X(h(y))|J(h(y))|^{-1}
$$

- **推论: **  
  假设 $X$ 是 $k$ 维连续随机变量，具有概率密度函数 $f_X(\cdot)$   
  给定可逆矩阵 $A\in \mathbb R^{k\times k}$ 和向量 $b\in \mathbb R^k$，记 $Y=AX+b$  
  则 $f_Y(y) = \frac{1}{|\det(A)|}f_X(A^{-1}(y-b))$ 



# 1.4 矩母函数 (Moment Generating Function)  

随机变量 $X$ 的**矩母函数** $M_X(t)$ 对任意 $t\in \mathbb R$ 定义为:  

$$
M_X(t) = \mathbb{E}[e^{tX}] = 
\begin{cases}
\sum_{x:\text{pmf}(x)>0} e^{tx}\text{pmf}(x), &\text{Discrete case}\\
\int_{-\infty}^{\infty}e^{tx}f(x)\mathrm{d}x, &\text{Continuous case}\end{cases}
$$
我们称 $\phi(t)$ 为**矩母函数**，是因为 $X$ 所有的**矩** (moment) 都能由 $\phi(t)$ 求微分得到.  
一般地，对于任意 $k\geq 1$，都有 $M^{(k)}_X(t) = \mathbb{E}[X^ke^{tX}]$  
因此有 $\mathbb{E}[X^k] = M_X^{(k)}(0)$ 成立.  
**矩母函数唯一地确定了分布: ** 即随机变量的矩母函数和分布函数之间存在唯一对应.   

对于非负随机变量 $X$，我们也可以定义它的 **Laplace 变换**:   
$$
\mathcal L_X(t) = \mathbb{E}[e^{-tX}] = M_X(-t)\quad (t\geq 0)
$$
当随机变量非负时，与矩母函数相比，Laplace 变换的优点是:   
如果 $X\geq 0$ 且 $t\geq 0$，则 $0\leq e^{-tX}\leq 1$，即 Laplace 变换永远在 $0,1$ 之间.  
**Laplace 变换同样能唯一确定分布.**

## 1.4.1 常见分布的矩母函数:     

### (Ⅰ) 离散情况: 

- **① Bernoulli 随机变量 (二项随机变量的特例)**  
  $X\sim \text B(1,p)$ 有 $\begin{cases}\text{pmf}(1) = p\\ \text{pmf}(0) = 1-p \end{cases}$   且 $\begin{cases}
  \mathbb{E} (X) =  p\\
  \mathbb{E}(X^2) = p\\
  \text{Var}(X) = p(1-p)
  \end{cases}$  
  $$
  \begin{align}
  M_X(t) 
  &=\mathbb{E}[e^{tX}]\\
  &= e^{t\cdot 0}(1-p)+e^{t\cdot 1}p\\
  &= pe^t + 1-p\end{align}
  $$
  因此 $\mathbb{E}[X^k] = M_X^{(k)}(0) = p\ \ (\forall\ k=1,2,\dots)$  
  
- **② 二项随机变量 (Binomial r.v.)**  
  $X\sim \text B(n,p)$ 有 $\text{pmf}(i) =\binom{n}{i}p^i(1-p)^{n-i}\ \ (i=0,1,\dots,n)$ 且 $\begin{cases}
  \mathbb{E} (X) =  np\\
  \mathbb{E}(X^2) = n(n-1)p^2+np\\
  \text{Var}(X) = np(1-p)\end{cases}$  
  $$
  \begin{align}
  M_X(t) 
  &= \mathbb{E}[e^{tX}]\\
  &= \sum_{i=0}^ne^{t\cdot i}\binom{n}{i}p^i(1-p)^{n-i}\\
  &= \sum_{i=0}^n\binom{n}{i}(pe^t)^i(1-p)^{n-i}\\
  &= (pe^t+1-p)^n
  \end{align}
  $$
  
- **③ 几何随机变量 (Geometric r.v.)**  
  $X\sim \text{Geo}(p)$ 有 $\text{pmf}(i) = (1-p)^{i-1}p\ \ (i=1,2,\dots)$ 而 $\begin{cases}
  \mathbb{E} (X) =  \frac{1}{p}\\
  \mathbb{E}(X^2) = \frac{2}{p^2}-\frac{1}{p}\\
  \text{Var}(X) = \frac{1-p}{p^2}\end{cases}$  
  $$
  \begin{align}
  M_X(t) 
  &= \mathbb{E} [e^{tX}]\\
  &= \underset{i=0}{\overset{\infty}\sum}e^{t\cdot i}(1-p)^{i-1}p\\
  &= p\cdot\underset{i=0}{\overset{\infty}\sum}[(1-p)e^t]^i\\
  &=\frac{p}{1 - (1-p)e^t}\end{align}
  $$
  
- **④ Poisson 随机变量**  
  $X\sim \text{Poisson}(\lambda)$ 有 $\text{pmf}(i)= e^{-\lambda} \frac{\lambda^i}{i!}\ \ (i=0,1,\dots)$ 且 $\begin{cases}
  \mathbb{E} (X) =  \lambda\\
  \mathbb{E}(X^2) = \lambda^2 + \lambda\\
  \text{Var}(X) = \lambda\end{cases}$  
  $$
  \begin{align}
  M_X(t) 
  &= \mathbb{E}[e^{tX}]\\
  &= \underset{i=0}{\overset{\infty}\sum} e^{t\cdot i}e^{-\lambda}\frac{\lambda^i}{i!}\\
  &= e^{-\lambda}\underset{i=0}{\overset{\infty}\sum} \frac{(\lambda e^t)^i}{i!}\\
  &= e^{-\lambda}e^{\lambda e^t}\\
  &= e^{\lambda(e^t-1)}\end{align}
  $$

### (Ⅱ) 连续情况:

- **① 均匀随机变量 (Uniform r.v.)**  
  $X\sim \text{Uniform}(a,b)$ 有 $f(x) =\begin{cases} \frac{1}{b-a}, \ \ \text{if}\ \ a<x<b\\ 0,\ \ \ \ \ \ \text{otherwise} \end{cases}$ 且 $\begin{cases}
  \text {E}(X) =\frac{a+b}2\\
  \mathbb{E}(X^2) = \frac{a^2+ab+b^2}{3}\\
  \text{Var}(X) = \frac{(a-b)^2}{12}\end{cases}$  
  $$
  \begin{align}
  M_X(t) &= \mathbb{E}[e^{tX}]\\
  &= \int_a^b e^{tx} \frac{1}{b-a}\mathrm{d}x\\
  &= \frac{e^{bt}-e^{at}}{(b-a)t}\end{align}
  $$
  
- **② 指数随机变量 (Exponential r.v.) (Gamma 随机变量的特例)**  
  $X\sim \text{exp}(\lambda) = \text{Gamma} (1,\lambda)$ 有 $f(x) = \begin{cases} \lambda e^{-\lambda x},\ \ x\geq 0\\ 0,\ \ \ \ \ \ \ \ \ \ \text{otherwise} \end{cases}$ 且 $\begin{cases}
  \mathbb{E}(X) = \frac1\lambda\\
  \mathbb{E}(X^2) = \frac{2}{\lambda^2}\\
  \text{Var}(X)=\frac{1}{\lambda^2}\end{cases}$  
  $$
  \begin{align}
  M_X(t) &= \mathbb{E}[e^{tX}]\\
  &= \int_0^{\infty} e^{tx}\cdot \lambda e^{-\lambda x}\\
  &= \lambda \int_0^{\infty} e^{-(\lambda-t)x}\mathrm{d}x\\
  &= \frac{\lambda}{\lambda-t}\end{align}
  $$
  对于指数分布，$M_X(t)$ 只对小于 $\lambda$ 的 $t$ 值定义.
  (一般形式 Gamma 随机变量的矩母函数参见 ④)
  
- **③ 正态随机变量 (Normal r.v.)**  
  $X\sim \text N(\mu,\sigma^2)$ 有 $f(x)  = \frac{1}{\sqrt{2\pi} \sigma} \exp\{-\frac{(x-\mu)^2}{2\sigma^2}\}\ \ (\forall\ x\in \mathbb R)$ 且 $\begin{cases}
  \mathbb{E}(X) = \mu\\
  \mathbb{E}(X^2) = \sigma^2 + \mu^2\\
  \text{Var}(X) = \sigma^2\end{cases}$  
  考虑标准正态随机向量 $Z\sim \text{N}(0,1)$，其矩母函数为:   
  $$
  \begin{align}
  M_X(t) &= \mathbb{E}[e^{tX}]\\
  &= \int_{-\infty}^{\infty} e^{tx}\cdot \frac{1}{\sqrt{2\pi}} e^{-x^2/2}\mathrm{d}x\\
  &= \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} e^{-(x^2-2tx)/2}\mathrm{d}x\\
  &= e^{t^2/2}\frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} e^{-(x-t)^2/2}\mathrm{d}(x-t)\\
  &= e^{t^2/2} \int_{-\infty}^\infty \frac{1}{\sqrt{2\pi}} e^{-u^2/2} \mathrm{d}u\quad (u:= x-t)\\
  &= e^{t^2/2}\cdot 1\\
  &= e^{t^2/2}\end{align}
  $$
  则 $X = \sigma Z+\mu \sim \text{N}(\mu,\sigma^2)$ 的矩母函数为:   
  $$
  \begin{align}
  M_X(t) &= \mathbb{E}[e^{tX}]\\
  &= \mathbb{E}[e^{t(\sigma Z+\mu)}]\\
  &= e^{t\mu}\mathbb{E}[e^{t\sigma Z}]\\
  &= \exp\left\{\frac{\sigma^2t^2}{2}+\mu t\right\}\end{align}
  $$
  
  (一元正态的矩母函数需要记忆)
  
- **④ Gamma 随机变量**  
  $X\sim \text{Gamma}(\alpha,\lambda)$ 有 $f(x) = \begin{cases} \frac{\lambda e^{-\lambda x}(\lambda x)^{\alpha-1}}{\Gamma(\alpha)},\ \ x\geq 0\\ 0,\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \text{otherwise}  \end{cases}$ 且 $\begin{cases}
  \mathbb{E}(X) = \frac{\alpha}{\lambda}\\
  \mathbb{E}(X^2)= \frac{(\alpha+1)\alpha}{\lambda^2}\\
  \text{Var}(X)=\frac{\alpha}{\lambda^2}\end{cases}$  
  其中 $\Gamma(\alpha)= \int_0^{\infty}e^{-t}t^{\alpha-1}\mathrm{d}t$  
  $$
  \begin{align}
  M_X(t) 
  &= \mathbb{E}[e^{tX}]\\
  &= \int_0^{\infty} e^{tx}\cdot\frac{\lambda e^{-\lambda x}(\lambda x)^{\alpha-1}}{\Gamma(\alpha)}\mathrm{d}x\\
  &=\lambda^\alpha \int_0^{\infty} \frac{e^{-(\lambda-t)x} x^{\alpha-1}}{\Gamma(\alpha)}\mathrm{d}x\\
  &=\frac{\lambda^\alpha}{(\lambda-t)^\alpha} 
  \int_0^{\infty} \frac{e^{-(\lambda-t)x}((\lambda-t)x)^{\alpha-1}}{\Gamma(\alpha)} \mathrm{d}((\lambda-t)x)\\
  &=\frac{\lambda^\alpha}{(\lambda-t)^\alpha}\int_0^\infty \frac{e^{-u} u^{\alpha-1}}{\Gamma(\alpha)} \mathrm{d}u\quad (u:= (\lambda-t)x)\\
  &=\frac{\lambda^\alpha}{(\lambda-t)^\alpha}\cdot 1\\
  &=\left(\frac{\lambda}{\lambda-t}\right)^\alpha \end{align}
  $$
  (对于 Gamma 分布，$M_X(t)$ 只对小于 $\lambda$ 的 $t$ 值定义) 
  
- **⑤ 卡方随机变量 (Chi-Squared r.v.) (Gamma 随机变量的特例)**  
  $X\sim \chi^2(k) = \text{Gamma}(\frac k 2, \frac 12)$ 有 $f(x) = \begin{cases} \frac{e^{-x/2 }x^{k/2-1}}{2^{k/2}\Gamma(k/2)},\ \ x\geq 0\\ 0,\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \text{otherwise}  \end{cases}$  
  根据 ④ 的结论可知 $\begin{cases} \mathbb{E}(X) = \frac{k/2}{1/2} = k\\ \mathbb{E}(X^2) = \frac{(k/2+1)(k/2)}{(1/2)^2} = k(k+2)\\ \text{Var}(X) = \frac{k/2}{(1/2)^2} = 2k \end{cases}$  
  且 $M_X(t) = (\frac{1/2}{1/2-t})^{k/2} = (\frac{1}{1-2t})^{k/2} = (1-2t)^{-k/2}$  



## 1.4.2 独立随机变量之和的矩母函数

矩母函数的一个重要性质是:   
**独立随机变量之和的矩母函数正是单个矩母函数的乘积.**   
具体来说，假设 $X,Y$ 独立，分别具有矩母函数 $M_X(t)$ 和 $M_Y(t)$  
则 $X+Y$ 的矩母函数为:   
$$
\begin{align}
M_{X+Y}(t) 
&= \mathbb{E}[e^{t(X+Y)}]\\
&= \mathbb{E}[e^{tX}e^{tY}]\quad (\text{note that }X\ \bot\ Y)\\
&= \mathbb{E}[e^{tX}]\cdot \mathbb{E}[e^{tY}]\\
&= M_X(t)\cdot M_Y(t)\end{align}
$$

- **① 独立二项随机变量的和: **  

  给定 $\begin{cases}
  X\sim \text{B}(n,p)\\
  Y\sim \text{B}(m,p)\\
  X\ \bot\ Y\end{cases}$  有 $\begin{cases}
  M_X(t) = (pe^t + 1-p)^n\\
  M_Y(t) = (pe^t + 1-p)^m\end{cases}$  
  于是 $M_{X+Y}(t) = M_X(t)\cdot M_Y(t) = (pe^t+1-p)^{m+n}$   
  从而 $X+Y \sim \text{B}(m+n,p)$ 

- **② 独立 Poisson 随机变量的和: **  
  给定 $\begin{cases}
  X\sim \text{Poisson}(\lambda_1)\\
  Y\sim \text{Poisson}(\lambda_2)\\
  X\ \bot\ Y\end{cases}$  有 $\begin{cases}
  M_X(t) = e^{\lambda_1(e^t-1)}\\
  M_Y(t) = e^{\lambda_2(e^t-1)}\end{cases}$  
  于是 $M_{X+Y}(t) = M_X(t)\cdot M_Y(t) = e^{(\lambda_1+\lambda_2)(e^t-1)}$   
  从而 $X+Y \sim \text{Poisson}(\lambda_1+\lambda_2)$
  
- **③ Poisson 范例 (paradigm): **  
  在 $1.2.1$ 节中，我们说明过 Poisson 随机变量可用于近似 $n$ 很大而 $p$ 很小的二项随机变量.   
  即在 $n$ 很大而 $p$ 很小时，二项分布 $\text B(n,p) \approx \text{Poisson}(np)$  
  我们可以继续加强这个结论:   

  - **首先，每次独立试验不必拥有相同的成功概率，只要所有的成功概率都很小即可.**  
    考虑 $n$ 次独立试验 $X_i \sim \text{B}(1,p_i)$，其中 $p_i$ 都很小 $(i=1,\dots,n)$  
    $X_i$ 的矩母函数为:
    $$
    \begin{align}
    M_{X_i}(t) 
    &= p_ie^t + 1-p_i \\
    &= 1 + p_i(e^t-1)\\
    &\approx \exp\{p_i(e^t-1)\}\end{align}
    $$
    (这说明 $X_i\sim \text{B}(1,p_i)$ 近似于 $\text{Poisson}(p_i)$)
    记成功的总次数为 $X = \sum_{i=1}^n X_i$  
    $X$ 的矩母函数为:   
    $$
    \begin{align}
    M_X(t) &= \prod_{i=1}^n M_{X_i}(t)\\
    &\approx \exp\left\{\sum_{i=1}^n p_i \cdot (e^t-1)\right\}\\  
    \end{align}
    $$
    (这说明 $X$ 近似于 $\text{Poisson}(\sum_{i=1}^np_i)$) 
    
  - 其次，对于成功次数近似服从 Poisson 分布的试验，  
    **不仅每次试验不必有相同的成功概率，甚至不需要是独立的，只要它们的依赖性是弱的即可.**  
    一个具体的例子:   
    假设有 $n$ 个人把帽子放在一起，再各自随机选取一个帽子.  
    将随机选取看出 $n$ 次试验，记第 $i$ 次试验成功的事件为 $X_i$，代表第 $i$ 人恰好选取了自己的帽子.  
    由此推出 $\begin{cases}
    \text{P}(X_i) = \frac{1}{n}\\
    \text{P}(X_i|X_j) = \frac{1}{n-1}\quad(\forall\ j\neq i)\end{cases}$  
    虽然这些试验不是相互独立的，但当 $n$ 很大时，它们显示弱的依赖性和小的成功概率.  
    所有当 $n$ 很大时，匹配总数 $X = \sum_{i=1}^n X_i$ 近似服从 $\text{Poisson}(1)$ 分布.  
    
  - 总而言之:   
    **当每次试验成功概率都很小的时候，在 $n$ 次独立 (或弱相依) 的试验中，  
    其成功次数近似是一个 Poisson 随机变量.**  
    这个陈述称为 **Poisson 范例**.
  
- **④ 独立 Gamma 随机变量的和: **   
  给定 $\begin{cases}
  X\sim \text{Gamma}(\alpha_1,\lambda)\\
  Y\sim \text{Gamma}(\alpha_2,\lambda)\\
  X\ \bot\ Y\end{cases}$  有 $\begin{cases}
  M_X(t) = (\frac{\lambda}{\lambda-t})^{\alpha_1}\\
  M_Y(t) = (\frac{\lambda}{\lambda-t})^{\alpha_2}
  \end{cases}$  
  于是 $M_{X+Y}(t) = M_X(t)\cdot M_Y(t) = (\frac{\lambda}{\lambda-t})^{\alpha_1+\alpha_2}$   
  从而 $X+Y \sim \text{Gamma}(\alpha_1+\alpha_2,\lambda)$  
  根据上述结论我们也可以知道:   
  对于独立卡方随机变量 $\begin{cases}
  X\sim \chi^2(k_1) = \text{Gamma}(\frac{k_1}{2},\frac12)\\
  Y\sim \chi^2(k_2) = \text{Gamma}(\frac{k_2}{2},\frac12)\\
  X\ \bot\ Y\end{cases}$  
  有 $X+Y \sim \chi^2(k_1+k_2) = \text{Gamma}(\frac{k_1+k_2}{2},\frac12)$  

- **⑤ 独立正态随机变量的和: **  
  给定 $\begin{cases}
  X\sim \text{N}(\mu_1,\sigma_1^2)\\
  Y\sim \text{N}(\mu_2,\sigma_2^2)\\
  X\ \bot\ Y\end{cases}$  有 $\begin{cases}
  M_X(t) = \exp\{\frac{\sigma_1^2t^2}{2}+\mu_1 t\}\\
  M_Y(t) = \exp\{\frac{\sigma_2^2t^2}{2}+\mu_2 t\}
  \end{cases}$  
  于是 $M_{X+Y}(t) = M_X(t)\cdot M_Y(t) = (\frac{\lambda}{\lambda-t})^{\alpha_1+\alpha_2}$   
  从而 $X+Y \sim \text{N}(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$  



## 1.4.3 联合矩母函数

我们可以定义两个或更多的随机变量的联合矩母函数.  
对于任意 $n$ 个随机变量 $X_1,\dots,X_n$ 和所有的实值 $t_1,\dots,t_n\in \mathbb R$  
联合矩母函数为 $M(t_1,\dots,t_n)= \mathbb{E}[e^{t_1X_1+\dotsm+t_nX_n}]$  
可以证明 $M(t_1,\dots,t_n)$ 唯一地确定 $X_1,\dots,X_n$ 的联合分布.  

- **① 多元正态分布: **  
  令 $Z_1,\dots,Z_n$ 是 $n$ 个相互独立的标准正态随机向量.  
  若对于某些常数 $a_{ij}\ (1\leq i\leq m,1\leq j\leq n)$ 和 $\mu_i\ (1\leq i\leq m)$  
  有 $\begin{bmatrix}
  X_1\\
  \vdots\\
  X_m\end{bmatrix}=
  \begin{bmatrix}
  a_{11} & \dotsm & a_{1n}\\
  \vdots && \vdots\\
  a_{m1} & \dotsm & a_{mn}\end{bmatrix}
  \begin{bmatrix}
  Z_1\\
  \vdots\\
  Z_n\end{bmatrix} + 
  \begin{bmatrix}
  \mu_1\\
  \vdots\\
  \mu_m\end{bmatrix}$ 成立 (简记为 $X=AZ+\mu$)，
  则称 $X_1,\dots,X_m$ 具有(联合)**多元正态分布**.  
  由于独立正态随机变量的和本身就是一个正态随机变量，  
  因此 $X_1,\dots,X_m$ 各自也都是正态随机变量.   
  易知 $X = [X_1,\dots,X_m]^\mathrm{T} \sim \text{N}(\mu,AA^\mathrm{T})$  
  **对比一元正态分布的概率密度函数形式，我们可以写出多元正态分布的概率密度函数: **  
  $$
  f(X) = \frac{1}{(2\pi)^{m/2}\det{(AA^\mathrm{T})}}\exp\left\{-\frac12 (X-\mu)^\mathrm{T} (AA^\mathrm{T})^{-1}(X-\mu)\right\}
  $$
  **下面我们确定 $X_1,\dots,X_m$ 的联合矩母函数: **  
  记 $t = \begin{bmatrix} t_1\\\vdots\\ t_m
  \end{bmatrix}$，则 $Y:=\sum_{i=1}^m t_iX_i = t^\mathrm{T}X \sim \text{N}(t^\mathrm{T}\mu ,t^\mathrm{T}\text{Cov}(X)t) = \text{N}(t^\mathrm{T}\mu,t^\mathrm{T}AA^\mathrm{T}t)$   
  因此有:
  $$
  \begin{align}
  M(t) &= \mathbb{E}[e^{t^\mathrm{T}X}]\\
  &= \mathbb{E}[e^Y]\\
  &= M_Y(1)\\
  &= \exp\left\{\frac{\text{Var}(Y)\cdot 1^2}{2} + \mathbb{E}(Y)\cdot 1\right\}\\
  &= \exp\left\{\frac{1}{2} t^\mathrm{T}AA^\mathrm{T}t + t^\mathrm{T} \mu\right\}\\
  &= \exp\left\{\frac{1}{2} t^\mathrm{T}\text{Cov}(Y)t + t^\mathrm{T}\mathbb{E}(Y)\right\}\end{align}
  $$
  
  这就证明了 $X = [X_1,\dots,X_m]^\mathrm{T}$ 的联合分布由 $\text{E(X)}=\mu$ 和 $\text{Cov}(X)=AA^\mathrm{T}$ 完全确定.    
  
- **② 多元正态随机变量做线性映射后仍是多元正态的**  
  具体来说，若 $X\sim \text{N}(\mu,\Sigma)$，则 $Ax+b \sim \text{N}(A\mu+b,A\Sigma A^\mathrm{T})$  
  **证明: **  
  $$
  \begin{align}
  M_{AX+b}(t) &= \mathbb{E}[e^{t^\mathrm{T}(AX+b)}]\\
  &= e^{b^\mathrm{T}t}\cdot \mathbb{E}[e^{(A^\mathrm{T}t)^\mathrm{T}X}]\\
  &= e^{b^\mathrm{T}t}\cdot M_X(A^{\mathrm{T}}t)\\
  &= e^{b^\mathrm{T}t}\cdot \exp\left\{\frac12 (A^{\mathrm{T}}t)^\mathrm{T}\Sigma (A^{\mathrm{T}}t) + (A^{\mathrm{T}}t)^\mathrm{T}\mu\right\}\quad (X\sim \text{N}(\mu,\Sigma))\\
  &= \exp\{\frac12 t^\mathrm{T}(A\Sigma A^\mathrm{T})t +t^\mathrm{T}(A\mu+b)\}\\
  &= M_{\text{N}(A\mu+b,A\Sigma A^\mathrm{T})}(t)\end{align}
  $$
  得证 $Ax+b \sim \text{N}(A\mu+b,A\Sigma A^\mathrm{T})$  
  **推论: **
  
  - 多元正态随机向量的任意分量都是正态随机向量.
  
  - $n$ 维随机变量 $X\sim \text{N}(\mu,\Sigma)$ 当且仅当对于任意 $\alpha \in \mathbb R^n$ 都有 $\alpha^\mathrm{T}X\sim \text{N}(\alpha^\mathrm{T}\mu,\alpha^\mathrm{T}\Sigma\alpha)$ 成立.  
    **证明: **  
    必要性显然成立，下面验证充分性:   
    $$
    \begin{align} 
    M_{\alpha^\mathrm{T}X}(t) 
    &=\mathbb{E}[e^{t\alpha^\mathrm{T}X}]\\ 
    &= \exp\{\frac12\alpha^\mathrm{T}\Sigma\alpha\cdot t^2 + \alpha^\mathrm{T}\mu\cdot t\}\\
    &= \exp\{\frac12(t\alpha)^\mathrm{T}\Sigma (t\alpha) + (t\alpha)^\mathrm{T}\mu\}\\
    &= M_{\text{N}(\mu,\Sigma)}(t\alpha)\end{align}
    $$
    根据 $\alpha\in \mathbb R^n$ 的任意性，我们知道 $t\alpha$ 可以取到 $\mathbb R^n$ 中的任意一点，  
    说明有 $X\sim \text{N}(\mu,\Sigma)$ 成立.  
    
  - 若 $n$ 维随机变量 $X\sim \text{N}\left(\begin{bmatrix}\mu_1\\
    \vdots\\
    \mu_n\end{bmatrix},\begin{bmatrix}\sigma^2_1 & &\\
    &\ddots&\\
    &&\sigma^2_n \end{bmatrix}\right)$   
    则有 $\begin{cases}
    X_i \sim \text{N}(\mu_i,\sigma_i^2)\ \ (i=1,\dots,n)\\
    X_1,\dots,X_n \text{ are uncorrelated/independent}\end{cases}$ 成立.  
    这个结论可由上一个推论直接得到，也可通过矩母函数证明.  
    它也说明联合多元正态的随机向量之间，**不相关**和**独立**是等价的.  
    注意，必须在联合多元正态的前提条件下，考虑以下示例:   
    给定 $\begin{cases}
    X\sim \text{N}(0,1)\\
    \varepsilon = \begin{cases}
    1, &\frac12\\
    -1, & \frac12\end{cases}\\
    \varepsilon\ \bot\ X\end{cases}$，记 $Y=\varepsilon X$，显然有 $\begin{cases}
    Y\sim \text{N}(0,1)\\
    Y\ \not\bot\ X\end{cases}$ 成立  
    然而我们可以证明 $X,Y$ 不相关:   
    $$
    \begin{align}
    \text{Cov}(X,Y) 
    &= \mathbb{E}(XY)-\mathbb{E}(X)\mathbb{E}(Y)\\
    &= \mathbb{E}(X^2\varepsilon) - 0\cdot 0\\
    &= \mathbb{E}(X^2)\cdot \mathbb{E}(\varepsilon)\\
    &= (0^2+1)\cdot 0\\
    &= 0\end{align}
    $$
  
- **③ 正态总体的样本均值与样本方差的联合分布: **  
  回顾 $1.3.3$ 的内容:
  
  > 若 $X_1,\dots,X_n$ 独立同分布，则称随机变量 $\overline X=\frac1{n}\sum_{i=1}^nX_i$ 为**样本均值** (sample mean)    
  > 下面的命题说明了样本均值的一些性质:  
  > 若 $X_1,\dots,X_n$ 独立同分布，且具有期望 $\mu$ 和方差 $\sigma^2$，则有:   
  >
  >   - ① $\mathbb{E}(\overline X)=\mu$
  >   - ② $\text{Var}(\overline X)=\frac{\sigma^2}{n}$
  >   - ③ $\text{Cov}(\overline X,X_i-\overline X)=0\ \ (\forall\ i=1,2,\dots,n)$
  >
  > 这说明样本均值 $\overline X$ 是理论均值 $\mu$ 的**无偏估计量** (unbiased estimator).   
  > 也说明样本均值 $\overline X$ 与任一样本偏差 $X_i-\overline X$ 之间的协方差为零，即二者之间不存在线性依赖.
  
  我们进而定义**样本方差** (sample variance): 
  若 $X_1,\dots,X_n$ 独立同分布，则称随机变量 $S^2=\frac1{n-1}\sum_{i=1}^n(X_i - \overline X)^2$ 为**样本方差** (sample variance)      
  下面的命题说明了样本方差的一些性质:   
  若 $X_1,\dots,X_n$ 独立同分布，且具有期望 $\mu$ 和方差 $\sigma^2$，则有: 
  
  - ① $\mathbb{E}(S^2) = \sigma^2$ 
  - ② 当 $X_1,\dots,X_n \overset{\text{iid}}{\sim} \text{N}(\mu,\sigma^2)$ 时，我们有 $\begin{cases}
    \overline X \ \bot\ S^2\\
    \overline X \sim \text{N}(\mu,\frac{\sigma^2}{n})\\
    S^2 \sim \sigma^2 \frac{\chi^2(n-1)}{n-1}
    \end{cases}$
  
  **证明: **  
  
  - ① 证明 $\mathbb{E}(S^2) = \sigma^2$:  
    $$
    \begin{align}
    \sum_{i=1}^n (X_i - \overline{X})^2 
    &= \sum_{i=1}^n \left( X_i - \mu + \mu - \overline{X} \right)^2 \\
    &= \sum_{i=1}^n (X_i - \mu)^2 + n(\mu - \overline{X})^2 + 2(\mu - \overline{X})\sum_{i=1}^n (X_i - \mu) \\
    &= \sum_{i=1}^n (X_i - \mu)^2 + n(\mu - \overline{X})^2 + 2(\mu - \overline{X})(n\overline{X} - n\mu) \\
    &= \sum_{i=1}^n (X_i - \mu)^2 + n(\mu - \overline{X})^2 - 2n(\mu - \overline{X})^2\\
    &= \sum_{i=1}^n (X_i - \mu)^2 - n(\mu - \overline{X})^2
    \end{align}
    $$
    (这是一个相当重要的恒等式: $(n-1)S^2 = \sum_{i=1}^n (X_i - \overline{X})^2 = \sum_{i=1}^n (X_i - \mu)^2 - n(\mu - \overline{X})^2$)  
    因此有:   
    $$
    \begin{align}
    \mathbb{E}[(n-1)S^2] 
    &= \mathbb{E}\left[\sum_{i=1}^n (X_i - \mu)^2 - n(\mu - \overline{X})^2\right]\\
    &= \sum_{i=1}^n \mathbb{E}[(X_i - \mu)^2] - n\mathbb{E}[(\overline{X}-\mu)^2]\\
    &= n\sigma^2 - n \text{Var}(\overline X)\quad (\text{note that }\text{Var}(\overline{X})=\frac{\sigma^2}{n})\\
    &= n\sigma^2 - n \cdot \frac{\sigma^2}{n}\\
    &= (n-1)\sigma^2
    \end{align}
    $$
    
    于是有 $\mathbb{E}(S^2) = \sigma^2$  
    
  - ② 证明当 $X_1,\dots,X_n \overset{\text{iid}}{\sim} \text{N}(\mu,\sigma^2)$ 时，有 $\begin{cases}
    \overline X \ \bot\ S^2\\
    \overline X \sim \text{N}(\mu,\frac{\sigma^2}{n})\\
    S^2 \sim \sigma^2 \frac{\chi^2(n-1)}{n-1}
    \end{cases}$ 成立  
    - 证明 $\overline X \ \bot\ S^2$:  
      **在正态假设下，独立性等价于线性无关性.**   
      根据前文结论 $\text{Cov}(\overline X,X_i-\overline X)=0\ \ (\forall\ i=1,2,\dots,n)$  
      可知 $\overline X$ 与偏差序列 $X_i-\overline X\ \ (i=1,\dots,n)$ 是线性无关的，因而是独立的.  
      由此推出 $\overline X$ 独立于样本方差 $S^2=\frac1{n-1}\sum_{i=1}^n(X_i - \overline X)^2$ 
      (对于一般情况，$\overline X$ 与 $S^2$ 一定线性无关，但不一定相互独立)
    - 证明 $\overline X \sim \text{N}(\mu,\frac{\sigma^2}{n})$:   
      由于 $\overline X=\frac1{n}\sum_{i=1}^nX_i$ 是正态随机向量 $X_1,\dots,X_n$ 的线性组合，故也是正态随机向量.  
      根据前文结论 $\begin{cases}
      \mathbb{E}(\overline X)=\mu\\
      \text{Var}(\overline X)=\frac{\sigma^2}{n}
      \end{cases}$ 可知 $\overline X \sim \text{N}(\mu,\frac{\sigma^2}{n})$  
    - 证明 $S^2 \sim \sigma^2 \frac{\chi^2(n-1)}{n-1}$:  
      利用恒等式 $(n-1)S^2 = \sum_{i=1}^n (X_i - \overline{X})^2 = \sum_{i=1}^n (X_i - \mu)^2 - n(\mu - \overline{X})^2$    
      可知 $\frac{(n-1)S^2}{\sigma^2} + (\frac{\overline X-\mu}{\sigma/\sqrt{n}})^2 = \sum_{i=1}^n \frac{(X_i-\mu)^2}{\sigma^2}$    
      由于 $\begin{cases}
      (\frac{\overline X-\mu}{\sigma/\sqrt{n}})^2\sim \text{N}(0,1) = \chi^2(1)\\
      \sum_{i=1}^n \frac{(X_i-\mu)^2}{\sigma^2} \sim \chi^2(n)\\
      (\frac{\overline X-\mu}{\sigma/\sqrt{n}})^2\ \bot\ \sum_{i=1}^n \frac{(X_i-\mu)^2}{\sigma^2}\end{cases}$  
      因此 $\frac{(n-1)S^2}{\sigma^2}\sim \chi^2(n-1)$   
      即 $S^2 \sim \sigma^2 \frac{\chi^2(n-1)}{n-1}$  



# 1.5 发生事件数的分布

考虑任意 $n$ 个事件 $E_1,\dots,E_n$，记这些事件中发生的个数为 $X$  
对于任意 $1\leq k\leq n$，记所有 $\binom{n}{k}$ 组 $k$ 个不同事件的交的概率之和为 $S_k = \sum_{i_1< \dotsm < i_k} \text{P}(E_{i_1}\dotsm E_{i_k})$  
我们希望证明 $\begin{cases}
\text{pmf}(k) = \text{P}(X=k) = \sum_{j=k}^n (-1)^{j-k}\binom{j}{k}S_j\\
\text{P}(X\geq k) = \sum_{j=k}^n (-1)^{j-k}\binom{j-1}{k-1}S_j
\end{cases}$ 

**(Ⅰ) 证明** $\text{P}(X=k) = \sum_{j=k}^n (-1)^{k+j}\binom{j}{k}S_j$  
我们固定 $n$ 个事件中的 $k$ 个，记为 $E_{i_1},\dots,E_{i_k}$  
记这 $k$ 个事件都发生的事件为 $A = \bigcap_{j\in \{i_1,\dots,i_k\}} E_{j}$  
记其余 $n-k$ 个事件都不发生的事件为 $B = \bigcap_{j\in \{i_1,\dots,i_k\}} E_{j}^c$  
于是我们可以用 $AB$ 代表恰好只有 $E_{i_1},\dots,E_{i_k}$ 发生的事件.  
$$
\begin{align}
\text{P}(AB) 
&= \text{P}(A)- \text{P}(AB^c)\\
&= \text{P}(A) - \text{P}[A(\bigcup_{j\notin \{i_1,\dots,i_k\}} E_{j})]\\
&= \text{P}(A) - \text{P}(\bigcup_{j\notin \{i_1,\dots,i_k\}} AE_{j})\\
&= \text{P}(A) - \sum_{s=1}^{n-k}
[(-1)^{s+1} \sum_{j_1<\dotsm<j_s\notin \{i_1,\dots,i_k\}}\text{P}(AE_{j_1}\dotsm E_{j_s})]\qquad(容斥原理)
\end{align}
$$
在所有 $k$ 个不同下标的集合上对上式求和可得:   
$$
\begin{align}
\text{P}(X=k) &= \underset{i_1<\dotsm<i_k}{\sum} \text{P}(E_{i_1}\dotsm E_{i_k}) - \underset{i_1<\dotsm < i_k}{\sum}
\underset{s=1}{\overset{n-k}\sum}
[(-1)^{s+1}\underset{j_1<\dotsm < j_s \notin \{i_1,\dots,i_k\}}{\sum} \text{P}(E_{i_1}\dotsm E_{i_k}\cdot E_{j_1}\dotsm E_{j_s})]\\
&= S_k - \underset{s=1}{\overset{n-k}\sum}(-1)^{s+1}[\underset{i_1<\dotsm < i_k
}{\sum}\underset{j_1<\dotsm < j_s \notin \{i_1,\dots,i_k\}}{\sum}\text{P}(E_{i_1}\dotsm E_{i_k}\cdot E_{j_1}\dotsm E_{j_s})]\\
&= S_k - \underset{s=1}{\overset{n-k}\sum}(-1)^{s+1}[\binom{k+s}{k}\underset{m_1< \dotsm <m_{k+s}}{\sum}\text{P}(E_{m_1}\dotsm E_{m_{k+s}})]\\
&= S_k - \underset{s=1}{\overset{n-k}\sum}(-1)^{s+1}[\binom{k+s}{k}S_{k+s}]\\
&= \underset{s=0}{\overset{n-k}\sum}(-1)^{s}\binom{k+s}{k}S_{k+s}\\
&= \sum_{j=k}^n(-1)^{j-k}\binom{j}{k}S_{j}
\end{align}
$$
**(Ⅱ) 数学归纳法证明** $\text{P}(X\geq k) = \sum_{j=k}^n (-1)^{j-k}\binom{j-1}{k-1}S_j$   

- 当 $k=n$ 时，根据 (Ⅰ) 中的结论有 $\text{P}(X\geq n) = \text{P}(X=n) = S_n$  
  此时命题成立.
  
- 假设命题对 $k+1$ 成立，即 $\text{P}(X\geq k+1) = \sum_{j=k+1}^n (-1)^{j-(k+1)}\binom{j-1}{(k+1)-1}S_j$ 
  则有:
  $$
  \begin{align}
  \text{P}(X\geq k) &= \text{P}(X=k)+\text{P}(X\geq k+1)\\
  &= \sum_{j=k}^n(-1)^{j-k}\binom{j}{k}S_{j} +
  \sum_{j=k+1}^n(-1)^{j-(k+1)}\binom{j-1}{(k+1)-1}S_j\\
  &= S_k + \sum_{j=k+1}^n (-1)^{j-k}[\binom{j}{k}-\binom{j-1}{k}]S_j\\
  &= S_k + \sum_{j=k+1}^n (-1)^{j-k}\binom{j-1}{k-1}S_j\\
  &= \sum_{j=k}^n (-1)^{j-k}\binom{j-1}{k-1}S_j\\\end{align}
  $$

这样就完成了证明.



# 1.6 极限定理

## 1.6.1 Markov 不等式 & Chebyshev 不等式

> 统计学家只懂两种不等式，一是 Markov 不等式，二是 Chebyshev 不等式.

**(Markov 不等式)**  
若 $X$ 为非负随机向量，且 $-\infty<\mathbb{E}(X)<\infty$，  
则对于任意 $a>0$ 都有 $\text{P}\{X\geq a\} \leq \frac{\mathbb{E}[X]}{a}$ 成立.  
**证明: **

- 我们首先假设 $X$ 为连续型随机变量，概率密度函数为 $f$:
  $$
  \begin{align}
  \mathbb{E}[X] 
  &= \int_0^{\infty} xf(x)\mathrm{d}x\\
  &= \int_0^axf(x) \mathrm{d}x + \int_a^{\infty} xf(x)\mathrm{d}x\\
  &\geq \int_a^{\infty} xf(x)\mathrm{d}x\\
  &\geq \int_a^{\infty} af(x)\mathrm{d}x\\
  & = a\int_a^{\infty} f(x)\mathrm{d}x\\
  & = a\cdot \text{P}\{X\geq a\}\end{align}
  $$
  
  于是有 $\text{P}\{X\geq a\} \leq \frac{\mathbb{E}[X]}{a}$ 成立.
  
- 离散情况的证明十分类似，假设概率质量函数为 $\text{pmf}$: 
  $$
  \begin{align}
  \mathbb{E}[X] 
  &= \underset{x: \text{pmf}(x)>0}{\sum} x\cdot\text{pmf}(x)\\
  &= \underset{\begin{subarray}{} x: \text{pmf}(x)>0\\ 0\leq x<a\end{subarray}}{\sum} x\cdot\text{pmf}(x) + \underset{\begin{subarray}{} x: \text{pmf}(x)>0\\ x\geq a\end{subarray}}{\sum} x\cdot\text{pmf}(x)\\
  &\geq \underset{\begin{subarray}{} x: \text{pmf}(x)>0\\ x\geq a\end{subarray}}{\sum} x\cdot\text{pmf}(x)\\
  &\geq \underset{\begin{subarray}{} x: \text{pmf}(x)>0\\ x\geq a\end{subarray}}{\sum} a\cdot\text{pmf}(x)\\
  & = a\underset{\begin{subarray}{} x: \text{pmf}(x)>0\\ x\geq a\end{subarray}}{\sum} \text{pmf}(x)\\
  & = a\cdot \text{P}\{X\geq a\}\end{align}
  $$

作为 Markov 不等式的直接推论，我们有:

- **命题 (Chernoff 不等式)**  
  若 $X$ 为非负随机向量，且矩母函数 $M_X(t)$ 存在，  
  则对于任意 $a>0$ 都有 $\text{P}(X\geq a) \leq \underset{t>0}{\inf} M_X(t) e^{-at}$ 成立.  
  **证明: **  
  任意给定 $a>0$  
  对于任意 $t >0 $，$e^{tX}$ 都是非负随机变量，因此对其应用 Markov 不等式可得:     
  $$
  \begin{align}
  \text{P}(X\geq a) &= \text{P}(e^{tX} \geq e^{ta})\\
  &\leq \frac{\mathbb{E}[e^{tX}]}{e^{ta}}\quad (\text{Markov Inequality})\\
  &= M_X(t)e^{-at}\end{align}
  $$
  因此我们有:
  $$
  \text{P}(X\geq a) \leq \underset{t>0}{\inf} M_X(t) e^{-at}
  $$
  
- **(Chebyshev 不等式)**  
  若 $X$ 是具有均值 $\mu$ 和 $k\geq 2$ 阶中心距 $\mathbb{E}[|X-\mu|^k]<\infty$ 的随机变量，  
  则对于任意 $a>0$ 都有 $\text{P}\{|X-\mu|\geq a\}\leq \frac{\mathbb{E}[|X-\mu|^k]}{a^k}$ 成立.  
  
  **证明: **  
  由于 $|X-\mu|^k$ 是非负随机变量，因此对其应用 Markov 不等式可得:   
  $$
  \begin{align}
  \text{P}\{|X-\mu|\geq a\} 
  &= \text{P}\{|X-\mu|^k\geq a^k\}\\
  &\leq \frac{\mathbb{E}[|X-\mu|^k]}{a^k}\quad(\text{Markov Inequality})
  \end{align}
  $$

***

**Markov 不等式和 Chebyshev 不等式的重要性在于: **  
在只有概率分布的均值 (或者均值、方差) 已知时，  
它们能为我们提供待求概率的上界，尽管这些上界可能不是非常紧，  
但在许多情况下，这些简单的工具足以提供有用的信息或证明.  
当然，如果真实分布已知，那么我们直接精确计算待求概率就好了.  

**一个具体的例子: **  
设某工厂每日生产的产品数 (记为 $X$) 的均值是 $500$  

- ① 估计某一天生产的产品数至少有 $1000$ 的概率.  
  解: 由于 $X$ 是非负随机变量，故可应用 Markov 不等式，  
  有 $\text{P}\{X\geq 1000\} \leq \frac{\mathbb{E}[X]}{1000} = \frac{500}{1000} = \frac 12$ 成立.  
  
- ② 若今天生产的产品数的方差已知等于 $100$，  
  那么如何估计今天生产的产品数在 $400$ 到 $600$ 之间的概率？  
  解: 应用 Chebyshev 不等式，可得:   
  $$
  \begin{align}
  \text{P}\{|X -500|\geq 100\}
  &\leq \frac{\text{Var}(X)}{100^2}\\
  &= \frac{100}{100^2}\\
  &= \frac{1}{100}\end{align}
  $$
  因此:   
  $$
  \begin{align}
  \text{P}\{400<X<600\} &= 1 -\text{P}\{|X -500|\geq 100\}\\
  &\geq 1-\frac{1}{100}\\
  &= \frac{99}{100}\end{align}
  $$

 

## 1.6.2 随机变量序列的收敛性

### (1) 依分布收敛 (Convergence in Distribution)

**(依分布收敛 & 弱收敛)**  
设随机变量序列 $\{X_n\}$ 的分布函数序列为 $\{F_n\}$，随机变量 $X$ 的分布函数为 $F$  
若对于 $F$ 的任意**连续点** $x$ (即我们不考虑**间断点**)，都有 $\underset{n\to \infty}\lim F_n(x) = F(x)$ 成立  
则称分布函数序列 $\{F_n\}$ **弱收敛**于 $F$，记为 $F_n \overset{\mathrm{w}}\to F$   
也称随机变量序列 $\{X_n\}$ **依分布收敛**于 $X$，记为 $X_n \overset{\mathrm{d}}\to X$   
(它之所以称为弱收敛，是因为它比逐点收敛要弱一些)

### (2) 依概率收敛 (Convergence in Probability)

**(依概率收敛)**  
设 $\{X_n\}$ 为一个随机变量序列，$X$ 为一个随机变量  
若对于任意 $\varepsilon>0$ 都有 $\underset{n\rightarrow \infty}\lim\text{P}(|X_n-X|>\varepsilon) = 0$  
则称序列 $\{X_n\}$ 依概率收敛于 $X$，记作 $X_n \overset{\mathrm{p}}\to X$  
特别地，当 $X$ 退化为常数 $c$ 处的单点分布 (即 $\text{P}(X=c)=1$) 时，  
则称序列 $\{X_n\}$ 收敛于常数 $c$，记作 $X_n \overset{\mathrm{p}}\to c$  

- **依概率收敛**是一种比**依分布收敛**更强的收敛性，即 $X_n\overset{\mathrm{p}}\to X\ \Rightarrow\ X_n\overset{\mathrm{d}}\to X$  
  **证明: **  
  设随机变量序列 $\{X_n\}$ 的分布函数序列为 $\{F_n\}$，随机变量 $X$ 的分布函数为 $F$    
  假设 $X_n\overset{\mathrm{p}}\to X$，我们要证明 $X_n\overset{\mathrm{d}}\to X$，即要证 $F_n \overset{\mathrm{w}}\to F$  
  **只需证明: **  
  对于任意 $x\in \mathbb R$ 都有 $F(x-0) \leq \underset{n\to\infty}{\underline{\lim}} F_n(x)\leq 
  \underset{n\to\infty}{\overline{\lim}} F_n(x) \leq F(x+0)\ \ (\star)$ 成立.  
  因为若上式成立，  
  则当 $x$ 为 $F$ 的连续点时，有 $F(x-0)=F(x+0)$，进而有 $\underset{n\to \infty}\lim F_n(x) = F(x)$ 成立，  
  即可得 $F_n \overset{\mathrm{w}}\to F$  
  **为证明 $(\star)$ 式成立，不妨取 $x’<x$，则有: **   
  $$
  \begin{align}
  \{X\leq x'\} &= \{X\leq x',X_n\leq x\}\cup \{X\leq x',X_n> x\}\\
  &\subset \{X_n\leq x\} \cup \{|X_n-X|> x-x'\}\end{align}
  $$
  从而有 $F(x') \leq F_n(x) + \text{P}(|X_n-X|> x-x')$  
  由 $X_n\overset{\mathrm{p}}\to X$ 可知 $\underset{n\to\infty}\lim\text{P}(|X_n-X|> x-x') = 0$  
  故有 $F(x') \leq \underset{n\to\infty}{\underline{\lim}} F_n(x)$  
  再令 $x'\rightarrow x$，即有 $F(x-0) \leq \underset{n\to\infty}{\underline{\lim}} F_n(x)$  
  **同理可证**  $\underset{n\to\infty}{\overline{\lim}} F_n(x) \leq F(x+0)$  
  因此对于任意 $x\in \mathbb R$ 都有 $F(x-0) \leq \underset{n\to\infty}{\underline{\lim}} F_n(x)\leq 
  \underset{n\to\infty}{\overline{\lim}} F_n(x) \leq F(x+0)\ \ (\star)$ 成立.  
  进而得证 $F_n \overset{\mathrm{w}}\to F$，即 $X_n\overset{\mathrm{d}}\to X$  

- **上述命题的逆命题不成立**，即 $X_n\overset{\mathrm{d}}\to X\quad\not\Rightarrow\quad X_n\overset{\mathrm{p}}\to X$  
  这说明一般情况下，依概率收敛、依分布收敛是不等价的.  
  一个具体的例子:   
  设随机变量 $X\sim \text{N}(0,1)$   
  显然随机变量序列 $\{X_n = -X\}$ 的每一项都与 $X$ 同分布，于是有 $X_n \overset{\mathrm{d}}\to X$  
  但对于任意给定的 $\varepsilon>0$ 和任意 $n=1,2,\dots$，
  $\begin{align}
  \text{P}(|X_n-X|> \varepsilon)
  &=\text{P}(2|X|> \varepsilon)\\
  &= \text{P}(|X|> \frac{\varepsilon}{2})\\
  &\neq 0\end{align}$  
  因此 $\underset{n\to\infty}{\lim} \text{P}(|X_n-X|> \varepsilon) = \text{P}(|X|> \frac{\varepsilon}2)\neq 0$  
  说明 $X_n \overset{\mathrm{p}}{\not\to} X$  
  
- 当极限随机变量 $X$ 退化为单点分布时，依分布收敛等价于依概率收敛.  
  即若 $c$ 为常数，则 $X_n \overset{\mathrm{d}}\to c$ 等价于 $X_n\overset{\mathrm{p}}\to c$ 
  **证明: **  
  必要性已由上一个命题给出，下证充分性:   
  记随机变量序列 $\{X_n\}$ 的分布函数序列为 $\{F_n\}$  
  假设 $X_n \overset{\mathrm{d}}\to c$，则根据定义可知 $F_n \overset{\mathrm{w}}\to F$，其中 $F(x) = \begin{cases}0,&x<c\\1,&x\geq c\end{cases}$  
  所有对于任意 $\varepsilon>0$ 都有:   
  $$
  \begin{align}
  \text{P}(|X_n-c|> \varepsilon) 
  &= \text{P}(X_n> c+\varepsilon) + \text{P}(X_n< c-\varepsilon)\\
  &\leq \text{P}(X_n>c+\frac{\varepsilon}2) + \text{P}(X_n\leq c-\frac{\varepsilon}{2})\\
  &= 1-F_n(c+\frac{\varepsilon}2) + F_n(c-\frac{\varepsilon}2)\end{align}
  $$
  由于 $c+\frac{\varepsilon}2$ 和 $c-\frac{\varepsilon}2$ 均为 $F$ 的连续点，  
  因此根据 $F_n \overset{\mathrm{w}}\to F$ 有 $\begin{cases}
  \underset{n\to \infty}\lim F_n(c+\frac{\varepsilon}2) = F(c+\frac{\varepsilon}2) = 1\\
  \underset{n\to \infty}\lim F_n(c-\frac{\varepsilon}2) = F(c-\frac{\varepsilon}2) = 0\end{cases}$  
  于是 $\underset{n\to \infty}\lim\text{P}(|X_n-c|> \varepsilon)=0$  
  即有 $X_n\overset{\mathrm{p}}\to c$，命题得证.  

### (3) $k$-阶矩收敛 (Convergence in the k-th Moment)

**($k$-阶矩收敛)**  
若随机变量序列 $\{X_n\}$ 和随机变量 $X$ 满足 $\underset{n\rightarrow \infty}{\lim} \mathbb{E}[|X_n-X|^k] = 0$ (其中 $k\in \mathbb N_+$)，  
则称 $\{X_n\}$ **$k$-阶矩收敛**于 $X$，记为 $X_n \overset{L_k}\to X$    
特殊地，若 $k=2$，则称为**均方收敛**，记为 $X_n \overset{L_2}\to X$ 

- **k阶收敛**是一种比**依概率收敛**更强的收敛性，即 $X_n\overset{L_k}\to X\ \Rightarrow\ X_n\overset{\mathrm{p}}\to X$  
  **证明: **  
  假设 $X_n\overset{L_k}\to X$ 成立  
  对于任意给定的 $\varepsilon>0$，根据 Markov 不等式可知:   
  $$
  \begin{align}
  \text{P}(|X_n-X|>\varepsilon) 
  &= \text{P}(|X_n-X|^k>\varepsilon^k)\\
  &\leq \frac{\mathbb{E}[|X_n-X|^k]}{\varepsilon^k}\rightarrow 0\quad(n\rightarrow \infty)\end{align}
  $$
  说明有 $X_n\overset{\mathrm{p}}\to X$  成立，命题得证.
  
- **上述命题的逆命题不成立**，即 $X_n\overset{\mathrm{p}}\to X\ \not\Rightarrow\ X_n\overset{L_k}\to X$  
  这说明**一般情况下，$k$-阶矩收敛、依概率收敛是不等价的**.    
  一个具体的例子:   
  设 $U \sim \text{Uniform}(0,1)$，记 $X_n = \sqrt{n}\ I_{(0,\frac1n)}(U) = \begin{cases}
  \sqrt{n},&U\in (0,\frac1n)\\
  0,&\text{otherwise}\end{cases}\ \ (\forall\ n=1,2,\dots)$  
  对于任意给定的 $\varepsilon>0$，都存在正整数 $N=\lceil\varepsilon^2\rceil$ 使得对于任意 $n\geq N$ 都有: 
  $$
  \begin{align}
  \text{P}(|X_n|> \varepsilon) 
  &= \text{P}(\sqrt{n}\ I_{(0,\frac1n)}(U)> \varepsilon)\\
  &= \text{P}(0< U < \frac1n)\\
  &= \frac1n\to 0\quad(n\rightarrow \infty)\end{align} 
  $$
  因此 $\underset{n\to\infty}\lim \text{P}(|X_n|> \varepsilon) = 0$，说明 $X_n\overset{\mathrm{p}}\to 0$  
  但是对于任意 $n=1,2,\dots$ 都有 $\mathbb{E}[|X_n|^k] = \int_0^{\frac1n} (\sqrt{n})^k \mathrm{d}x = n^{\frac{k}{2}}\cdot \frac1n = n^{\frac{k}{2}-1}$  
  因此 $\underset{n\rightarrow \infty}{\lim} \mathbb{E}[|X_n|^k] = \underset{n\rightarrow \infty}{\lim}n^{\frac{k}{2}-1} = \begin{cases}
  0,&\text{if }k=1\\
  1, &\text{if }k=2\\
  \infty, &\text{if }k=3,4,\dots\end{cases}$  
  说明除了 $X_n\overset{L_1}\to X$ 成立以外，对于任意 $k=2,3,\dots$，$k$-阶矩收敛都不成立，即 $X_n\overset{L_k}{\not\to}X\ \ (k\geq 2)$  
  
- **($k$-阶矩收敛的性质)**   
  
  - 若 $\begin{cases}
    X_n \overset{\mathrm{p}}\to X\\
    X_n \overset{\mathrm{p}}\to Y\end{cases}$，则有 $X\overset{\text{a.s.}}=Y$  
    根据上一个命题得到推论: 若 $\begin{cases}
    X_n \overset{L_k}\to X\\
    X_n \overset{L_k}\to Y\end{cases}$，则有 $X\overset{\text{a.s.}}=Y$ 
  - 若 $X_n \overset{\mathrm{p}}\to X$，且存在常数 $M>0$ 使得 $|X_n|\overset{\text{a.s.}}\leq M$，则有 $X_n \overset{L_k}\to X$ 成立.  
  - 若 $X_n \overset{L_k}\to X$，则 $\underset{n\to\infty}{\lim}\mathbb{E}[X_n] = \mathbb{E}[X]$  
  
- **(随机变量序列收敛性的变换规则)**  
  设 $\{X_n\},\{ Y_n\}$ 为随机变量序列，$X,Y$ 为随机变量.
  
  - **① 加减: **  
    - 若 $\begin{cases}
      X_n\overset{L_k}\to X\\
      Y_n\overset{L_k}\to Y\end{cases}$ ，则 $X_n\pm Y_n \overset{L_k}\to X\pm Y$  
    - 若 $\begin{cases}
      X_n\overset{\mathrm{p}}\to X\\
      Y_n\overset{\mathrm{p}}\to Y\end{cases}$ ，则 $X_n\pm Y_n \overset{\mathrm{p}}\to X\pm Y$  
    - 若 $\begin{cases}
      X_n\overset{\mathrm{d}}\to X\\
      Y_n\overset{\mathrm{d}}\to c\ \ \ (\text{i.e. }Y_n\overset{\mathrm{p}}\to c)\end{cases}$ ，则 $X_n\pm Y_n \overset{\mathrm{d}}\to X\pm c\ \ \text{(Slutzky)}$   
      一般来说，对于依分布收敛，$\begin{cases}
      X_n\overset{\mathrm{d}}\to X\\
      Y_n\overset{\mathrm{d}}\to Y\end{cases} \ \ \not\Rightarrow\ \ X_n\pm Y_n \overset{\mathrm{d}}\to X\pm Y$ 
  - **② 乘除: ** 
    - 若 $\begin{cases}
      X_n\overset{\mathrm{p}}\to X\\
      Y_n\overset{\mathrm{p}}\to Y\end{cases}$ ，则 $X_nY_n \overset{\mathrm{p}}\to XY$   
      当 $Y\neq 0$ 时，进一步成立 $\frac{X_n}{Y_n} \overset{\mathrm{p}}\to \frac{X}{Y}$ 
    - 若 $\begin{cases}
      X_n\overset{\mathrm{d}}\to X\\
      Y_n\overset{\mathrm{d}}\to c\ \ \ (\text{i.e. }Y_n\overset{\mathrm{p}}\to c)\end{cases}$ ，则 $X_nY_n \overset{\mathrm{d}}\to cX\ \ \text{(Slutzky)}$    
      当 $c\neq 0$ 时，进一步成立 $\frac{X_n}{Y_n} \overset{\mathrm{d}}\to \frac{X}{c}$   
      一般来说，对于依分布收敛，$\begin{cases}
      X_n\overset{\mathrm{d}}\to X\\
      Y_n\overset{\mathrm{d}}\to Y\end{cases} \ \ \not\Rightarrow\ \ X_nY_n \overset{\mathrm{d}}\to XY$ 
  - **③ 对于连续函数 $g$: **  
    - 若 $X_n\overset{\mathrm{p}}\to X$，则 $g(X_n)\overset{\mathrm{p}}\to g(X)$
    - 若 $X_n\overset{\mathrm{d}}\to X$，则 $g(X_n)\overset{\mathrm{d}}\to g(X)$   

### (4) 几乎处处收敛 (Almost Sure Convergence)

**(几乎处处收敛)**   
若随机变量序列 $\{X_n\}$ 和随机变量 $X$ 满足 $\underset{n\to \infty}{\lim}\text{P}(X_n=X) = 1$，
则称 $\{X_n\}$ **几乎处处收敛**于 $X$，记作 $X_n\overset{\text{a.s.}}\to X$  (又称为**依概率 $1$ 收敛**)

- **等价定义: **  
  若存在零概率事件 $\Omega_0\in \Omega\ (\text{P}(\Omega_0) = 0)$，  
  使得对于任意事件 $\omega\in \Omega\backslash\Omega_0$ 都有 $\underset{n\rightarrow \infty}{\lim} X_n(\omega)=X(\omega)$ (数列收敛) 成立，  
  则称 $\{X_n\}$ **几乎处处收敛**于 $X$，记作 $X_n\overset{\text{a.s.}}\to X$ 


**(几乎处处收敛的充要条件)**  
$X_n\overset{\text{a.s.}}\to X$ 当且仅当对于任意 $\varepsilon>0$   
都有 $\underset{N\to \infty}{\lim} \text{P}\{\omega\in \Omega:\underset{n\geq N}{\sup}|X_n(\omega)-X(\omega)| > \varepsilon\} =  \underset{N\to \infty}{\lim}\text{P} \{\underset{n=N}{\overset{\infty}\bigcup}\{|X_n-X|>\varepsilon\}\} = 0$ 成立.  
**证明: **  

- 对于任意给定的 $\omega \in \Omega$，$\underset{n\rightarrow \infty}{\lim} X_n(\omega)=X(\omega)$ (**收敛**) 意味着:   
  对于任意 $\varepsilon>0$，都存在 $N\in \mathbb N_+$ 使得对于任意 $n>N$ 都有 $|X_n(\omega)-X(\omega)|\leq\varepsilon$ 成立  
  即事件 $\underset{\varepsilon>0}{\bigcap}\underset{N\in\mathbb N_+}{\bigcup}\underset{n>N}{\bigcap}\{|X_n(\omega)-X(\omega)|\leq\varepsilon\}$ 成立
- 相反地，对于任意给定的 $\omega \in \Omega$，$\{X_n(\omega)\}$ **不收敛**到 $X(\omega)$ 意味着:   
  存在 $\varepsilon_0>0$，对于任意 $N\in \mathbb N_+$ 都存在 $n_0>N$ 使得 $|X_n(\omega)-X(\omega)|>\varepsilon$ 成立  
  即事件 $\underset{\varepsilon>0}{\bigcup}\underset{N\in\mathbb N_+}{\bigcap}\underset{n>N}{\bigcup}\{|X_n(\omega)-X(\omega)|>\varepsilon\}$ 成立 

- 因此 $X_n\overset{\text{a.s.}}{\to} X$ 等价于 $\text{P}\{\omega\in \Omega:\underset{\varepsilon>0}{\bigcup}\underset{N\in\mathbb N_+}{\bigcap}\underset{n>N}{\bigcup}\{|X_n(\omega)-X(\omega)|>\varepsilon\}\}=0$   
  - 对于任意给定的 $\varepsilon>0$，记 $E_\varepsilon=\{\omega\in\Omega:\underset{N\in\mathbb N_+}{\bigcap}\underset{n>N}{\bigcup}\{|X_n(\omega)-X(\omega)|>\varepsilon\}\}$   
    易知对于任意 $0<\varepsilon_1<\varepsilon_2$ 都有 $E_{\varepsilon_1}\supset E_{\varepsilon_2}$ 成立  
    因此 $\text{P}\{\underset{\varepsilon>0}{\bigcup} E_\varepsilon\} = 0$ 等价于 $\underset{\varepsilon \to 0}{\lim}\text{P}\{E_\varepsilon\}= 0$    
    也就是说，对于任意 $\varepsilon>0$ 都有 $\text{P}\{E_\varepsilon\} = \text{P}\{\omega\in\Omega:\underset{N\in\mathbb N_+}{\bigcap}\underset{n>N}{\bigcup}\{|X_n(\omega)-X(\omega)|>\varepsilon\}\}=0$
    
  - 对于任意给定的 $N\in \mathbb N_+$，记 $E_N=\{\omega\in\Omega:\underset{n>N}{\bigcup}\{|X_n(\omega)-X(\omega)|>\varepsilon\}\}$   
    易知对于任意 $N_1<N_2 \in \mathbb N_+$ 都有 $E_{N_1}\subset E_{N_2}$ 成立  
    因此 $\text{P}\{E_\varepsilon\} =\text{P}\{\underset{N\in \mathbb N_+}{\bigcap}E_{N}\} = 0$ 等价于:   
    $$
    \begin{align}
    \underset{N\to \infty}{\lim}\text{P}\{E_N\} 
    &= \text{P}\{\omega\in\Omega:\underset{n>N}{\bigcup}\{|X_n(\omega)-X(\omega)|>\varepsilon\}\}\\
    &= \underset{N\to \infty}{\lim}\text{P} \{\underset{n=N}{\overset{\infty}\bigcup}\{|X_n-X|>\varepsilon\}\}\\
    &= 0\end{align}
    $$

这就说明了 $X_n\overset{\text{a.s.}}{\to} X$ 等价于对于任意 $\varepsilon>0$  都有 $\underset{N\to \infty}{\lim}\text{P} \{\underset{n=N}{\overset{\infty}\bigcup}\{|X_n-X|>\varepsilon\}\} = 0$ 成立.  
命题得证.

***

**推论: (几乎处处收敛的充分条件)**  
若对于任意 $\varepsilon>0$，正项级数 $\underset{n=1}{\overset{\infty}\sum} \text{P}(|X_n-X|>\varepsilon) < +\infty$，则 $X_n\overset{\text{a.s.}}\to X$         
**证明: **    
对于任意 $\varepsilon>0$，正项级数 $\underset{n=1}{\overset{\infty}\sum} \text{P}(|X_n-X|>\varepsilon) < +\infty$   
等价于 $\underset{N\to\infty}{\lim} \underset{n=N}{\overset{\infty}\sum} \text{P}\{|X_n-X|> \varepsilon\}=0$  
注意到: 对于任意 $N\in\mathbb N_+$ 都有 $\text{P}\{\underset{n=N}{\overset{\infty}\bigcup}\{|X_n-X|> \varepsilon\}\}\leq\underset{n=N}{\overset{\infty}\sum} \text{P}\{|X_n-X|> \varepsilon\}$ 成立  
因此有 $\underset{N\to\infty}{\lim}\text{P}\{\underset{n=N}{\overset{\infty}\bigcup}\{|X_n-X|> \varepsilon\}\}=0$ 成立.  
根据**几乎处处收敛的充要条件**可知 $X_n\overset{\text{a.s.}}\to X$，推论得证.



## 1.6.3 极限定理

**(独立同分布 Bernoulli 随机变量序列的极限定理)**   
设 $\{X_n\}$ 是一列独立同分布的 Bernoulli 随机变量，即 $\{X_n\}\overset{\text{iid}}\sim \text{B}(1,p)$  (其中 $0<p<1$)  
记它们的部分和为 $S_n = \sum_{i=1}^nX_i\sim \text{B}(n,p)$，记样本均值 $\overline {X_n} = \frac{S_n}{n}$，则有:   

- **① (Borel 强大数定律): **   
  部分和 $S_n$ 的样本均值 $\overline X_n=\frac{S_n}{n} \overset{\text{a.s.}}\to p$  (**频率收敛到概率**)         
  
  - **证明: **   
    我们利用**几乎处处收敛的充分条件**进行证明，即要证 $\underset{n=1}{\overset{\infty}\sum} \text{P}(|\frac{S_n}{n}-p|>\varepsilon) < +\infty$     
    也即要证 $\underset{n=1}{\overset{\infty}\sum} \text{P}(|\underset{i=1}{\overset{n}{\sum}}(X_i-p)|>n\varepsilon) < +\infty$  
    令 $Z_i:= X_i-p$ (**中心化**)，则 $Z_1,Z_2,\dots$ 独立同分布，且 $Z_i=\begin{cases}
    1-p,&p\\
    -p,&(1-p)\end{cases}$  
    于是问题转化为证明 $\underset{n=1}{\overset{\infty}\sum} \text{P}(|\underset{i=1}{\overset{n}{\sum}}Z_i|>n\varepsilon) < +\infty$  
    中心化的 Bernoulli 随机变量 $Z_i$ 的性质相当好，其任意阶中心矩都是有限的，  
    即对于任意 $k=1,2,\dots$ 都有:   
    $$
    \mathbb{E}[Z_i^k]\leq\mathbb{E}[|Z_i|^k] = (1-p)^kp+p^k(1-p)<p+(1-p)=1
    $$
    在这里，我们只需要使用 **Chebyshev 不等式**将上界放到四阶中心矩即可:   
    对于任意 $n=1,2,\dots$ 都有:  
    $$
    \begin{align}
    \text{P}(|\underset{i=1}{\overset{n}{\sum}}Z_i|>n\varepsilon)
    &\leq \frac{\mathbb{E}[|\sum_{i=1}^nZ_i-0|^4]}{(n\varepsilon)^4}\quad(\text{Chebyshev Inequality})\\
    &= \frac{1}{n^4\varepsilon^4}\mathbb{E}[(\sum_{i=1}^nZ_i)^4]\\
    &= \frac{1}{n^4\varepsilon^4}\mathbb{E}[\underset{i,j,k,l}{\overset{n}\sum}Z_iZ_jZ_kZ_l]\\
    &= \frac{1}{n^4\varepsilon^4}\underset{i,j,k,l}{\overset{n}\sum}\mathbb{E}[Z_iZ_jZ_kZ_l]\quad(\mathbb{E}[Z_i]=0\text{ for all }i=1,\dots,n)\\
    &= \frac{1}{n^4\varepsilon^4}\{0+\underset{i\neq j}{\overset{n}\sum}\mathbb{E}[Z_i^2]\cdot\mathbb{E}[Z_j^2] + \sum_{i=1}^n\mathbb{E}[Z_i^4]\}\\
    &< \frac{1}{n^4\varepsilon^4}\{2\binom{n}{2}(1\cdot 1) + n\cdot 1\}\\
    &= O(\frac{1}{n^2})\end{align}
    $$
    这说明 $\text{P}(|\underset{i=1}{\overset{n}{\sum}}Z_i|>n\varepsilon)$ 是一个 $O(\frac{1}{n^2})$ 的量，  
    因此有 $\underset{n=1}{\overset{\infty}\sum} \text{P}(|\underset{i=1}{\overset{n}{\sum}}Z_i|>n\varepsilon) = O(\frac{1}{n})< +\infty$ 成立  
    这就证明了 $\frac{S_n}{n}\overset{\text{a.s.}}\to p$ (**频率收敛到概率**) 
  
- **② (De Moivre-Laplace 中心极限定理): **   
  部分和 $S_n$ 经过中心化、标准化后有 $\frac{S_n-np}{\sqrt{np(1-p)}} \overset{\mathrm{d}}\to \text{N}(0,1)$ (即 $\frac{\overline X_n-p}{\sqrt{p(1-p)/n}} \overset{\mathrm{d}}\to \text{N}(0,1)$)     
  (其证明过程包含在 **Feller-Lévy 中心极限定理**的证明之中)

***

**(小概率事件的 Poisson 极限定理)**   

- 若实数序列 $\{p_n\}$ 满足 $\begin{cases}
  p_n \in [0,1]\quad(\forall\ n=1,2,\dots)\\
  \underset{n\to\infty}{\lim} p_n = 0\\
  \underset{n\to\infty}{\lim} n\cdot p_n = \lambda >0\\
  \end{cases}$  
  则对于任意整数 $0\leq k\leq n$，都有 $\underset{n\to\infty}{\lim} \binom{n}{k}p_n^k(1-p_n)^k = e^{-\lambda}\frac{\lambda^k}{k!}$ 成立.  
- **推广: ** 对于任意给定的 $n = 1,2,\dots$  
  设 $n$ 个相互独立的 Bernoulli 随机变量 $X_1^{(n)},\dots,X_n^{(n)}$ 分别服从 $\text{B}(1,p^{(n)}_1),\dots,\text{B}(1,p_n^{(n)})$  
  记 $S_n = \sum_{k=1}^nX_k^{(n)}\\$  
  若有 $\begin{cases}
  \underset{n\to\infty}{\lim} \underset{1\leq k\leq n}{\sup} p^{(n)}_k = 0\\
  \underset{n\to\infty}{\lim} \sum_{k=1}^np_k^{(n)} = \lambda>0\\
  \end{cases}$  
  则有 $S_n \overset{\mathrm{d}}\to \text{Poisson}(\lambda)$ 成立.

****

**(弱相关随机变量序列的 Markov 弱大数定律)**    
给定随机变量序列 $\{X_n\}$，记部分和 $S_n = \sum_{i=1}^n X_i$   
若 **Markov 条件**成立，即 $\underset{n\to\infty}{\lim} \frac{\text{Var}(S_n)}{n^2}=0$，  
则有 $\frac{S_n-\mathbb{E}[S_n]}{n}\overset{\mathrm{p}}\to 0$  (即 $\frac{S_n}{n}\overset{\mathrm{p}}\to \frac{\mathbb{E}(S_n)}{n}$)   

- **证明: **  
  要证明 $\frac{S_n-\mathbb{E}[S_n]}{n}\overset{\mathrm{p}}\to 0$，即要证对于任意 $\varepsilon>0$ 都有 $\underset{n\to\infty}{\lim} \text{P}\{|\frac{S_n-\mathbb{E}[S_n]}{n}|> \varepsilon\}=0$ 成立.  
  对于任意 $\varepsilon>0$ 都有:    
  $\begin{align}
  \text{P}\{|\frac{S_n-\mathbb{E}[S_n]}{n}|> \varepsilon\}
  &= \text{P}\{|S_n-\mathbb{E}[S_n]|> n\varepsilon\}\\
  &\leq \frac{\mathbb{E}[|S_n-\mathbb{E}[S_n]|^2]}{(n\varepsilon)^2}\\
  &=\frac{\text{Var}(S_n)}{n^2\varepsilon^2}
  \to 0 \quad (n\to\infty) \end{align}$  
  命题得证.   
  
- **理解 Markov 条件: **  
  $$
  \begin{align}
  \frac{\text{Var}(S_n)}{n^2}
  &= \frac{1}{n^2}\text{Var}(\sum_{i=1}^n X_i)\\
  &= \frac{1}{n^2}\{\sum_{i=1}^n\text{Var}(X_i) + \underset{i\neq j}{\overset{n}\sum} \text{Cov}(X_i,X_j)\}\end{align}
  $$
  因此 $\underset{n\to\infty}{\lim} \frac{\text{Var}(S_n)}{n^2}=0$ 意味着:   
  如果我们限定随机变量序列 $\{X_i\}$ 每一项的方差都有限，   
  (即 $\text{Var}(X_i)< \infty\text{ for all }i=1,2,\dots$，也即 $\begin{cases}
  \mathbb{E}[X_i^2]<\infty\\
  -\infty< \mathbb{E}[X_i] <\infty\end{cases}\text{ for all }i=1,2,\dots$)    
  那么 Markov 条件将限制协方差 $\text{Cov}(X_i,X_j) = o(1) < \infty\text{ for all }i\neq j=1,2,\dots$  
  也就是说，任意两个不同项之间的协方差都比常数阶小，即所谓 **"弱相关性"**

***

**(独立同分布随机变量序列的极限定理)**   
设 $X_1,X_2,\dots$ 是一列独立同分布的随机变量，  
记部分和为 $S_n = \sum_{i=1}^nX_i$，记样本均值 $\overline {X}_n = \frac{S_n}{n}$，则有: 

- **① (Khinchin 弱大数定律): **  
  当且仅当对于任意 $i=1,2,\dots$ 都有 $\mathbb{E}[X_i]=\mu<\infty$ 时，有 $\overline{X}_n=\frac{S_n}{n}\overset{\mathrm{p}}\to \mu = \mathbb{E}[\frac{S_n}{n}]$ 成立  
  
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
  
- **② (Kolmogorov 强大数定律): **
  当且仅当对于任意 $i=1,2,\dots$ 都有 $\begin{cases}
  \mathbb{E}[|X_i|] < \infty\\
  \mathbb{E}[X_i] = \mu\end{cases}$ 时，有 $\overline{X}_n=\frac{S_n}{n}\overset{\text{a.s.}}\to \mu = \mathbb{E}[\frac{S_n}{n}]$ 成立
  
- **③ (Feller-Lévy 中心极限定理): **   
  当且仅当对于任意 $n=1,2,\dots$ 都有 $\begin{cases}
  0<\text{Var}[X_n] = \sigma^2<\infty\\
  -\infty <\mathbb{E}[X_n] = \mu < \infty\end{cases}$ 时，  
  有 $\frac{S_n-\mathbb{E}[S_n]}{\sqrt{\text{Var}(S_n)}}=\frac{S_n-n\mu}{\sqrt{n\sigma^2}}\overset{\mathrm{d}}\to \text{N}(0,1)$ 成立 (即 $\frac{\overline{X_n}-\mu}{\sqrt{\sigma^2/n}}\overset{\mathrm{d}}\to \text{N}(0,1)$)    
  
  - **证明: **  
    要证明 $\frac{S_n-n\mu}{\sqrt{n\sigma^2}}\overset{\mathrm{d}}\to \text{N}(0,1)$，等价于证明当 $n\to\infty$ 时，  
    $\frac{S_n-n\mu}{\sqrt{n\sigma^2}}$ 的**特征函数** $\varphi_n(t)= \mathbb{E}[e^{it\frac{S_n-n\mu}{\sqrt{n\sigma^2}}}]$ **弱收敛**于标准正态分布 $\text{N}(0,1)$ 的**特征函数** $e^{\frac12(it)^2}=e^{-\frac12t^2}$    
    即要说明对于任意 $t\in \mathbb R$ (特征函数在整个 $\mathbb R$ 上连续) 都有 $\underset{n\to\infty}{\lim}\varphi_n(t)= e^{-\frac12 t^2}$ 成立.    
    记 $Z_i:= \frac{X_i-\mu}{\sigma}$，则有 $\frac{S_n-n\mu}{\sqrt{n\sigma^2}} = \frac{\sum_{i=1}^nX_i-n\mu}{\sqrt n \sigma} = \frac{1}{\sqrt n}\sum_{i=1}^n\frac{X_i-\mu}{\sigma} = \frac{1}{\sqrt n}\sum_{i=1}^nZ_i$   
    易知 $\{Z_i\}$ 独立同分布，且 $\begin{cases}
    \mathbb{E}[Z_i]=0\\
    \text{Var}[Z_i]=1\\
    \mathbb{E}[Z_i^2]=1\end{cases}$，它们拥有相同的特征函数，记为 $\varphi(t)$  
    下面计算 $\varphi(t) = \mathbb{E}[e^{itZ}]$ 的**二阶 Taylor 展开式**:     
    根据 $\begin{cases}
    \varphi(0) = \mathbb{E}[e^0] = 1\\
    \varphi'(0) = \mathbb{E}[e^{itZ}iZ]|_{t=0} = i\mathbb{E}[Z]=0\\
    \varphi''(0) = \mathbb{E}[e^{itZ}(iZ)^2]|_{t=0} = i^2\mathbb{E}[Z] = -1\end{cases}$  可知 $\varphi(t) = 1 - \frac12 t^2 + o(t^2)$   
    因此对于任意 $t\in\mathbb R$ 都有:   
    $$
    \begin{align}
    \varphi_n(t) 
    &= \mathbb{E}[e^{it\frac{S_n-n\mu}{\sqrt{n\sigma^2}}}]\\
    &= \mathbb{E}[\exp\{i\frac{t}{\sqrt n}\sum_{i=1}^nZ_i\}]\quad(Z_i\ \bot\ Z_j\text{ for all }i\neq j=1,2,\dots)\\
    &= \prod_{i=1}^n \mathbb{E}[e^{i\frac{t}{\sqrt{n}}Z_i}]\\
    &= [\varphi(\frac{t}{\sqrt{n}})]^n\\
    &= [1-\frac{1}{2}(\frac{t}{\sqrt{n}})^2+o((\frac{t}{\sqrt{n}})^2)]^n\\
    &= [1-\frac{t^2}{2n} + o(\frac{1}{n})]^n\to e^{-\frac12 t^2}\quad(n\to \infty) \end{align}
    $$
    命题得证.

***

**(独立不同分布随机变量序列的极限定理) **  
设 $X_1,X_2,\dots$ 是一列独立的随机变量，  
记部分和为 $S_n = \sum_{i=1}^nX_i$，记样本均值 $\overline {X_n} = \frac{S_n}{n}$，则有: 

- **① (Kolmogorov 强大数定律的推广): **  
  若 $\begin{cases}
  \mathbb{E}[X_k^2] < \infty\ \ (\forall\ k=1,2,\dots)\\
  \underset{k=1}{\overset{\infty}\sum}\frac{1}{k^2} \text{Var}[X_k] < \infty\end{cases}$，则有 $\overline {X_n} - \mathbb{E}[\overline {X_n}] = \overline{X_n} -\frac{1}{n}\sum_{k=1}^n \mathbb{E}[X_k] 
  \overset{\text{a.s.}}{\to} 0$  
- **② (Lindeberg-Feller 中心极限定理): **  
  记 $\begin{cases}
  \mu_i = \mathbb{E}(X_i)\ \ (\forall\ i=1,\dots,n)\\
  \sigma_i^2 = \text{Var}(X_i)\ \ (\forall\ i=1,\dots,n)\\
  V_n = \sqrt{\text{Var}(S_n)} = \sqrt{\sum_{i=1}^n\sigma_i^2} \end{cases}$  
  若 $\{X_n\}$ 满足 **Lindeberg 条件**，  
  即对于任意 $\varepsilon>0$ 都有 $\underset{n\rightarrow \infty}\lim \frac{1}{(\varepsilon V_n)^2} \sum_{i=1}^n \int_{|x-\mu_i|>\varepsilon V_n} (x-\mu_i)^2 f_{X_i}(x)\mathrm{d}x = 0$  
  则有 $\frac{S_n -\mathbb{E}(S_n)}{V_n} = \frac{S_n - \sum_{i=1}^n \mu_i }{\sqrt{\sum_{i=1}^n\sigma_i^2}} \overset{\mathrm{d}}\to \text{N}(0,1)$  
  - **Lindeberg 条件的理解: **  
    $S_n$ 经过中心化、标准化后得到 $\frac{S_n -\mathbb{E}(S_n)}{V_n} = \sum_{i=1}^n\frac{X_i-\mu_i}{V_n}$   
    我们要求每一项 $\frac{X_i-\mu_i}{V_n}$ 均匀地小，  
    即对于任意 $\varepsilon>0$，要求 $\underset{n\to\infty}\lim \text{P}\{\underset{1\leq i\leq n}{\sup} \frac{|X_i-\mu_i|}{V_n} > \varepsilon\} = \underset{n\to\infty}\lim \text{P}\{\underset{1\leq i\leq n}{\sup}|X_i-\mu_i| > \varepsilon V_n\} = 0$  
    由于:   
    $\begin{align}
    \text{P}\{\underset{1\leq i\leq n}{\sup}|X_i-\mu_i| > \varepsilon V_n\}
    &= \text{P}\{\bigcup_{i=1}^n(|X_i-\mu_i|>\varepsilon V_n)\}\\
    &\leq \sum_{i=1}^n\text{P}\{|X_i-\mu_i|>\varepsilon V_n\}\\
    &=\sum_{i=1}^n\int_{|X_i-\mu_i|>\varepsilon V_n} f_{X_i}(x)\mathrm{d}x\\
    &\leq \frac{1}{(\varepsilon V_n)^2}\sum_{i=1}^n\int_{|X_i-\mu_i|>\varepsilon V_n} (x-\mu_i)^2f_{X_i}(x)\mathrm{d}x
    \end{align}$  
    因此我们只需要求 $\underset{n\rightarrow \infty}\lim \frac{1}{(\varepsilon V_n)^2} \sum_{i=1}^n \int_{|x-\mu_i|>\varepsilon V_n} (x-\mu_i)^2 f_{X_i}(x)\mathrm{d}x = 0$   
    这就是 **Lindeberg 条件**.  
    Lindeberg 条件虽然比较一般，但难以验证，  
    下面的 **Lyapunov 条件**则比较容易验证，它只对矩提要求，因而便于应用.  
- **③ (Lyapunov 中心极限定理): **  
  记 $\begin{cases}
  \mu_i = \mathbb{E}(X_i)\ \ (\forall\ i=1,\dots,n)\\
  \sigma_i^2 = \text{Var}(X_i)\ \ (\forall\ i=1,\dots,n)\\
  V_n = \sqrt{\text{Var}(S_n)} = \sqrt{\sum_{i=1}^n\sigma_i^2} \end{cases}$  
  若 $\{X_n\}$ 满足 **Lyapunov 条件**，  
  即存在 $\delta>0$ 满足 $\underset{n\rightarrow \infty}\lim \frac{1}{V_n^{2+\delta}} \sum_{i=1}^n \mathbb{E}(|X_i - \mu_i|^{2+\delta}) = 0$  
  则有 $\frac{S_n -\mathbb{E}(S_n)}{V_n} = \frac{S_n - \sum_{i=1}^n \mu_i }{\sqrt{\sum_{i=1}^n\sigma_i^2}} \overset{\mathrm{d}}\to \text{N}(0,1)$    
  (其中 $\delta>0$ 确保随机变量的高阶绝对中心矩 (比方差高的矩) 相对于它们方差的和增长得足够慢)



# 1.7 条件概率与条件期望

## 1.7.1 离散情形

对于任意两个事件 $E_1,E_2\subseteq \Omega$  
当 $E_2$ 给定且 $\text{P}(E_2)>0$ 时，$E_1$ 的**条件概率**定义为 $\text{P}(E_1|E_2) = \frac{\text{P}(E_1E_2)}{\text{P}(E_2)}$  
若 $X,Y$ 都是离散随机变量，且具有联合概率质量函数 $p(x,y)$，  
则对于所有使 $\text{P}(Y=y)>0$ 的 $y$ 值，在 $Y=y$ 给定的条件下:   

- 定义 $X$ 的**条件概率质量函数**为:   
  $$
  \begin{align}
  p_{X|Y}(x|y) &= \text{P}\{X=x|Y=y\}\\
  &= \frac{\text{P}\{X=x,Y=y\}}{\text{P}\{Y=y\}}\\
  &= \frac{p(x,y)}{p_Y(y)}\end{align}
  $$
  
- 定义 $X$ 的**条件概率分布函数**为:   
  $$
  \begin{align}
  F_{X|Y}(x|y) & = \text{P}\{X\leq x\mid Y= y\}\\
  &= \underset{a\leq x}{\sum}p_{X|Y}(a|y)\end{align}
  $$
  
- 定义 $X$ 的**条件期望**为:   
  $$
  \begin{align}
  \mathbb{E}[X|Y=y] &= \underset{x}\sum x\cdot\text{P}\{X=x|Y=y\}\\
  &= \underset{x}\sum x\cdot p_{X|Y}(x|y)\end{align}
  $$
  **条件期望**具有一般期望的一切性质，诸如 $\begin{cases}
  \mathbb{E}[\sum_{i=1}^nX_i|Y=y] = \sum_{i=1}^n \mathbb{E}[X_i|Y=y]\\
  \mathbb{E}[h(X)|Y=y] = \sum_x h(x)\cdot p_{X|Y}(x|y)\end{cases}$ 
  
- 定义 $X$ 的**条件方差**为:
  $$
  \begin{align}
  \text{Var}(X|Y=y)
  &= \mathbb{E}[(X-\mathbb{E}[X|Y=y])^2|Y=y]\\
  &= \mathbb{E}[X^2|Y=y] - (\mathbb{E}[X|Y=y])^2\end{align}
  $$

换句话说，除了给定事件 $Y=y$ 以外，定义恰如以前所述.  
如果 $X,Y$ 独立，那么条件概率质量函数、条件概率分布函数、条件期望、条件方差都与无条件时一样.  

**两个具体的例子: **  

- 给定$\begin{cases}
  X_1\sim \text{B}(n_1,p)\\
  X_2\sim \text{B}(n_2,p)\\
  X_1\ \bot\ X_2\end{cases}$，计算在 $X_1+X_2 = m$ 给定条件下 $X_1$ 的条件概率质量函数.  
  记 $q = 1-p$  
  易知 $X_1+X_2 \sim \text{B}(n_1+n_2,p)$   
  则对于任意 $0\leq k\leq m$，我们有:   
  $$
  \begin{align}
  \text{P}\{X_1=k|X_1+X_2=m\}
  &= \frac{\text{P}\{X_1=k,X_1+X_2 = m\}}{\text{P}\{X_1+X_2=m\}}\\
  &= \frac{\text{P}\{X_1=k,X_2 = m-k\}}{\text{P}\{X_1+X_2=m\}}\\
  &= \frac{\text{P}\{X_1=k\}\text{P}\{X_2=m-k\}}{\text{P}\{X_1+X_2=m\}}\\
  &= \frac{\binom{n_1}{k}p^kq^{n_1-k}\binom{n_2}{m-k}p^{m-k}q^{n_2-m+k}}{\binom{n_1+n_2}{m}p^m q^{n_1+n_2-m}}\\
  &= \frac{\binom{n_1}{k}\binom{n_2}{m-k}}{\binom{n_1+n_2}{m}}\end{align}
  $$
  这个分布首次见于 $1.3.4 (2)$ 中，名为**超几何分布**   
  直观地看为什么此条件是超几何分布:   
  即使限定总成功次数为 $m$，$n_1+n_2$ 次独立试验依然具有相同的成功概率，  
  因此前 $n_1$ 次试验的成功次数是**超几何随机变量**.    
  我们也可以直观地得到**条件期望** $\mathbb{E}[X_1|X_1+X_2=m] = n_1\cdot \frac{m}{n_1+n_2}$ 的结论，  
  这一点可由定义式 $\mathbb{E}[X_1|X_1+X_2=m]=\underset{k=0}{\overset{m}\sum}\frac{\binom{n_1}{k}\binom{n_2}{m-k}}{\binom{n_1+n_2}{m}}$ 验证.
  
- 给定$\begin{cases}
  X_1\sim \text{Poisson}(\lambda_1)\\
  X_2\sim \text{Poisson}(\lambda_2)\\
  X_1\ \bot\ X_2\end{cases}$，计算在 $X_1+X_2=n$ 给定条件下 $X_1$ 的条件概率质量函数.  
  易知 $X_1+X_2\sim \text{Poisson}(\lambda_1+\lambda_2)$   
  则对于任意 $k = 0,1,\dots$，我们有:   
  $$
  \begin{align}
  \text{P}\{X_1 = k \mid X_1 + X_2 = n\} 
  &= \frac{\text{P}\{X_1 = k, X_1 + X_2 = n\}}{\text{P}\{X_1 + X_2 = n\}} \\ 
  &= \frac{\text{P}\{X_1 = k, X_2 = n - k\}}{\text{P}\{X_1 + X_2 = n\}} \\ 
  &= \frac{\text{P}\{X_1 = k\}\text{P}\{X_2 = n - k\}}{\text{P}\{X_1 + X_2 = n\}}\\
  &= \frac{e^{-\lambda_1} \lambda_1^k}{k!} \frac{e^{-\lambda_2} \lambda_2^{n-k}}{(n - k)!} \left[ \frac{e^{-(\lambda_1+\lambda_2)} (\lambda_1 + \lambda_2)^n}{n!} \right]^{-1} \\ 
  &= \frac{n!}{(n - k)!k!} \frac{\lambda_1^k \lambda_2^{n-k}}{(\lambda_1 + \lambda_2)^n} \\ 
  &= \binom{n}{k} \left( \frac{\lambda_1}{\lambda_1 + \lambda_2} \right)^k \left( \frac{\lambda_2}{\lambda_1 + \lambda_2} \right)^{n-k}
  \end{align}
  $$
  因此在 $X_1+X_2=n$ 给定条件下 $X_1$ 的条件分布是 $\text{B}(n,\frac{\lambda_1}{\lambda_1+\lambda_2})$  
  进而可直接得到 $\mathbb{E}\{X_1|X_1+X_2=n\}=n\frac{\lambda_1}{\lambda_1+\lambda_2}$  



## 1.7.2 连续情形

若 $X,Y$ 都是连续随机变量，且具有联合概率密度函数 $f(x,y)$，  
则对于所有使 $f_Y(y)>0$ 的 $y$ 值，在 $Y=y$ 给定的条件下:   

- 定义 $X$ 的**条件概率密度函数**定义为 $f_{X|Y}(x|y) = \frac{f(x,y)}{f_Y(y)}$  
  上述定义的动机如下:   
  $$
  \begin{align}
  f_{X|Y}(x|y)\mathrm{d}x 
  &= \frac{f(x,y)\mathrm{d}x\mathrm{d}y}{f_Y(y)\mathrm{d}y}\\
  &\approx \frac{\text{P}\{x\leq X\leq x+\mathrm{d}x,y\leq Y\leq y+\mathrm{d}y\}}{\text{P}\{y\leq Y\leq y+\mathrm{d}y\}}\\
  &=\text{P}\{x\leq X\leq x+\mathrm{d}x\mid y\leq Y\leq y+\mathrm{d}y\}\end{align}
  $$
  换句话说，对于小的值 $\mathrm{d}x,\mathrm{d}y$，  
  $f_{X|Y}(x|y)\mathrm{d}x$ 近似地是 $y\leq Y\leq y+\mathrm{d}y$ 给定条件下 $x\leq X\leq x+\mathrm{d}x$ 的条件概率.  
  
- 定义 $X$ 的**条件概率分布函数**为:   
  $$
  \begin{align}
  F_{X|Y}(a|y) & = \text{P}\{X\leq a\mid Y= y\}\\
  &= \int_{-\infty}^{a} f_{X|Y}(x|y)\mathrm{d}x\end{align}
  $$
  
- 定义 $X$ 的**条件期望**为:   
  $$
  \mathbb{E}[X|Y=y] = \int_{-\infty}^{\infty} xf_{X|Y}(x|y)\mathrm{d}x
  $$
  
- 定义 $X$​ 的**条件方差**为:   
  $$
  \begin{align}
  \text{Var}(X|Y=y)
  &= \mathbb{E}[(X-\mathbb{E}[X|Y=y])^2|Y=y]\\
  &= \mathbb{E}[X^2|Y=y] - (\mathbb{E}[X|Y=y])^2\end{align}
  $$

**一些具体的例子: **  

- 设 $X$ 和 $Y$ 有联合密度 $f(x,y) = \begin{cases}
  6xy(2-x-y), &0<x,y<1\\
  0, &\text{otherwise}\end{cases}$   
  对于 $0<y<1$，计算条件期望 $\mathbb{E}[X|Y=y]$:   
  **首先计算条件密度: **  
  $$
  \begin{align}
  f_{X|Y}(x|y) 
  &= \frac{f(x,y)}{f_Y(y)}\\
  &= \frac{6xy(2-x-y)}{\int_0^1 6xy(2-x-y)\mathrm{d}x}\\
  &= \frac{6xy(2-x-y)}{y(4-3y)}\\
  &= \frac{6x(2-x-y)}{4-3y}
  \end{align}
  $$
  **其次计算条件期望:**
  $$
  \begin{align}
  \mathbb{E}[X|Y=y]
  &= \int_0^1 \frac{6x(2-x-y)}{4-3y}\mathrm{d}x\\
  &= \frac{1}{4-3y}[(2-y)2-\frac{6}{4}]\\ 
  &= \frac{5-4y}{8-6y}\end{align}
  $$
  
- 设 $X$ 和 $Y$ 有联合密度 $f(x,y) = \begin{cases}
  \frac12 y e^{-xy}, &0<x<\infty, 0<y<2\\
  0, &\text{otherwise}\end{cases}$  
  对于 $0<y<2$，计算 $e^{X/2}$ 的条件期望 $\mathbb{E}[e^{X/2}|Y=y]$:   
  **首先计算条件密度: **
  $$
  \begin{align}
  f_{X|Y}(x|y) 
  &= \frac{f(x,y)}{f_Y(y)}\\
  &= \frac{\frac{1}{2}ye^{-xy}}{\int_0^{\infty}\frac12 ye^{-xy}\mathrm{d}x}\\
  &= \frac{ye^{-xy}}{-e^{-t}|_0^{\infty}}\quad (t:= xy)\\
  &= ye^{-xy}\end{align}
  $$
  (这里可以看出 $Y$ 的边际分布是 $(0,2)$ 上的均匀分布)
  
  **其次计算条件期望:**
  $$
  \begin{align}
  \mathbb{E}[e^{X/2}|Y=y]
  &= \int_0^{\infty} e^{x/2} f_{X|Y}(x|y)\mathrm{d}x\\
  &= \int_0^{\infty} e^{x/2} \cdot ye^{-xy} \mathrm{d}x\\
  &= y\int_0^{\infty} e^{(\frac12 -y)x} \mathrm{d}x\\
  &= \begin{cases} 
  \frac12 \int_0^{\infty} 1\mathrm{d}x = \infty,\qquad\text{if }y=\frac12\\
  \frac{y}{\frac12-y}\int_0^{\infty} e^{(\frac12 - y)x}\mathrm{d}(\frac12 - y)x=
  \begin{cases}
  \frac{2y}{1-2y} e^t|_0^{\infty} = \infty, &\text{if } 0<y<\frac12\\ 
  \frac{2y}{1-2y} e^t|_{0}^{-\infty} = \frac{2y}{2y-1}, &\text{if } \frac12 < y <2
  \end{cases}\end{cases}\\
  &= 
  \begin{cases}
  \infty, &\text{if } 0<y\leq \frac12\\
  \frac{2y}{2y-1}, &\text{if } \frac12 < y <2
  \end{cases}\end{align}
  $$
  
- **($t$ 分布)**  
  若 $\begin{cases}
  X\sim \text{N}(0,1)\\
  Y\sim \chi^2(n) = \text{Gamma}(\frac{n}{2},\frac12)\\
  X\ \bot\ Y\end{cases}$   
  则称 $T = \frac{X}{\sqrt{Y/n}} = \sqrt{n} \frac{X}{\sqrt{Y}}\sim t(n)$ 为具有 $n$ 个自由度的 $t$ 随机变量.  
  **为计算 $T$ 的概率密度函数，我们首先推导给定 $Y=y$ 时 $T$ 的条件分布: **  
  由于给定 $Y=y$ 时 $T$ 的条件分布是 $T_{Y=y}=\sqrt{\frac{n}{y}}X\sim \text{N}(0,\frac{n}{y})$    
  因此 $f_{T|Y}(t|y) 
  = \sqrt{\frac{y}{2\pi n}}\exp\{-\frac{t^2y}{2n}\}\ \ (\forall\ t\in \mathbb R)$  
  而卡方随机变量 $Y$ 的概率密度函数为 $f_Y(y) = \frac{e^{-y/2}y^{n/2-1}}{2^{n/2}\Gamma(n/2)}\ \ (\forall\ y>0)$  
  于是有:
  $$
  \begin{align}
  f_T(t) 
  &=\int_0^{\infty} f_{T,Y}(t,y)\mathrm{d}y\\
  &=\int_0^{\infty} f_{T|Y}(t|y)f_Y(y)\mathrm{d}y\\
  &=\int_0^{\infty} \sqrt{\frac{y}{2\pi n}}\exp\{-\frac{t^2y}{2n}\}\cdot \frac{e^{-y/2}y^{n/2-1}}{2^{n/2}\Gamma(n/2)} \mathrm{d}y\\
  \end{align}
  $$
  记 $\begin{cases}
  K = \frac{1}{\sqrt{\pi n} 2^{(n+1)/2} \Gamma (n/2)}\\
  c = \frac{t^2}{2n} + \frac{1}{2}\end{cases}$ 则有:   
  $$
  \begin{align}
  f_T(t) 
  &= \frac{1}{K}\int_0^{\infty} e^{-cy}y^{(n-1)/2}\mathrm{d}y\\
  &= \frac{c^{-(n-1)/2 + 1}}{K} \int_0^{\infty} e^{-s} s^{(n-1)/2} \mathrm{d}s\quad (s:= cy)\\
  &= \frac{c^{-(n+1)/2}}{K}\Gamma(\frac{n+1}{2})\\
  &= \frac{\Gamma(\frac{n+1}{2})}{\Gamma(\frac{n}{2})\sqrt{n\pi}} (1+\frac{t^2}{n})^{-(n+1)/2}\quad (\forall\ t\in\mathbb R)\end{align}
  $$
  
- **(例 $3.8$ 待补充)**



## 1.7.3 通过取条件计算期望和方差

值得注意的是，条件期望 $\mathbb{E}[X|Y]$ 本身是一个随机变量.
我们可以将 $\mathbb{E}[X|Y]$ 看作随机变量 $Y$ 的函数，  
它在 $Y=y$ 处的取值是 $\mathbb{E}[X|Y=y]$  
条件期望的一个重要性质是 $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}(X|Y)]$，称为**全期望公式** (Law of Total Expectation)  
用于在给定随机变量的某些条件分布的情况下，计算该随机变量的期望值.  
下面我们证明 $\mathbb{E}[X]=\mathbb{E}[\mathbb{E}[X|Y]] =\begin{cases}
\underset{y}{\sum} \mathbb{E}[X|Y=y] \text{P}\{Y=y\},&若\ Y\ 离散\\
\int_{-\infty}^{\infty} \mathbb{E}[X|Y=y]f_Y(y)\mathrm{d}y,&若\ Y\ 连续\end{cases}$ 

- ① 若 $Y$ 是**离散**随机变量，  
  则有 $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}(X|Y)] = \sum_y \mathbb{E}[X|Y=y]\text{P}\{Y=y\}$    
  **证明: **
  $$
  \begin{align}
  \underset{y}{\sum} \mathbb{E}[X|Y=y]\text{P}\{Y=y\} 
  &= \underset{y}{\sum}\{\underset{x}{\sum}x\text{P}\{X=x|Y=y\}\}\text{P}\{Y=y\}\\  
  &= \underset{y}{\sum}\underset{x}{\sum} x\frac{\text{P}\{X=x,Y=y\}}{\text{P}\{Y=y\}} \text{P}\{Y=y\}\\
  &= \underset{y}{\sum}\underset{x}{\sum} x\text{P}\{X=x,Y=y\}\\
  &= \underset{x}{\sum} x \underset{y}{\sum}\text{P}\{X=x,Y=y\}\\
  &= \underset{x}{\sum} x \text{P}\{X=x\}\\
  &= \mathbb{E}[X]\end{align}
  $$
  
- ② 若 $Y$ 是**连续**随机变量，  
  则有 $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}(X|Y)] = \int_{-\infty}^{\infty} \mathbb{E}[X|Y=y]f_Y(y)\mathrm{d}y$   
  **证明: **
  $$
  \begin{align}
  \int_{-\infty}^{\infty} \mathbb{E}[X|Y=y]f_Y(y)\mathrm{d}y
  &= \int_{-\infty}^{\infty} \{\int_{-\infty}^{\infty} x\cdot f_{X|Y}(x|y)\mathrm{d}x\}f_Y(y)\mathrm{d}y\\
  &= \int_{-\infty}^{\infty}\int_{-\infty}^{\infty}  
  x\frac{f(x,y)}{f_Y(y)}f_Y(y)\mathrm{d}x\mathrm{d}y\\
  &= \int_{-\infty}^{\infty}\int_{-\infty}^{\infty}  
  xf(x,y)\mathrm{d}x\mathrm{d}y\\
  &= \int_{-\infty}^{\infty}x\int_{-\infty}^{\infty}  
  f(x,y)\mathrm{d}y\mathrm{d}x\\
  &= \int_{-\infty}^{\infty} xf_X(x)\mathrm{d}x\\
  &= \mathbb{E}[X]\end{align}
  $$

条件期望也可以用于计算随机变量的方差:   
**全方差公式** (Law of Total Variance): $\text{Var}(X) = \mathbb{E}(\text{Var}(X|Y)) + \text{Var}(\mathbb{E}(X|Y))$   
**证明: **  

- 计算 $\mathbb{E}(\text{Var}(X|Y))$:
  $$
  \begin{align}
  \mathbb{E}(\text{Var}(X|Y))
  &= \mathbb{E}[\mathbb{E}(X^2|Y) - (\mathbb{E}(X|Y))^2]\\
  &= \mathbb{E}[\mathbb{E}(X^2|Y)] - \mathbb{E}[(\mathbb{E}(X|Y))^2]\\
  &= \mathbb{E}(X^2) - \mathbb{E}[(\mathbb{E}(X|Y))^2]\end{align}
  $$
  
- 计算 $\text{Var}(\mathbb{E}(X|Y))$:
  $$
  \begin{align}
  \text{Var}(\mathbb{E}(X|Y))
  &= \mathbb{E}[(\mathbb{E}(X|Y))^2] - (\mathbb{E}[\mathbb{E}(X|Y)])^2\\
  &= \mathbb{E}[(\mathbb{E}(X|Y))^2] - (\mathbb{E}(X))^2\end{align}
  $$
  
- 因此有:
  $$
  \begin{align}
  \mathbb{E}(\text{Var}(X|Y)) + \text{Var}(\mathbb{E}(X|Y))
  &= \mathbb{E}(X^2) - \mathbb{E}[(\mathbb{E}(X|Y))^2] + \mathbb{E}[(\mathbb{E}(X|Y))^2] - (\mathbb{E}(X))^2\\
  &= \mathbb{E}(X^2)- (\mathbb{E}(X))^2\\
  &= \text{Var}(X)\end{align}
  $$

***

**一些具体的例子:**  
**(复合随机变量的期望与方差)**  
设 $X_1,X_2,\dots$ 是一列独立同分布的随机变量，具有均值 $\mathbb{E}[X]=\mu$ 和方差 $\text{Var}(X)=\sigma^2$   
假设它们与取非负整数值的随机变量 $N$ 独立.  
我们称 $S= \sum_{i=1}^N X_i$ 为**复合随机变量**  

- ① 计算 $S= \sum_{i=1}^N X_i$ 的**期望**:     
  注意到:   
  $$
  \begin{align}
  \mathbb{E}[S|N=n]&=\mathbb{E}[\sum_{i=1}^N X_i|N=n]\\
  &= \mathbb{E}[\sum_{i=1}^n X_i | N=n]\\
  &= \mathbb{E}[\sum_{i=1}^n X_i]\quad (N\ \bot\ X_i,\forall \ i=1,2,\dots)\\
  &= \sum_{i=1}^n\mathbb{E}[X_i]\\
  &= n\mu  
  \end{align}
  $$
  因此有:
  $$
  \begin{align}
  \mathbb{E}[S] 
  &= \mathbb{E}[\sum_{i=1}^N X_i]\\
  &= \mathbb{E}[\mathbb{E}[\sum_{i=1}^N X_i|N]]\\
  &= \mathbb{E}[N\mu]\\
  &= \mu \mathbb{E}[N]\end{align}
  $$
  
- ② 计算 $S= \sum_{i=1}^N X_i$ 的**方差**:    
  注意到:   
  $$
  \begin{align}
  \text{Var}(S|N=n)
  &= \text{Var}(\sum_{i=1}^NX_i |N=n)\\
  &= \text{Var}(\sum_{i=1}^nX_i |N=n)\\
  &= \text{Var}(\sum_{i=1}^nX_i)\quad (N\ \bot\ X_i,\forall \ i=1,2,\dots)\\
  &= \sum_{i=1}^n \text{Var}(X_i)\\
  &= n\sigma^2\end{align}
  $$
  因此有:
  $$
  \begin{align}
  \text{Var}(S) 
  &= \mathbb{E}(\text{Var}(S|N)) + \text{Var}(\mathbb{E}(S|N))\\
  &= \mathbb{E}[N\sigma^2] + \text{Var}(N\mu)\\
  &= \sigma^2 \mathbb{E}[N] + \mu^2\text{Var}(N)\end{align}
  $$
  
- 特殊地，如果 $N \sim \text{Possion}(\lambda)$  
  则我们称 $S= \sum_{i=1}^N X_i$ 为**复合 Poisson 随机变量** 
  根据 $\begin{cases}
  \mathbb{E}[S] = \mu\mathbb{E}[N]\\
  \text{Var}(S) = \sigma^2 \mathbb{E}[N] + \mu^2\text{Var}(N)\\
  \mathbb{E}[N] = \text{Var}[N]=\lambda\\
  \mathbb{E}[X^2] = \text{Var}(X) + (\mathbb{E}[X])^2 = \sigma^2 + \mu^2\end{cases}$  
  可知 $\begin{cases}
  \mathbb{E}[S] = \mu\lambda = \lambda \mathbb{E}[X]\\
  \text{Var}(S) = (\sigma^2+ \mu^2)\lambda =\lambda \mathbb{E}[X^2]\end{cases}$ 

****

**(匹配轮数问题)**   
设有 $N$ 个人将帽子混杂后各自抽取一顶帽子，  
取到自己帽子的人离开，而其余的人将帽子混杂后重新抽取，  
这个过程持续进行到所有人都取到了自己的帽子为止.  

- **① 首先我们证明，无论有多少人，平均每轮抽取都有一次匹配: **  
  不失一般性地，考虑第一轮抽取，暂时将 $N$ 视为一个确定的数.  
  记 $X = \sum_{i=1}^N X_i$ 为第一轮抽取的总匹配数，其中 $X_i = \begin{cases}
  1, &第\ i\ 人取到了自己的帽子\\
  0, &\text{otherwise}\end{cases}$   
  对于任意 $i=1,\dots,N$ 都有:    
  $$
  \begin{align}
  \mathbb{E}[X_i] 
  &= 1\cdot\text{P}\{X_i=1\} + 0\cdot \text{P}\{X_i=0\}\\
  &= 1\cdot \frac{1}{N} + 0 \cdot (1-\frac{1}{N})\\
  &=\frac{1}{N} \end{align}
  $$
  因此有 $\mathbb{E}[X] = \mathbb{E}[\sum_{i=1}^N X_i] = \sum_{i=1}^N \mathbb{E}[X_i] = N\cdot \frac{1}{N} = 1$   
  说明对于任一轮抽取，无论有多少人，平均每轮抽取都有一次匹配.
  
- **② 其次我们证明，无论有多少人，总匹配数的方差都是 $1$: **  
  对于任意 $i=1,\dots,N$ 都有 $\text{P}\{X_i=1\} = \frac{1}{N}$，因此有 $\text{Var}(X_i) = \frac{1}{N}(1-\frac{1}{N})$  
  对于任意 $i\neq j \in \{1,\dots,N\}$ 都有 $\text{P}\{X_i=1|X_j=1\} = \frac{1}{N-1}$，  
  于是有:   
  $$
  \begin{align}
  \mathbb{E}[X_iX_j] 
  &= 1^2 \cdot\text{P}\{X_i=1,X_j=1\}\\ 
  &= \text{P}\{X_j=1\}\text{P}\{X_i=1|X_j=1\}\\
  &= \frac{1}{N}\frac{1}{N-1} \\
  &= \frac{1}{N(N-1)}\end{align}
  $$
  因此有:   
  $$
  \begin{align}
  \text{Cov}(X_i,X_j)
  &= \mathbb{E}[X_iX_j] - \mathbb{E}[X_i]\mathbb{E}[X_j]\\
  &= \frac{1}{N(N-1)} - \frac{1}{N}\frac{1}{N}\\
  &= \frac{1}{N^2(N-1)}\end{align}
  $$
  综上所述:   
  $$
  \begin{align}
  \text{Var}(X) 
  &= \underset{i=1}{\overset{N}{\sum}} \text{Var}(X_i) + 2\underset{i<j}{\sum}
  \text{Cov}(X_i,X_j)\\
  &= N\cdot \frac{1}{N}(1-\frac1N) + 2\binom{N}{2}\frac{1}{N^2(N-1)}\\
  &= (1-\frac1N) + \frac1N\\
  &= 1\end{align}
  $$
  说明对于任一轮抽取，无论有多少人，总匹配数的方差都是 $1$.  
  综合①②我们知道，无论 $N$ 为多大，都有 $\begin{cases}
  \mathbb{E}[X]=\text{Var}(X)=1\\
  \mathbb{E}[X^2] = 2\end{cases}$ 
  
- **③ 记 $R_n$ 为最开始有 $n$ 个人时，完成全部匹配所需的总轮数**

  - 计算 $\mathbb{E}[R_n]$:      
    根据 ① 的结论，直觉上我们能够猜到 $\mathbb{E}[R_n] = n$.   
    这个猜测是正确的，现在给出一个归纳性的证明:   

    - 对于 $n=1$ 的情况，显然有 $\mathbb{E}[R_1] = 1$  
    
    - 对于 $n\geq 2$ 的情况，假定对于任意 $k=1,\dots,n-1$ 都有 $\mathbb{E}[R_k]=k$    
      记第一轮的总匹配数为 $X$，我们对 $X$ 取条件，则有:   
      $$
      \begin{align}
      \mathbb{E}[R_n]
      &= \sum_{i=0}^n\mathbb{E}[R_n|X=i]\cdot\text{P}\{X=i\}\\
      &= \sum_{i=0}^n
      (1+\mathbb{E}[R_{n-i}])\cdot \text{P}\{X=i\}\\
      &= 1 + \mathbb{E}[R_{n}]\text{P}\{X=0\} + 
      \sum_{i=1}^n(n-i)\text{P}\{X=i\}\quad (代入归纳假设)\\
      &= 1 + \mathbb{E}[R_{n}]\text{P}\{X=0\} + n(1-\text{P}\{X=0\}) - \mathbb{E}[X]\quad (\text{note that }\mathbb{E}[X]=1)\\
      &= \mathbb{E}[R_n] \text{P}\{X=0\} + 
      n(1-\text{P}\{X=0\})
      \end{align}
      $$
      于是我们解出 $\mathbb{E}[R_n] = n$  
    
    根据数学归纳法，我们证明了 $\mathbb{E}[R_n] = n$ 
    
  - 计算 $\text{Var}(R_n)$:   
    根据 ② 的结论，直觉上我们能够猜到 $\text{Var}[R_n]= n\ \ (n\geq 2)$.  
    这个猜测是正确的，现在给出一个归纳性的证明:   

    - 对于 $n=0,1$ 的情况，我们设 $\text{Var}(R_0)=\text{Var}(R_1) = 0$  
    
    - 对于 $n=2$ 的情况，完全匹配所需的轮数 $R_2$ 服从几何分布 $\text{Geo}(\frac12)$   
      (这很好理解，当 $n=2$ 时，首次成功匹配就代表着完全匹配，成功概率为 $\frac12$)  
      因此 $\text{Var}[R_2] = \frac{1-\frac12}{(\frac12)^2} = 2$  
      
    - 对于 $n\geq 3$ 的情况，假定对于任意 $k=1,\dots,n-1$ 都有 $\text{Var}(R_k)=k$   
      记第一轮的总匹配数为 $X$，我们对 $X$ 取条件，则有:     
      $$
      \begin{align}
      \text{Var}(R_n)
      &= \mathbb{E}[\text{Var}(R_n|X)] + \text{Var}(\mathbb{E}[R_n|X])\\
      &= \mathbb{E}[\text{Var}(R_{n-X})] + \text{Var}(1+\mathbb{E}[R_{n-X}])\\
      &= \sum_{i=0}^n\text{Var}(R_{n-i})\text{P}\{X=i\} +\text{Var}(1+n-X)\\
      &= \text{Var}(R_n)\text{P}\{X=0\} + 
      \sum_{i=1}^n\text{Var}(R_{n-i})\text{P}\{X=i\} + \text{Var}(X)\qquad (代入归纳假设)\\
      &= \text{Var}(R_n)\text{P}\{X=0\} + 
      \sum_{i=1}^n(n-i)\text{P}\{X=i\} + \text{Var}(X)\\
      &(对于\ i=n-1 \ 的情况我们注意到\ \text{P}\{X=n-1\}=0, 因此对应系数可以取为\ n-i\ 的通式形式)\\
      &= \text{Var}(R_n)\text{P}\{X=0\} + 
      n(1-\text{P}\{X=0\}) - \mathbb{E}[X] + \text{Var}(X)\quad(代入\ \mathbb{E}[X]=\text{Var}(X)=1)\\ 
      &= \text{Var}(R_n)\text{P}\{X=0\} + 
      n(1-\text{P}\{X=0\})\end{align}
      $$
      
      于是我们解出 $\text{Var}(R_n)=n$  
    
    根据数学归纳法，我们证明了 $\text{Var}[R_n] = n\ \ (n\geq 2)$  

- **④ 记 $S_n$ 为最开始有 $n (n\geq 2)$ 个人时，完成全部匹配所需的总抽取次数**  

  - 计算 $\mathbb{E}[S_n]$:   
    根据 ① 的结论，直觉上我们能够猜到 $\mathbb{E}[S_n] = \sum_{i=2}^{n-2}i = \frac{n(n+2)}{2} = \frac{n^2}{2} + n$     
    这个猜测是正确的，现在给出一个归纳性的证明:  

    - 对于 $n=0,1$ 的情况，我们设 $\mathbb{E}[S_0] = \mathbb{E}[S_1] = 0$

    - 对于 $n=2$ 的情况，完全匹配所需的轮数 $R_2$ 服从几何分布 $\text{Geo}(\frac{1}{2})$       
      (这很好理解，当 $n=2$ 时，首次成功匹配就代表着完全匹配，成功概率为 $\frac12$)    
      因此 $\mathbb{E}[S_2] = \mathbb{E}[2R_2] = 2\mathbb{E}[R_2] = 2\cdot 2 = 4$，命题成立. 
      
    - 对于 $n\geq 3$ 的情况，假定对于任意 $k=1,\dots,n-1$ 都有 $\mathbb{E}[S_k]=\frac{k^2}{2}+k$   
      记第一轮的总匹配数为 $X$，我们对 $X$ 取条件，则有:   
      $$
      \begin{align}
      \mathbb{E}[S_n]
      &= \sum_{i=0}^n\mathbb{E}[S_n|X=i]\cdot\text{P}\{X=i\}\\
      &= \sum_{i=0}^n
      (n+\mathbb{E}[S_{n-i}])\cdot \text{P}\{X=i\}\\
      &= n + \mathbb{E}[S_n]\text{P}\{X=0\} + \sum_{i=1}^n\mathbb{E}[S_{n-i}]\text{P}\{X=i\}\quad (代入归纳假设)\\
      &= n + \mathbb{E}[S_n]\text{P}\{X=0\} + 
      \sum_{i=1}^n[\frac{(n-i)^2}{2}+(n-i)]\text{P}\{X=i\}\\
      &(对于\ i=n-1 \ 的情况我们注意到\ \text{P}\{X=n-1\}=0, 因此对应系数可以取为\ \frac{(n-i)^2}{2}+(n-i)\ 的通式形式)\\ 
      &= n + \mathbb{E}[S_n]\text{P}\{X=0\} + 
      (\frac{n^2}{2}+n)(1-\text{P}\{X=0\}) - (n+1)\mathbb{E}[X] + \frac{1}{2}\mathbb{E}[X^2]\\ 
      &= n + \mathbb{E}[S_n]\text{P}\{X=0\} + 
      (\frac{n^2}{2}+n)(1-\text{P}\{X=0\}) - (n+1)\cdot 1 + \frac{1}{2}\cdot 2\\
      &= \mathbb{E}[S_n]\text{P}\{X=0\} + 
      (\frac{n^2}{2}+n)(1-\text{P}\{X=0\})\end{align}
      $$
      于是我们解出 $\mathbb{E}[S_n] = \frac{n^2}{2} + n$    
    
    根据数学归纳法，我们证明了 $\mathbb{E}[S_n] = \frac{n^2}{2} + n\ \ (n\geq 2)$
  



## 1.7.4 通过取条件计算概率

我们不仅可以通过对合适的随机变量先取条件得到期望和方差，而且也可用此方法计算概率.   
具体来说，记 $E$ 为一个任意事件且定义**示性随机变量** $X = \begin{cases}
1, &若\ E\ 发生\\
0, &若\ E\ 不发生 \end{cases}$  
由 $X$ 的定义推出 $\begin{cases}
\mathbb{E}[X]= \text{P}(E)\\
\mathbb{E}[X|Y=y] = \text{P}(E|Y=y)\quad (对于任意随机变量\ Y)\end{cases}$  
因此根据**全期望公式** $\mathbb{E}[X]=\mathbb{E}[\mathbb{E}[X|Y]] =\begin{cases}
\underset{y}{\sum} \mathbb{E}[X|Y=y] \text{P}\{Y=y\},&若\ Y\ 离散\\
\int_{-\infty}^{\infty} \mathbb{E}[X|Y=y]f_Y(y)\mathrm{d}y,&若\ Y\ 连续\end{cases}$   
可知有 $\text{P}[E]=\begin{cases}
\underset{y}{\sum} \text{P}[E|Y=y] \text{P}\{Y=y\},&若\ Y\ 离散\\
\int_{-\infty}^{\infty} \text{P}[E|Y=y]f_Y(y)\mathrm{d}y,&若\ Y\ 连续\end{cases}$ 

**一些具体的例子: **  

- 设 $X,Y$ 为独立的连续随机变量，分别具有密度 $f_X,f_Y$，计算 $\text{P}\{X<Y\}$:     
  对 $Y$ 取条件得:   
  $$
  \begin{align}
  \text{P}\{X<Y\}
  &= \int_{-\infty}^{\infty} \text{P}\{X<Y|Y=y\}f_Y(y)\mathrm{d}y\\
  &= \int_{-\infty}^{\infty} \text{P}\{X<y|Y=y\}f_Y(y)\mathrm{d}y\qquad(X\ \bot\ Y)\\
  &= \int_{-\infty}^{\infty} \text{P}\{X<y\} f_Y(y)\mathrm{d}y\\
  &= \int_{-\infty}^{\infty} F_X(y)f_Y(y)\mathrm{d}y\end{align}
  $$
  其中 $F_X(y) = \int_{-\infty}^y f_X(x)\mathrm{d}x$ 
  
- **(Poisson 均值的随机性)**
  设 $\begin{cases}
  Y \sim \text{Poisson}(X)\\
  X \sim \exp(1) = \text{Gamma}(1,1)\end{cases}$ 计算 $\text{P}\{Y=n\}$:   
  首先，Poisson 均值 $X$ 的概率密度函数为 $f_X(x) = xe^{-x}\ \ (x\geq 0)$  
  其次，对 $X$ 取条件可得: $(\forall\ n=0,1,\dots)$   
  $$
  \begin{align}
  \text{P}\{Y=n\}
  &= \int_{-\infty}^{\infty} \text{P}\{Y=n|X=x\}f_X(x)\mathrm{d}x\\
  &= \int_{-\infty}^{\infty}\text{P}\{\text{Poisson}(x)=n\}f_X(x)\mathrm{d}x\\
  &= \int_{0}^{\infty}e^{-x}\frac{x^n}{n!}\cdot x e^{-x} \mathrm{d}x\\ 
  &= \frac{1}{n!}\int_0^{\infty}x^{n+1}e^{-2x}\mathrm{d}x\\
  &= \frac{n+1}{2^{n+2}}\int_0^{\infty} \frac{2e^{-2x}(2x)^{n+1}}{(n+1)!} \mathrm{d}x\qquad(凑出 \ \text{Gamma}(n+2,2)\ 的概率密度函数)\\
  &= \frac{n+1}{2^{n+2}}\end{align}
  $$
  
- **(相互独立的 Poisson 随机事件的分类)**   
  记 $N\sim \text{Poisson}(\lambda)$ 为每天参与训练的人数，假定每个人是否参加是相互独立的，  
  其中是女性的概率为 $p$，是男性的概率为 $1-p$  
  求今天恰有 $n$ 个女性和 $m$ 个男性参与训练的联合概率:   
  记今天参加的女、男性人数分别为 $N_1,N_2$，满足 $N=N_1+N_2$  
  对 $N$ 取条件可得: $(\forall\ n,m = 0,1,\dots)$  
  $$
  \begin{align}
  \text{P}\{N_1=n,N_2=m\}
  &= \underset{i=0}{\overset{\infty}\sum}
  \text{P}\{N_1=n,N_2=m|N=i\}\text{P}\{N=i\}\\
  &= 0 + \text{P}\{N_1=n,N_2=m|N=n+m\}\text{P}\{N=n+m\}\\
  &= \binom{n+m}{n}p^n(1-p)^m\cdot e^{-\lambda}\frac{\lambda^n\lambda^m}{(n+m)!}\\
  &= \frac{(n+m)!}{n!m!}p^n(1-p)^m e^{-\lambda  p}e^{-\lambda(1-p)}\frac{\lambda^n\lambda^m}{(n+m)!}\\
  &= e^{-\lambda p}\frac{(\lambda p)^n}{n!}\cdot e^{-\lambda (1-p)}\frac{(\lambda(1-p))^m}{m!}\\
  &= \text{P}\{\text{Poisson}(\lambda p)=n\}\cdot \text{P}\{\text{Poisson}(\lambda (1-p))=m\}\end{align}
  $$
  这说明 $\begin{cases}
  N_1\sim \text{Poisson}(\lambda p)\\
  N_2\sim \text{Poisson}(\lambda (1-p))\\
  N_1\ \bot\ N_2\end{cases}$   
  **这是一个重要的结论: **  
  当每一个均值为 $\lambda$ 的 Poisson 随机事件**独立地**以概率 $p$ 分入第一类，或以概率 $1-p$ 分入第二类时，  
  第一类与第二类中的事件数是独立的 Poisson 随机变量，且 Poisson 均值分别为 $\lambda p$ 和 $\lambda(1-p)$   
  **这个结论的推广形式: **  
  $N$ 个均值为 $\lambda$ 的 Poisson 随机事件**独立地**分为 $k$ 类， 
  其中分入第 $i$ 类的概率是 $p_i$，满足 $\underset{i=1}{\overset{k}{\sum}}p_i=1$.  
  记 $N_i$ 为分入第 $i$ 类的事件数，满足 $\underset{i=1}{\overset{k}{\sum}}N_i=N$  
  则 $N_1,\dots,N_k$ 是独立的 Poisson 随机变量，均值分别为 $\lambda p_1,\dots,\lambda p_k$  
  **证明: **  
  对于 $n = \underset{i=1}{\overset{k}{\sum}}n_i$:   
  $$
  \begin{align}
  \text{P}\{N_1=n_1,\dots,N_k=n_k\}
  &= \text{P}\{N_1=n_1,\dots,N_k=n_k|N=n\}\cdot\text{P}\{N=n\}\\
  &= \frac{n!}{n_1!\dotsm n_k!}p_1^{n_1}\dotsm p_k^{n_k}\cdot e^{-\lambda}\frac{\lambda^n}{n!}\\
  &= \underset{i=1}{\overset{k}\prod}
  e^{-\lambda p_i}\frac{(\lambda p_i)^{n_i}}{n_i!}\\&= \underset{i=1}{\overset{k}\prod}
  \text{P}\{\text{Poisson}(\lambda p_i)=n_i\}\end{align}
  $$
  说明 $\begin{cases}
  N_i\sim \text{Poisson}(\lambda p_i) &(\forall\ i=1,\dots,n)\\
  N_i\ \bot\ N_j&(\forall \ i\neq j = 1,\dots,n)\end{cases}$ 

**The End**
