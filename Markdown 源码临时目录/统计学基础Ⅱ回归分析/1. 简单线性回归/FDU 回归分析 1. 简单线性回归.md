# FDU 回归分析 1. 简单线性回归

本文根据王勤文老师课堂笔记整理而成，并参考以下教材:

- Statistical Inference (2nd Edition, G. Casella) Chapter $11$
- 统计推断 (第 $2$ 版, G. Casella) 第 $11$ 章
- All of Statistics (L. Wasserman) 
- 统计学完全教程 (L. Wasserman) 
- 应用回归分析 (第 $5$ 版, 何晓群, 刘文卿) 第 $2$ 章
- 数理统计讲义 (郑明, 陈子毅, 汪嘉冈) 第 $1$ 章

欢迎批评指正!

## 1.1 简单线性回归模型

给定随机变量 $(X,Y)$ 的 $n$ 个观测 $(x_1,y_1),\dots,(x_n,y_n)$，我们希望推断 $X,Y$ 之间的关系.  
在通常的应用中我们通常把 $X$ 称为**自变量** (independent variable) 或**解释变量** (predictor)  
而把 $Y$ 称为**因变量** (dependent variable) 或**响应变量** (response variable)

具体来说，在简单线性回归中我们假设 $X,Y$ 有如下关系:
$$
Y = \alpha + \beta X + \varepsilon,
$$
其中 $\alpha,\beta$ 是固定的未知参数，分别称为回归方程的**截距** (intercept) 和**斜率** (slope).  
我们通常假设 $\varepsilon$ 为零均值随机噪音 (若其均值非零，则我们可将其均值并入 $\alpha$)，因此我们有:  
$$
\mathbb{E}(Y|X) = \alpha + \beta X.
$$
值得注意的是，线性回归中的 "线性" 一词是指**对于参数是线性的**.  
因此 $\mathbb{E}(Y|X)=\alpha+\beta X^2$ 和 $\mathbb{E}(\log(Y)|X) = \alpha + \beta \frac{1}{X}$ 也都是线性回归.



### 1.1.1 基础模型

设观测样本是 $n$ 个数据对 $(x_1,y_1),\dots,(x_n,y_n)$，  
其中 $x_1,\dots,x_n$ 为随机变量 $X_1,\dots,X_n$ 的实现 (但暂时视其为已知的固定常数)，  
而 $y_1,\dots,y_n$ 是随机变量 $Y_1,\dots,Y_n$ 的观测值.   
我们假设:
$$
\mathbb{E}(Y_i|X=x_i) = \alpha + \beta x_i\ \ (i=1,\dots,n).
$$
这个模型也可等价表示为:
$$
Y_i = \alpha + \beta x_i + \varepsilon_i\ \ (i=1,\dots,n),
$$
其中 $\varepsilon_1,\dots,\varepsilon_n$ 是零均值随机变量 (我们暂且不对它们的相关性或分布做任何假设).  
基于观测样本 $(x_1,y_1),\dots,(x_n,y_n)$，我们定义下面的量:
$$
\begin{align}
\bar x &= \frac1n \sum_{i=1}^n x_i = \frac1n 1_n^{\mathrm T}x\\
\bar y &= \frac1n \sum_{i=1}^n y_i = \frac1n 1_n^{\mathrm T}y\\
\hline
S_{xx} &= \sum_{i=1}^n (x_i-\bar x)^2\\ 
&= (x-\bar x1_n)^{\mathrm T} (x-\bar x1_n)\\
&= x^{\mathrm T}x-2\bar x 1_n^{\mathrm T}x + \bar x^2 1_n^{\mathrm T}1_n\\
&= x^{\mathrm T}x - 2n\bar x^2 + n\bar x^2\\
&= x^{\mathrm T}x - n\bar x^2\\
S_{yy} &= \sum_{i=1}^n (y_i-\bar y)^2\\
&= (y-\bar y1_n)^{\mathrm T} (y-\bar y1_n)\\ 
&= y^{\mathrm T}y-2\bar y 1_n^{\mathrm T}y + \bar y^2 1_n^{\mathrm T}1_n\\
&= y^{\mathrm T}y - 2n\bar y^2 + n\bar y^2\\
&= y^{\mathrm T}y - n\bar y^2\\
S_{xy} &= \sum_{i=1}^n (x_i-\bar x)(y_i - \bar y)\\ 
&= (x-\bar x1_n)^{\mathrm T} (y-\bar y1_n)\\ 
&= x^{\mathrm T}y - \bar x 1_n^{\mathrm T} y - \bar y 1_n^{\mathrm T}x + \bar x \bar y 1_n^{\mathrm T}1_n\\ 
&= x^{\mathrm T}y - n\bar x \bar y
\end{align}
$$




### 1.1.2 条件正态模型

**条件正态模型** (conditional normal model) 是最常用的简单线性回归模型，也是最容易分析的模型.  
设观测样本是 $n$ 个数据对 $(x_1,y_1),\dots,(x_n,y_n)$，  
其中 $x_1,\dots,x_n$ 为随机变量 $X_1,\dots,X_n$ 的实现 (但暂时视其为已知的固定常数)，  
而 $y_1,\dots,y_n$ 是独立随机变量 $Y_1,\dots,Y_n$ 的观测值.

进一步，我们假定 $Y_1,\dots,Y_n$ 服从正态分布 (结合独立性可知它们联合正态) 且具有相同方差:
$$
Y_i \sim \mathcal{N}(\alpha + \beta x_i,\sigma^2)\ \ (i=1,\dots,n)\\
\begin{bmatrix}
Y_1\\
\vdots\\
Y_n
\end{bmatrix}
\sim
\mathcal{N}
\left(
\begin{bmatrix}
\alpha + \beta x_1\\
\vdots\\
\alpha + \beta x_n
\end{bmatrix},
\sigma^2 I_n
\right)
$$
由此可知总体回归函数为 $\mathbb{E}(Y|X=x) = \alpha + \beta x$.  
具体来说，条件分布 $(Y|X=x) \sim \mathcal{N}(\alpha + \beta x,\sigma^2)$.

条件正态模型还具有如下等价形式:
$$
Y_i = \alpha + \beta x_i + \varepsilon_i\ \ (i=1,\dots,n)\text{ where }\varepsilon_1,\dots,\varepsilon_n \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,\sigma^2),
$$
它相当于基础模型额外加上了 $\varepsilon_1,\dots,\varepsilon_n \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,\sigma^2)$ 的假设.



### 1.1.3 二元正态模型

在之前的模型中，我们都将随机变量 $X_1,\dots,X_n$ 的实现 $x_1,\dots,x_n$ 视其为已知的固定常数.  
现在我们感到有必要在模型中考虑 $X_1,\dots,X_n$ 的随机性，二元正态模型就是一个相对简单的这种模型.    
我们将观测样本 $(x_1,y_1),\dots,(x_n,y_n)$ 视为独立的二元正态随机变量 $(X_1,Y_1),\dots,(X_n,Y_n)$ 的实现.  
具体来说，我们假定:  
$$
(X_i,Y_i)\sim \text{bivariate normal}(\mu_X,\mu_Y,\sigma^2_X,\sigma^2_Y,\rho) 
= 
\mathcal{N}\left(
\begin{bmatrix}
\mu_X\\
\mu_Y
\end{bmatrix},
\begin{bmatrix}
\sigma_X^2 & \rho\sigma_X\sigma_Y\\
\rho\sigma_X\sigma_Y & \sigma_Y^2
\end{bmatrix}
\right).
$$
对于二元正态分布，给定 $X=x$ 时 $Y$ 的条件分布是正态分布.  
具体来说，其总体回归函数为:  
$$
\mathbb{E}(Y|X=x) 
= \mu_Y + \rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X)
= \left(\mu_Y - \rho\frac{\sigma_Y}{\sigma_X}\mu_X\right) + \left(\rho\frac{\sigma_Y}{\sigma_X}\right)x = \alpha + \beta x,
$$
其中:
$$
\begin{cases}
\alpha = \mu_Y - (\rho \sigma_Y/\sigma_X)\mu_X=\mu_Y- \beta \mu_X\\
\beta = \rho \sigma_Y/\sigma_X.
\end{cases}
$$

> 证明过程:
> $$
> \frac{X-\mu_X}{\sigma_X} := Z_1 \sim \mathcal{N}(0,1)\\
> \frac{Y-\mu_Y}{\sigma_Y} := \rho Z_1 + \sqrt{1-\rho^2} Z_2\quad (\text{where }Z_1\ \bot\ Z_2\text{ and }Z_2\sim \mathcal{N}(0,1))\\
> \hline
> 
> \mathbb{E}\left(\frac{Y-\mu_Y}{\sigma_Y}{\Large|}X=x\right) = \rho\cdot \frac{x-\mu_X}{\sigma_X} + 0 =  \rho\cdot \frac{x-\mu_X}{\sigma_X}\\
> \Downarrow\\
> \begin{align}
> \mathbb{E}(Y|X=x) 
> &=
> \mu_Y + \rho \frac{x-\mu_X}{\sigma_X} \sigma_Y\\
> &=
> \left(\mu_Y - \rho \frac{\sigma_Y}{\sigma_X}\mu_X\right) + \left(\rho\frac{\sigma_Y}{\sigma_X}\right)x
> \end{align}
> $$

换言之，二元正态模型的假设自然保证了总体回归 $\mathbb{E}(Y|X=x)$ 是 $x$ 的线性函数，  
我们不需要像在前面的模型中那样假定这一点.

值得注意的是，线性回归分析几乎总是可以用给定 $X_1=x_1,\dots,X_n=x_n$ 时 $(Y_1,\dots,Y_n)$ 的条件分布来进行，  
而不是用 $(X_1,Y_1),\dots,(X_n,Y_n)$ 的无条件分布来进行，这就和条件正态模型别无二致了.  
如果我们仅考虑给定 $X_1=x_1,\dots,X_n=x_n$ 时 $(Y_1,\dots,Y_n)$ 的条件分布，  
那么 $x_1,\dots,x_n$ 作为随机变量的观测值这一点就无关紧要了，    
此时我们便可以将其视为已知的固定常数 (我们在基础模型和条件正态模型中就是这么做的).

事实上，二元正态性这个假定除了自然地保证了条件分布的线性性以外，就对后续的推断没有什么作用了.  
在基于点估计、区间估计和假设检验的推断中，$(X,Y)$ 中 $X$ 的边缘分布什么也不影响.  
在线性回归中，正是条件分布起关键作用.



## 1.2 参数的点估计

参数 $\alpha,\beta$ 的最常见的估计 $\hat \alpha,\hat \beta$ 为:
$$
\begin{cases}
\hat \beta = S_{xy}/S_{xx} = (x^{\mathrm T}y-n\bar x\bar y)/(x^{\mathrm T}x-n\bar x^2)\\
\hat \alpha = \bar y - \hat \beta \bar x.
\end{cases}
$$
我们将在各种模型下证明这个估计的表达式.

### 1.2.1 最小二乘估计

首先我们从基础模型开始:  

> 设观测样本是 $n$ 个数据对 $(x_1,y_1),\dots,(x_n,y_n)$，  
> 其中 $x_1,\dots,x_n$ 为随机变量 $X_1,\dots,X_n$ 的实现 (但暂时视其为已知的固定常数)，  
> 而 $y_1,\dots,y_n$ 是随机变量 $Y_1,\dots,Y_n$ 的观测值.   
> 我们假设:  
> $$
> \mathbb{E}(Y_i|X=x_i) = \alpha + \beta x_i\ \ (i=1,\dots,n).
> $$
> 这个模型也可等价表示为:  
> $$
> Y_i = \alpha + \beta x_i + \varepsilon_i\ \ (i=1,\dots,n),
> $$
> 其中 $\varepsilon_1,\dots,\varepsilon_n$ 是零均值随机变量 (我们暂且不对它们的相关性或分布做任何假设).

我们定义 $(x_1,y_1),\dots,(x_n,y_n)$ 关于直线 $\mathbb{E}(Y|X=x)=\alpha + \beta x$ 的**残差平方和** (residual sum of squares) 为:  
$$
\text{RSS}(\alpha,\beta) := \sum_{i=1}^n (y_i - (\alpha + \beta x_i))^2 = \|y-\alpha 1_n - \beta x\|_2^2.
$$
最小二乘的观点表明:  
$$
(\hat \alpha ,\hat \beta) = \arg \min_{\alpha,\beta} \text{RSS}(\alpha,\beta) = \arg \min_{\alpha,\beta} \|y-\alpha 1_n - \beta x\|_2^2.
$$
注意到目标函数 $\|y-\alpha 1_n - \beta x\|_2^2$ 是关于 $\alpha,\beta$ 的严格凸函数，因此上述优化问题的最优解是唯一的.   
计算目标函数关于 $\alpha,\beta$ 的偏导数:
$$
\begin{align}
\frac{\partial}{\partial\alpha} \|y-\alpha 1_n - \beta x\|_2^2 
&= 1_n^{\mathrm T}\cdot 2(y-\alpha 1_n -\beta x) \\
&= 2(1_n^{\mathrm T}y - \alpha 1_n^{\mathrm T}1_n - \beta 1_n^{\mathrm T}x)\\
&= 2(n\bar y - \alpha n - \beta n\bar x)\\
&= 2n(\bar y - \alpha - \beta \bar x)\\

\frac{\partial }{\partial \beta} \|y-\alpha 1_n - \beta x\|_2^2
&=
x^{\mathrm T}\cdot 2(y-\alpha 1_n -\beta x)\\
&=
2(x^{\mathrm T}y - \alpha x^{\mathrm T}1_n - \beta x^{\mathrm T}x)\\
&=
2(x^{\mathrm T}y - \alpha n \bar x - \beta x^{\mathrm T}x).
\end{align}
$$
令上述偏导数等于零，我们便得到方程组:
$$
\begin{cases}
\frac{\partial}{\partial\alpha} \|y-\alpha 1_n - \beta x\|_2^2=
2n(\bar y - \alpha - \beta \bar x)=0\\

\frac{\partial}{\partial\beta} \|y-\alpha 1_n - \beta x\|_2^2 = 
 2(x^{\mathrm T}y - \alpha n \bar x - \beta x^{\mathrm T}x) =0.

\end{cases}
$$
根据第一个等式我们得到 $\alpha = \bar y - \beta \bar x$，代入第二个等式我们有:  
$$
x^{\mathrm T}y - (\bar y - \beta \bar x) n\bar x - \beta x^{\mathrm T}x = 0\\
\Updownarrow\\
x^{\mathrm T}y - n \bar x\bar y = \beta (x^{\mathrm T}x - n \bar x^2)\\
\Updownarrow\\
\beta = \frac{x^{\mathrm T}y - n\bar x\bar y}{x^{\mathrm T}x-n\bar x^2} = \frac{S_{xy}}{S_{xx}}
$$
综上所述，最小二乘问题 $(\hat \alpha ,\hat \beta) = \arg \min_{\alpha,\beta} \|y-\alpha 1_n - \beta x\|_2^2$ 的解为:  
$$
\begin{cases}
\hat \beta = S_{xy}/S_{xx} = (x^{\mathrm T}y-n\bar x\bar y)/(x^{\mathrm T}x-n\bar x^2)\\
\hat \alpha = \bar y - \hat \beta \bar x.
\end{cases}
$$

****

下面我们证明最小二乘估计量 $\hat \alpha,\hat \beta$ 可以写成 $y_1,\dots,y_n$ 的线性组合，  
并且 $\hat \alpha-\alpha$ 和 $\hat\beta - \beta$ 可以写成 $\varepsilon_1,\dots,\varepsilon_n$ 的线性组合:
$$
\begin{align}
\hat \beta 
&=
\frac{S_{xy}}{S_{xx}}\\
&=
\frac{x^{\mathrm T}y - n \bar x \bar y}{S_{xx}}\\
&=
\frac{(x - \bar x 1_n)^{\mathrm T}y}{S_{xx}}\\
&=
\frac{(x - \bar x 1_n)^{\mathrm T}(\alpha 1_n + \beta x + \varepsilon)}{S_{xx}}\\
&=
\frac{1}{S_{xx}}(\alpha(x^{\mathrm T}1_n - \bar x 1_n^{\mathrm T}1_n) + \beta (x^{\mathrm T}x - \bar x 1_n^{\mathrm T}x) + (x-\bar x 1_n)^{\mathrm T}\varepsilon)\\
&=
\frac{1}{S_{xx}}(\alpha(n\bar x - n\bar x) + \beta (x^{\mathrm T}x - n\bar x^2) + (x-\bar x 1_n)^{\mathrm T}\varepsilon)\\
&=
\frac{1}{S_{xx}}(\alpha\cdot 0 + \beta \cdot S_{xx} + (x-\bar x 1_n)^{\mathrm T}\varepsilon)\\
&=
\beta + \frac{(x-\bar x 1_n)^{\mathrm T}\varepsilon}{S_{xx}}\\

\hline

\hat \alpha 
&=
\bar y - \hat \beta \bar x\\
&=
\frac1n 1_n^{\mathrm T} y - \left(\beta + \frac{(x-\bar x 1_n)^{\mathrm T}\varepsilon}{S_{xx}}\right) \bar x\\
&=
\frac1n 1_n^{\mathrm T}(\alpha 1_n + \beta x + \varepsilon) - \beta \bar x - \frac{(x-\bar x 1_n)^{\mathrm T}\varepsilon}{S_{xx}} \bar x\\
&=
\alpha\left(\frac1n 1_n^{\mathrm T} 1_n\right) + \beta \left(\frac1n 1_n^{\mathrm T}x-\bar x\right) + \frac{1}{n} 1_n^{\mathrm T} \varepsilon - \frac{\bar x(x-\bar x 1_n)^{\mathrm T}\varepsilon}{S_{xx}}\\
&=
\alpha + \left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)^{\mathrm T} \varepsilon.
\end{align}
$$
因此 $\hat \alpha-\alpha$ 和 $\hat\beta - \beta$ 可以写成 $\varepsilon_1,\dots,\varepsilon_n$ 的线性组合:  
$$
\begin{align}
\hat \alpha -\alpha 
&= \left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)^{\mathrm T} \varepsilon\\
\hat \beta-\beta 
&= \frac{(x-\bar x 1_n)^{\mathrm T}}{S_{xx}}\varepsilon.
\end{align}
$$
如果我们额外假设:
$$
\begin{cases}
\mathbb{E}(\varepsilon) = 0_n\\
\text{Var}(\varepsilon) = \sigma^2 I_n,
\end{cases}
$$
即零均值、互不相关且方差相同，则我们有:  
$$
\begin{align}
\mathbb{E}(\hat \alpha)
&=
\mathbb{E}\left(\alpha + \left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)^{\mathrm T} \varepsilon\right)\\
&=
\alpha + \left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)^{\mathrm T} \mathbb{E}(\varepsilon)\\
&=
\alpha + \left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)^{\mathrm T} 0_n\\
&=
\alpha\\

\hline
\mathbb{E}(\hat \beta)
&=
\mathbb{E}\left(\beta + \frac{(x-\bar x 1_n)^{\mathrm T}\varepsilon}{S_{xx}}\right)\\
&=
\beta + \frac{(x-\bar x 1_n)^{\mathrm T}\mathbb{E}(\varepsilon)}{S_{xx}}\\
&=
\beta + \frac{(x-\bar x 1_n)^{\mathrm T}0_n}{S_{xx}}\\
&=
\beta\\

\hline
\text{Var}(\hat \alpha) 
&= 
\text{Var}\left(\alpha + \left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)^{\mathrm T} \varepsilon\right)\\
&=
\left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)^{\mathrm T}\text{Var}(\varepsilon) \left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)\\
&=
\left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)^{\mathrm T} \sigma^2 I_n \left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)\\
&=
\left(\frac1{n^2}1_n^{\mathrm T}1_n - 2\frac{\bar x}{nS_{xx}}(x-\bar x1_n)^{\mathrm T} 1_n + \frac{\bar x^2}{S_{xx}^2} (x-\bar x 1_n)^{\mathrm T}(x-\bar x 1_n)\right) \sigma^2\\
&=
\left(\frac1n - 2\frac{\bar x}{nS_{xx}}(n\bar x-n\bar x1_n) + \frac{\bar x^2}{S_{xx}^2} S_{xx}\right) \sigma^2\\
&=
\left(\frac1n + \frac{\bar x^2}{S_{xx}}\right) \sigma^2\\
&=
\frac{x^{\mathrm T}x}{nS_{xx}} \sigma^2\\

\hline

\text{Var}(\hat \beta)
&=
\text{Var}\left(\beta + \frac{(x-\bar x 1_n)^{\mathrm T}\varepsilon}{S_{xx}}\right)\\
&=
\frac{(x-\bar x 1_n)^{\mathrm T}}{S_{xx}} \text{Var}(\varepsilon) \frac{x-\bar x 1_n}{S_{xx}}\\
&=
\frac{(x-\bar x 1_n)^{\mathrm T}}{S_{xx}} \sigma^2 I_n \frac{x-\bar x 1_n}{S_{xx}}\\
&=
\frac{S_{xx}}{S_{xx}^2} \sigma^2\\
&=
\frac{1}{S_{xx}} \sigma^2\\

\hline

\text{Cov}(\hat \alpha ,\hat \beta)
&=
\text{Cov}\left(\alpha + \left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)^{\mathrm T} \varepsilon ,\beta + \frac{(x-\bar x 1_n)^{\mathrm T}\varepsilon}{S_{xx}}\right)\\
&=
\left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)^{\mathrm T} \text{Var}(\varepsilon)\frac{x-\bar x 1_n}{S_{xx}}\\
&=
\left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)^{\mathrm T} \sigma^2I_n\frac{x-\bar x 1_n}{S_{xx}}\\
&=
\left(\frac{\frac1n 1_n^{\mathrm T} (x-\bar x 1_n)}{S_{xx}} - \frac{\bar x (x-\bar x1_n)^{\mathrm T}(x-\bar x1_n)}{S_{xx}^2}\right)\sigma^2\\
&=
\left(\frac{\bar x- \bar x}{S_{xx}} - \frac{\bar x S_{xx}}{S_{xx}^2}\right)\sigma^2\\
&=
- \frac{\bar x}{S_{xx}}\sigma^2

\end{align}
$$



### 1.2.1 最佳线性无偏估计

现在我们为基础模型添加假设: 
$$
\begin{cases}
\mathbb{E}(\varepsilon) = 0_n\\
\text{Var}(\varepsilon) = \sigma^2 I_n,
\end{cases}
$$
即零均值、互不相关和方差相同，则我们有:
$$
\begin{cases}
\mathbb{E}(Y|X=x) = \alpha1_n + \beta x\\
\text{Var}(Y|X=x) = \sigma^2 I_n.
\end{cases}
$$

本节中我们不需指明随机噪音 $\varepsilon$ 的分布.

我们将证明最小二乘估计量
$$
\begin{cases}
\hat \beta = S_{xy}/S_{xx} = (x^{\mathrm T}y-n\bar x\bar y)/(x^{\mathrm T}x-n\bar x^2)\\
\hat \alpha = \bar y - \hat \beta \bar x
\end{cases}
$$
是 $(\alpha,\beta)$ 的线性无偏估计类中最优的估计量.

****

我们先对 $\beta$ 进行分析:

- 首先，$\beta$ 的一个线性估计量具有形式 $\hat\beta= c^{\mathrm T} y$ (其中 $c\in \mathbb R^n$ 是已知且固定的向量).

- 其次，$\beta$ 的一个线性无偏估计量还满足 $\mathbb{E}(\hat \beta) =\beta$，也就是说:  
  $$
  \begin{align}
  \mathbb{E}(\hat \beta) 
  &= \mathbb{E}(c^{\mathrm T}y)\\
  &= \mathbb{E}(c^{\mathrm T}(\alpha 1_n + \beta x + \varepsilon))\\
  &= \alpha (c^{\mathrm T}1_n) + \beta (c^{\mathrm T}x) + c^{\mathrm T}\mathbb{E}(\varepsilon) \quad (\text{note that }\mathbb{E}(\varepsilon)=0_n)\\
  &= (1_n^{\mathrm T} c) \alpha + (c^{\mathrm T}x)\beta\\
  &=\beta.
  \end{align}
  $$
  根据 $(1_n^{\mathrm T} c) \alpha + (c^{\mathrm T}x)\beta = \beta$ 可知 $c$ 满足:
  $$
  \begin{cases}
  1_n^{\mathrm T} c =0\\
  c^{\mathrm T}x = 1.
  \end{cases}
  $$
  
- 最后，$\beta$ 的最佳线性无偏估计量还满足:   
  $$
  \text{Var}(\hat \beta) = \underset{
  \begin{subarray}{}
  1_n^{\mathrm T}c = 0\\
  c^{\mathrm T}x=1
  \end{subarray}
  }\min \text{Var}(c^{\mathrm T}y).
  $$
  注意到目标函数可进行如下化简:
  $$
  \begin{align}
  \text{Var}(c^{\mathrm T}y) 
  &= \text{Var}(c^{\mathrm T}(\alpha 1_n + \beta x + \varepsilon))\\ 
  &= c^{\mathrm T}\text{Var}(\varepsilon )c\\ 
  &= c^{\mathrm T} \sigma^2 I_n c\\
  &= \sigma^2 \|c\|_2^2.
  \end{align}
  $$
  因此问题变为求解最优化问题:  
  $$
  \hat c = \underset{
  \begin{subarray}{}
  1_n^{\mathrm T}c = 0\\
  c^{\mathrm T}x=1
  \end{subarray}
  }{\arg\min} \|c\|_2^2
  $$
  定义 Lagrange 函数为:   
  $$
  L(c,\lambda_1,\lambda_2) = \|c\|_2^2 + \lambda_1 (1_n^{\mathrm T}c-0) + \lambda_2 (c^{\mathrm T}x-1).
  $$
  注意到目标函数是凸函数，因此其 $\text{KKT}$ 点就对应其最优解.  
  其 $\text{KKT}$ 条件为: 
  $$
  \begin{cases}
  \nabla_c L(c,\lambda_1,\lambda_2) = 2c + \lambda_1 1_n + \lambda_2 x = 0_n\\
  \frac{\partial }{\partial \lambda_1}L(c,\lambda_1,\lambda_2) = 1_n^{\mathrm T}c = 0\\
  \frac{\partial }{\partial \lambda_2} L(c,\lambda_1,\lambda_2) = c^{\mathrm T}x-1 = 0.
  \end{cases}
  $$
  分别对第一个等式左乘 $1_n^{\mathrm T}$ 和 $x^{\mathrm T}$，并代入第二个和第三个等式就可以得到 (只有一个 $\text{KKT}$ 点): 
  $$
  \begin{cases}
  n\lambda_1 + n\bar x\lambda_2=0\\
  2 + n\bar x\lambda_1 + x^{\mathrm T}x \lambda_2 =0 
  \end{cases}
  \ \ \Rightarrow\ \ 
  \begin{cases}
  \lambda_1 = -\bar x\lambda_2 = 2\bar{x}/S_{xx}\\
  \lambda_2 = -2/(x^{\mathrm T}x-n\bar x^2) = -2/S_{xx}.
  \end{cases}
  $$
  于是唯一的最优解为:  
  $$
  \hat c=-\frac12(\lambda_1 1_n + \lambda_2 x) = \frac{x-\bar x 1_n}{S_{xx}}.
  $$
  因此最佳线性无偏估计量为:  
  $$
  \begin{align}
  \hat \beta 
  &= \hat c^{\mathrm T} y\\ 
  &= \left(\frac{x-\bar x 1_n}{S_{xx}}\right)^{\mathrm T} y\\ 
  &= \frac{x^{\mathrm T}y-\bar x 1_n^{\mathrm T} y}{S_{xx}}\\ 
  &= \frac{x^{\mathrm T}y - n\bar x\bar y}{S_{xx}}\\ 
  &= \frac{S_{xy}}{S_{xx}}.
  \end{align}
  $$
  其方差为 $\beta$ 的线性无偏估计量所能具有的最小方差 (因而 $\hat \beta$ 是最稳定的线性无偏估计量): 
  $$
  \begin{align}
  \text{Var}(\hat \beta) 
  &= \sigma^2 \|\hat c\|_2^2\\
  &= \sigma^2 \left(\frac{x-\bar x 1_n}{S_{xx}}\right)^{\mathrm T} \left(\frac{x-\bar x 1_n}{S_{xx}}\right) \\
  &= \sigma^2 \frac{S_{xx}}{S_{xx}^2}\\
  &= \frac{\sigma^2}{S_{xx}}.
  \end{align}
  $$

****

我们对 $\alpha$ 进行类似的分析:

- 首先，$\alpha$ 的一个线性估计量具有形式 $\hat\alpha= c^{\mathrm T} y$ (其中 $c\in \mathbb R^n$ 是已知且固定的向量) 

- 其次，$\alpha$ 的一个线性无偏估计量还满足 $\mathbb{E}(\hat \alpha) =\alpha$，也就是说:  
  $$
  \begin{align}
  \mathbb{E}(\hat \alpha) 
  &= \mathbb{E}(c^{\mathrm T}y)\\
  &= \mathbb{E}(c^{\mathrm T}(\alpha 1_n + \beta x + \varepsilon))\\
  &= \alpha (c^{\mathrm T}1_n) + \beta (c^{\mathrm T}x) + c^{\mathrm T}\mathbb{E}(\varepsilon) \quad (\text{note that }\mathbb{E}(\varepsilon)=0_n)\\
  &= (1_n^{\mathrm T} c) \alpha + (c^{\mathrm T}x)\beta\\
  &=\alpha.
  \end{align}
  $$
  根据 $(1_n^{\mathrm T} c) \alpha + (c^{\mathrm T}x)\beta = \alpha$ 可知 $c$ 满足:
  $$
  \begin{cases}
  1_n^{\mathrm T} c =1\\
  c^{\mathrm T}x = 0.
  \end{cases}
  $$
  
- 最后，$\alpha$ 的一个最佳线性无偏估计量还满足:   
  $$
  \text{Var}(\hat \alpha) = \underset{
  \begin{subarray}{}
  1_n^{\mathrm T}c = 1\\
  c^{\mathrm T}x=0
  \end{subarray}
  }\min \text{Var}(c^{\mathrm T}y).
  $$
  注意到目标函数可进行如下化简:   
  $$
  \begin{align}
  \text{Var}(c^{\mathrm T}y) 
  &= \text{Var}(c^{\mathrm T}(\alpha 1_n + \beta x + \varepsilon))\\ 
  &= c^{\mathrm T}\text{Var}(\varepsilon )c\\ 
  &= c^{\mathrm T} \sigma^2 I_n c\\ 
  &= \sigma^2 \|c\|_2^2.
  \end{align}
  $$
  因此问题变为求解最优化问题:
  $$
  \hat c = \underset{
  \begin{subarray}{}
  1_n^{\mathrm T}c = 1\\
  c^{\mathrm T}x=0
  \end{subarray}
  }{\arg \min} \|c\|_2^2.
  $$
  定义 Lagrange 函数为:   
  $$
  L(c,\lambda_1,\lambda_2) = \|c\|_2^2 + \lambda_1 (1_n^{\mathrm T}c-1) + \lambda_2 (c^{\mathrm T}x-0).
  $$
  注意到目标函数是凸函数，因此其 $\text{KKT}$ 点就对应其最优解.  
  其 $\text{KKT}$ 条件为: 
  $$
  \begin{cases}
  \nabla_c L(c,\lambda_1,\lambda_2) = 2c + \lambda_1 1_n + \lambda_2 x = 0_n\\
  \frac{\partial }{\partial \lambda_1}L(c,\lambda_1,\lambda_2) = 1_n^{\mathrm T}c-1 = 0\\
  \frac{\partial }{\partial \lambda_2} L(c,\lambda_1,\lambda_2) = c^{\mathrm T}x = 0.
  \end{cases}
  $$
  分别对第一个等式左乘 $1_n^{\mathrm T}$ 和 $x^{\mathrm T}$，并代入第二个和第三个等式就可以得到 (只有一个 $\text{KKT}$ 点): 
  $$
  \begin{cases}
  2+n\lambda_1 + n\bar x\lambda_2=0\\
  n\bar x\lambda_1 + x^{\mathrm T}x \lambda_2 =0 
  \end{cases}
  \ \ \Rightarrow\ \ 
  \begin{cases}
  \lambda_1 = -2/(n-\frac{n^2 \bar x^2}{x^{\mathrm T}x}) = -2x^{\mathrm T}x/(nS_{xx})\\
  \lambda_2 = -(n\bar x/(x^{\mathrm T}x))\cdot\lambda_1 = 2\bar x/S_{xx}.
  \end{cases}
  $$
  于是唯一的最优解为:  
  $$
  \begin{align}
  \hat c
  &= -\frac12(\lambda_1 1_n + \lambda_2 x)\\
  &= \frac{x^{\mathrm T}x1_n}{nS_{xx}} - \frac{\bar x x}{S_{xx}}\\ 
  &= \frac1n1_n - \frac{\bar x x - \bar x^2 1_n}{S_{xx}}.
  \end{align}
  $$
  因此唯一的最佳线性无偏估计量为:  
  $$
  \begin{align}
  \hat \alpha 
  &= \hat c^{\mathrm T} y\\ 
  &= \left(\frac1n1_n - \frac{\bar x x - \bar x^2 1_n}{S_{xx}}\right)^{\mathrm T} y\\ 
  &= \frac1n 1_n^{\mathrm T}y - \frac{x^{\mathrm T}y-n\bar x\bar y}{S_{xx}}\bar x\\
  &= \bar y - \frac{S_{xy}}{S_{xx}}\bar x\\ 
  &= \bar y - \hat \beta \bar x.
  \end{align}
  $$
  其方差为 $\alpha$ 的线性无偏估计量所能具有的最小方差 (因而 $\hat \alpha$ 是最稳定的线性无偏估计量): 
  $$
  \begin{align}
  \text{Var}(\hat \alpha) 
  &= \sigma^2 \|\hat c\|_2^2\\
  &= \sigma^2 \left(\frac1n1_n - \frac{\bar x x - \bar x^2 1_n}{S_{xx}}\right)^{\mathrm T} \left(\frac1n1_n - \frac{\bar x x - \bar x^2 1_n}{S_{xx}}\right) \\
  &= \sigma^2 \left(\frac1{n^2}1_n^{\mathrm T}1_n - 2\frac{1}{n}1_n^{\mathrm T} \left(\frac{\bar x x - \bar x^2 1_n}{S_{xx}}\right) + \frac{(\bar x x - \bar x^2 1_n)^{\mathrm T} (\bar x x - \bar x^2 1_n)}{S_{xx}^2}\right)\\
  &= \sigma^2 \left(\frac1n - 2\frac{\bar x^2 - \bar x^2}{S_{xx}} + \bar x^2\frac{(x-\bar x1_n)^{\mathrm T}(x-\bar x 1_n)}{S_{xx}^2}\right)\\
  &= 
  \sigma^2 \left(\frac1n - 0 + \bar x^2 \frac{S_{xx}}{S_{xx}^2}\right)\\
  &=
  \sigma^2 \left(\frac1n + \frac{\bar x^2}{S_{xx}}\right)\\
  &=
  \sigma^2 \frac{(x^{\mathrm T}x-n\bar x^2)+ n\bar x^2}{nS_{xx}}\\
  &=
  \frac{\sigma^2 x^{\mathrm T}x}{nS_{xx}}.
  \end{align}
  $$

***

**(简单线性回归中的 Gauss-Markov 定理, Statistical Inference 第 $11.3.2$ 节)**  
给定数据点 $(x_1,y_1),\dots,(x_n,y_n)$ 和简单线性回归模型 $Y=\alpha + \beta X + \varepsilon$   
若 $\mathbb{E}(\varepsilon) = 0_n$ 且 $\text{Var}(\varepsilon) = \sigma^2 I_n$ (零均值和互不相关, 无需给出分布上的假设)，  
则最小二乘估计量
$$
\begin{cases}
\hat \beta = S_{xy}/S_{xx} = (x^{\mathrm T}y-n\bar x\bar y)/(x^{\mathrm T}x-n\bar x^2)\\
\hat \alpha = \bar y - \hat \beta \bar x
\end{cases}
$$
是参数 $(\alpha,\beta)$ 的**最佳线性无偏估计量** (Best Linear Unbiased Estimator, BLUE):
$$
\mathbb{E}(\hat \alpha) = \alpha \\
\mathbb{E}(\hat \beta) = \beta\\
\text{Var}(\hat \alpha ) = \left(\frac1n + \frac{\bar x^2}{S_{xx}}\right)\sigma^2 = \frac{x^{\mathrm T}x}{nS_{xx}}\sigma^2\\
\text{Var}(\hat \beta) = \frac{1}{S_{xx}}\sigma^2\\
\text{Cov}(\hat \alpha ,\hat \beta) = -\frac{\bar x}{ S_{xx}}\sigma^2.
$$




### 1.2.3 极大似然估计

现在我们考虑条件正态模型:  

> **条件正态模型** (conditional normal model) 是最常用的简单线性回归模型，也是最容易分析的模型.  
> 设观测样本是 $n$ 个数据对 $(x_1,y_1),\dots,(x_n,y_n)$，  
> 其中 $x_1,\dots,x_n$ 为随机变量 $X_1,\dots,X_n$ 的实现 (但暂时视其为已知的固定常数)，  
> 而 $y_1,\dots,y_n$ 是独立随机变量 $Y_1,\dots,Y_n$ 的观测值.
>
> 进一步，我们假定 $Y_1,\dots,Y_n$ 服从正态分布 (结合独立性可知它们联合正态) 且具有相同方差:
> $$
> Y_i \sim \mathcal{N}(\alpha + \beta x_i,\sigma^2)\ \ (i=1,\dots,n)\\
> \begin{bmatrix}
> Y_1\\
> \vdots\\
> Y_n
> \end{bmatrix}
> \sim
> \mathcal{N}
> \left(
> \begin{bmatrix}
> \alpha + \beta x_1\\
> \vdots\\
> \alpha + \beta x_n
> \end{bmatrix},
> \sigma^2 I_n
> \right)
> $$
> 由此可知总体回归函数为 $\mathbb{E}(Y|X=x) = \alpha + \beta x$.  
> 具体来说，条件分布 $(Y|X=x) \sim \mathcal{N}(\alpha + \beta x,\sigma^2)$.
>
> 条件正态模型还具有如下等价形式:
> $$
> Y_i = \alpha + \beta x_i + \varepsilon_i\ \ (i=1,\dots,n)\text{ where }\varepsilon_1,\dots,\varepsilon_n \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,\sigma^2),
> $$
> 它相当于基础模型额外加上了 $\varepsilon_1,\dots,\varepsilon_n \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,\sigma^2)$ 的假设.

#### (1) 似然解

下面我们求解三个参数 $\alpha,\beta,\sigma^2$ 的极大似然估计量.  
首先我们证明对于任意固定的 $\sigma^2$，$(\alpha,\beta)$ 的极大似然估计量正是最小二乘估计量:
$$
\begin{cases}
\hat \beta = S_{xy}/S_{xx} = (x^{\mathrm T}y-n\bar x\bar y)/(x^{\mathrm T}x-n\bar x^2)\\
\hat \alpha = \bar y - \hat \beta \bar x.
\end{cases}
$$
注意到 $(Y_1,\dots,Y_n)$ 服从 $n$ 元正态分布 $\mathcal{N}(\mu,\Sigma)$，其中 $\mu = \alpha 1_n + \beta x$ 而 $\Sigma = \sigma^2 I_n$.  
于是似然函数 $\mathcal{L}(\alpha,\beta,\sigma^2|x,y)$ 为:
$$
\begin{align}
\mathcal{L}(\alpha,\beta,\sigma^2|x,y) 
&:=
\text{P}(Y_1=y_1,\dots,Y_n=y_n)\\
&=
\text{P}(\mathcal{N}(\mu,\Sigma)= y)\\
&=
\frac{1}{\sqrt{(2\pi)^n\det(\Sigma)}} \exp\left(-\frac12(y-\mu)^{\mathrm T}\Sigma^{-1} (y-\mu)\right)\\
&=
\frac{1}{\sqrt{(2\pi)^n(\sigma^2)^n}} \exp\left(-\frac1{2\sigma^2}\|y-\alpha 1_n -\beta x\|_2^2\right).
\end{align}
$$
因此对数似然函数 $\log \mathcal{L}(\alpha,\beta,\sigma^2|x,y)$ 为:
$$
\log \mathcal{L}(\alpha,\beta,\sigma^2|x,y) = -\frac{n}{2} \log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2} \|y-\alpha 1_n -\beta x\|_2^2.
$$
注意到对于任意固定的 $\sigma^2$，对数似然函数 $\log \mathcal{L}(\alpha,\beta,\sigma^2|x,y)$ 作为 $\alpha,\beta$ 的函数最大化的问题，  
就等价于残差平方和 $\text{RSS}(\alpha,\beta)= \|y-\alpha 1_n -\beta x\|_2^2$ 作为 $\alpha,\beta$ 的函数最小化的问题.    
因此对于任意固定的 $\sigma^2$，$(\alpha,\beta)$ 的极大似然估计量正是最小二乘估计量:
$$
\begin{cases}
\hat \beta = S_{xy}/S_{xx} = (x^{\mathrm T}y-n\bar x\bar y)/(x^{\mathrm T}x-n\bar x^2)\\
\hat \alpha = \bar y - \hat \beta \bar x.
\end{cases}
$$
 由于 $\hat \alpha,\hat \beta$ 可以表示为联合正态随机变量 $y_1,\dots, y_n$ 的线性组合，故它们也是联合正态的:  
$$
\begin{bmatrix}
\hat \alpha\\
\hat \beta
\end{bmatrix} \sim 
\mathcal{N}\left(
\begin{bmatrix}
\alpha\\
\beta
\end{bmatrix},
\frac{\sigma^2}{S_{xx}}\begin{bmatrix}
x^{\mathrm T}x/n & -\bar x\\
-\bar x & 1
\end{bmatrix}
\right) 
= 
\mathcal{N}\left(
\begin{bmatrix}
\alpha\\
\beta
\end{bmatrix},
\sigma^2 (X^{\mathrm T}X)^{-1}
\right),
$$

其中 $X=[1_n,x]\in \mathbb R^{n\times 2}$.

> 回忆起:
> $$
> \text{Var}(\hat \alpha ) = \left(\frac1n + \frac{\bar x^2}{S_{xx}}\right)\sigma^2 = \frac{x^{\mathrm T}x}{nS_{xx}}\sigma^2\\
> \text{Var}(\hat \beta) = \frac{1}{S_{xx}}\sigma^2\\
> \text{Cov}(\hat \alpha ,\hat \beta) = -\frac{\bar x}{ S_{xx}}\sigma^2.
> $$

*****

现在我们将最小二乘估计量
$$
\begin{cases}
\hat \beta = S_{xy}/S_{xx} = (x^{\mathrm T}y-n\bar x\bar y)/(x^{\mathrm T}x-n\bar x^2)\\
\hat \alpha = \bar y - \hat \beta \bar x
\end{cases}
$$
代入对数似然函数 $\log \mathcal{L}(\alpha,\beta,\sigma^2|x,y)$ 以求得 $\sigma^2$ 的极大似然估计量.    
注意到对数似然函数 $\log \mathcal{L}(\hat\alpha,\hat \beta,\sigma^2|x,y)$ 关于 $\sigma^2$ 是严格凹函数，因此其最大值点唯一且为驻点. 
我们令:
$$
\frac{\partial }{\partial \sigma^2}\log \mathcal{L}(\alpha,\beta,\sigma^2|x,y) = -\frac{n}{2\sigma^2} + \frac12\|y-\hat \alpha 1_n - \hat \beta x\|_2^2 \frac{1}{\sigma^4} = 0,
$$
解得:  
$$
\hat\sigma^2 = \frac1{n}\|y-\hat \alpha 1_n - \hat \beta x\|_2^2 = \frac1n\text{RSS}(\hat \alpha,\hat \beta),
$$
即为在最小二乘线 $y=\hat \alpha + \hat \beta x$ 处算得的残差平方和 $\text{RSS}(\hat \alpha,\hat \beta)$ 除以样本量 $n$.
$$
\begin{align}
\hat\sigma^2
&=
\frac1{n}\|y-\hat \alpha 1_n - \hat \beta x\|_2^2\\
&=
\frac1n \|(\alpha 1_n + \beta x + \varepsilon)-\hat \alpha 1_n - \hat \beta x\|_2^2\\
&=
\frac1n \|\varepsilon - (\hat \alpha - \alpha)1_n - (\hat \beta - \beta)x\|_2^2

\quad \left(\text{recall that}\begin{cases}
\hat \alpha -\alpha = (\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}})^{\mathrm T} \varepsilon\\
\hat \beta-\beta = \frac{(x-\bar x 1_n)^{\mathrm T}}{S_{xx}}\varepsilon
\end{cases}\right)\\

&=
\frac1n \left\|\varepsilon - \left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)^{\mathrm T} \varepsilon 1_n - \frac{(x-\bar x 1_n)^{\mathrm T}}{S_{xx}}\varepsilon x\right\|_2^2\\

&=
\frac1n \left\|
\varepsilon - \frac1n 1_n1_n^{\mathrm T} \varepsilon - \frac{(x-\bar x1_n)(x-\bar x1_n)^{\mathrm T}}{S_{xx}} \varepsilon\right\|_2^2\\
&=
\frac1n \left\|
\left(I_n - \frac1n 1_n1_n^{\mathrm T} - \frac{(x-\bar x1_n)(x-\bar x1_n)^{\mathrm T}}{S_{xx}}\right) \varepsilon\right\|_2^2\\

&=
\frac1n \|(I_n-H)\varepsilon\|_2^2
\quad \left(\text{note that } H := \frac1n 1_n 1_n^{\mathrm T} + \frac{(x-\bar x1_n)(x-\bar x1_n)^{\mathrm T}}{S_{xx}}\text{ is symmetric and idempotent}\right)\\

&=
\frac1n \varepsilon^{\mathrm T} (I_n- H)^{\mathrm T}(I_n-H) \varepsilon\\

&=
\frac1n \varepsilon^{\mathrm T} (I_n- H)^2 \varepsilon\\
&=
\frac1n \varepsilon^{\mathrm T} (I_n - 2H + H^2)\varepsilon\\
&=
\frac1n \varepsilon^{\mathrm T} (I_n - 2H + H)\varepsilon\\
&=
\frac1n \varepsilon^{\mathrm T} (I_n - H)\varepsilon\\
\end{align}
$$

其中投影矩阵 $H\in \mathbb{R}^{n\times n}$ 的定义为:
$$
H := \frac1n 1_n 1_n^{\mathrm T} + \frac{(x-\bar x1_n)(x-\bar x1_n)^{\mathrm T}}{S_{xx}}.
$$



#### (2) 纠偏

下面我们证明 $\sigma^2$ 的极大似然估计量 $\hat\sigma^2 = \frac1{n}\|y-\hat \alpha 1_n - \hat \beta x\|_2^2 = \frac1n \varepsilon^{\mathrm T} (I_n - H)\varepsilon$ 不是无偏的.    

$$
\begin{align}
\mathbb{E}(\hat \sigma^2)
&=
\mathbb{E}(\frac1n \varepsilon^{\mathrm T} (I_n - H)\varepsilon)\\
&=
\frac1n\mathbb{E}(\tr(\varepsilon^{\mathrm T} (I_n - H)\varepsilon))\\
&=
\frac1n\mathbb{E}(\tr((I_n - H)\varepsilon\varepsilon^{\mathrm T}))\\
&=
\frac1n \tr(\mathbb{E}((I_n-H)\varepsilon \varepsilon^{\mathrm T}))\\
&=
\frac1n \tr((I_n-H)\mathbb{E}(\varepsilon \varepsilon^{\mathrm T}))\\
&=
\frac1n \tr((I_n-H)\sigma^2 I_n)\\
&=
\frac1n \sigma^2 (n - \tr(H))
\end{align}
$$
注意到:
$$
\begin{align}
\tr(H) 
&= \tr(\frac1n 1_n 1_n^{\mathrm T} + \frac{(x-\bar x1_n)(x-\bar x1_n)^{\mathrm T}}{S_{xx}})\\ 
&= \frac1n\cdot n + \frac{(x-\bar x1_n)^{\mathrm T}(x-\bar x1_n)}{S_{xx}}\\
&= 1+1\\
&= 2,
\end{align}
$$
于是我们有:
$$
\begin{align}
\mathbb{E}(\hat \sigma^2)
&=
\frac1n \sigma^2 (n - \tr(H))\\
&=
\frac{1}{n}\sigma^2(n-2)\\
&=
\frac{n-2}{n}\sigma^2.
\end{align}
$$
因此我们可以构造 $\sigma^2$ 的无偏估计量为:
$$
s^2 := \frac{n}{n-2}\hat\sigma^2 =\frac{1}{n-2}\|y-\hat \alpha 1_n - \hat \beta x\|_2^2.
$$


#### (3) 分布

在 $1.2.3 (1)$ 中我们已经说明了最小二乘估计量
$$
\begin{cases}
\hat \beta = S_{xy}/S_{xx} = (x^{\mathrm T}y-n\bar x\bar y)/(x^{\mathrm T}x-n\bar x^2)\\
\hat \alpha = \bar y - \hat \beta \bar x
\end{cases}
$$
服从联合正态分布:
$$
\begin{bmatrix}
\hat \alpha\\
\hat \beta
\end{bmatrix} \sim 
\mathcal{N}\left(
\begin{bmatrix}
\alpha\\
\beta
\end{bmatrix},
\frac{\sigma^2}{S_{xx}}\begin{bmatrix}
x^{\mathrm T}x/n & -\bar x\\
-\bar x & 1
\end{bmatrix}
\right) 
= 
\mathcal{N}\left(
\begin{bmatrix}
\alpha\\
\beta
\end{bmatrix},
\sigma^2 (X^{\mathrm T}X)^{-1}
\right),
$$
其中 $X=[1_n,x]\in \mathbb R^{n\times 2}$.

下面我们证明 $\hat\sigma^2 = \frac1{n}\|y-\hat \alpha 1_n - \hat \beta x\|_2^2 = \frac1n \varepsilon^{\mathrm T} (I_n - H)\varepsilon$ 与 $(\hat \alpha,\hat \beta)$ 独立.

> 在证明之前我们先给出两个有用的引理.  
> **(Statistical Inference 引理 $11.3.2$)**  
> 设 $Y_1,\dots,Y_n$ 是互不相关的随机变量，$\text{Var}(Y_i)=\sigma^2_i\ (i=1,\dots,n)$，$a,b\in \mathbb R^n$ 为常数向量.  
> 若记 $Y=(Y_1,\dots,Y_n)^{\mathrm T},\Sigma = \text{Cov}(Y)=\text{diag}(\sigma_1^2,\dots,\sigma^2_n)$，则我们有:    
> $$
> \begin{align}
> \text{Cov}(a^{\mathrm T}Y,b^{\mathrm T}Y) 
> &= \text{Cov}\left(\sum_{i=1}^n a_i Y_i, \sum_{j=1}^n b_j Y_j\right)\\ 
> &= \sum_{i=1}^n\sum_{j=1}^n a_ib_j\text{Cov}(Y_i,  Y_j)\quad 
> \left(\text{note that }
> \text{Cov}(Y_i,  Y_j)
> =
> \begin{cases}
> \sigma_i^2, &\text{if }i=j\\
> 0, &\text{otherwise}
> \end{cases}
> \right)\\ 
> &= \sum_{i=1}^n a_i b_i\sigma_i^2\\
> &= a^{\mathrm T} \Sigma b
> \end{align}
> $$
> **(Statistical Inference 引理 $5.3.3$)**  
> 设 $X_i\sim \mathcal{N}(\mu_i,\sigma_i^2)\ (i=1,\dots,n)$ 相互独立，$k+m\leq n$，记:  
> $$
> X = \begin{bmatrix}
> X_1\\
> \vdots\\
> X_n
> \end{bmatrix},
> \quad 
> U =\begin{bmatrix}
> U_1\\
> \vdots\\
> U_k
> \end{bmatrix},
> \quad
> V=\begin{bmatrix}
> V_1\\
> \vdots\\
> V_m
> \end{bmatrix}\\
> 
> A=(a_1,\dots,a_k) \in \mathbb R^{n\times k},\quad B=(b_1,\dots,b_m)\in \mathbb R^{n\times m}\\
> 
> U=A^{\mathrm T} X = \begin{bmatrix}
> a_1^{\mathrm T}X\\
> \vdots\\
> a_k^{\mathrm T}X
> \end{bmatrix},
> 
> \quad
> 
> V=B^{\mathrm T} X = \begin{bmatrix}
> b_1^{\mathrm T}X\\
> \vdots\\
> b_m^{\mathrm T}X
> \end{bmatrix}\\
> 
> \Sigma = \text{Cov}(X) =\text{diag}(\sigma_1^2,\dots,\sigma_n^2)
> $$
> 则我们有:
>
> - $U_i$ 和 $V_j$ 相互独立当且仅当 $\text{Cov}(U_i,V_j) = \text{Cov}(a_i^{\mathrm T}X,b_j^{\mathrm T}X) = a_i^{\mathrm T}\Sigma b_j=0$，  
>   其中 $i=1,\dots,k$ 而 $j=1,\dots,m$.
> - $U$ 和 $V$ 相互独立当且仅当对于任意 $i=1,\dots,k$ 和 $j=1,\dots,m$ 都有 $\text{Cov}(U_i,V_j)=0$，  
>   即当且仅当 $\text{Cov}(U,V) = \text{Cov}(A^{\mathrm T}X,B^{\mathrm T}X) = A^{\mathrm T} \Sigma B = 0_{k\times m}$.

回忆起 $\hat \alpha-\alpha,\hat \beta-\beta$ 可以写成 $\varepsilon$ 的线性组合:
$$
\begin{align}
\hat \alpha -\alpha 
&= \left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)^{\mathrm T} \varepsilon\\
\hat \beta-\beta 
&= \frac{(x-\bar x 1_n)^{\mathrm T}}{S_{xx}}\varepsilon.
\end{align}
$$
要证明 $\hat\sigma^2 = \frac1n \|(I_n - H)\varepsilon\|_2^2$ 与 $(\hat \alpha,\hat \beta)$ 独立，  
只需证明 $(I_n -H)\varepsilon$ 与 $(\hat \alpha,\hat \beta)$ 独立即可.  
根据 **Statistical Inference 引理 $5.3.3$** 可知，  
只需证明 $\text{Cov}((I_n -H)\varepsilon,\hat \alpha)=0_n$ 和 $\text{Cov}((I_n -H)\varepsilon,\hat \beta)=0_n$ 即可.
$$
\begin{align}
\text{Cov}((I_n -H)\varepsilon,\hat \alpha)
&=
\text{Cov}((I_n -H)\varepsilon,\hat \alpha - \alpha)\\
&=
\text{Cov}\left((I_n -H)\varepsilon,\left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)^{\mathrm T} \varepsilon\right)\\
&=
(I_n-H)\cdot \sigma^2 I_n \cdot \left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)\\
&=
\sigma^2\left(\left(I_n-\frac1n 1_n 1_n^{\mathrm T} - \frac{(x-\bar x1_n)(x-\bar x1_n)^{\mathrm T}}{S_{xx}}\right)\left(\frac{1}{n} 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)\right)\\
&=
\sigma^2\left(\frac1n 1_n - \frac{\bar x(x-\bar x 1_n)}{S_{xx}} -\frac1n 1_n + 0_n + 0_n + \frac{\bar x(x-\bar x 1_n)}{S_{xx}}\right)\\
&=
0_n\\

\hline
\text{Cov}((I_n -H)\varepsilon,\hat \beta)
&=
\text{Cov}((I_n -H)\varepsilon,\hat \beta - \beta)\\
&=
\text{Cov}\left((I_n- H)\varepsilon, \frac{(x-\bar x 1_n)^{\mathrm T}}{S_{xx}}\varepsilon\right)\\
&=
(I_n-H)\cdot \sigma^2 I_n \cdot \frac{(x-\bar x 1_n)}{S_{xx}}\\
&=
\sigma^2\left(\left(I_n-\frac1n 1_n 1_n^{\mathrm T} - \frac{(x-\bar x1_n)(x-\bar x1_n)^{\mathrm T}}{S_{xx}}\right)\frac{(x-\bar x 1_n)}{S_{xx}}\right)\\
&=
\sigma^2 \left(\frac{(x-\bar x 1_n)}{S_{xx}} - 0_n - \frac{(x-\bar x 1_n)}{S_{xx}}\right)\\
&=
0_n

\end{align}
$$
根据 $\hat \sigma^2\ \bot \ (\hat \alpha,\hat \beta)$ 可知 $s^2 = \frac{n}{n-2}\hat\sigma^2\ \bot\ (\hat \alpha,\hat \beta)$.

*****

下面我们证明 $n\hat\sigma^2/\sigma^2 = \varepsilon^{\mathrm T} (I_n - H)\varepsilon/\sigma^2$ 服从自由度为 $n-2$ 的 $\chi^2$ 分布:  
在前面的证明中，我们说明了矩阵 $H$ 的几个性质:  

- $H$ 对称且幂等
- $\tr(H)=2$ 

因此 $H$ 的 $n$ 个特征值中，有 $2$ 个是 $1$，其余 $n-2$ 个是 $0$.  
进而可知 $I_n - H$ 的 $n$ 个特征值中，有 $2$ 个是 $0$，其余 $n-2$ 个是 $1$.   
设 $I_n-H$ 的谱分解 (实对称阵一定具有谱分解) 为:  
$$
U^{\mathrm T}(I_n-H)U = \Lambda = \text{diag}(\underset{n-2}{\underbrace{1,\dots,1}},0,0).
$$
记 $\eta := U^{\mathrm T}\varepsilon/\sigma$，根据 $\varepsilon\sim \mathcal{N}(0_n,\sigma^2 I_n)$ 可知:   
$$
\eta = \frac{1}{\sigma}U^{\mathrm T}\varepsilon \sim \mathcal{N}\left(\frac{1}{\sigma}U^{\mathrm T}0_n,\frac{1}{\sigma^2}U^{\mathrm T}\sigma^2 I_n U\right) = \mathcal{N}(0_n, I_n).
$$
这表明 $\eta = (\eta_1,\dots,\eta_n)^{\mathrm T}$ 的分量是独立同分布的标准正态随机变量.  
于是我们有:  
$$
\begin{align}
\frac{n}{\sigma^2}\hat \sigma^2 
&=
 \frac{1}{\sigma^2}\varepsilon^{\mathrm T} (I_n - H)\varepsilon\\
&=
 \frac{1}{\sigma^2}\varepsilon^{\mathrm T} U\Lambda U^{\mathrm T}\varepsilon\\
&=
\eta^{\mathrm T} \Lambda \eta\\
&=
 \sum_{i=1}^{n-2} \eta_i^2\sim \chi^2_{(n-2)}\quad (\text{note that }\Lambda=\text{diag}(\underset{n-2}{\underbrace{1,\dots,1}},0,0))

\end{align}
$$
即得证 $n\hat\sigma^2/\sigma^2 = \varepsilon^{\mathrm T} (I_n - H)\varepsilon/\sigma^2$ 服从自由度为 $n-2$ 的 $\chi^2$ 分布.  
据此我们也可以自然地得到:   
$$
(n-2) s^2 = n\hat \sigma^2 \sim \sigma^2\chi^2_{(n-2)}\\
\mathbb{E}(\hat \sigma^2) = \frac{\sigma^2}{n}(n-2) = \frac{n-2}{n}\sigma^2
\ \Rightarrow\ 
\mathbb{E}(s^2) = \sigma^2\\

\text{Var}(\hat \sigma^2) = \frac{\sigma^4}{n^2}2(n-2) = \frac{2(n-2)}{n^2}\sigma^4
\ \Rightarrow\ 
\text{Var}(s^2) = \frac{2}{n-2}\sigma^4
$$

****

总之我们有如下定理:  
**(Statistical Inference 定理 $11.3.2$)**  
在条件正态模型下，$\hat \alpha,\hat \beta,s^2$ 分别是 $\alpha,\beta,\sigma^2$ 的无偏估计量，它们满足:
$$
{\begin{cases}
\hat \alpha = \bar y - \hat \beta \bar x\\
\hat \beta = S_{xy}/S_{xx} = (x^{\mathrm T}y-n\bar x\bar y)/(x^{\mathrm T}x-n\bar x^2)
\end{cases}}\Rightarrow 

\begin{bmatrix}
\hat \alpha\\
\hat \beta
\end{bmatrix} = (X^{\mathrm T}X)^{-1}X^{\mathrm T}y\\

s^2 = \frac1{n-2}\|y-\hat \alpha 1_n - \hat \beta x\|_2^2 = \frac1{n-2} \varepsilon^{\mathrm T} (I_n - H)\varepsilon\\

\hline 

\begin{bmatrix}
\hat \alpha\\
\hat \beta
\end{bmatrix} \sim 
\mathcal{N}\left(
\begin{bmatrix}
\alpha\\
\beta
\end{bmatrix},
\frac{\sigma^2}{S_{xx}}\begin{bmatrix}
x^{\mathrm T}x/n & -\bar x\\
-\bar x & 1
\end{bmatrix}
\right) 
= 
\mathcal{N}\left(
\begin{bmatrix}
\alpha\\
\beta
\end{bmatrix},
\sigma^2 (X^{\mathrm T}X)^{-1}
\right)\\

\text{SSE} = (n-2)s^2 \sim \sigma^2\chi^2_{(n-2)}\\

\hline
\mathbb{E}(s^2) = \sigma^2\\
\text{Var}(s^2) = \frac{2}{n-2}\sigma^4
$$

其中设计矩阵 $X$ 和投影矩阵 $H$ 的定义为:
$$
X:=[1_n,x]\in \mathbb R^{n\times 2}\\

H := \frac1n 1_n 1_n^{\mathrm T} + \frac{(x-\bar x1_n)(x-\bar x1_n)^{\mathrm T}}{S_{xx}} = X(X^{\mathrm T}X)^{-1}X^{\mathrm T}\in \mathbb{R}^{n\times n}.
$$


## 1.3 假设检验

### 1.3.1 $\beta=0$ 的检验

关于参数 $\alpha$ 和 $\beta$ 的推断通常基于下面两个 $t$ 分布:
$$
\frac{\hat \alpha - \alpha}{s\sqrt{x^{\mathrm T}x/(nS_{xx})}} = 
\frac{(\hat\alpha - \alpha)/(\sigma \sqrt{x^{\mathrm T}x/(nS_{xx})})}{s/\sigma}
\sim 
\frac{\mathcal{N}(0,1)}{\sqrt{\chi_{(n-2)}^2/(n-2)}} = t_{n-2}\\

\frac{\hat \beta - \beta}{s\sqrt{1/S_{xx}}} 
=
\frac{(\hat \beta -\beta)/(\sigma \sqrt{1/S_{xx}})}{s/\sigma}

\sim \frac{\mathcal{N}(0,1)}{\sqrt{\chi_{(n-2)}^2/(n-2)}}= t_{n-2}
$$
事实上这两个量 (注意它们并不是统计量，而是枢轴量) 的联合分布是二变量 $t$ 分布，  
它们可用来对 $\alpha,\beta$ 进行同时推断.  
然而在实际应用中我们习惯每次只对一个参数进行推断，其中我们通常对参数 $\beta$ 更感兴趣.  
参数 $\alpha = \mathbb{E}(Y|X=0)$ 是否值得推断取决于具体的问题.  
参数 $\beta$ 是 $\mathbb{E}(Y|X=x)$ 作为 $x$ 的函数的变化率，它包含了 $Y$ 与 $X$ 之间线性关系的信息.

**(Statistical Inference 习题 $11.33$)**  
在二元正态模型的假设下，$\beta =0$ 当且仅当 $\rho = 0$.  
此外，当 $\beta=0$ 时，枢轴量 $\frac{\hat \beta - \beta}{s\sqrt{1/S_{xx}}} = \frac{\sqrt{n-2}r}{\sqrt{1-r^2}}$ (可用于检验 $\beta=0$)，  
其中 $r=S_{xy}/\sqrt{S_{xx}S_{yy}}$ 为样本相关系数，可以证明 $r$ 为 $\rho$ 的极大似然估计量.

> **(回顾二元正态模型)**  
> 我们将观测样本 $(x_1,y_1),\dots,(x_n,y_n)$ 视为独立的二元正态随机变量 $(X_1,Y_1),\dots,(X_n,Y_n)$ 的实现.  
> 具体来说，我们假定:  
> $$
> (X_i,Y_i)\sim \text{bivariate normal}(\mu_X,\mu_Y,\sigma^2_X,\sigma^2_Y,\rho) 
> = 
> \mathcal{N}\left(
> \begin{bmatrix}
> \mu_X\\
> \mu_Y
> \end{bmatrix},
> \begin{bmatrix}
> \sigma_X^2 & \rho\sigma_X\sigma_Y\\
> \rho\sigma_X\sigma_Y & \sigma_Y^2
> \end{bmatrix}
> \right).
> $$
> 对于二元正态分布，给定 $X=x$ 时 $Y$ 的条件分布是正态分布.  
> 具体来说，其总体回归函数为:  
> $$
> \mathbb{E}(Y|X=x) 
> = \mu_Y + \rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X)
> = \left(\mu_Y - \rho\frac{\sigma_Y}{\sigma_X}\mu_X\right) + \left(\rho\frac{\sigma_Y}{\sigma_X}\right)x = \alpha + \beta x,
> $$
> 其中:
> $$
> \begin{cases}
> \alpha = \mu_Y - (\rho \sigma_Y/\sigma_X)\mu_X=\mu_Y- \beta \mu_X\\
> \beta = \rho \sigma_Y/\sigma_X.
> \end{cases}
> $$
>
> 换言之，二元正态模型的假设自然保证了总体回归 $\mathbb{E}(Y|X=x)$ 是 $x$ 的线性函数，  
> 我们不需要像在前面的模型中那样假定这一点.



#### (1) 基本记号

考虑第一类型错误概率界限为 $\alpha$ 的检验问题 $H_0:\beta = 0\ \leftrightarrow\ H_1:\beta \neq 0$.

**① 总平方和 (Total Sum of Squares)**
$$
\begin{align}
\text{SST}
&= S_{yy}\\
&= \|y- \bar y 1_n\|_2^2\\
&= (y-\bar y 1_n)^{\mathrm T} (y-\bar y 1_n)\\
&= y^{\mathrm T}y - n\bar y^2\\
&= y^{\mathrm T}y - n (\frac1n1_n^{\mathrm T}y)^2\\
&= y^{\mathrm T}(I_n - \frac1n 1_n 1_n^{\mathrm T})y
\end{align}
$$
显然 $\text{SST}$ 只与样本有关，与模型无关.  

当零假设 $H_0:\beta =0$ 成立时，我们有:  
$$
\begin{align}
\text{SST}
&= y^{\mathrm T}\left(I_n - \frac1n 1_n 1_n^{\mathrm T}\right) y\\
&\overset{H_0}=
(\alpha 1_n + \varepsilon)^{\mathrm T}\left(I_n - \frac1n 1_n 1_n^{\mathrm T}\right) (\alpha 1_n + \varepsilon)\\
&= \varepsilon^{\mathrm T} \left(I_n - \frac1n 1_n 1_n^{\mathrm T}\right) \varepsilon
\end{align}
$$
注意到 $I_n - \frac1n 1_n 1_n^{\mathrm T}$ 对称、幂等且 $\tr(I_n - \frac1n 1_n 1_n^{\mathrm T})=n-1$.  
因此它存在谱分解，且有 $n-1$ 个特征值是 $1$，$1$ 个特征值是 $0$.  
我们可设其谱分解为:   
$$
U^{\mathrm T}(I_n - \frac1n 1_n 1_n^{\mathrm T})U = \Lambda = \text{diag}(\underset{n-1}{\underbrace{1,\dots,1}},0).
$$
记 $\eta := U^{\mathrm T} \varepsilon/\sigma$，根据 $\varepsilon\sim \mathcal{N}(0_n,\sigma^2 I_n)$ 可知:   
$$
\eta = \frac{1}{\sigma}U^{\mathrm T}\varepsilon \sim \mathcal{N}(\frac{1}{\sigma}U^{\mathrm T}0_n,\frac{1}{\sigma^2}U^{\mathrm T}\sigma^2 I_n U) = \mathcal{N}(0_n, I_n).
$$
这表明 $\eta = (\eta_1,\dots,\eta_n)^{\mathrm T}$ 的分量是独立同分布的标准正态随机变量.  
于是我们有:
$$
\begin{align}
\frac{1}{\sigma^2}\text{SST}
&= 
\frac{1}{\sigma^2}\|y- \bar y 1_n\|_2^2\\
&\overset{H_0}= 
\frac{1}{\sigma^2}\varepsilon^{\mathrm T} \left(I_n - \frac1n 1_n 1_n^{\mathrm T}\right) \varepsilon\\
&=
\frac{1}{\sigma^2}\varepsilon^{\mathrm T} U\Lambda U^{\mathrm T} \varepsilon\\
&=
\eta^{\mathrm T}\Lambda \eta\\
&=
\sum_{i=1}^{n-1} \eta_i^2\sim \chi^2_{(n-1)}\quad (\text{note that }\Lambda = \text{diag}(\underset{n-1}{\underbrace{1,\dots,1}},0))
\end{align}
$$
此时我们定义**总均方和** (Total Mean Squares) 为:
$$
\text{MST}=\frac{\text{SST}}{\text{df}_T} = \frac{\text{SST}}{n-1}=\frac1{n-1}S_{yy},
$$
则我们有:  
$$
\text{SST} = S_{yy}\overset{H_0}\sim \sigma^2\chi^2_{(n-1)}\\
\text{MST} = \frac{1}{n-1}S_{yy} \overset{H_0}\sim \sigma^2 \frac{\chi_{(n-1)}^2}{n-1}.
$$

****

**② 回归平方和 (Regression Sum of Squares)**
$$
\begin{align}
\text{SSR} 
&= \|\hat y - \bar y 1_n\|_2^2\\
&= \|\hat \alpha 1_n + \hat \beta x - \bar y1_n\|_2^2\\
&= \|(\bar y-\hat \beta \bar x)1_n + \hat\beta x - \bar y 1_n\|_2^2\\
&= \|\hat \beta (x-\bar x 1_n)\|_2^2\\
&= \hat \beta^2 S_{xx}\\
&= (\frac{S_{xy}}{S_{xx}})^2 S_{xx}\\
&= \frac{S_{xy}^2}{S_{xx}}
\end{align}
$$
当零假设 $H_0:\beta =0$ 成立时，我们有:
$$
\begin{align}
\frac{1}{\sigma^2}\text{SSR}
&=
\frac{1}{\sigma^2}\hat \beta^2 S_{xx}\\
&=
\left(\frac{\hat \beta}{\frac{\sigma}{\sqrt{S_{xx}}}}\right)^2
\overset{H_0}\sim (\mathcal{N}(0,1))^2 = \chi^2_{(1)}\\

&(\text{note that }\hat \beta \sim \mathcal{N}(\beta,\frac{\sigma^2}{S_{xx}}) \overset{H_0}= \mathcal{N}(0,\frac{\sigma^2}{S_{xx}}))\\
\end{align}
$$
此时我们定义**回归均方和** (Regression Mean Squares) 为:
$$
\text{MSR}=\frac{\text{SSR}}{\text{df}_R} = \frac{\text{SSR}}{1} = \frac{S_{xy}^2}{S_{xx}},
$$
则我们有:
$$
\text{MSR} = \text{SSR} = \frac{S_{xy}^2}{S_{xx}}\overset{H_0}\sim \sigma^2\chi^2_{(1)}.
$$

****

**③ 误差平方和 (Sum of Squared Errors)**
$$
\begin{align}
\text{SSE}
&=
\text{RSS}(\hat \alpha,\hat \beta)\\
&=
\|y-\hat y\|_2^2\\
&=
\|y-\hat\alpha 1_n - \hat \beta x\|_2^2\\
&=
(n-2)s^2\sim \sigma^2\chi^2_{(n-2)}
\end{align}
$$
值得注意的是 $\text{SSE}\sim \sigma^2\chi^2_{(n-2)}$ 并不依赖于零假设 $H_0:\beta =0$ 成立.  
可以证明 $\text{SSE} = \text{SST}-\text{SSR}$:  
$$
\begin{align}
\text{SSE}
&=
\|y-\hat \alpha 1_n - \hat \beta x\|_2^2\\
&=
\|y-(\bar y-\hat \beta \bar x)1_n - \hat \beta x \|_2^2\\
&=
 \|(y-\bar y 1_n) -\hat\beta (x-\bar x1_n)\|_2^2\\
&=
(y-\bar y1_n)^{\mathrm T}(y-\bar y 1_n) -2\hat \beta (y-\bar y1_n)^{\mathrm T} (x-\bar x 1_n) + \hat \beta^2 (x-\bar x 1_n)^{\mathrm T}(x-\bar x 1_n)\\
&=
S_{yy} - 2\hat \beta S_{xy} + \hat \beta^2 S_{xx}\\
&=
S_{yy} - 2\frac{S_{xy}}{S_{xx}} S_{xy} + \left(\frac{S_{xy}}{S_{xx}}\right)^2 S_{xx}\\
&=
S_{yy}-\frac{S_{xy}^2}{S_{xx}}\\
&=
\text{SST}-\text{SSR}
\end{align}
$$
此时我们定义**均方误差** (Mean Square Error) 为:
$$
\text{MSE}=\frac{\text{SSE}}{\text{df}_E} = \frac{\text{SSE}}{n-2} = \frac{1}{n-2}\left(S_{yy}-\frac{S_{xy}^2}{S_{xx}}\right),
$$
则我们有:  
$$
\text{SSE} = S_{yy}-\frac{S_{xy}^2}{S_{xx}}\sim \sigma^2\chi^2_{(n-2)}\\
\text{MSE} = \frac{1}{n-2}\left(S_{yy}-\frac{S_{xy}^2}{S_{xx}}\right)\sim \sigma^2 \frac{\chi_{(n-2)}^2}{n-2}.
$$

*****

总结如下:
$$
\begin{cases}
H_0: \beta = 0\\
\hline
\text{SST} = \|y-\bar y 1_n\|_2^2= S_{yy} \overset{H_0}\sim \sigma^2\chi^2_{(n-1)}
&\quad \text{MST} = \frac{\text{SST}}{\text{df}_T} = \frac{1}{n-1}S_{yy} \overset{H_0}\sim \sigma^2 \frac{\chi^2_{(n-1)}}{n-1}\\

\text{SSR} = \|\hat y - \bar y1_n\|_2^2 = \frac{S_{xy}^2}{S_{xx}} \overset{H_0}\sim \sigma^2 \chi^2_{(1)}
&\quad
\text{MSR} = \frac{\text{SSR}}{\text{df}_R} = \frac{S_{xy}^2}{S_{xx}} \overset{H_0}\sim \sigma^2 \chi^2_{(1)}\\

\text{SSE} = \|y-\hat y\|_2^2 = (S_{yy}-\frac{S_{xy}^2}{S_{xx}}) \sim \sigma^2 \chi^2_{(n-2)}
&\quad
\text{MSE} = \frac{\text{SSE}}{\text{df}_E} = \frac{1}{n-2}(S_{yy}-\frac{S_{xy}^2}{S_{xx}}) \sim \sigma^2 \frac{\chi^2_{(n-2)}}{n-2}\\

\begin{cases}
\text{SST} = \text{SSR} + \text{SSE}\\
\text{df}_T = \text{df}_R + \text{df}_E\\
\text{SSR}\ \bot \ \text{SSE}
\end{cases}

\end{cases}
$$



#### (2) 检验法

考虑第一类型错误概率界限为 $\alpha$ 的检验问题 $H_0:\beta = 0\ \leftrightarrow\ H_1:\beta \neq 0$     
我们使用如下的 $t$ 统计量或等价的 $F$ 统计量:
$$
\frac{\hat \beta}{s/\sqrt{S_{xx}}} \overset{H_0}\sim t_{n-2}\\
\frac{\hat \beta^2}{s^2/S_{xx}} \overset{H_0}\sim F_{1,n-2}
$$
我们记 $t_{n-2,\frac{\alpha}{2}}$ 为 $t_{n-2}$ 分布的 $1-\frac{\alpha}{2}$ (右尾) 分位数，$F_{1,n-2,\alpha}$ 为 $F_{1,n-2}$ 分布的 $1-\alpha$ 分位数.  
根据定义我们知道 $F_{1,n-2} = (t_{n-2})^2$，因而有 $F_{1,n-2,\alpha} = (t_{n-2,\frac{\alpha}{2}})^2$. 

值得注意的是，$F$ 统计量的表达式可以写为:  
$$
\frac{\hat \beta^2}{s^2/S_{xx}} = \frac{\hat \beta^2S_{xx}}{s^2} = \frac{\text{MSR}}{\text{MSE}} \overset{H_0}\sim F_{1,n-2}
\text{ where }\begin{cases}
\text{MSR} = \hat \beta^2 S_{xx} = S_{xy}^2/S_{xx} \overset{H_0}\sim \sigma^2\chi^2_{(1)}\\
\text{MSE} = s^2 = \frac{1}{n-2}(S_{yy}-S_{xy}^2/S_{xx})\sim \sigma^2 \chi^2_{(n-2)}
\end{cases}\\

\Downarrow\\
\frac{\text{MSR}}{\text{MSE}} 
= \frac{S_{xy}^2/S_{xx}}{\frac{1}{n-2}(S_{yy}-S_{xy}^2/S_{xx})} 
= \frac{(n-2)S_{xy}^2}{S_{xx}S_{yy}-S_{xy}^2}
$$

其中 $\text{MSE}=s^2\ \bot\ \text{MSR}=\hat \beta^2 S_{xx}$ 由 **Statistical Inference 定理 $11.3.2$** 中的 $s^2\ \bot\ \hat \beta^2$ 保证.

因此 $F$-检验法和等价的 $t$-检验法为:

- ($F$-检验法) 若 $\frac{\text{MSR}}{\text{MSE}}=\frac{(n-2)S_{xy}^2}{S_{xx}S_{yy}-S_{xy}^2}> F_{1,n-2,\alpha}$，则我们拒绝零假设 $H_0:\beta = 0$.
- ($t$-检验法) 若 $\sqrt{\frac{\text{MSR}}{\text{MSE}}}=\sqrt{\frac{(n-2)S_{xy}^2}{S_{xx}S_{yy}-S_{xy}^2}}> t_{n-2,\frac{\alpha}2}$，则我们拒绝零假设 $H_0:\beta = 0$.



#### (3) 推广: $\beta=\beta_0$ 的检验

考虑第一类型错误概率界限为 $\alpha$ 的检验问题 $H_0:\beta = \beta_0\ \leftrightarrow\ H_1:\beta \neq \beta_0$.  
注意到:
$$
\frac{\hat \beta - \beta_0}{s\sqrt{1/S_{xx}}} 
=
\frac{(\hat \beta -\beta_0)/(\sigma \sqrt{1/S_{xx}})}{s/\sigma}

\overset{H_0}\sim \frac{\mathcal{N}(0,1)}{\sqrt{\chi_{(n-2)}^2/(n-2)}}= t_{n-2}\\

\left(\text{note that }\hat \beta \sim \mathcal{N}\left(\beta,\frac{\sigma^2}{S_{xx}}\right) \overset{H_0}= \mathcal{N}\left(\beta_0,\frac{\sigma^2}{S_{xx}}\right)\right)
$$
因此我们可以使用如下的 $t$ 统计量或等价的 $F$ 统计量:
$$
\frac{\hat \beta - \beta_0}{s/\sqrt{S_{xx}}} \overset{H_0}\sim t_{n-2}\\
\frac{(\hat \beta-\beta_0)^2}{s^2/S_{xx}} \overset{H_0}\sim F_{1,n-2}
$$
值得注意的是，$F$ 统计量的表达式可以写为:   
$$
\frac{(\hat \beta-\beta_0)^2}{s^2/S_{xx}} 
= 
\frac{(S_{xy}/S_{xx} - \beta_0)^2S_{xx}}{\frac{1}{n-2}(S_{yy}-S_{xy}^2/S_{xx})} 
=
\frac{(n-2)(S_{xy}-\beta_0 S_{xx})^2}{S_{xx}S_{yy}-S_{xy}^2}.
$$
因此 $F$-检验法和等价的 $t$-检验法为:  

- ($F$-检验法) 若 $\frac{(\hat \beta-\beta_0)^2}{s^2/S_{xx}}=\frac{(n-2)(S_{xy}-\beta_0 S_{xx})^2}{S_{xx}S_{yy}-S_{xy}^2}> F_{1,n-2,\alpha}$，则我们拒绝零假设 $H_0:\beta = \beta_0$ 
- ($t$-检验法) 若 $\sqrt{\frac{(\hat \beta-\beta_0)^2}{s^2/S_{xx}}}=\sqrt{\frac{(n-2)(S_{xy}-\beta_0 S_{xx})^2}{S_{xx}S_{yy}-S_{xy}^2}}> t_{n-2,\frac{\alpha}2}$，则我们拒绝零假设 $H_0:\beta = \beta_0$ 



### 1.3.2 响应的估计

#### (1) 平均响应

考虑条件正态模型 $(Y|X=x) \sim \mathcal{N}(\alpha + \beta x,\sigma^2)$.  
关于预测变量的一个给定值 $x=x_0$ 有一个 $Y$ 值的总体，  
在条件正态模型的假设下我们有 $(Y|X=x_0) \sim \mathcal{N}(\alpha + \beta x_0,\sigma^2)$.

在观测到数据 $(x_1,y_1),\dots,(x_n,y_n)$ 并得到 $\alpha,\beta,\sigma^2$ 的估计量 $\hat \alpha,\hat \beta,s^2$ 之后，  
试验者可能要设置 $x=x_0$ 来得到一个新的观测值 $y_0$，它是随机变量 $Y_0\sim \mathcal{N}(\alpha+\beta x_0,\sigma^2)$ 的一个实现.

平均响应 $\mu=\mathbb{E}(Y|X=x_0)=\alpha + \beta x_0$ 点估计的一个显然的选择就是 $\hat\mu=\hat \alpha + \hat \beta x_0$，因为它是无偏的.
$$
\begin{align}
\mathbb{E}(\hat\mu)
&=
\mathbb{E}(\hat \alpha + \hat \beta x_0)\\
&=
\mathbb{E}(\hat \alpha) + \mathbb{E}(\hat \beta) x_0\\
&=
\alpha + \beta x_0\\

\hline

\text{Var}(\hat \mu)
&=
\text{Var}(\hat \alpha + \hat \beta x_0)\\
&=
\text{Var}(\hat \alpha) + 2x_0 \text{Cov}(\hat \alpha ,\hat \beta) + x_0^2 \text{Var}(\hat \beta)\\
&=
\left(\frac1n + \frac{\bar x^2}{S_{xx}}\right)\sigma^2 + 2x_0 \cdot \left(-\frac{\bar x \sigma^2}{S_{xx}}\right) + x_0^2 \frac{\sigma^2}{S_{xx}}\\
&=
\sigma^2 \left(\frac1n + \frac{(x_0-\bar x)^2}{S_{xx}}\right)..
\end{align}
$$

> 回忆起:
> $$
> \text{Var}(\hat \alpha ) = \left(\frac1n + \frac{\bar x^2}{S_{xx}}\right)\sigma^2 = \frac{x^{\mathrm T}x}{nS_{xx}}\sigma^2\\
> \text{Var}(\hat \beta) = \frac{1}{S_{xx}}\sigma^2\\
> \text{Cov}(\hat \alpha ,\hat \beta) = -\frac{\bar x}{ S_{xx}}\sigma^2.
> $$

由于 $(\hat \alpha ,\hat \beta)$ 联合正态，故 $\hat\mu=\hat \alpha + \hat \beta x_0$ 也服从正态分布:  
$$
\hat\mu=\hat \alpha + \hat \beta x_0 \sim \mathcal{N}\left(\alpha + \beta x_0, \sigma^2 \left(\frac1n + \frac{(x_0-\bar x)^2}{S_{xx}}\right)\right)
$$
由于 $s^2$ 与 $(\hat \alpha ,\hat \beta)$ 独立，故 $s^2$ 也与 $\hat\mu=\hat \alpha + \hat \beta x_0$ 独立，因此我们有:  
$$
\frac{\hat \alpha + \hat \beta x_0 - (\alpha + \beta x_0)}{s\sqrt{\frac1n + \frac{(x_0-\bar x)^2}{S_{xx}}}} 
=
\frac{
{
\frac{\hat \alpha + \hat \beta x_0 - (\alpha + \beta x_0)}{\sigma\sqrt{\frac1n + \frac{(x_0-\bar x)^2}{S_{xx}}}
}}}

{\frac{s}{\sigma}}
\sim
\frac{\mathcal{N}(0,1)}{\sqrt{\chi_{(n-2)}^2/(n-2)}}= t_{n-2}.
$$
根据这个枢轴量便可得到平均响应 $\mu = \alpha + \beta x_0$ 的 $(1-\alpha)$ 置信区间:  
$$
\alpha + \beta x_0 \in \left(\hat \alpha + \hat \beta x_0 \pm s\sqrt{\frac1n + \frac{(x_0-\bar x)^2}{S_{xx}}}t_{n-2,\frac{\alpha}{2}}\right)
$$
其中 $t_{n-2,\frac{\alpha}{2}}$ 为 $t_{n-2}$ 分布的 $1-\frac{\alpha}{2}$ (右尾) 分位数.



#### (2) 个体响应

现在我们考虑个体响应 $Y_0\sim \mathcal{N}(\alpha+\beta x_0,\sigma^2)$ 的估计.  

- **(预测区间)**    
  未观测随机变量 $Y$ 的一个基于样本 $X$ 的 $(1-\alpha)$ 预测区间是一个随机区间 $(L(X),U(X))$，满足:  
  $$
  \text{P}_\theta (L(X)\leq Y\leq U(X)) \geq 1-\alpha \ \ (\forall\ \theta\in \Theta).
  $$
  预测区间的定义与置信区间是相似的，区别在于预测区间是针对随机变量的，而不是针对参数的.  
  直观上，由于随机变量与参数 (常数) 相比具有变异性，故预测区间会比相同水平的置信区间更宽.

由于估计量 $\hat \alpha,\hat\beta,s^2$ 是根据以前的数据计算出来的，故 $Y_0$ 与 $\hat \alpha,\hat\beta,s^2$ 独立.
$$
{\begin{cases}
Y_0\sim \mathcal{N}(\alpha+\beta x_0,\sigma^2)\\
\hat \alpha + \hat \beta x_0 \sim \mathcal{N}(\alpha + \beta x_0, \sigma^2 (\frac1n + \frac{(x_0-\bar x)^2}{S_{xx}}))
\end{cases}}\\

\Downarrow\\

\begin{align}
Y_0- (\hat \alpha + \hat \beta x_0)
&\sim \mathcal{N}\left(\alpha + \beta x_0-\alpha -\beta x_0, \sigma^2 + \sigma^2 \left(\frac1n + \frac{(x_0-\bar x)^2}{S_{xx}}\right)\right)\\ 
&= \mathcal{N}\left(0,\sigma^2\left(1+\frac1n + \frac{(x_0-\bar x)^2}{S_{xx}}\right)\right)

\end{align}
$$
利用 $s^2$ 与 $Y_0- (\hat \alpha + \hat \beta x_0)$ 的独立性，我们有:  
$$
\frac{Y_0- (\hat \alpha + \hat \beta x_0)}{s\sqrt{1+\frac1n + \frac{(x_0-\bar x)^2}{S_{xx}}}} 
=
\frac{\frac{Y_0- (\hat \alpha + \hat \beta x_0)}{\sigma\sqrt{1+\frac1n + \frac{(x_0-\bar x)^2}{S_{xx}}}} }{\frac{s}{\sigma}}
\sim
\frac{\mathcal{N}(0,1)}{\sqrt{\frac1{n-2}\chi^2_{(n-2)}}}
=
t_{n-2}.
$$
于是我们得到个体响应 $Y_0\sim \mathcal{N}(\alpha+\beta x_0,\sigma^2)$ 的 $(1-\alpha)$ 预测区间:  
$$
Y_0 \in \left(\hat \alpha + \hat \beta x_0 \pm s\sqrt{1+\frac1n + \frac{(x_0-\bar x)^2}{S_{xx}}}t_{n-2,\frac{\alpha}{2}}\right)
$$
其中 $t_{n-2,\frac{\alpha}{2}}$ 为 $t_{n-2}$ 分布的 $1-\frac{\alpha}{2}$ (右尾) 分位数.



#### (3) 同时估计

在 $1.3.2(1)$ 中我们已经看到，与 $x_0$ 相联系的总体 $Y_0\sim \mathcal{N}(\alpha+\beta x_0,\sigma^2)$ 的均值，  
即平均响应 $\mu=\mathbb{E}(Y|X=x_0)=\alpha + \beta x_0$ 的 $(1-\alpha)$ 置信区间为:  
$$
\alpha + \beta x_0 \in (\hat \alpha + \hat \beta x_0 \pm s\sqrt{\frac1n + \frac{(x_0-\bar x)^2}{S_{xx}}}t_{n-2,\frac{\alpha}{2}})

\text{ where }
\begin{cases}
\hat \beta = \frac{S_{xy}}{S_{xx}}\\
\hat \alpha = \bar y - \hat \beta \bar x\\
s^2 = \frac{1}{n-2}(S_{yy}-\frac{S_{xy}^2}{S_{xx}})
\end{cases}
$$
现假定我们要在若干点处对于 $Y$ 总体的均值进行推断.  
具体来说，我们要求包含 $\mu_i = \mathbb{E}(Y|X=x_0^{(i)}) = \alpha + \beta x_0^{(i)}\ (i=1,\dots,m)$ 的区间.  
如果按上面的方法分别给出 $\mu_i\ (i=1,\dots,m)$ 的 $(1-\alpha)$ 置信区间，  
那么合并得到的区间的置信水平将不是 $1-\alpha$.

一个简单且有用的处理方式是应用 Bonferroni 不等式.

> 当事件交的概率难以 (甚至无法) 计算，而我们只需知道其大致范围时，Bonferroni 不等式非常有用.  
> **(Bonferroni 不等式, Statistical Inference 例 $1.2.10$)**  
> 设 $A,B$ 是两个事件，则我们有 $\text{P}(A\cap B)\geq P(A) + P(B)-1$.  
> 注意: 若 $A,B$ 发生概率不足够大，则 Bonferroni 下界可能是一个毫无用处 (尽管仍然正确) 的负数.

由 Bonferroni 不等式可知，我们有:
$$
\begin{align}
&\text{P}\left(\alpha + \beta x_0^{(i)}\in \left(\hat \alpha + \hat \beta x_0^{(i)}\pm s\sqrt{\frac1n + \frac{(x_0^{(i)}-\bar x)^2}{S_{xx}}}t_{n-2,\frac{\alpha}{2m}}\right)\text{ for all }i=1,\dots,m\right)\\
&\geq
\sum_{i=1}^m \text{P}\left(\alpha + \beta x_0^{(i)}\in \left(\hat \alpha + \hat \beta x_0^{(i)}\pm s\sqrt{\frac1n + \frac{(x_0^{(i)}-\bar x)^2}{S_{xx}}}t_{n-2,\frac{\alpha}{2m}}\right)\right) - (m-1)\\
&=
m\cdot (1-\frac{\alpha}{m}) - (m-1)\\
&=
1-\alpha
\end{align}
$$
换言之，我们至少以概率 $1-\alpha$ 有事件 $\alpha + \beta x_0^{(i)}\in (\hat \alpha + \hat \beta x_0^{(i)}\pm s\sqrt{\frac1n + \frac{(x_0^{(i)}-\bar x)^2}{S_{xx}}}t_{n-2,\frac{\alpha}{2m}})\ (\forall\ i=1,\dots,m)$ 发生.  
(结合 $1.3.2(2)$ 的内容，我们可以通过类似的方法给出 $Y_0^{(i)}\sim \mathcal{N}(\alpha + \beta x_0^{(i)},\sigma^2)\ (i=1,\dots,m)$ 的同时预测区间)

****

我们还可进一步对所有 $x$ 进行同时推断.  
**(Scheffe 置信带, Statistical Inference 定理 $11.3.6$)**    
在条件正态模型 $(Y|X=x) \sim \mathcal{N}(\alpha + \beta x,\sigma^2)$ 下，我们至少以概率 $1-\tau$ 对所有 $x$ 同时成立:  
$$
\alpha + \beta x \in \left(\hat \alpha + \hat \beta x \pm s\sqrt{\frac1n + \frac{(x-\bar x)^2}{S_{xx}}}M_\tau\right)\text{ where }M_\tau = \sqrt{2F_{2,n-2,\alpha}}.
$$

- **注解:**   
  由于上式至少以概率 $1-\tau $ 对所有 $x$ 成立  
  故它给出的是整条总体回归线 $\mathbb{E}(Y|X=x)=\alpha + \beta x$ 的一个 $(1-\tau)$ 置信带 (称为 **Scheffe 带**).  
  就像一个置信区间覆盖一个参数一样，这个置信带覆盖了整条总体回归线.

  下图给出了一个 Scheffe 带的例子，同时给出了两个 Bonferroni 区间和一个 $t$ 区间:

  <img src="Scheffe 带.png" style="zoom:30%;" />

  实际上 Bonferroni 区间有可能比 Scheffe 带更宽 (尽管图中没有给出这样的例子)  
  因此即使只对少数几个 $x$ 感兴趣，我们仍更倾向于选择 Scheffe 带而不是 Bonferroni 区间.  
  这是由于 Bonferroni 区间是通用的界 (而且实际覆盖率要比 $1-\tau$ 高)，而 Scheffe 带是问题的精确解.

  此外，理论上我们可以用类似方法得到针对所有 $x$ 的**同时预测区间**，但导出的统计量没有特别好的分布.


**证明:**  
上述问题等价于寻找一个常数 $M_\alpha$ 使得:  
$$
\text{P}\left(\frac{((\hat \alpha + \hat \beta x)-(\alpha + \beta x))^2}{s^2 (\frac1n + \frac{(x-\bar x)^2}{S_{xx}})}\leq M_\tau^2 \text{ for all }x\right) = 1-\tau,
$$
即等价于使得:  
$$
\text{P}\left(\max_{x}\frac{((\hat \alpha + \hat \beta x)-(\alpha + \beta x))^2}{s^2 (\frac1n + \frac{(x-\bar x)^2}{S_{xx}})}\leq M_\tau^2\right) = 1-\tau.
$$

> **(Statistical Inference 习题 $11.40$)**  
> 若 $a,b,c,d$ 为常数且 $c,d>0$，则我们有:   
> $$
> \max_t \frac{(a+bt)^2}{c+dt^2}=\frac{a^2}{c} + \frac{b^2}{d}.
> $$
> 这个结论是下面引理的直接推论.
>
> **Lemma: (广义 Raylaigh 商)**  
> 若 $b\in \mathbb R^n$ 为给定向量且 $A\in \mathbb R^{n\times n}$ 正定，则我们有:
> $$
> \underset{x\neq 0_n\in \mathbb R^n}\max \frac{(b^{\mathrm T}x)^2}{x^{\mathrm T}Ax}=b^{\mathrm T}A^{-1}b.
> $$
>
>  证明:    
> $$
> \begin{align}
> \max_{x\neq 0_n\in \mathbb R^n} \frac{(b^{\mathrm T}x)^2}{x^{\mathrm T}Ax}
> &=
> \max_{y\neq 0_n\in \mathbb R^n} \frac{(b^{\mathrm T}A^{-\frac12} y)^2}{y^{\mathrm T}y}\quad (y:= A^{\frac12}x)\\
> &=
> \max_{y \neq 0_n\in \mathbb R^n}\frac{y^{\mathrm T}(A^{-\frac12} bb^{\mathrm T}A^{-\frac12})y}{y^{\mathrm T}y}
> \quad (\text{Raylaigh theorem})\\
> &=
> \lambda_\max(A^{-\frac12} bb^{\mathrm T} A^{-\frac12})\qquad\ \ \  (\text{note that }A\text{ is positive definite, hence symmetric})\\
> &=
> \lambda_\max((A^{-\frac12} b)(A^{-\frac12} b)^{\mathrm T})\quad(\text{note that rank-one matrix }zz^{\mathrm T} \text{'s only non-zero eigenvalue is }\|z\|_2)\\
> &=
> \|A^{-\frac12}b\|_2^2\\
> &=
> b^{\mathrm T}A^{-1}b
> \end{align}
> $$
> 其中最大值可以在 $x=\alpha A^{-1}b\ (\forall\ \alpha\neq 0\in \mathbb R)$ 取到.

$$
\begin{align}
\max_{x}\frac{((\hat \alpha + \hat \beta x)-(\alpha + \beta x))^2}{s^2 (\frac1n + \frac{(x-\bar x)^2}{S_{xx}})}
&=
\frac1{s^2} \max_x 
\frac{((\bar y - \hat \beta \bar x + \hat \beta x) - (\alpha + \beta \bar x- \beta \bar x + \beta x))^2}
{\frac1n + \frac{(x-\bar x)^2}{S_{xx}}}\\
&=
\frac1{s^2} \max_x 
\frac{((\bar y - \alpha - \beta \bar x) +(\hat \beta - \beta)(x-\bar x))^2}{\frac1n + \frac{1}{S_{xx}}(x-\bar x)^2}\quad (\text{denote }t=x-\bar x)\\
&=
\frac1{s^2} \max_t 
\frac{((\bar y - \alpha - \beta \bar x) +(\hat \beta - \beta)t)^2}{\frac1n + \frac{1}{S_{xx}}t^2}
\quad(\text{use lemma})\\
&=
\frac1{s^2} \left(\frac{(\bar y - \alpha -\beta \bar x)^2}{\frac1n} + \frac{(\hat \beta - \beta)^2}{\frac{1}{S_{xx}}}\right)\\
&=
\frac{\frac{(\bar y - \alpha -\beta \bar x)^2}{\sigma^2/n} + \frac{(\hat \beta - \beta)^2}{\sigma^2/S_{xx}}}
{s^2/\sigma^2}\\
&(\text{note that }
\begin{cases}
\bar y \sim \mathcal{N}(\alpha + \beta \bar x , \frac{\sigma^2}{n})\\
\hat \beta \sim \mathcal{N}(\beta,\frac{\sigma^2}{S_{xx}})\\
s^2 \sim \frac{1}{n-2}\sigma^2 \chi^2_{(n-2)}
\end{cases}
\text{ and they are independent})\\

&\sim
\frac{\chi_{(2)}^2}{\chi^2_{(n-2)}/(n-2)}\\
&=
2\frac{\chi_{(2)}^2/2}{\chi^2_{(n-2)}/(n-2)}\\
&=
2F_{2,n-2}
\end{align}
$$

因此 $M_\tau^2 = 2F_{2,n-2,\tau}$，即 $M_\tau = \sqrt{2F_{2,n-2,\tau}}$.   
命题得证.

**The End**

