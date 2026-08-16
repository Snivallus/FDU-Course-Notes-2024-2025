# FDU 回归分析 5. 方差分析

本文根据王勤文老师课堂笔记整理而成，欢迎批评指正!

## 5.1 An Introduction

在多元线性回归模型 $y=X\beta+\varepsilon$ 中，  
无论是响应变量 $Y$ 还是解释变量 $X_1,X_2,\dots,X_k$，它们都是连续变量.  
但在实际问题的研究中，我们会遇到离散变量，也称为定性变量，  
例如性别、职业、颜色等等，此时多元线性回归模型就不适用了.

Logistic 回归所解决的二分类问题中，响应变量 $Y$ 为 $\text{0-1}$ 型离散变量，而解释变量均为连续变量.  
这里我们考虑另一类问题:  
响应变量 $Y$ 为连续变量，而解释变量 $X_1,\dots,X_k$ 为离散变量的问题，  
例如研究 "调查问卷纸张的颜色" 对其 "回收率" 的影响.  
再例如 "定价" 和 "包装" 对商品 "销售量" 的影响.

处理这类问题通常采用方差分析方法，  
只考虑单个解释变量的方差分析称为**单因子方差分析** (Single\-Factor ANOVA)  
同时考虑多个解释变量的方差分析称为**多因子方差分析** (Multi\-Factor ANOVA)  
本课程里，我们只研究到**双因子方差分析** (Two\-Factor ANOVA)

所谓 **"因子"** (Factor)，即离散的解释变量  
所谓 **"因子水平"** (Factor Level)，即这个因子可能的取值  
在单因子方差分析中，我们通常记模型中唯一的因子为 $A$，记其 $a$ 个因子水平为 $A_1,A_2,\dots,A_a$   
设在这 $a$ 个因子水平下分别有 $n_1,n_2,\dots,n_a$ 个观测，记**观测总数** $n := \sum_{i=1}^a n_i$   
我们将这些观测记为:
$$
\begin{matrix} 
\text{Factor Level:} & A_1 & \dots & A_i & \dots & A_n\\ 
\text{Observation:} &  
\begin{matrix}y_{1,1}\\ \vdots\\ y_{1,n_1} \end{matrix} 
& \dots  
& \begin{matrix}y_{i,1}\\ \vdots\\ y_{i,n_i} \end{matrix} 
& \dots & \begin{matrix}y_{a,1}\\ \vdots\\ y_{a,n_a} \end{matrix} 
\end{matrix}
$$
其中 $y_{ij}$ 代表因子水平 $A_i$ 下的第 $j$ 个观测.  
我们记 $y_{ij} = \mu_i + \varepsilon_{ij}$   
其中 $\mu_i$ 为因子水平 $A_i$ 下观测的**理论均值**  
**扰动项** $\varepsilon_{ij}$ 独立于下标 $(i,j)$，服从某个零均值的分布 $\text{Distribution}(0,\sigma^2)$  
我们通常做正态假设: $\{\varepsilon_{ij}\} \overset{\text{i.i.d.}}\sim N(0,\sigma^2)$  
于是上述观测可表示成缺少截距项的线性回归模型:
$$
\begin{align}
\begin{bmatrix}
y^{(1)}\\
y^{(2)}\\
\vdots\\
y^{(a)}
\end{bmatrix}
&=
\begin{bmatrix} 
\begin{bmatrix}y_{1,1}\\ \vdots\\ y_{1,n_1}\end{bmatrix}\\ 
\begin{bmatrix}y_{2,1}\\ \vdots\\ y_{2,n_2}\end{bmatrix}\\ 
\vdots\\ 
\begin{bmatrix}y_{a,1}\\ \vdots\\ y_{a,n_a}\end{bmatrix}\\ 
\end{bmatrix}
= 
\begin{bmatrix} \mu_11_{n_1}\\ \mu_21_{n_2}\\ \vdots\\ \mu_a 1_{n_a} \end{bmatrix}
+ 
\begin{bmatrix} 
\begin{bmatrix}\varepsilon_{1,1}\\ \vdots\\ \varepsilon_{1,n_1}\end{bmatrix}\\ \begin{bmatrix}\varepsilon_{2,1}\\ \vdots\\ \varepsilon_{2,n_2}\end{bmatrix}\\ 
\vdots\\ 
\begin{bmatrix}\varepsilon_{a,1}\\ \vdots\\ \varepsilon_{a,n_a}\end{bmatrix}\\ 
\end{bmatrix}
= 
\begin{bmatrix} 1_{n_1} &&&\\  &1_{n_2}&&\\ &&\ddots&\\ &&&1_{n_a}\\ \end{bmatrix}_{n\times a} 
\begin{bmatrix}\mu_1 \\ \mu_2 \\\vdots\\ \mu_a\end{bmatrix}
+
\begin{bmatrix}
\varepsilon^{(1)}\\
\varepsilon^{(2)}\\
\vdots\\
\varepsilon^{(a)}
\end{bmatrix}


\end{align}
$$
进一步，我们记:   
$$
\begin{cases} 
y_{i\cdot} = \sum_{j=1}^{n_i} y_{ij}\\ 
\bar y_{i\cdot} = \frac{1}{n_i} y_{i\cdot} = \frac{1}{n_i}\sum_{j=1}^{n_i}y_{ij} \\ 
y_{\cdot\cdot} = \sum_{i=1}^a y_{i\cdot} = 
\sum_{i=1}^a \sum_{j=1}^{n_i} y_{ij}\\ 
\bar y_{\cdot\cdot} = \frac{1}{n} y_{\cdot\cdot} = \frac1n \sum_{i=1}^a \sum_{j=1}^{n_i} y_{ij} \end{cases}
$$
其中 $y_{i\cdot}$ 称为**因子水平 $A_i$ 下的观测和**，$\bar y_{i\cdot}$ 称为**因子水平 $A_i$ 下的样本均值**  
而 $y_{\cdot\cdot}$ 称为因子 $A$ 的**总观测和**，$\bar y_{\cdot\cdot}$ 称为因子 $A$ 的**总样本均值**

类似地记:   
$$
\begin{cases} 
\varepsilon_{i\cdot} = \sum_{j=1}^{n_i} \varepsilon_{ij} \\ 
\bar \varepsilon_{i\cdot} =  \frac{1}{n_i} \varepsilon_{i\cdot} = \frac{1}{n_i}\sum_{j=1}^{n_i} \varepsilon_{ij}\\ 
\varepsilon_{\cdot\cdot}  = \sum_{i=1}^a \sum_{j=1}^{n_i} \varepsilon_{ij}\\ 
\bar \varepsilon_{\cdot\cdot} = \frac{1}{n} \varepsilon_{\cdot\cdot} 
=
\frac1n \sum_{i=1}^a \sum_{j=1}^{n_i} \varepsilon_{ij}\end{cases}
$$
根据 $y_{ij} = \mu_i + \varepsilon_{ij}$ 可得:
$$
\begin{cases} 
y_{i\cdot}= n_i\mu_i + \varepsilon_{i\cdot}\\ 
\bar y_{i\cdot} = \mu_i + \bar\varepsilon_{i\cdot}\\ 
y_{\cdot\cdot} = \sum_{i=1}^a n_i\mu_i + \varepsilon_{\cdot\cdot}\\ 
\bar y_{\cdot\cdot} =  \sum_{i=1}^a \frac{n_i}n\mu_i + \bar \varepsilon_{\cdot\cdot}
\end{cases}
$$


## 5.2 单因子方差分析

### 5.2.1 点估计

注意到因子水平 $A_i$ 下的观测 $y_{i,1},y_{i,2},\dots,y_{i,n_i}$ 相互独立，且同分布于 $N(\mu_i,\sigma^2)$    
考虑优化问题:  
$$
\min_{\mu \in \mathbb R}\|y^{(i)} - \mu 1_{n_i}\|^2:= \sum_{j=1}^{n_i} (y_{ij}-\mu)^2
$$
令 $\nabla_\mu \|y^{(i)}-\mu 1_{n_i}\|^2 = -2\cdot 1_{n_i}^{\mathrm T}(y^{(i)}-\mu 1_{n_i}) = 0_{n_i}$ 可知 $\mu_i$ 的最小二乘估计量为:  
$$
\hat \mu_i = (1_{n_i}^{\mathrm T}1_{n_i})^{-1} 1_{n_i}^{\mathrm T} y^{(i)} = \frac{1}{n_i}\sum_{j=1}^{n_i}y_{ij} = \bar y_{i\cdot}
$$
根据多元线性回归的 **Gauss-Markov 定理**可知 $\bar y_{i\cdot}$ 是 $\mu_i$ 的最佳线性无偏估计量 $(\text{BLUE})$  
它服从正态分布:  
$$
\bar y_{i\cdot} = \frac{1}{n_i}\sum_{j=1}^{n_i}y_{ij} \sim N(\mu_i,\frac{\sigma^2}{n_i})
$$

> **(多元线性回归中的 Gauss-Markov 定理)**  
> 给定数据点 $(x^{(1)},y_1),\dots,(x^{(n)},y_n)$ 和多元线性回归模型 $Y= \beta^{\mathrm T}x + \varepsilon$   
> (其中 $\beta\in \mathbb R^{p+1}$ 为参数向量，$\varepsilon$ 为随机噪音)    
> 记样本关系为 $y=X\beta + \varepsilon$   
> (其中 $X=[x^{(1)},\dots,x^{(n)}]^{\mathrm T}\in \mathbb R^{n\times (p+1)}$ 为设计矩阵，$\varepsilon\in \mathbb R^n$ 为随机误差构成的列向量)
>
> 若 $\begin{cases}
> \text{E}[\varepsilon] = 0_n\\
> \text{Var}[\varepsilon] = \sigma^2 I_n
> \end{cases}$ (零均值和互不相关, 无需给出分布上的假设)，  
> 则最小二乘估计量 $\hat\beta = (X^{\mathrm T}X)^{-1}X^{\mathrm T}y$ 是参数向量 $\beta$ 的最佳线性无偏估计量 (Best Linear Unbiased Estimator, BLUE)  
> $$
> \text{E}[\hat \beta] = \beta\\
> \text{Cov}[\hat \beta] = \sigma^2 (X^{\mathrm T}X)^{-1}
> $$
>

*****

考虑单因子方差分析的多元线性回归形式:
$$
\begin{align}
\begin{bmatrix}
y^{(1)}\\
y^{(2)}\\
\vdots\\
y^{(a)}
\end{bmatrix}
= 
\begin{bmatrix} 1_{n_1} &&&\\  &1_{n_2}&&\\ &&\ddots&\\ &&&1_{n_a}\\ \end{bmatrix}_{n\times a} 
\begin{bmatrix}\mu_1 \\ \mu_2 \\\vdots\\ \mu_a\end{bmatrix}
+
\begin{bmatrix}
\varepsilon^{(1)}\\
\varepsilon^{(2)}\\
\vdots\\
\varepsilon^{(a)}
\end{bmatrix}
\end{align}
$$
其中:  
$$
y^{(i)} = 
\begin{bmatrix}
y_{i,1}\\
y_{i,2}\\
\vdots\\
y_{i,n_i}
\end{bmatrix}
\quad
\varepsilon^{(i)}
=
\begin{bmatrix}
\varepsilon_{i,1}\\
\varepsilon_{i,2}\\
\vdots\\
\varepsilon_{i,n_i}
\end{bmatrix}\quad (i=1,\dots,a)\\
n=n_1+n_2+\dotsm + n_a\\
\{\varepsilon_{i,j}\} \overset{i.i.d}\sim N(0,\sigma^2)
$$
根据多元线性回归的结论可知 $\sigma^2$ 的无偏估计量为:  
$$
\begin{align}
s^2 
&=
\frac{1}{n-a}
\left\|
\begin{bmatrix}
y^{(1)}\\
y^{(2)}\\
\vdots\\
y^{(a)}
\end{bmatrix}
-
\begin{bmatrix} 1_{n_1} &&&\\  &1_{n_2}&&\\ &&\ddots&\\ &&&1_{n_a}\\ \end{bmatrix}_{n\times a} 
\begin{bmatrix}\bar y_{1\cdot} \\ \bar y_{2\cdot} \\\vdots\\ \bar y_{a\cdot}\end{bmatrix}
\right\|^2\\
&=
\frac1{n-a} 
\left\|
\begin{bmatrix}
y^{(1)} - \bar y_{1\cdot} 1_{n_1}\\
y^{(2)} - \bar y_{2\cdot} 1_{n_2}\\
\vdots\\
y^{(a)} - \bar y_{a\cdot} 1_{n_a}
\end{bmatrix}
\right\|^2\\

&=
\frac1{n-a} \sum_{i=1}^a \|y^{(i)}-\bar y_{i\cdot}1_{n_i}\|^2\\
&=
\frac{1}{n-a}\sum_{i=1}^a (n_i-1)s_i^2
\end{align}
$$
其中 $s_i^2 = \frac{1}{n_i-1}\|y^{(i)}-\bar y_{i\cdot}1_{n_i}\|^2 = \frac{1}{n_i-1}\sum_{j=1}^{n_i}(y_{ij}-\bar y_{i\cdot})^2$ 为因子水平 $A_i$ 下的样本方差.

****

根据多元线性回归的结论可知:

- ① $\bar y_{i\cdot} = \frac{1}{n_i}\sum_{j=1}^{n_i}y_{ij} \sim N(\mu_i,\frac{\sigma^2}{n_i})$ 
- ② $s_i^2 = \frac{1}{n_i-1}\sum_{j=1}^{n_i}(y_{ij}-\bar y_{i\cdot})^2 \sim \frac{1}{n_i-1}\sigma^2\chi^2_{(n_i-1)}$ 且 $\bar y_{i\cdot}\perp s_i^2$   
  (事实上对于任意 $i,j=1,\dots,a$ 都有 $\bar y_{i\cdot}\perp s_j^2$ 成立)
- ③ $s^2 = \frac{1}{n-a}\sum_{i=1}^a (n_i-1)s_i^2\sim \frac{1}{n-a} \sigma^2\chi^2_{(n-a)}$ 且 $s^2\perp \bar y_{i\cdot}\ (\forall\ i=1,\dots,a)$ 



### 5.2.2 置信区间估计

给定置信水平为 $\alpha$  

#### (1) 理论均值

因子水平 $A_i$ 下的理论均值 $\mu_i$ 的 $t$ 检验统计量为:
$$
\begin{align}
\frac{\sqrt{n_i}(\bar y_{i\cdot}-\mu_i)}{s}
&=
\frac{\frac{\sqrt{n_i}(\bar y_{i\cdot}-\mu_i)}{\sigma}}{\frac{s}{\sigma}}
\quad\ (\text{note that }
\begin{cases}
\bar y_{i\cdot} \sim N(\mu_i,\frac{\sigma^2}{n_i})\\
s^2 \sim \frac{1}{n-a}\sigma^2\chi^2_{(n-a)}
\end{cases})\\
&\sim
\frac{N(0,1)}{\frac{1}{n-a}\chi^2_{(n-a)}}\quad(\text{note that }
s^2\perp \bar y_{i\cdot})\\
&=
t_{n-a}
\end{align}
$$
因此 $\mu_i$ 的 $(1-\alpha)\cdot 100\%$ 置信区间为 $\bar y_{i\cdot} \pm \frac{s}{\sqrt{n_i}}t_{n-a}(\frac{\alpha}{2})$  
其中 $t_{n-a}(\frac{\alpha}{2})$ 为 $t_{n-a}$ 分布的 $1-\frac{\alpha}{2}$ 分位数.



#### (2) 组间偏差

记因子水平 $A_i$ 和 $A_j$ 的理论偏差为 $d_{ij} := \mu_i - \mu_j$    
显然其无偏估计量为 $\hat d_{ij}:= \bar y_{i\cdot}-\bar y_{j\cdot}$   
注意到:  
$$
\begin{cases} 
\bar y_{i\cdot} \sim N(\mu_i,\frac{\sigma^2}{n_i})\\ 
\bar y_{j\cdot} \sim N(\mu_j,\frac{\sigma^2}{n_j})\\ 
\bar y_{i\cdot} \perp \bar y_{j\cdot} \end{cases}
\quad\Rightarrow\quad
\begin{bmatrix}
\bar y_{i\cdot}\\
\bar y_{j\cdot}
\end{bmatrix}
\sim N
\left(
\begin{bmatrix}
\mu_i\\
\mu_j
\end{bmatrix},
\sigma^2
\begin{bmatrix}
\frac{1}{n_i} & \\
& \frac{1}{n_j}
\end{bmatrix}
\right)
$$
因此我们有:  
$$
\begin{align}
\hat d_{ij}
&=
\bar y_{i\cdot}-\bar y_{j\cdot}\\
&=
\begin{bmatrix}
1\\
-1
\end{bmatrix}^{\mathrm T}
\begin{bmatrix}
\bar y_{i\cdot}\\
\bar y_{j\cdot}
\end{bmatrix}\\
&\sim 
N
\left(
\begin{bmatrix}
1\\
-1
\end{bmatrix}^{\mathrm T}
\begin{bmatrix}
\mu_i\\
\mu_j
\end{bmatrix},
\sigma^2
\begin{bmatrix}
1\\
-1
\end{bmatrix}^{\mathrm T}
\begin{bmatrix}
\frac{1}{n_i} & \\
& \frac{1}{n_j}
\end{bmatrix}
\begin{bmatrix}
1\\
-1
\end{bmatrix}
\right)\\
&=
N\left(\mu_i - \mu_j , \sigma^2 \left(\frac1{n_i}+\frac{1}{n_j}\right)\right)
\end{align}
$$
标准化后得到:  
$$
\frac{\hat d_{ij}-d_{ij}}{\sigma \sqrt{\frac{1}{n_i}+\frac{1}{n_j}}} \sim N(0,1)
$$
因此理论偏差 $d_{ij} = \mu_i - \mu_j$ 的 $t$ 检验统计量为:  
$$
\begin{align}
\frac{\hat d_{ij}-d_{ij}}{s\sqrt{\frac{1}{n_i}+\frac{1}{n_j}}}
&=
\frac{\frac{\hat d_{ij}-d_{ij}}{\sigma \sqrt{\frac{1}{n_i}+\frac{1}{n_j}}}}{\frac{s}{\sigma}}
\quad\ \ (\text{note that }
\begin{cases}
\frac{\hat d_{ij}-d_{ij}}{\sigma \sqrt{\frac{1}{n_i}+\frac{1}{n_j}}} \sim N(0,1)\\
s^2 \sim \frac{1}{n-a}\sigma^2\chi^2_{(n-a)}
\end{cases})\\
&\sim
\frac{N(0,1)}{\frac{1}{n-a}\chi^2_{(n-a)}}\quad (\text{note that }
s^2\perp \bar y_{i\cdot}\text{ for all }i=1,\dots,a\text{ so that }
s^2\perp \hat d_{ij}:= \bar y_{i\cdot}-\bar y_{j\cdot})\\
&=
t_{n-a}
\end{align}
$$
因此 $d_{ij} = \mu_i - \mu_j$ 的 $(1-\alpha)\cdot 100\%$ 置信区间为 $\hat d_{ij} \pm s\sqrt{\frac{1}{n_i} + \frac{1}{n_j}} t_{n-a}(\frac{\alpha}{2})$  
其中 $t_{n-a}(\frac{\alpha}{2})$ 为 $t_{n-a}$ 分布的 $1-\frac{\alpha}{2}$ 分位数.



#### (3) 对比度

给定系数 $c_1,\dots,c_a$ 满足 $\sum_{i=1}^a c_i=0$   
定义对比度 $L:= \sum_{i=1}^a c_i \mu_i = c^{\mathrm T}\mu$   
其中 $\mu:= [\mu_1,\dots,\mu_a]^{\mathrm T}$    
显然其无偏估计量为:  
$$
\begin{align}
\hat L
&:=
\sum_{i=1}^a c_i \bar y_{i\cdot}\\
&=
\begin{bmatrix}
c_1\\
\vdots\\
c_a
\end{bmatrix}^{\mathrm T}
\begin{bmatrix}
\bar y_{1\cdot}\\
\vdots\\
\bar y_{a\cdot}
\end{bmatrix}\quad (\text{note that }
\begin{bmatrix}
\bar y_{1\cdot}\\
\vdots\\
\bar y_{a\cdot}
\end{bmatrix}
\sim
N\left(
\begin{bmatrix}
\mu_1\\
\vdots\\
\mu_a
\end{bmatrix},
\sigma^2\begin{bmatrix}
\frac{1}{n_1}\\
& \ddots & \\
& & \frac{1}{n_a}
\end{bmatrix}
\right)
)\\
&=
N\left(
\begin{bmatrix}
c_1\\
\vdots\\
c_a
\end{bmatrix}^{\mathrm T}
\begin{bmatrix}
\mu_1\\
\vdots\\
\mu_a
\end{bmatrix},
\sigma^2
\begin{bmatrix}
c_1\\
\vdots\\
c_a
\end{bmatrix}^{\mathrm T}
\begin{bmatrix}
\frac{1}{n_1}\\
& \ddots & \\
& & \frac{1}{n_a}
\end{bmatrix}
\begin{bmatrix}
c_1\\
\vdots\\
c_a
\end{bmatrix}
\right)\\
&=
N
\left(
\sum_{i=1}^a c_i \mu_i,
\sigma^2 \sum_{i=1}^a 
\frac{1}{n_i} c_i^2
\right)
\end{align}
$$
标准化后得到: 
$$
\frac{\hat L - L}{\sigma \sqrt{\sum_{i=1}^a \frac{c_i^2}{n_i}}} \sim N(0,1)
$$
因此对比度 $L= \sum_{i=1}^a c_i \mu_i = c^{\mathrm T}\mu$ 的 $t$ 检验统计量为:  
$$
\begin{align}
\frac{\hat L - L}{s \sqrt{\sum_{i=1}^a \frac{c_i^2}{n_i}}}
&=
\frac{\frac{\hat L - L}{\sigma \sqrt{\sum_{i=1}^a \frac{c_i^2}{n_i}}}}
{\frac{s}{\sigma}}
\quad\ (\text{note that }
\begin{cases}
\frac{\hat L - L}{\sigma \sqrt{\sum_{i=1}^a \frac{c_i^2}{n_i}}} \sim N(0,1)\\
s^2 \sim \frac{1}{n-a}\sigma^2\chi^2_{(n-a)}
\end{cases})\\
&\sim
\frac{N(0,1)}{\frac{1}{n-a}\chi^2_{(n-a)}}\quad (\text{note that }
s^2\perp \bar y_{i\cdot}\text{ for all }i=1,\dots,a\text{ so that }
s^2\perp \hat L := \sum_{i=1}^a c_i \bar y_{i\cdot}
)\\
&=
t_{n-a}
\end{align}
$$
因此 $L= \sum_{i=1}^a c_i \mu_i = c^{\mathrm T}\mu$ 的 $(1-\alpha)\cdot 100\%$ 置信区间为 $\hat L \pm s\sqrt{\sum_{i=1}^a \frac{c_i^2}{n_i}} t_{n-a}(\frac{\alpha}{2})$  
其中 $t_{n-a}(\frac{\alpha}{2})$ 为 $t_{n-a}$ 分布的 $1-\frac{\alpha}{2}$ 分位数.



### 5.2.3 主效应检验

考虑单因子方差分析的多元线性回归形式:
$$
y
=\begin{align}
\begin{bmatrix}
y^{(1)}\\
y^{(2)}\\
\vdots\\
y^{(a)}
\end{bmatrix}
= 
\begin{bmatrix} 1_{n_1} &&&\\  &1_{n_2}&&\\ &&\ddots&\\ &&&1_{n_a}\\ \end{bmatrix}_{n\times a} 
\begin{bmatrix}\mu_1 \\ \mu_2 \\\vdots\\ \mu_a\end{bmatrix}
+
\begin{bmatrix}
\varepsilon^{(1)}\\
\varepsilon^{(2)}\\
\vdots\\
\varepsilon^{(a)}
\end{bmatrix}=X\mu + \varepsilon
\end{align}
$$
其中:  
$$
y^{(i)} = 
\begin{bmatrix}
y_{i,1}\\
y_{i,2}\\
\vdots\\
y_{i,n_i}
\end{bmatrix}
\quad
\varepsilon^{(i)}
=
\begin{bmatrix}
\varepsilon_{i,1}\\
\varepsilon_{i,2}\\
\vdots\\
\varepsilon_{i,n_i}
\end{bmatrix}\quad (i=1,\dots,a)\\
n=n_1+n_2+\dotsm + n_a\\
\{\varepsilon_{i,j}\} \overset{\text{i.i.d.}}\sim N(0,\sigma^2)
$$
设计矩阵 $X = 1_{n_1}\oplus \dotsm \oplus 1_{n_a}\in \mathbb R^{n\times a}$   
我们定义投影矩阵为:  
$$
\begin{align}
H 
&:= X(X^{\mathrm T}X)^{-1}X^{\mathrm T}\\
&=
\begin{bmatrix} 1_{n_1} &&&\\  &1_{n_2}&&\\ &&\ddots&\\ &&&1_{n_a}\\ \end{bmatrix}
\begin{bmatrix}
n_1\\
& n_2\\
& & \ddots & \\
& & & n_a
\end{bmatrix}^{-1}
\begin{bmatrix} 1_{n_1} &&&\\  &1_{n_2}&&\\ &&\ddots&\\ &&&1_{n_a}\\ \end{bmatrix}^{\mathrm T}\\
&=
\begin{bmatrix}
\frac{1}{n_1}1_{n_1}1_{n_1}^{\mathrm T}\\
& \frac{1}{n_2}1_{n_2}1_{n_2}^{\mathrm T}\\
& & \ddots\\
& & & \frac{1}{n_a} 1_{n_a}1_{n_a}^{\mathrm T}
\end{bmatrix}
\end{align}
$$
我们想知道在不同因子水平 $A_1,A_2,\dots,A_a$ 的理论均值是否完全一致.  
因此零假设和备择假设分别为: 
$$
H_0: \mu_1 = \mu_2 =\dots = \mu_a \\ 
\Updownarrow\\
H_1: \exists\ i,j \text{ such that }\mu_i \neq \mu_j
$$

#### (1) 基本记号

根据多元线性回归的结论我们有:
$$
\begin{align}
\text{SST}
&=
\|y-\bar y_{\cdot\cdot}1_n\|^2\\
&=\sum_{i=1}^a \sum_{j=1}^{n_i} (y_{ij}-\bar y_{\cdot\cdot})^2\\
\hline
\text{SSE}
&=
\|y-\hat y\|^2\\
&=
\|y-Hy\|^2\\
&=
\left\|
\begin{bmatrix}
y^{(1)}\\
\vdots\\
y^{(a)}
\end{bmatrix}
-
\begin{bmatrix}
\frac1{n_1} 1_{n_1}1_{n_1}^{\mathrm T}\\
&\ddots\\
& & \frac{1}{n_a} 1_{n_a}1_{n_a}^{\mathrm T}
\end{bmatrix}
\begin{bmatrix}
y^{(1)}\\
\vdots\\
y^{(a)}
\end{bmatrix}
\right\|^2\\
&=
\left\|
\begin{bmatrix}
y^{(1)} - \frac{1}{n_1}1_{n_1}1_{n_1}y^{(1)}\\
\vdots\\
y^{(a)} - \frac{1}{n_a} 1_{n_a} 1_{n_a} y^{(a)}
\end{bmatrix}
\right\|^2\\
&=
\left\|
\begin{bmatrix}
y^{(1)} - \bar y_{1\cdot}1_{n_1}\\
\vdots\\
y^{(a)} - \bar y_{a\cdot}1_{n_a}
\end{bmatrix}
\right\|^2\\
&=
\sum_{i=1}^a \|y^{(i)} - \bar y_{i\cdot}1_{n_i}\|^2\\
&=
\sum_{i=1}^a \sum_{j=1}^{n_i} (y_{ij}-\bar y_{i\cdot})^2\\
\hline

\text{SSR}
&=
\|\hat y - \bar y_{\cdot\cdot}1_n\|^2\\
&=
\|Hy - \bar y_{\cdot\cdot}1_n\|^2\quad (\text{use conclusion above})\\
&=
\left\|
\begin{bmatrix}
\bar y_{1\cdot}1_{n_1} - \bar y_{\cdot\cdot}1_{n_1}\\
\vdots\\
\bar y_{a\cdot}1_{n_a} - \bar y_{\cdot\cdot}1_{n_a}
\end{bmatrix}
\right\|^2\\
&=
\sum_{i=1}^a \|(\bar y_{i\cdot}-\bar y_{\cdot\cdot})1_{n_i}\|^2\\
&=
\sum_{i=1}^a n_i (\bar y_{i\cdot}-\bar y_{\cdot\cdot})^2

\end{align}
$$
而且它们的分布满足:
$$
\begin{cases}
H_0:\mu_1=\dotsm = \mu_a\\
\hline
\text{SST} = \|y-\bar y_{\cdot\cdot}1_n\|^2 = \sum_{i=1}^a \sum_{j=1}^{n_i} (y_{ij}-\bar y_{\cdot\cdot})^2
\overset{H_0}\sim \sigma^2\chi^2_{(n-1)}
&\quad \text{MST} = \frac{\text{SST}}{\text{df}_{\text{T}}} \overset{H_0}\sim \sigma^2 \frac{\chi^2_{(n-1)}}{n-1}\\

\text{SSR} = \|\hat y - \bar y_{\cdot\cdot} 1_n\|^2 = 
\sum_{i=1}^a n_i(\bar y_{i\cdot}-\bar y_{\cdot\cdot})^2
\overset{H_0}\sim \sigma^2 \chi^2_{(a-1)}
&\quad
\text{MSR} = \frac{\text{SSR}}{\text{df}_{\text{R}}}\overset{H_0}\sim \sigma^2 \frac{\chi^2_{(a-1)}}{a-1}\\

\text{SSE} = \|y-\hat y\|^2 = \sum_{i=1}^a \sum_{j=1}^{n_i}(y_{ij}-\bar y_{i\cdot})^2 \sim \sigma^2 \chi^2_{(n-a)}
&\quad
\text{MSE} = \frac{\text{SSE}}{\text{df}_\text{E}} \sim \sigma^2 \frac{\chi^2_{(n-a)}}{n-a}\\

\begin{cases}
\text{SST} = \text{SSR} + \text{SSE}\\
\text{df}_{\text{T}} = \text{df}_{\text{R}} + \text{df}_{\text{E}}\\
\text{SSR}\perp\text{SSE}
\end{cases}
\end{cases}
$$


#### (2) 主效应

注意到:  
$$
\begin{align}
\text{SSR}
&=
\|\hat y - \bar y_{\cdot\cdot} 1_n\|^2\\
&=
\sum_{i=1}^a n_i (\bar y_{i\cdot}-\bar y_{\cdot\cdot})^2\\
&=
\sum_{i=1}^a n_i \left(
(\mu_i + \bar \varepsilon_{i\cdot}) - 
\left(
\frac{1}{n}\sum_{k=1}^a n_k \mu_k + \bar \varepsilon_{\cdot\cdot}
\right)
\right)^2\\
&=
\sum_{i=1}^a n_i \left(
\mu_i - \frac{1}{n}\sum_{k=1}^a n_k \mu_k 
+ \bar \varepsilon_{i\cdot} - \bar \varepsilon_{\cdot\cdot}
\right)^2\\

&=
\sum_{i=1}^a n_i (\text{ME}_{A_i} + \varepsilon_{i\cdot}-\bar \varepsilon_{\cdot\cdot})^2
\end{align}
$$
其中 $\text{ME}_{A_i} = \mu_i - \frac1n \sum_{k=1}^a n_k \mu_k= \mu_i-\mu_\cdot$ 称为因子水平 $A_i$ 的**主效应** (main effect).    
于是我们可以将零假设和备择假设等价表示为:
$$
H_0 : \text{ME}_{A_1} = \text{ME}_{A_2} = \dots = \text{ME}_{A_a} = 0\\
\Updownarrow\\
H_1: \exists\ i\text{ such that } \text{ME}_{A_i} \neq 0
$$
这就和多元线性回归的**回归方程显著性检验**之间建立了联系.



#### (3) 检验法

设第一类型错误概率界限为 $\alpha$.  
多元线性回归的**回归方程显著性检验**指导我们使用如下的 $F$ 统计量:
$$
F:= \frac{\text{MSR}}{\text{MSE}} \overset{H_0}\sim \frac{\sigma^2 \chi^2_{(a-1)}/(a-1)}
{\sigma^2 \chi^2_{(n-a)}/(n-a)} = F_{a-1,n-a},
$$
其中分子 $\text{MSR}$ 和分母 $\text{MSE}$ 是相互独立的 (无论零假设 $H_0$ 是否成立).

记 $F_{a-1,n-a}(\alpha)$ 为 $F_{a-1,n-a}$ 分布的 $1-\alpha$ 分位数.  
主效应检验的 $F$-检验法为:

- 若 $F=\frac{\text{MSR}}{\text{MSE}}= \frac{\sum_{i=1}^a n_i (\bar y_{i\cdot}-\bar y_{\cdot\cdot})^2 / (a-1)}{\sum_{i=1}^{a}\sum_{j=1}^{n_i} (y_{ij}-\bar y_{i\cdot})^2/(n-a)} > F_{a-1,n-a}(\alpha)$，  
  则我们拒绝零假设 $H_0 : \text{ME}_{A_1} = \text{ME}_{A_2} = \dots = \text{ME}_{A_a} = 0$，  
  即拒绝零假设 $H_0: \mu_1 = \mu_2 =\dots = \mu_a$.  
  说明当前所研究的因子 $A$ 对响应变量 $Y$ 有解释作用.



## 5.3 双因子方差分析

### 5.3.1 模型假设

在双因子方差分析中，我们记模型中唯二的因子为 $A,B$  
记 $A$ 的 $a$ 个因子水平为 $A_1,A_2,\dots,A_a$   
记 $B$ 的 $b$ 个因子水平为 $B_1,B_2,\dots,B_b$   
我们称 $A,B$ 因子水平的组合 $(A_i,B_j)$ 为一个 $\text{treatment}$    
易知模型共计 $ab$ 个 $\text{treatment}$:
$$
\begin{array}{|c|c|c|c|} 
\hline A/B & B_1 & \dotsm&  B_j & \dotsm & B_b \\ 
\hline A_1 & \mu_{11} & & &  & \mu_{1b} \\
\hline \vdots\\
\hline A_i &  & &\mu_{ij} & &  \\
\hline \vdots\\
\hline A_a & \mu_{a1} &  & & & \mu_{ab} \\ 
\hline \end{array}
$$
其中 $(i,j)$ 位置上的单元格对应的就是 $\text{treatment}\ (A_i,B_j)$   
我们设每个 $\text{treatment}$ 下都有 $n$ 个观测  
设 $\text{treatment}\ (A_i,B_j)$ 下的 $n$ 个观测为 $y_{ij1},\dots,y_{ijn}$   
其中 $y_{ijk}$ 的下标 $(i,j,k)$ 表明 $y_{ijk}$ 是 $\text{treatment}\ (A_i,B_j)$ 下的第 $k$ 个**观测**   

记 $y_{ijk}= \mu_{ij} + \varepsilon_{ijk}$  
其中 $\mu_{ij}$ 为 $\text{treatment}\ (A_i,B_j)$ 下观测的**理论均值**  
**随机项** $\{\varepsilon_{ijk}\}\overset{\text{i.i.d.}}\sim N(0,\sigma^2)$   
我们记**行均值**、**列均值**和**总体均值**为:
$$
\begin{cases} 
\mu_{i\cdot} = \frac{1}b \sum_{j=1}^b\mu_{ij}\\ 
\mu_{\cdot j} = \frac{1}a \sum_{i=1}^a\mu_{ij}\\ 
\mu_{\cdot\cdot} = \frac{1}{ab} \sum_{i=1}^a \sum_{j=1}^b \mu_{ij}\\ \end{cases}
$$
我们分别记 $A_i$ 和 $B_j$ 以及 $(A_i,B_j)$ 的**主效应** (main effect) 为:  
$$
\begin{cases} 
\text{ME}_{A_i} = \mu_{i\cdot}- \mu_{\cdot\cdot}\\ 
\text{ME}_{B_j} = \mu_{\cdot j}-\mu_{\cdot\cdot}\\
\text{ME}_{(A_i,B_j)} = \mu_{ij} - \mu_{\cdot\cdot}
\end{cases}
$$
- 若 $\mu_{1\cdot} = \dotsm = \mu_{a\cdot} = \mu_{\cdot\cdot}$，则主效应 $\text{ME}_{A_1} = \dotsm = \text{ME}_{A_a} = 0$ 
- 若 $\mu_{\cdot 1} = \dotsm = \mu_{\cdot b} = \mu_{\cdot\cdot}$，则主效应 $\text{ME}_{B_1}=\dotsm = \text{ME}_{B_b} = 0$ 

我们称因子 $A,B$ 是**可加的**，如果它们满足:  
$$
\text{ME}_{(A_i,B_j)} = \text{ME}_{A_i} + \text{ME}_{B_j}\quad(\text{for all }\begin{cases} i= 1,2,\dots,a\\ j=1,2,\dots,b \end{cases})
$$
此时我们也称因子 $A,B$ 之间不存在**交互效应** (interaction)  
否则，我们称因子 $A,B$ 是**不可加的**，代表因子 $A,B$ 之间**存在交互效应**  
双因子方差分析的模型通常建立在 "存在交互效应" 的基本假设下.  
我们定义 $\text{IA}_{(A_i,B_j)}$ 用于衡量交互效应:
$$
\begin{align}
\text{IA}_{(A_i,B_j)} 
&:= \text{ME}_{(A_i,B_j)} - \text{ME}_{A_i} - \text{ME}_{B_j}\\
&= 
(\mu_{ij} - \mu_{\cdot\cdot}) 
- 
(\mu_{i\cdot}-\mu_{\cdot\cdot})
-
(\mu_{\cdot j}-\mu_{\cdot\cdot})\\
&=
\mu_{ij} + \mu_{\cdot\cdot} - \mu_{i\cdot} - \mu_{\cdot j}
\end{align}
$$
值得注意的是:
$$
\begin{align}
\sum_{i=1}^a 
\text{ME}_{A_i}
&=
\sum_{i=1}^a (\mu_{i\cdot} - \mu_{\cdot\cdot})\\
&=
a \mu_{\cdot\cdot} - a\mu_{\cdot\cdot}\\
&=
0\\
\hline
\sum_{j=1}^b 
\text{ME}_{B_j}
&=
\sum_{j=1}^b (\mu_{\cdot j} - \mu_{\cdot\cdot})\\
&=
 b\mu_{\cdot\cdot} -  b \mu_{\cdot\cdot}\\
&=
0\\
\hline
\sum_{i=1}^a \sum_{j=1}^b \text{ME}_{(A_i,B_j)}
&=
\sum_{i=1}^a \sum_{j=1}^b (\mu_{ij} - \mu_{\cdot\cdot})\\
&= ab \mu_{\cdot\cdot} - ab \mu_{\cdot\cdot}\\
&= 0\\
\hline
\sum_{i=1}^a \text{IA}_{(A_i,B_j)} 
&=
\sum_{i=1}^a (\mu_{ij} + \mu_{\cdot\cdot} - \mu_{i\cdot} - \mu_{\cdot j})\\
&=
a\mu_{\cdot j} + a \mu_{\cdot\cdot} - a \mu_{\cdot\cdot} - a \mu_{\cdot j}\\
&=
0\\
\hline
\sum_{j=1}^b \text{IA}_{(A_i,B_j)} 
&=
\sum_{j=1}^b (\mu_{ij} + \mu_{\cdot\cdot} - \mu_{i\cdot} - \mu_{\cdot j})\\
&=
b\mu_{i\cdot} + b \mu_{\cdot\cdot} - b \mu_{\cdot\cdot} - b \mu_{i\cdot}\\
&=
0
\end{align}
$$

### 5.3.2 点估计

#### (1) 记号

给定 $abn$ 个样本观测 $\{y_{ijk}\}$ (其中 $i=1,\dots,a$, $j=1,\dots,b$, $k=1,\dots,n$)  
我们定义:  
$$
\begin{align}
\bar y_{ij\cdot} 
&:= \frac1n \sum_{k=1}^n y_{ijk}\\
\bar y_{i\cdot\cdot} 
&:= \frac{1}{bn} \sum_{j=1}^b \sum_{k=1}^n y_{ijk} = \frac{1}{b}\sum_{j=1}^b \bar y_{ij\cdot}\\
\bar y_{\cdot j\cdot}
&:= \frac{1}{an} \sum_{i=1}^a \sum_{k=1}^n y_{ijk} = \frac{1}{a}\sum_{i=1}^a \bar y_{ij\cdot}\\
\bar y_{\cdot\cdot\cdot}
&:=
\frac{1}{abn}\sum_{i=1}^a \sum_{j=1}^b \sum_{k=1}^n y_{ijk} = \frac{1}{ab}\sum_{i=1}^a \sum_{j=1}^b \bar y_{ij\cdot}
\end{align}
$$
设 $y_{ijk} = \mu_{ij} + \varepsilon_{ijk}$   
其中 $\mu_{ij}$ 为 $\text{treatment}$ $(A_i,B_j)$ 的理论均值，而 $\{\varepsilon_{ijk}\}\overset{\text{iid}}\sim N(0,\sigma^2)$   
类似地，我们定义:  
$$
\begin{align}
\bar \varepsilon_{ij\cdot} 
&:= \frac1n \sum_{k=1}^n \varepsilon_{ijk}\\
\bar \varepsilon_{i\cdot\cdot} 
&:= \frac{1}{bn} \sum_{j=1}^b \sum_{k=1}^n \varepsilon_{ijk} = \frac{1}{b}\sum_{j=1}^b \bar \varepsilon_{ij\cdot}\\
\bar \varepsilon_{\cdot j\cdot}
&:= \frac{1}{an} \sum_{i=1}^a \sum_{k=1}^n \varepsilon_{ijk} = \frac{1}{a}\sum_{i=1}^a \bar \varepsilon_{ij\cdot}\\
\bar \varepsilon_{\cdot\cdot\cdot}
&:=
\frac{1}{abn}\sum_{i=1}^a \sum_{j=1}^b \sum_{k=1}^n \varepsilon_{ijk} = \frac{1}{ab}\sum_{i=1}^a \sum_{j=1}^b \bar \varepsilon_{ij\cdot}
\end{align}
$$
于是我们有:  
$$
\begin{align}
\bar y_{ij\cdot} 
&= \frac1n \sum_{k=1}^n y_{ijk} = \frac{1}{n}\sum_{k=1}^n (\mu_{ij} + \varepsilon_{ijk}) = \mu_{ij} + \bar\varepsilon_{ij\cdot}\\
\bar y_{i\cdot\cdot} 
&= \frac{1}{b}\sum_{j=1}^n \bar y_{ij\cdot} = \frac{1}{b}\sum_{j=1}^b (\mu_{ij} + \bar \varepsilon_{ij\cdot}) = \mu_{i\cdot} + \bar\varepsilon_{i\cdot\cdot}\\
\bar y_{\cdot j\cdot}
&= \frac{1}{a}\sum_{i=1}^a \bar y_{ij\cdot}
= \frac{1}{a} \sum_{i=1}^a (\mu_{ij} + \bar \varepsilon_{ij\cdot}) = \mu_{\cdot j} + \bar\varepsilon_{\cdot j\cdot}\\
\bar y_{\cdot\cdot\cdot}
&=
\frac{1}{ab}\sum_{i=1}^a \sum_{j=1}^b \bar y_{ij\cdot}
=
\frac{1}{ab}\sum_{i=1}^a \sum_{j=1}^b (\mu_{ij} + \bar \varepsilon_{ij\cdot}) = \mu_{\cdot\cdot} + \bar\varepsilon_{\cdot\cdot\cdot}
\end{align}
$$

#### (2) 均值

注意到 $\text{treatment}$ $(A_i,B_j)$ 下的观测 $y_{i,j,1},y_{i,j,2},\dots,y_{i,j,n}$ 相互独立，且同分布于 $N(\mu_{ij},\sigma^2)$    
考虑优化问题:  
$$
\min_{\mu \in \mathbb R}\|y^{(i,j)} - \mu 1_{n}\|^2:= \sum_{k=1}^{n} (y_{ijk}-\mu)^2\\
\text{where }y^{(i,j)} = 
\begin{bmatrix}
y_{i,j,1}\\
\vdots\\
y_{i,j,n}
\end{bmatrix}
$$
令 $\nabla_\mu \|y^{(i,j)}-\mu 1_{n}\|^2 = -2\cdot 1_{n}^{\mathrm T}(y^{(i,j)}-\mu 1_{n}) = 0_{n}$ 可知 $\mu_{ij}$ 的最小二乘估计量为:  
$$
\hat \mu_{ij} := (1_{n}^{\mathrm T}1_{n})^{-1} 1_{n}^{\mathrm T} y^{(i,j)} = \frac{1}{n}\sum_{k=1}^{n}y_{ijk} = \bar y_{ij\cdot}
$$
根据多元线性回归的 **Gauss-Markov 定理**可知 $\bar y_{ij\cdot}$ 是 $\mu_{ij}$ 的最佳线性无偏估计量 $(\text{BLUE})$  
它服从正态分布:  
$$
\bar y_{ij\cdot} = \frac{1}{n}\sum_{k=1}^{n}y_{ijk} \sim N(\mu_{ij},\frac{\sigma^2}{n})
$$

****

基于上述结论我们有:

- $\hat \mu_{ij}:= \bar y_{ij\cdot}\sim N(\mu_{ij},\frac{\sigma^2}{n})$ 是 $\mu_{ij}$ 的 $\text{BLUE}$ 

- $\hat \mu_{i\cdot} := \bar y_{i\cdot\cdot}\sim N(\mu_{i\cdot},\frac{\sigma^2}{bn})$ 是 $\mu_{i\cdot}$ 的 $\text{BLUE}$ 

- $\hat \mu_{\cdot j} :=\bar y_{\cdot j\cdot}\sim N(\mu_{\cdot j},\frac{\sigma^2}{an})$ 是 $\mu_{\cdot j}$ 的 $\text{BLUE}$ 

- $\hat \mu_{\cdot\cdot} :=\bar y_{\cdot\cdot\cdot}\sim N(\mu_{\cdot\cdot},\frac{\sigma^2}{abn})$ 是 $\mu_{\cdot\cdot}$ 的 $\text{BLUE}$ 

- $\widehat {\text{ME}}_{A_i} := \bar y_{i\cdot\cdot}-\bar y_{\cdot\cdot\cdot}$ 是 $\text{ME}_{A_i} = \mu_{i\cdot}-\mu_{\cdot\cdot}$ 的 $\text{BLUE}$ 

- $\widehat {\text{ME}}_{B_j}:= \bar y_{\cdot j\cdot} -\bar y_{\cdot\cdot\cdot}$ 是 $\text{ME}_{B_j}= \mu_{\cdot j}-\mu_{\cdot\cdot}$ 的 $\text{BLUE}$ 

- $\widehat {\text{ME}}_{(A_i,B_j)}:= \bar y_{ij\cdot}-\bar y_{\cdot\cdot\cdot}$ 是 $\text{ME}_{(A_i,B_j)}=\mu_{ij}-\mu_{\cdot\cdot}$ 的 $\text{BLUE}$ 

- $\widehat{\text{IA}}_{(A_i,B_j)} := \widehat {\text{ME}}_{(A_i,B_j)} - 
  \widehat{\text{ME}}_{A_i} - \widehat{\text{ME}}_{B_j}=\bar y_{ij\cdot} + \bar y_{\cdot\cdot\cdot} - \bar y_{i\cdot\cdot} - \bar y_{\cdot j\cdot}$ 是  
  $\text{IA}_{(A_i,B_j)}:= \text{ME}_{(A_i,B_j)} - \text{ME}_{A_i} - \text{ME}_{B_j}=\mu_{ij} + \mu_{\cdot\cdot} - \mu_{i\cdot} - \mu_{\cdot j}$ 的 $\text{BLUE}$
  
- 类似地，我们有:  
  $$
  \begin{align}
  \sum_{i=1}^a 
  \widehat{\text{ME}}_{A_i}
  &=
  \sum_{i=1}^a (\bar y_{i\cdot\cdot} - \bar y_{\cdot\cdot\cdot})\\
  &=
  a\bar y_{\cdot\cdot\cdot} - a\bar y_{\cdot\cdot\cdot}\\
  &=
  0\\
  \hline
  \sum_{j=1}^b 
  \widehat{\text{ME}}_{B_j}
  &=
  \sum_{j=1}^b (\bar y_{\cdot j\cdot} - \bar y_{\cdot\cdot})\\
  &=
   b\bar \mu_{\cdot\cdot} -  b\bar \mu_{\cdot\cdot}\\
  &=
  0\\
  \hline
  \sum_{i=1}^a \sum_{j=1}^b \widehat {\text{ME}}_{(A_i,B_j)}
  &=
  \sum_{i=1}^a \sum_{j=1}^b (\bar y_{ij\cdot} - \bar y_{\cdot\cdot\cdot})\\
  &= ab \bar y_{\cdot\cdot\cdot} - ab \bar y_{\cdot\cdot\cdot}\\
  &= 0\\
  \hline
  \sum_{i=1}^a \widehat{\text{IA}}_{(A_i,B_j)} 
  &=
  \sum_{i=1}^a (\bar y_{ij\cdot} + \bar y_{\cdot\cdot\cdot} - \bar y_{i\cdot\cdot} - \bar y_{\cdot j\cdot})\\
  &=
  a\bar y_{\cdot j\cdot} + a \bar y_{\cdot\cdot\cdot} - a \bar y_{\cdot\cdot\cdot} - a \bar y_{\cdot j\cdot}\\
  &=
  0\\
  \hline
  \sum_{j=1}^b \widehat{\text{IA}}_{(A_i,B_j)} 
  &=
  \sum_{j=1}^b (\bar y_{ij\cdot} + \bar y_{\cdot\cdot\cdot} - \bar y_{i\cdot\cdot} - \bar y_{\cdot j\cdot})\\
  &=
  b\bar y_{i\cdot\cdot} + b \bar y_{\cdot\cdot\cdot} - b \bar y_{\cdot\cdot\cdot} - b \bar y_{i\cdot\cdot}\\
  &=
  0
  \end{align}
  $$



#### (3) 方差

考虑双因子方差分析的多元线性回归形式:
$$
\begin{align}
\begin{bmatrix}
y^{(\cdot 1)}\\
y^{(\cdot 2)}\\
\vdots\\
y^{(\cdot b)}
\end{bmatrix}
= 
\begin{bmatrix} I_a \otimes 1_n &&&\\  &I_a \otimes 1_n&&\\ &&\ddots&\\ &&&I_a \otimes 1_n\\ \end{bmatrix}_{abn\times ab} 
\begin{bmatrix}\mu^{(\cdot 1)} \\ \mu^{(\cdot 2)} \\\vdots\\ \mu^{(\cdot b)}\end{bmatrix}
+
\begin{bmatrix}
\varepsilon^{(\cdot 1)}\\
\varepsilon^{(\cdot 2)}\\
\vdots\\
\varepsilon^{(\cdot b)}
\end{bmatrix}
\end{align}
$$
其中 $\otimes $ 代表 Kronecker 乘积，而其他记号定义如下:
$$
I_a \otimes 1_n =
\begin{bmatrix}
1_n \\
& 1_n\\
& & \ddots\\
& & & 1_n
\end{bmatrix}_{an\times a}\\
y^{(i,j)}
=
\begin{bmatrix}
y_{i,j,1}\\
y_{i,j,2}\\
\vdots\\
y_{i,j,n}
\end{bmatrix}\quad

y^{(\cdot j)} = 
\begin{bmatrix}
y^{(1,j)}\\
y^{(2,j)}\\
\vdots\\
y^{(a,j)}
\end{bmatrix}\\

\varepsilon^{(i,j)}
=
\begin{bmatrix}
\varepsilon_{i,j,1}\\
\varepsilon_{i,j,2}\\
\vdots\\
\varepsilon_{i,j,n}
\end{bmatrix}
\quad
\varepsilon^{(\cdot j)}
=
\begin{bmatrix}
\varepsilon^{(1,j)}\\
\varepsilon^{(2,j)}\\
\vdots\\
\varepsilon^{(a,j)}
\end{bmatrix}\\
\{\varepsilon_{ijk}\} \overset{\text{iid}}\sim N(0,\sigma^2)\\

\mu^{(\cdot j)} = 
\begin{bmatrix}
\mu_{1j}\\
\mu_{2j}\\
\vdots\\
\mu_{aj}
\end{bmatrix}
\quad
\bar y^{(\cdot j)}
=
\begin{bmatrix}
\bar y_{1j\cdot}\\
\bar y_{2j\cdot}\\
\vdots\\
\bar y_{aj\cdot}
\end{bmatrix}
$$
根据多元线性回归的结论可知 $\sigma^2$ 的无偏估计量为:  
$$
\begin{align}
s^2 
&=
\frac{1}{abn-ab}
\left\|
\begin{bmatrix}
y^{(\cdot 1)}\\
y^{(\cdot 2)}\\
\vdots\\
y^{(\cdot b)}
\end{bmatrix}
-
\begin{bmatrix} I_a \otimes 1_n &&&\\  &I_a \otimes 1_n&&\\ &&\ddots&\\ &&&I_a \otimes 1_n\\ \end{bmatrix}_{abn\times ab} 
\begin{bmatrix}\bar y^{(\cdot1)} \\ \bar y^{(\cdot2)} \\\vdots\\ \bar y^{(\cdot b)}\end{bmatrix}
\right\|^2\\
&=
\frac1{ab(n-1)} 
\left\|
\begin{bmatrix}
y^{(\cdot 1)} - \bar y^{(\cdot1)} \odot 1_{an}\\
y^{(\cdot 2)} - \bar y^{(\cdot2)} \odot 1_{an}\\
\vdots\\
y^{(\cdot b)} - \bar y^{(\cdot b)} \odot 1_{an}
\end{bmatrix}
\right\|^2\quad (\text{where }\odot \text{ denote elementwise product})\\

&=
\frac1{ab(n-1)} \sum_{i=1}^a \sum_{j=1}^b \|y^{(i,j)}-\bar y_{ij\cdot}1_{n}\|^2\\
&=
\frac{1}{ab(n-1)}\sum_{i=1}^a \sum_{j=1}^b (n-1)s_{ij}^2
\end{align}
$$
其中 $s_{ij}^2 = \frac{1}{n-1}\|y^{(i,j)}-\bar y_{ij\cdot}1_{n}\|^2 = \frac{1}{n-1}\sum_{k=1}^{n}(y_{ijk}-\bar y_{ij\cdot})^2$ 为 $\text{treatment}$ $(A_i,B_j)$ 下的样本方差.



#### (4) 分布

根据多元线性回归的结论可知:

- ① 关于样本均值:
  $$
  \bar y_{ij\cdot} = \frac{1}{n}\sum_{k=1}^{n}y_{ijk} \sim N\left(\mu_{ij},\frac{\sigma^2}{n}\right)\\
  \bar y_{i\cdot\cdot} = \frac{1}{b}\sum_{j=1}^b \bar y_{ij\cdot}\sim N\left(\mu_{i\cdot},\frac{\sigma^2}{bn}\right)\\
  \bar y_{\cdot j\cdot} = \frac{1}{a}\sum_{i=1}^a \bar y_{ij\cdot} \sim N\left(\mu_{\cdot j},\frac{\sigma^2}{an}\right)\\
  \bar y_{\cdot\cdot\cdot} = \frac{1}{ab}\sum_{i=1}^a \sum_{j=1}^b \bar y_{ij\cdot} \sim N
  \left(\mu_{\cdot\cdot},\frac{\sigma^2}{abn}\right)
  $$

- ② $s_i^2 = \frac{1}{n-1}\sum_{k=1}^{n}(y_{ijk}-\bar y_{ij\cdot})^2 \sim \frac{1}{n-1}\sigma^2\chi^2_{(n-1)}$ 且 $\bar y_{ij\cdot}\perp s_{ij}^2$   
  (事实上对于任意 $i_1,i_2$ 和 $j_1,j_2$，我们都有 $\bar y_{i_1j_1\cdot}\perp s_{i_2j_2}^2$ 成立)
  
- ③ $s^2 = \frac{1}{ab(n-1)}\sum_{i=1}^a\sum_{j=1}^b (n-1)s_{ij}^2\sim \frac{1}{ab(n-1)} \sigma^2\chi^2_{(ab(n-1))}$ 且 $s^2\perp \bar y_{ij\cdot}\ (\forall\ i=1,\dots,a,j=1,\dots,b)$ 



### 5.3.3 交互效应存在性检验

首先我们想知道因子 $A,B$ 之间是否存在交互效应.  
因此零假设和备择假设分别为: 
$$
H_0: \widehat{\text{IA}}_{(A_i,B_j)} = 0 \text{ for all }
\begin{cases}
i=1,\dots,a\\
j=1,\dots,b
\end{cases}\\
\Updownarrow\\
H_1: \exists\ i,j\text{ such that }\widehat {\text{IA}}_{(A_i,B_j)} = 0
$$

#### (1) 基本记号

为了记号的简便，我们记:  
$$
\begin{align}
\sum_{i=1}^a \sum_{j=1}^b \sum_{k=1}^n &\equiv \sum_{i,j,k}^{a,b,n}\\
\sum_{i=1}^a \sum_{j=1}^b &\equiv \sum_{i,j}^{a,b}
\end{align}
$$
考虑总平方和 $\text{SST}$ (Total Sum of Squares) 的分解:  
$$
\begin{align}
\text{SST}
&=
\sum_{i,j,k}^{a,b,n} (y_{ijk}-\bar y_{\cdot\cdot\cdot})^2\\
&=
\sum_{i,j,k}^{a,b,n}(y_{i,j,k} - \bar y_{ij\cdot} + \bar y_{ij\cdot} - \bar y_{\cdot\cdot\cdot})^2\\
&=
\sum_{i,j,k}^{a,b,n} (y_{ijk} - \bar y_{ij\cdot})^2 
+
\sum_{i,j,k}^{a,b,n} (\bar y_{ij\cdot} - \bar y_{\cdot\cdot\cdot})^2
+
2\sum_{i,j,k}^{a,b,n} (y_{ijk} - \bar y_{ij\cdot})(\bar y_{ij\cdot} - \bar y_{\cdot\cdot\cdot})\\
&=
\sum_{i,j,k}^{a,b,n} (y_{ijk} - \bar y_{ij\cdot})^2 
+
n\sum_{i,j}^{a,b} (\bar y_{ij\cdot} - \bar y_{\cdot\cdot\cdot})^2
+
2 \sum_{i,j}^{a,b}\left\{(\bar y_{ij\cdot} - \bar y_{\cdot\cdot\cdot})\sum_{k=1}^{n} (y_{ijk} - \bar y_{ij\cdot})\right\}\\

&=
\sum_{i,j,k}^{a,b,n} (y_{ijk} - \bar y_{ij\cdot})^2 
+
n\sum_{i,j}^{a,b}  (\bar y_{ij\cdot} - \bar y_{\cdot\cdot\cdot})^2
+
2 \sum_{i,j}^{a,b}\left\{(\bar y_{ij\cdot} - \bar y_{\cdot\cdot\cdot})(n\bar y_{ij\cdot} - n\bar y_{ij\cdot})\right\}\\
&=
\sum_{i,j,k}^{a,b,n} (y_{ijk} - \bar y_{ij\cdot})^2 
+
n\sum_{i,j}^{a,b} (\bar y_{ij\cdot} - \bar y_{\cdot\cdot\cdot})^2\\
&=
\text{SSE} + \text{SAB}
\end{align}
$$
注意到 $\text{SST}$ 有 $abn$ 个变量 $\{y_{ijk}\}$ 和一个约束 $\sum_{i,j,k}^{a,b,n} (y_{ijk} - \bar y_{\cdot\cdot\cdot})=0$   
因此其自由度 $\text{df}_{\text{SST}} = abn-1$ 

- 根据交叉项 $=0$ 可知 $\text{SSE}\perp \text{SAB}$ 

- **组内偏差平方和** (Within\-Group Deviation Sum of Squares)  
  $$
  \begin{align}
  \text{SSE}
  &=
  \sum_{i,j,k}^{a,b,n} (y_{ijk} - \bar y_{ij\cdot})^2\quad (\text{note that }s_{ij}^2:= \frac{1}{n-1} \sum_{k=1}^n (y_{ijk} - \bar y_{ij\cdot})^2)\\
  &=
  \sum_{i,j}^{a,b} (n-1) s_{ij}^2 \quad (\text{note that }s^2:= \frac{1}{ab(n-1)}\sum_{ij}^{a,b}(n-1)s_{ij}^2 \sim \frac{\sigma^2}{ab(n-1)}\chi^2_{ab(n-1)}) \\
  &=
  ab(n-1)s^2\\
  &\sim
  \sigma^2\chi^2_{ab(n-1)}
  \end{align}
  $$
  显然 $\text{SSE}$ 的自由度 $\text{df}_{\text{SSE}}= ab(n-1)$ 

- **组间偏差平方和** (Between\-Group Deviation Sum of Squares)  
  $$
  \begin{align}
  \text{SAB}
  &=
  \sum_{i,j,k}^{a,b,n} (\bar y_{ij\cdot} - \bar y_{\cdot\cdot\cdot})^2\\
  &=
  n\sum_{i,j}^{a,b}(\bar y_{ij\cdot} - \bar y_{\cdot\cdot\cdot})^2\\
  &=
  n\sum_{i,j}^{a,b} \widehat{\text{ME}}_{(A_i,B_j)}^2\quad (\text{note that }\widehat{\text{IA}}_{(A_i,B_j)} := \widehat {\text{ME}}_{(A_i,B_j)} - 
  \widehat{\text{ME}}_{A_i} - \widehat{\text{ME}}_{B_j})\\
  &=
  n\sum_{i,j}^{a,b} (\widehat{\text{IA}}_{(A_i,B_j)} + \widehat{\text{ME}}_{A_i} + \widehat{\text{ME}}_{B_j})^2\\
  &=
  n\sum_{i,j}^{a,b} \widehat{\text{IA}}_{(A_i,B_j)}^2
  +
  n\sum_{i,j}^{a,b} \widehat{\text{ME}}_{A_i}^2 
  +
  n\sum_{i,j}^{a,b} \widehat{\text{ME}}_{B_j}^2 \\
  &\qquad+
  2n \sum_{i,j}^{a,b} \widehat{\text{ME}}_{A_i} \widehat{\text{ME}}_{B_j}
  +
  2n \sum_{i,j}^{a,b} \widehat{\text{IA}}_{(A_i,B_j)} \widehat{\text{ME}}_{A_i}
  +
  2n \sum_{i,j}^{a,b} \widehat{\text{IA}}_{(A_i,B_j)} \widehat{\text{ME}}_{B_j}\\
  &=
  n\sum_{i,j}^{a,b} \widehat{\text{IA}}_{(A_i,B_j)}^2
  +
  bn\sum_{i=1}^{a} \widehat{\text{ME}}_{A_i}^2 
  +
  an\sum_{j=1}^{b} \widehat{\text{ME}}_{B_j}^2 \\
  &\qquad+
  2n \sum_{i=1}^a \widehat{\text{ME}}_{A_i} \sum_{j=1}^b \widehat{\text{ME}}_{B_j}
  +
  2n \sum_{i=1}^{a} \left\{\widehat{\text{ME}}_{A_i} \sum_{j=1}^b\widehat{\text{IA}}_{(A_i,B_j)} \right\}
  +
  2n \sum_{j=1}^{b} \left\{\widehat{\text{ME}}_{B_j} \sum_{i=1}^a\widehat{\text{IA}}_{(A_i,B_j)} \right\}\\
  
  &=
  n\sum_{i,j}^{a,b} \widehat{\text{IA}}_{(A_i,B_j)}^2
  +
  bn\sum_{i=1}^{a} \widehat{\text{ME}}_{A_i}^2 
  +
  an\sum_{j=1}^{b} \widehat{\text{ME}}_{B_j}^2 \\
  &\qquad+
  2n \cdot 0 \cdot 0
  +
  2n \sum_{i=1}^{a} \left\{\widehat{\text{ME}}_{A_i} \cdot 0 \right\}
  +
  2n \sum_{j=1}^{b} \left\{\widehat{\text{ME}}_{B_j} \cdot 0 \right\}\\
  &=
  n\sum_{i,j}^{a,b} \widehat{\text{IA}}_{(A_i,B_j)}^2
  +
  bn\sum_{i=1}^{a} \widehat{\text{ME}}_{A_i}^2 
  +
  an\sum_{j=1}^{b} \widehat{\text{ME}}_{B_j}^2\\
  &=
  \text{SSAB} + \text{SSA} + \text{SSB}
  \end{align}
  $$
  注意到 $\text{SAB}$ 有 $ab$ 个变量 $\{\bar y_{ij\cdot}\}$ 和一个约束 $\sum_{i,j}^{a,b} (\bar y_{ij\cdot} - \bar y_{\cdot\cdot\cdot})=0$   
  因此其自由度 $\text{df}_{\text{SAB}} = ab-1$  

- 根据交叉项 $=0$ 可知 $\text{SSAB},\text{SSA},\text{SSB}$ 相互独立.

- **交互效应平方和** (Interaction Sum of Squares)   
  $$
  \begin{align}
  \text{SSAB}
  &=
  n\sum_{i,j}^{a,b} \widehat{\text{IA}}_{(A_i,B_j)}^2\\
  &=
  n\sum_{i,j}^{a,b} (\bar y_{ij\cdot} + \bar y_{\cdot\cdot\cdot} - \bar y_{i\cdot\cdot} - \bar y_{\cdot j\cdot})^2
  \end{align}
  $$
  注意到 $\text{SSAB}$ 有 $ab$ 个变量 $\{\bar y_{ij\cdot}\}$ 和 $a+b-1$ 个约束 (注意不是 $a+b$ 个，因为有一个约束是冗余的):   
  $$
  \sum_{i=1}^a \widehat{\text{IA}}_{(A_i,B_j)} = 0\quad (j=1,\dots,b)\\
  \sum_{j=1}^b \widehat{\text{IA}}_{(A_i,B_j)} = 0\quad (i=1,\dots,a-1)
  $$
  因此其自由度 $\text{df}_{\text{SSAB}} = ab - (a+b-1) = (a-1)(b-1)$  

- 因子 $A$ 的因子水平 $A_1,\dots,A_a$ 之间的**偏差平方和**:   
  $$
  \begin{align}
  \text{SSA}
  &=
  bn\sum_{i=1}^{a} \widehat{\text{ME}}_{A_i}^2\\
  &=
  bn \sum_{i=1}^a (\bar y_{i\cdot\cdot} - \bar y_{\cdot\cdot\cdot})^2
  \end{align}
  $$
  注意到 $\text{SSA}$ 有 $a$ 个变量 $\{\bar y_{i\cdot\cdot}\}$ 和一个约束 $\sum_{i=1}^{a} (\bar y_{i\cdot\cdot} - \bar y_{\cdot\cdot\cdot})=0$   
  因此其自由度 $\text{df}_{\text{SSA}} = a-1$  

- 因子 $B$ 的因子水平 $B_1,\dots,B_b$ 之间的**偏差平方和**
  $$
  \begin{align}
  \text{SSB}
  &=
  an\sum_{j=1}^{b} \widehat{\text{ME}}_{B_j}^2\\
  &=
  an \sum_{j=1}^b (\bar y_{\cdot j \cdot} - \bar y_{\cdot\cdot\cdot})^2
  \end{align}
  $$
  注意到 $\text{SSB}$ 有 $b$ 个变量 $\{\bar y_{\cdot j \cdot}\}$ 和一个约束 $\sum_{j=1}^{b} (\bar y_{\cdot j\cdot} - \bar y_{\cdot\cdot\cdot})=0$   
  因此其自由度 $\text{df}_{\text{SSB}} = b-1$  

****

在存在交互效应的假设下，双因子方差分析的 ANOVA TABLE 如下:
$$
\text{ANOVA  TABLE  (with  interaction)}\\ 
\begin{array}{|c|c|c|c|}  
\hline 
\text{Sum of Squares} &  & \text{Degree of  Freedom} & \text{Mean Squares} \\  
\hline  
\text{SST} & \sum_{i,j,k}^{a,b,n}(y_{ijk}-\bar y_{\cdot\cdot\cdot})^2 & abn-1 & \text{MST} = \frac{\text{SST}}{abn-1} \\  
\hline  
\text{SSE} & \sum_{i,j,k}^{a,b,n}(y_{ijk}-\bar y_{ij\cdot})^2 & ab(n-1) &  \text{MSE} = \frac{\text{SSE}}{ab(n-1)}\\  
\hline  
\text{SAB} & n\sum_{i,j}^{a,b}\widehat{\text{ME}}_{(A_i,B_j)}^2 & ab-1 & \text{MAB} = \frac{\text{SAB}}{ab-1} \\  
\hline 
\text{SSAB} & n\sum_{i,j}^{a,b}\widehat {\text{IA}}_{(A_i,B_j)}^2 & (a-1)(b-1) & \text{MSAB} = \frac{\text{SSAB}}{(a-1)(b-1)}\\ 
\hline 
\text{SSA} &  bn\sum_{i=1}^a\widehat {\text{ME}}_{A_i}^2 & a-1 & \text{MSA} = \frac{\text{SSA}}{a-1}\\ \hline 
\text{SSB} &  an\sum_{j=1}^b \widehat {\text{ME}}_{B_j}^2 & b-1 & \text{MSB} = \frac{\text{SSB}}{b-1}\\ \hline \end{array}\\
\widehat {\text{ME}}_{A_i} = \bar y_{i\cdot\cdot}-\bar y_{\cdot\cdot\cdot}\\
\widehat {\text{ME}}_{B_j} = \bar y_{\cdot j\cdot} -\bar y_{\cdot\cdot\cdot}\\
\widehat {\text{ME}}_{(A_i,B_j)} = \bar y_{ij\cdot}-\bar y_{\cdot\cdot\cdot}\\
\widehat{\text{IA}}_{(A_i,B_j)} = \widehat {\text{ME}}_{(A_i,B_j)} - 
\widehat{\text{ME}}_{A_i} - \widehat{\text{ME}}_{B_j}=\bar y_{ij\cdot} + \bar y_{\cdot\cdot\cdot} - \bar y_{i\cdot\cdot} - \bar y_{\cdot j\cdot}
$$


#### (2) 零假设分布

考虑零假设 $H_0$:  
$$
H_0: \widehat{\text{IA}}_{(A_i,B_j)} = 0 \text{ for all }
\begin{cases}
i=1,\dots,a\\
j=1,\dots,b
\end{cases}
$$
在零假设 $H_0$ 成立的前提条件下，我们有:  
$$
\begin{align}
\widehat{\text{IA}}_{(A_i,B_j)} 
&= \widehat {\text{ME}}_{(A_i,B_j)} - 
\widehat{\text{ME}}_{A_i} - \widehat{\text{ME}}_{B_j}\\
&=\bar y_{ij\cdot} + \bar y_{\cdot\cdot\cdot} - \bar y_{i\cdot\cdot} - \bar y_{\cdot j\cdot}\\
&=
(\mu_{ij} + \bar \varepsilon_{ij\cdot}) + (\mu_{\cdot\cdot} + \bar \varepsilon_{\cdot\cdot\cdot})
-
(\mu_{i\cdot} + \bar\varepsilon_{i\cdot\cdot}) - (\mu_{\cdot j} + \bar \varepsilon_{\cdot j\cdot})\\
&=
(\mu_{ij} + \mu_{\cdot\cdot} - \mu_{i\cdot} - \mu_{\cdot j}) 
+ 
\bar \varepsilon_{ij\cdot} + \bar \varepsilon_{\cdot\cdot\cdot} - \bar \varepsilon_{i\cdot\cdot} - \bar \varepsilon_{\cdot j\cdot}\\
&=
\text{IA}_{(A_i,B_j)} + \bar \varepsilon_{ij\cdot} + \bar \varepsilon_{\cdot\cdot\cdot} - \bar \varepsilon_{i\cdot\cdot} - \bar \varepsilon_{\cdot j\cdot}\\
&\overset{H_0}= 
\bar \varepsilon_{ij\cdot} + \bar \varepsilon_{\cdot\cdot\cdot} - \bar \varepsilon_{i\cdot\cdot} - \bar \varepsilon_{\cdot j\cdot}
\end{align}
$$
定义 $\Epsilon:= [\bar \varepsilon_{ij\cdot}]\in \mathbb R^{a\times b}$  
用 $\Epsilon_{(i,:)}$ 代表 $\Epsilon$ 的第 $i$ 行，用 $\text{E}_{(:,j)}$ 代表 $\Epsilon$ 的第 $j$ 列.  
注意到: 
$$
\begin{align}
\bar\varepsilon_{i\cdot\cdot}
&= \frac1b \sum_{j=1}^b \bar \varepsilon_{ij\cdot} = \Epsilon_{(i,:)}\cdot \frac1b 1_b\\
\bar\varepsilon_{\cdot j\cdot}
&=
\frac{1}{a} \sum_{i=1}^a \bar\varepsilon_{ij\cdot} = \frac{1}{a}1_a^{\mathrm T}\cdot\Epsilon_{(:,j)} \\
\bar \varepsilon_{\cdot\cdot\cdot}
&=
\frac{1}{ab}\sum_{i=1}^{a}\sum_{j=1}^b \bar\varepsilon_{ij\cdot} = \frac{1}{a} 1_a^{\mathrm T} \cdot \Epsilon \cdot
\frac{1}{b} 1_b
\end{align}
$$
因此我们有:  
$$
\begin{align} 
\begin{bmatrix} 
\bar\varepsilon_{1\cdot\cdot} & \dotsm & \bar\varepsilon_{1\cdot\cdot}\\ \vdots &&\vdots\\ \bar \varepsilon_{a\cdot\cdot} & \dotsm & \bar\varepsilon_{a\cdot\cdot} \end{bmatrix} 
&= 
\begin{bmatrix} \bar \varepsilon_{1\cdot\cdot}1_b^{\mathrm T}\\ \vdots\\ \bar \varepsilon_{a\cdot\cdot}1_b^{\mathrm T} \end{bmatrix} 
= 
\Epsilon\cdot \frac{1}b1_b1_b^{\mathrm T}\\ 
\begin{bmatrix} \bar\varepsilon_{\cdot 1\cdot} & \dotsm & \bar\varepsilon_{\cdot b\cdot}\\ \vdots &&\vdots\\ \bar \varepsilon_{\cdot 1\cdot} & \dotsm & \bar\varepsilon_{\cdot b\cdot} \end{bmatrix}  
&= 
\begin{bmatrix}  \bar\varepsilon_{\cdot 1 \cdot} 1_a & \dotsm & \bar\varepsilon_{\cdot b\cdot} 1_a \end{bmatrix} 
= \frac1a 1_a1_a^{\mathrm T} \cdot \Epsilon\\ 
\begin{bmatrix} \bar\varepsilon_{\cdot\cdot\cdot} & \dotsm & \bar\varepsilon_{\cdot\cdot\cdot}\\ \vdots &&\vdots\\ \bar \varepsilon_{\cdot\cdot\cdot} & \dotsm & \bar\varepsilon_{\cdot\cdot\cdot} \end{bmatrix}  
&= 1_a \bar\varepsilon_{\cdot\cdot\cdot} 1_b^{\mathrm T} = \frac{1}a1_a1_a^{\mathrm T}\cdot \Epsilon\cdot \frac1b 1_b1_b^{\mathrm T} \end{align}
$$
于是我们有: 
$$
\begin{align}
&(I_a - \frac{1}{a}1_a1_a^{\mathrm T}) \Epsilon (I_b - \frac{1}{b} 1_b1_b^{\mathrm T})\\
&=\Epsilon +  \frac{1}a1_a1_a^{\mathrm T}\cdot \Epsilon\cdot \frac1b 1_b1_b^{\mathrm T} - \Epsilon\cdot \frac{1}b1_b1_b^{\mathrm T} - 
\frac1a 1_a1_a^{\mathrm T} \cdot \Epsilon\\
&=
[\bar\varepsilon_{ij\cdot}] + [\bar \varepsilon_{\cdot\cdot\cdot}] - [\bar \varepsilon_{i\cdot\cdot}]
-
[\bar \varepsilon_{\cdot j\cdot}]\\
&=
[\bar\varepsilon_{ij\cdot} + \bar \varepsilon_{\cdot\cdot\cdot} - \bar \varepsilon_{i\cdot\cdot} 
-
\bar \varepsilon_{\cdot j\cdot}]\\
&\overset{H_0}=
[\widehat{\text{IA}}_{(A_i,B_j)} ]
\end{align}
$$
记 $\begin{cases}
P_a =  I_a - \frac1a 1_a1_a^{\mathrm T}\\
P_b = I_b - \frac1b 1_b1_b^{\mathrm T}\end{cases}$   
显然它们是投影算子 (即自伴且幂等)，记其谱分解为:
$$
\begin{cases}
P_a = Q_a \Lambda_a Q_a^{\mathrm T}\\
P_b = Q_b \Lambda_b Q_b^{\mathrm T}
\end{cases}\text{ where }
\begin{cases}
\Lambda_a = \text{diag}\{\underset{a-1\text{ times}}{\underbrace{1,\dots,1}},0\}\\
\Lambda_b = \text{diag}\{\underset{b-1\text{ times}}{\underbrace{1,\dots,1}},0\}\\
\end{cases}
$$
则我们有:  
$$
\begin{align}
\text{SSAB}
&=
n\sum_{i,j}^{a,b} \widehat{\text{IA}}_{(A_i,B_j)}^2
\quad (\text{note that }\widehat{\text{IA}}_{(A_i,B_j)}\overset{H_0}= 
\bar \varepsilon_{ij\cdot} + \bar \varepsilon_{\cdot\cdot\cdot} - \bar \varepsilon_{i\cdot\cdot} - \bar \varepsilon_{\cdot j\cdot})\\
&\overset{H_0}= 
n\sum_{i,j}^{a,b} (\bar \varepsilon_{ij\cdot} + \bar \varepsilon_{\cdot\cdot\cdot} - \bar \varepsilon_{i\cdot\cdot} - \bar \varepsilon_{\cdot j\cdot})^2
\quad (\text{note that }\bar \varepsilon_{ij\cdot} + \bar \varepsilon_{\cdot\cdot\cdot} - \bar \varepsilon_{i\cdot\cdot} - \bar \varepsilon_{\cdot j\cdot} = [P_a \Epsilon P_b]_{(i,j)})\\
&=
n\sum_{i,j}^{a,b} ([P_a \Epsilon P_b]_{(i,j)})^2\\
&=
n \|P_a \Epsilon P_b\|_{\mathrm F}^2 \quad (\text{Frobenius norm})\\
&=
n\cdot \tr((P_a \Epsilon P_b)^{\mathrm T} P_a \Epsilon P_b)\\
&=
n\cdot \tr(P_b^{\mathrm T}\Epsilon^{\mathrm T} P_a^{\mathrm T} P_a \Epsilon P_b)\\
&=
n\cdot \tr(\Epsilon^{\mathrm T} P_a^{\mathrm T} P_a \Epsilon P_bP_b^{\mathrm T})\quad (\text{note that }
\begin{cases}
P_a^{\mathrm T}P_a = P_a^2 = P_a\\
P_b^{\mathrm T}P_b = P_b^2 = P_b
\end{cases})\\
&=
n\cdot \tr(\Epsilon^{\mathrm T} P_a \Epsilon P_b)\quad (\text{note that }\begin{cases}
P_a = Q_a \Lambda_a Q_a^{\mathrm T}\\
P_b = Q_b \Lambda_b Q_b^{\mathrm T}
\end{cases})\\
&=
n\cdot \tr(\Epsilon^{\mathrm T} Q_a\Lambda_a Q_a^{\mathrm T} \Epsilon Q_b \Lambda_b Q_b^{\mathrm T})\\
&=
n\cdot \tr(Q_b^{\mathrm T}\Epsilon^{\mathrm T} Q_a\Lambda_a Q_a^{\mathrm T} \Epsilon Q_b \Lambda_b)
\quad (\text{denote }\widetilde \Epsilon := Q_a^{\mathrm T}\Epsilon Q_b)\\
&=
n\cdot \tr(\widetilde \Epsilon ^{\mathrm T} \Lambda_a \widetilde \Epsilon \Lambda_b)\\
&=
n\cdot \tr({(\Lambda_a \widetilde \Epsilon)^{\mathrm T} (\widetilde \Epsilon \Lambda_b)})
\quad (\text{note that }
\Lambda_a = \text{diag}\{\underset{a-1\text{ times}}{\underbrace{1,\dots,1}},0\}\text{ and }
\Lambda_b = \text{diag}\{\underset{b-1\text{ times}}{\underbrace{1,\dots,1}},0\})\\
&=
n\sum_{i=1}^{a-1}\sum_{j=1}^{b-1} \widetilde \Epsilon_{(i,j)}^2
\end{align}
$$

注意到 $\Epsilon:= [\bar \varepsilon_{ij\cdot}]\in \mathbb R^{a\times b}$ 的元素独立同分布:  
$$
\{\bar \varepsilon_{ij\cdot}\} \overset{\text{iid}}\sim N\left(0,\frac{\sigma^2}{n}\right)
$$
我们可以将整个 $\Epsilon\in \mathbb R^{a\times b}$ 的分布表示为:  
$$
\text{vec}(\Epsilon) \sim N\left( 0_{ab}, \frac{\sigma^2}{n} I_b \otimes I_a\right)
$$
其中 $\text{vec}(\cdot)$ 是向量化操作符 (即将一个矩阵按列拉伸为向量)，而 $\otimes$ 代表 Kronecker 乘积.  
于是我们有:  
$$
\begin{align}
\text{vec}(\widetilde \Epsilon)
&=\text{vec}(Q_a^{\mathrm T} \Epsilon Q_b) \quad(\text{note that }\text{vec}(AXB) = (B^{\mathrm T}\otimes A)\text{vec}(X))\\
&= 
(Q^{\mathrm T}_b\otimes Q_a^{\mathrm T})\text{vec}(\Epsilon)\\
&\sim 
N\left((Q^{\mathrm T}_b\otimes Q_a^{\mathrm T}) 0_{ab}, (Q_b^{\mathrm T}\otimes Q_a^{\mathrm T})\cdot 
\frac{\sigma^2}{n}I_b\otimes I_a\cdot (Q_b^{\mathrm T}\otimes Q_a^{\mathrm T})^{\mathrm T}
\right)\\
&=
N\left(0_{ab}, (Q_b^{\mathrm T}\otimes Q_a^{\mathrm T})\cdot 
\frac{\sigma^2}{n}I_b\otimes I_a\cdot (Q_b\otimes Q_a)
\right)\\
&=
N\left(0_{ab}, \frac{\sigma^2}{n}[(Q_b^{\mathrm T} \cdot I_b\cdot Q_b)\otimes (Q_a^{\mathrm T} \cdot I_a\cdot Q_a)])
\right)\\
&=
N\left( 0_{ab}, \frac{\sigma^2}{n} I_b \otimes I_a\right)
\end{align}
$$
因此 $\widetilde \Epsilon = Q_a^{\mathrm T} \Epsilon Q_b\in \mathbb R^{a\times b}$ 的元素独立同分布:  
$$
\{\widetilde \Epsilon_{ij\cdot}\} \overset{\text{iid}}\sim N\left(0,\frac{\sigma^2}{n}\right)
$$

于是 $\text{SSAB}$ 的零假设分布为:  
$$
\text{SSAB} \overset{H_0} = n\sum_{i=1}^{a-1}\sum_{j=1}^{b-1} \widetilde \Epsilon_{(i,j)}^2 \sim n\cdot \frac{\sigma^2}{n} \chi^2_{(a-1)(b-1)} = \sigma^2 \chi^2_{(a-1)(b-1)}\\
\text{MSAB} := \frac{\text{SSAB}}{(a-1)(b-1)} \overset{H_0}\sim \frac{\sigma^2\chi^2_{(a-1)(b-1)}}{(a-1)(b-1)}
$$


#### (3) 检验法

零假设和备择假设分别为: 
$$
H_0: \widehat{\text{IA}}_{(A_i,B_j)} = 0 \text{ for all }
\begin{cases}
i=1,\dots,a\\
j=1,\dots,b
\end{cases}\\
\Updownarrow\\
H_1: \exists\ i,j\text{ such that }\widehat {\text{IA}}_{(A_i,B_j)} = 0
$$

设第一类型错误概率界限为 $\alpha$  
我们构造如下的 $F$ 检验统计量:
$$
F:= \frac{\text{MSAB}}{\text{MSE}} \overset{H_0}\sim \frac{\sigma^2\chi^2_{(a-1)(b-1)}/(a-1)(b-1)}
{\sigma^2 \chi^2_{ab(n-1)}/ab(n-1)} = F_{(a-1)(b-1),ab(n-1)}
$$
其中分子 $\text{MSAB}$ 和分母 $\text{MSE}$ 是相互独立的 (无论零假设 $H_0$ 是否成立)  

记 $F_{(a-1)(b-1),ab(n-1)}(\alpha)$ 为 $F_{(a-1)(b-1),ab(n-1)}$ 分布的 $1-\alpha$ 分位数.  
交互效应存在性检验的 $F$-检验法为:

- 若 $F= \frac{\text{MSAB}}{\text{MSE}}= \frac{n\sum_{i,j}^{a,b}(\bar y_{ij\cdot}-\bar y_{\cdot\cdot\cdot})^2 / (a-1)(b-1)}{\sum_{i,j,k}^{a,b,n}(y_{ijk}-\bar y_{ij\cdot})^2/ab(n-1)} > F_{(a-1)(b-1),ab(n-1)}(\alpha)$  
  则我们拒绝零假设 $H_0:\widehat{\text{IA}}_{(A_i,B_j)} = 0\text{ for all }i,j$  
  说明当前所研究的因子 $A,B$ 之间存在交互效应.



### 5.3.4 无交互效应时的主效应检验

假设交互效应存在性检验的零假设没有被拒绝 (注意: 没有被拒绝 $\neq$ 成立)  
考虑交互效应不存在时双因子方差分析的因子 $A$ 的主效应检验.  
零假设和备择假设分别为:
$$
H_0 : \mu_{1\cdot} = \dotsm = \mu_{a\cdot} (= \mu_{\cdot\cdot})\\
\Updownarrow\\
H_1 : \exists\ i_1,i_2\text{ such that }\mu_{i_1\cdot} \neq \mu_{i_2\cdot}
$$
回忆起因子 $A$ 的因子水平 $A_{i}$ 的主效应的定义为 $\text{ME}_{A_i}:= \mu_{i\cdot}-\mu_{\cdot\cdot}$   
于是零假设和备择假设可以等价表示为:  
$$
H_0 : \text{ME}_{A_1}=\dotsm = \text{ME}_{A_a} = 0\\
\Updownarrow\\
H_1 : \exists\ i=1,\dots,a\text{ such that }\text{ME}_{A_i}\neq 0
$$


#### (1) 基本记号

根据 $5.3.3(2)$ 的结论可知，在不存在交互效应的假设下，  
交互效应平方和 $\text{SSAB}=n\sum_{i,j}^{a,b}\widehat {\text{IA}}_{(A_i,B_j)}^2\sim \sigma^2\chi^2_{(a-1)(b-1)}$     
此时它已经没有单独存在的必要了，我们将它并入 $\text{SSE}$:  
$$
\begin{align}
\text{SSE}_{\text{reduced}}
&:=
\text{SSE} + \text{SSAB}\\
&=
\sum_{i,j,k}^{a,b,n} (y_{ijk} - \bar y_{ij\cdot})^2 + n\sum_{i,j}^{a,b} \widehat {\text{IA}}_{(A_i,B_j)}^2\\
&=
\sum_{i,j,k}^{a,b,n} (y_{ijk} - \bar y_{ij\cdot})^2 + n\sum_{i,j}^{a,b} (\bar y_{ij\cdot} + \bar y_{\cdot\cdot\cdot} - \bar y_{i\cdot\cdot} - \bar y_{\cdot j\cdot})^2\quad (\text{note that }\text{SSE}\perp \text{SSAB})\\
&=
\sum_{i,j,k}^{a,b,n} (y_{ijk} - \bar y_{ij\cdot}+\bar y_{ij\cdot} + \bar y_{\cdot\cdot\cdot} - \bar y_{i\cdot\cdot} - \bar y_{\cdot j\cdot})^2\\
&=
\sum_{i,j,k}^{a,b,n} (y_{ijk} + \bar y_{\cdot\cdot\cdot} - \bar y_{i\cdot\cdot} - \bar y_{\cdot j\cdot})^2
\quad 
(\text{note that }
\begin{cases}
\text{SSE}\sim \sigma^2 \chi^2_{ab(n-1)}\\
\text{SSAB}\sim \sigma^2_{(a-1)(b-1)}
\end{cases})\\
&\sim
\sigma^2\chi^2_{(abn - a - b +1)}
\end{align}
$$
在不存在交互效应的假设下，双因子方差分析的 ANOVA TABLE 如下:
$$
\text{ANOVA  TABLE  (without  interaction)}\\ 
\begin{array}{|c|c|c|c|}  
\hline 
\text{Sum of Squares} &  & \text{Degree of  Freedom} & \text{Mean Squares} \\  
\hline  
\text{SST} & \sum_{i,j,k}^{a,b,n}(y_{ijk}-\bar y_{\cdot\cdot\cdot})^2 & abn-1 & \text{MST} = \frac{\text{SST}}{abn-1} \\  
\hline  
\text{SSE}_{\text{reduced}} & \sum_{i,j,k}^{a,b,n} (y_{ijk} + \bar y_{\cdot\cdot\cdot} - \bar y_{i\cdot\cdot} - \bar y_{\cdot j\cdot})^2 & abn-a-b+1 &  \text{MSE}_{\text{reduced}} = \frac{\text{SSE}_{\text{reduced}}}{abn-a-b+1}\\  
\hline 
\text{SSA} &  bn\sum_{i=1}^a\widehat {\text{ME}}_{A_i}^2 & a-1 & \text{MSA} = \frac{\text{SSA}}{a-1}\\ \hline 
\text{SSB} &  an\sum_{j=1}^b \widehat {\text{ME}}_{B_j}^2 & b-1 & \text{MSB} = \frac{\text{SSB}}{b-1}\\ \hline \end{array}\\
\widehat {\text{ME}}_{A_i} = \bar y_{i\cdot\cdot}-\bar y_{\cdot\cdot\cdot}\\
\widehat {\text{ME}}_{B_j} = \bar y_{\cdot j\cdot} -\bar y_{\cdot\cdot\cdot}\\
$$



#### (2) 零假设分布

考虑零假设 $H_0$:  
$$
H_0 : \text{ME}_{A_1}=\dotsm = \text{ME}_{A_a} = 0
$$
在零假设 $H_0$ 成立的前提条件下，我们有:  
$$
\begin{align}
\widehat {\text{ME}}_{A_i}
&=
\bar y_{i\cdot\cdot} - \bar y_{\cdot\cdot\cdot}\\
&=
\mu_{i\cdot} + \bar \varepsilon_{i\cdot\cdot} - \mu_{\cdot\cdot} - \bar \varepsilon_{\cdot\cdot\cdot}\\
&=
\text{ME}_{A_i} + \bar \varepsilon_{i\cdot\cdot} -  \bar \varepsilon_{\cdot\cdot\cdot}\\
&\overset{H_0}=
\bar \varepsilon_{i\cdot\cdot} -  \bar \varepsilon_{\cdot\cdot\cdot}
\end{align}
$$
定义 $\Epsilon:= [\bar \varepsilon_{ij\cdot}]\in \mathbb R^{a\times b}$，用 $\Epsilon_{(i,:)}$ 代表 $\Epsilon$ 的第 $i$ 行  
注意到: 
$$
\begin{align}
\bar\varepsilon_{i\cdot\cdot}
&= \frac1b \sum_{j=1}^b \bar \varepsilon_{ij\cdot} = \Epsilon_{(i,:)}\cdot \frac1b 1_b\\
\bar \varepsilon_{\cdot\cdot\cdot}
&=
\frac{1}{ab}\sum_{i=1}^{a}\sum_{j=1}^b \bar\varepsilon_{ij\cdot} = \frac{1}{a} 1_a^{\mathrm T} \cdot \Epsilon \cdot
\frac{1}{b} 1_b
\end{align}
$$
因此我们有:  
$$
\begin{align} 
\begin{bmatrix} 
\bar\varepsilon_{1\cdot\cdot}\\ 
\vdots \\ 
\bar \varepsilon_{a\cdot\cdot}\end{bmatrix} 
&= 
\begin{bmatrix} \Epsilon_{(1,:)}\cdot \frac1b 1_b\\ \vdots\\ \Epsilon_{(a,:)}\cdot \frac1b 1_b \end{bmatrix} 
= 
\Epsilon\cdot \frac{1}b1_b1_b^{\mathrm T}\\ 
\begin{bmatrix} \bar\varepsilon_{\cdot\cdot\cdot}\\ \vdots \\ \bar \varepsilon_{\cdot\cdot\cdot} \end{bmatrix}  
&= 1_a \bar\varepsilon_{\cdot\cdot\cdot}  = \frac{1}a1_a1_a^{\mathrm T}\cdot \Epsilon\cdot \frac1b 1_b \end{align}
$$
于是我们有: 
$$
\begin{align}
&(I_a - \frac{1}{a}1_a1_a^{\mathrm T}) \cdot \Epsilon \cdot \frac{1}{b} 1_b\\
&=\Epsilon\cdot \frac{1}b1_b - \frac{1}a1_a1_a^{\mathrm T}\cdot \Epsilon\cdot \frac1b 1_b\\
&=
\begin{bmatrix} 
\bar\varepsilon_{1\cdot\cdot}\\ 
\vdots \\ 
\bar \varepsilon_{a\cdot\cdot}\end{bmatrix} 
-
\begin{bmatrix} 
\bar\varepsilon_{\cdot\cdot\cdot}\\ 
\vdots \\ 
\bar \varepsilon_{\cdot\cdot\cdot}\end{bmatrix} \\
&=
\begin{bmatrix} 
\bar\varepsilon_{1\cdot\cdot} - \bar\varepsilon_{\cdot\cdot\cdot}\\ 
\vdots \\ 
\bar \varepsilon_{a\cdot\cdot} - \bar\varepsilon_{\cdot\cdot\cdot}\end{bmatrix} \\
&\overset{H_0}=
\begin{bmatrix}
\widehat{\text{ME}}_{A_1}\\
\vdots\\
\widehat{\text{ME}}_{A_a}
\end{bmatrix}
\end{align}
$$
记 $P_a =  I_a - \frac1a 1_a1_a^{\mathrm T}$   
显然它是投影算子 (即自伴且幂等)，记其谱分解为:
$$
P_a = Q_a \Lambda_a Q_a^{\mathrm T}
\text{ where }
\Lambda_a = \text{diag}\{\underset{a-1\text{ times}}{\underbrace{1,\dots,1}},0\}
$$
则我们有:  
$$
\begin{align}
\text{SSA}
&=
bn\sum_{i=1}^{a} \widehat{\text{ME}}_{A_i}^2
\quad (\text{note that }\widehat{\text{ME}}_{A_i}\overset{H_0}= 
\bar \varepsilon_{i\cdot\cdot} - \bar \varepsilon_{\cdot\cdot\cdot})\\
&\overset{H_0}= 
bn\sum_{i=1}^{a} (\bar \varepsilon_{i\cdot\cdot} - \bar \varepsilon_{\cdot\cdot\cdot})^2
\quad (\text{note that }\bar \varepsilon_{i\cdot\cdot} - \bar \varepsilon_{\cdot\cdot\cdot}= \left[P_a \Epsilon \cdot \frac1b 1_b \right]_{(i)})\\
&=
bn\sum_{i=1}^{a} \left(\left[P_a \Epsilon \cdot \frac{1}{b}1_b \right]_{(i)}\right)^2\\
&=
bn \left\|P_a \Epsilon \cdot \frac1b1_b \right\|_2^2\\
&=
bn\cdot \frac{1}{b}1_b^{\mathrm T} \Epsilon^{\mathrm T} P_a^{\mathrm T} P_a \Epsilon \cdot \frac{1}{b}1_b \quad (\text{note that }
P_a^{\mathrm T} P_a = P_a^2=P_a)\\
&=
\frac{n}{b} 1_b^{\mathrm T}\Epsilon^{\mathrm T} P_a \Epsilon 1_b\quad (\text{note that }P_a = Q_a \Lambda_a Q_a^{\mathrm T})\\
&=
\frac{n}{b} 1_b^{\mathrm T} \Epsilon^{\mathrm T}Q_a \Lambda_a Q_a^{\mathrm T}\Epsilon  1_b\quad (\text{denote }\tilde e:= Q_a^{\mathrm T}\Epsilon 1_b\in \mathbb R^{a})\\
&=
\frac{n}{b} \tilde e^{\mathrm T} \Lambda_a \tilde e \quad (\text{note that }\Lambda_a = \text{diag}\{\underset{a-1\text{ times}}{\underbrace{1,\dots,1}},0\})\\
&=
\frac{n}{b} \sum_{i=1}^a \tilde e_i^2
\end{align}
$$

注意到 $\Epsilon:= [\bar \varepsilon_{ij\cdot}]\in \mathbb R^{a\times b}$ 的元素独立同分布:  
$$
\{\bar \varepsilon_{ij\cdot}\} \overset{\text{iid}}\sim N\left(0,\frac{\sigma^2}{n}\right)
$$
我们可以将整个 $\Epsilon\in \mathbb R^{a\times b}$ 的分布表示为:  
$$
\text{vec}(\Epsilon) \sim N\left( 0_{ab}, \frac{\sigma^2}{n} I_b \otimes I_a\right)
$$
其中 $\text{vec}(\cdot)$ 是向量化操作符 (即将一个矩阵按列拉伸为向量)，而 $\otimes$ 代表 Kronecker 乘积.  
于是我们有:  
$$
\begin{align}
\widetilde e
&=Q_a^{\mathrm T} \Epsilon 1_b \quad(\text{note that }\text{vec}(AXB) = (B^{\mathrm T}\otimes A)\text{vec}(X))\\
&= 
(1_b^{\mathrm T}\otimes Q_a^{\mathrm T})\text{vec}(\Epsilon)\\
&\sim 
N\left((1_b^{\mathrm T}\otimes Q_a^{\mathrm T}) 0_{ab}, (1_b^{\mathrm T}\otimes Q_a^{\mathrm T})\cdot 
\frac{\sigma^2}{n}I_b\otimes I_a\cdot (1_b^{\mathrm T}\otimes Q_a^{\mathrm T})^{\mathrm T}
\right)\\
&=
N\left(0_{ab}, (1_b^{\mathrm T}\otimes Q_a^{\mathrm T})\cdot 
\frac{\sigma^2}{n}I_b\otimes I_a\cdot (1_b\otimes Q_a)
\right)\\
&=
N\left(0_{ab}, \frac{\sigma^2}{n}[(1_b^{\mathrm T} \cdot I_b\cdot 1_b)\otimes (Q_a^{\mathrm T} \cdot I_a\cdot Q_a)])
\right)\\
&=
N\left( 0_{ab}, \frac{\sigma^2 b}{n}I_a\right)
\end{align}
$$
因此 $\tilde e = Q_a^{\mathrm T} \Epsilon 1_b\in \mathbb R^{a}$ 的元素独立同分布:  
$$
\{\tilde e_i\} \overset{\text{iid}}\sim N\left(0,\frac{\sigma^2b}{n}\right)
$$

于是 $\text{SSA}$ 的零假设分布为:  
$$
\text{SSA} \overset{H_0} 
= \frac{n}{b} \sum_{i=1}^a \tilde e_i^2 
\sim \frac{n}{b}\cdot \frac{\sigma^2b}{n} \chi^2_{(a-1)} = \sigma^2 \chi^2_{(a-1)}\\
\text{MSA} := \frac{\text{SSA}}{a-1} \overset{H_0}\sim \frac{\sigma^2\chi^2_{(a-1)}}{a-1}
$$


#### (3) 检验法

零假设和备择假设分别为: 
$$
H_0 : \text{ME}_{A_1}=\dotsm = \text{ME}_{A_a} = 0\\
\Updownarrow\\
H_1 : \exists\ i=1,\dots,a\text{ such that }\text{ME}_{A_i}\neq 0
$$

设第一类型错误概率界限为 $\alpha$  
我们构造如下的 $F$ 检验统计量:
$$
F:= \frac{\text{MSA}}{\text{MSE}_{\text{reduced}}} \overset{H_0}\sim \frac{\sigma^2\chi^2_{(a-1)}/(a-1)}
{\sigma^2 \chi^2_{abn-a-b+1}/(abn -a-b+1)} = F_{a-1,abn-a-b+1}
$$
其中分子 $\text{MSA}$ 和分母 $\text{MSE}_{\text{reduced}}$ 是相互独立的 (无论零假设 $H_0$ 是否成立)  

记 $F_{a-1,abn-a-b+1}(\alpha)$ 为 $F_{a-1,abn-a-b+1}$ 分布的 $1-\alpha$ 分位数.  
在无交互效应的假设下关于因子 $A$ 的主效应检验的 $F$-检验法为:

- 若 $F= \frac{\text{MSA}}{\text{MSE}_{\text{reduced}}}= \frac{bn\sum_{i=1}^{a}(\bar y_{i\cdot\cdot}-\bar y_{\cdot\cdot\cdot})^2 / (a-1)}{\sum_{i,j,k}^{a,b,n}(y_{ijk}+\bar y_{\cdot\cdot\cdot} - \bar y_{i\cdot\cdot} - \bar y_{\cdot j\cdot})^2/(abn-a-b+1)} > F_{a-1,abn-a-b+1}(\alpha)$  
  则我们拒绝零假设 $H_0 : \text{ME}_{A_1}=\dotsm = \text{ME}_{A_a} = 0$  
  即拒绝零假设 $H_0: \mu_{1\cdot} = \dots = \mu_{a\cdot}$    
  说明在无交互效应的假设下因子 $A$ 对响应变量 $Y$ 有解释作用.



### 5.3.5 置信区间估计

给定置信水平为 $\alpha$   
回忆起 $\hat \mu_{ij}:= \bar y_{ij\cdot} = \frac{1}{n}\sum_{k=1}^n y_{ijk}\sim N(\mu_{ij},\frac{\sigma^2}{n})$ 是 $\mu_{ij}$ 的最佳线性无偏估计量 $(\text{BLUE})$  
因此我们有:  
$$
\frac{\sqrt{n}(\hat \mu_{ij}-\mu_{ij})}{\sigma} = \frac{\sqrt{n}(\bar y_{ij\cdot} - \mu_{ij})}{\sigma} \sim N(0,1)
$$
关于 $\sigma^2$ 的无偏估计量，在不同假设下可以有不同选择.  
(类似地，我们可以对 $\mu_{i\cdot}$ 和 $\mu_{\cdot j}$ 及其线性组合进行置信区间估计)

#### (1) 有交互效应

在有交互效应的假设下，我们使用 $s^2=\text{MSE}$ 作为 $\sigma^2$ 的无偏估计量.  
根据 $5.3.2(3)$ 可知:  
$$
s^2=\text{MSE} = \frac{1}{ab(n-1)}\text{SSE} = \frac{1}{ab(n-1)}\sum_{i,j,k}^{a,b,n} (y_{ijk}-\bar y_{ij\cdot})^2 \sim \frac{\sigma^2\chi^2_{(ab(n-1))}}{ab(n-1)}\\

s^2\perp \mu_{ij}\text{ for all }
\begin{cases}
i=1,\dots,a\\
j=1,\dots,b
\end{cases}
$$
于是我们可以考虑以下枢轴量:  
$$
\frac{\sqrt{n}(\hat \mu_{ij}-\mu_{ij})}{\sqrt{\text{MSE}}}
=
\frac{\frac{\sqrt{n}(\hat \mu_{ij}-\mu_{ij})}{\sigma}}{\frac{\sqrt{\text{MSE}}}{\sigma}}
\sim 
\frac{N(0,1)}{\sqrt{\chi^2_{(ab(n-1))}/(ab(n-1))}} = t_{ab(n-1)}
$$
因此 $\mu_{ij}$ 的 $(1-\alpha)\cdot 100\%$ 置信区间为 $\bar y_{ij\cdot} \pm \frac{\sqrt{\text{MSE}}}{\sqrt{n}}t_{ab(n-1)}(\frac{\alpha}{2})$  
其中 $t_{ab(n-1)}(\frac{\alpha}{2})$ 为 $t_{ab(n-1)}$ 分布的 $1-\frac{\alpha}{2}$ 分位数.



#### (2) 无交互效应

在无交互效应的假设下，我们使用 $\text{MSE}_{\text{reduced}}$ 作为 $\sigma^2$ 的无偏估计量.  
根据 $5.3.4(1)$ 可知 $\text{MSE}_{\text{reduced}}$ 在无交互效应的假设下的分布为:  
$$
\begin{align}
\text{MSE}_{\text{reduced}} 
&= \frac{1}{abn-a-b+1} \text{SSE}_{\text{reduced}}\\
&= \frac{1}{abn-a-b+1} \sum_{i,j,k}^{a,b,n} (y_{ijk} + \bar y_{\cdot\cdot\cdot} - \bar y_{i\cdot\cdot} - \bar y_{\cdot j\cdot})^2\\
&\sim 
\frac{\sigma^2\chi^2_{(abn-a-b+1)}}{abn-a-b+1}
\end{align}
$$
**(待补充: $\text{MSE}_{\text{reduced}}\perp \bar y_{ij\cdot}$ 的证明)**  
于是我们可以考虑以下枢轴量:  
$$
\frac{\sqrt{n}(\hat \mu_{ij}-\mu_{ij})}{\sqrt{\text{MSE}_{\text{reduced}}}}
=
\frac{\frac{\sqrt{n}(\hat \mu_{ij}-\mu_{ij})}{\sigma}}{\frac{\sqrt{\text{MSE}_{\text{reduced}}}}{\sigma}}
\sim 
\frac{N(0,1)}{\sqrt{\chi^2_{(abn-a-b+1)}/(abn-a-b+1)}} = t_{abn-a-b+1}
$$
因此 $\mu_{ij}$ 的 $(1-\alpha)\cdot 100\%$ 置信区间为 $\bar y_{ij\cdot} \pm \frac{\sqrt{\text{MSE}_{\text{reduced}}}}{\sqrt{n}}t_{abn-a-b+1}(\frac{\alpha}{2})$  
其中 $t_{abn-a-b+1}(\frac{\alpha}{2})$ 为 $t_{abn-a-b+1}$ 分布的 $1-\frac{\alpha}{2}$ 分位数.

**The End**
