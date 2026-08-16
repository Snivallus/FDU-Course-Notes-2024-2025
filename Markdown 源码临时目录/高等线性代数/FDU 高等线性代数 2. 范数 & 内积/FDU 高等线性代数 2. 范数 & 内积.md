# FDU 高等线性代数 2. 范数 & 内积

本文根据邵美悦老师课堂笔记及纲要整理而成，并参考了以下资料:

* Matrix Analysis (R. Horn & C. Johnson) Chapter $5$ & Appendix $\text{E}$
* 矩阵分析 (R. Horn & C. Johnson) 第 $5$ 章, 附录 $\text{E}$
* [Matrix Norms and Spectral Radius (drexel.edu)](https://www.math.drexel.edu/~foucart/TeachingFiles/F12/M504Lect6.pdf)
* [Norms for Vectors and Matrices](https://www.cis.upenn.edu/~cis5150/cis515-11-sl4.pdf)
* 数值线性代数 (第 $2$ 版, 徐树方, 高立, 张平文) 第 $2.1$ 节
* Nonlinear Programming (D. Bertsekas) Appendix $\text{A}$
* 非线性规划 (D. Bertsekas) 附录 $\mathrm{A}$

欢迎批评指正!

## 2.1 赋范空间

### 2.1.1 范数

设 $V$ 是建立在域 $\mathbb F = \mathbb R\ \text{or }\mathbb C$ 上的线性空间.  
函数 $\|\cdot\|:V\mapsto \mathbb R$ 称为一个**范数** (norm)，如果它满足下列四条公理:  
对于任意 $x,y\in V$ 和 $\alpha\in \mathbb F$

- ① 非负性: $\|x\|\geq 0$    
- ② 正定性: $\|x\|=0$ 当且仅当 $x=0_V$   
- ③ 齐次性: $\|\alpha x\| = |\alpha| \|x\|$ 
- ④ 次可加性: $\|x+y\|\leq \|x\| + \|y\|$ (又称为三角不等式)

若函数 $\|\cdot\|:V\mapsto \mathbb R$ 不满足正定性，但满足其他三个性质，  
则称其为**半范数** (semi-norm).  
非零向量的半范数可能为零.

一个实的或复的线性空间 $V$ 与一个给定的范数 $\|\cdot\|:V\mapsto \mathbb R$ 合在一起，  
就称为一个**赋范空间** (normed linear space)  
此时我们可以定义函数 $d(\cdot,\cdot) :V\times V\mapsto \mathbb R$ 为 $d(x,y)=\|x-y\|$​，它满足:  

- 非负性: $d(x,y)\geq 0$ 当且仅当 $x=y$ 时取等.
- 对称性: $d(x,y)=d(y,x)$
- 三角不等式: $d(x,z)\leq d(x,y) + d(y,z)$ 

因此它是空间 $V$ 上的**度量 **(metric)，  
我们称空间 $V$ 和度量 $d(\cdot,\cdot)$ 构成一个**度量空间** (metric space).  
(注意度量空间和线性空间是两个独立的概念)  
这表明**赋范空间一定是度量空间**.

***

**(完备的三角不等式, Matrix Analysis 引理 $5.1.2$)**  
设 $V$ 是建立在域 $\mathbb F = \mathbb R\ \text{or }\mathbb C$ 上的线性空间.  
若函数 $\|\cdot\|:V\mapsto \mathbb R$ 是一个半范数，则我们有: 
$$
|\|x\|-\|y\|| \leq \|x-y\|\ \ (\forall\ x,y\in V),
$$
当且仅当 $x$ 是 $y$ 的非正实数倍时取等. 

- **证明:**  
  任意给定 $x,y\in V$.  
  一方面，根据 $y=x + (y-x)$ 我们有:  
  $$
  \|y\| \leq \|x\| + \|y-x\| = \|x\| + \|x-y\|\\
  \Updownarrow\\
  \|y\|-\|x\|\leq \|x-y\|
  $$
  另一方面，根据 $x=y+(x-y)$ 我们有:  
  $$
  \|x\| \leq \|y\| + \|x-y\| = \|y\| + \|y-x\|\\
  \Updownarrow\\
  \|x\|-\|y\| \leq \|y-x\|
  $$
  综上所述，我们有 $|\|x\|-\|y\|| \leq \|x-y\|\ \ (\forall\ x,y\in V)$ 成立.

****

从给定的范数出发，我们可以用若干种方法构造新的范数，例如:

- 两个范数的和是一个范数.
- 一个范数的任意正的倍数还是一个范数.
- 若 $\|\cdot\|_{\alpha}$ 和 $\|\cdot\|_\beta$ 是范数，则由 $\|x\|:= \max\{\|x\|_\alpha,\|x\|_\beta\}$ 定义的函数 $\|\cdot\|$ 也是范数.

上述结果都是如下定理的特例.  
**(Matrix Analysis 定理 $5.3.1$)**  
设 $\|\cdot\|_{\alpha_1},\dots,\|\cdot\|_{\alpha_m}$ 是域 $\mathbb F = \mathbb R \text{ or }\mathbb C$ 上的向量空间 $V$ 上给定的范数.  
令 $\|\cdot\|$ 是 $\mathbb {C}^m$ 上一个对所有满足 $u\leq v$ 的非负向量 $u,v$ 都满足 $\|u\|\leq \|v\|$ 的范数，  
(其中 $u\leq v$ 代表 $u$ 的每个分量都小于等于 $v$ 的对应分量，这是非负矩阵理论中的惯例)  
那么由 $f(x):=\|[\|x\|_{\alpha_1},\dots,\|x\|_{\alpha_m}]^{\mathrm T}\|$ 定义的函数 $f:V\mapsto \mathbb R$ 也是 $V$ 上的一个范数.

上述定理中对 $\|\cdot\|$ 的单调性要求是用来保证函数 $f$ 满足三角不等式的:  
$$
\begin{align}
f(x+y)
&=
\left\| 
\begin{bmatrix}
\|x + y\|_{\alpha_1}\\
\vdots\\
\|x + y\|_{\alpha_m}
\end{bmatrix}
\right\|\\
&\quad(\text{note that }\|u\|\leq \|v\|\ \text{whenever nonnegative vectors }u,v \text{ satisfy }u\leq v)\\

&\leq

\left\| 
\begin{bmatrix}
\|x\|_{\alpha_1} + \|y\|_{\alpha_1}\\
\vdots\\
\|x\|_{\alpha_m} + \|y\|_{\alpha_m}
\end{bmatrix}
\right\|\\

&=

\left\| 
\begin{bmatrix}
\|x\|_{\alpha_1}\\
\vdots\\
\|x\|_{\alpha_m}
\end{bmatrix}
+
\begin{bmatrix}
\|y\|_{\alpha_1}\\
\vdots\\
\|y\|_{\alpha_m}
\end{bmatrix}
\right\|\\

&\leq
\left\|
\begin{bmatrix}
\|x\|_{\alpha_1}\\
\vdots\\
\|x\|_{\alpha_m}
\end{bmatrix}
\right\|
+
\left\|
\begin{bmatrix}
\|y\|_{\alpha_1}\\
\vdots\\
\|y\|_{\alpha_m}
\end{bmatrix}
\right\|\\

&=
f(x) + f(y).
\end{align}
$$
显然 $l_p$ 范数满足这一单调性要求，但也有某些范数不满足. 
考虑 $m=2$ 的情况，容易验证 $\|x\|:= |x_1-x_2| + |x_2|$ 是 $\mathbb R^2$ 上不满足单调性要求的一个范数.  
例如取 $u=[0,1]^{\mathrm{T}}$ 和 $v = [1,1]^{\mathrm{T}}$ 时，  
我们有 $\|u\| = 2 > 1 = \|v\|$，说明 $\|\cdot\|$ 不满足单调性要求.  
因此函数
$$
\begin{align}
f(x)
&:=\|[\|x\|_\infty,\|x\|_1]^{\mathrm T}\|\\ 
&= |\|x\|_\infty - \|x\|_1| + |\|x\|_1|\\
&= (|x_1| + |x_2| - \max\{|x_1|,|x_2|\}) + |x_1| + |x_2|\\
&= \min\{|x_1|,|x_2|\} + |x_1| + |x_2|
\end{align}
$$
并不满足三角不等式，因而不是一个范数.



### 2.1.2 $l_p$ 范数

任意给定 $p\geq 1$，我们定义 $\mathbb C^n$ 上的 $l_p$ 范数为:
$$
\|x\|_p:= \left(\sum_{i=1}^n |x_i|^p\right)^{1/p} = (|x_1|^p + \dots + |x_n|^p)^{\frac1p}.
$$

- **① 任意给定 $x\in \mathbb C^n$，范数 $\|x\|_p$ 都关于 $p$ 连续且严格单调递减.** 

  <img src="lp 范数关于p递减.jpg" style="zoom:40%;" />

  这幅图也可以形象地展示范数的等价性 (参见后文)，  
  因为范数的单位球之间可以通过缩放而相互包含.

- **② 不同 $l_p$ 范数之间转换时的界.** (这也是范数等价性的体现)  
  任意给定 $1<p<q$，我们有:
  $$
  \|x\|_q \leq \|x\|_p \leq n^{\frac1p-\frac1q}\|x\|_q\ \ (\forall\ x\in \mathbb C^n).
  $$
  对于三种常用的 $l_p$ 范数，我们有:
  $$
  \begin{cases} \|x\|_2 \leq \|x\|_1 \leq n^{\frac12}\|x\|_2\\ \|x\|_\infty \leq \|x\|_1 \leq n\|x\|_\infty\\ \|x\|_\infty \leq \|x\|_2 \leq n^\frac12\|x\|_\infty. \end{cases}
  $$

- **③ $l_p$ 范数是置换不变的 (permutation invariant)**  
  换言之，对于任意 $x\in \mathbb C^n$ 和排列矩阵 $P\in \mathbb C^{n\times n}$，都有 $\|Px\|_p = \|x\|_p$ 成立.

***

三种常用的 $l_p$ 范数:

- $l_1$ 范数，又称**和范数**或 Manhattan 范数: 
  $$
  \|x\|_1 := \sum_{i=1}^n |x_i|\ \ (\forall\ x\in \mathbb C^n).
  $$
  
- $l_2$ 范数，又称 **Euclid 范数**:
  $$
  \|x\|_2 := \left(\sum_{i=1}^n |x_i|^2\right)^{1/2}\ \ (\forall\ x\in \mathbb C^n).
  $$
  它是**酉不变的** (unitarily invariant)，  
  即对于任意 $x\in \mathbb C^n$ 和酉矩阵 $U\in \mathbb C^{n\times n}$ 都有 $\|Ux\|_2 = \|x\|_2$ 成立.  
  事实上，$l_2$ 范数的正倍数是 $\mathbb C^n$ 上仅有的酉不变范数.
  
- $l_\infty$ 范数，又称**最大值范数**:
  $$
  \|x\|_1 := \max_{i=1,\dots,n}|x_i|\ \ (\forall\ x\in \mathbb C^n).
  $$

****

其他有用的范数:

- $k$-范数是 $\mathbb C^n$ 上的一个重要的离散范数族，它填补了 $l_1$ 范数和 $l_\infty$ 范数之间的空隙.  
  对于任意 $k=1,\dots,n$，向量 $x\in \mathbb C^n$ 的 $k$-范数是 $x$ 的前 $k$ 大分量模长的和:  
  $$
  \|x\|_{[k]}:= |x_{\pi(1)}| + \dotsm + |x_{\pi(k)}|\text{where }\pi\text{ such that }|x_{\pi(1)}|\geq \dotsm \geq |x_{\pi(n)}|
  $$
  它们在酉不变相容范数的理论中起着重要作用 (Matrix Analysis 7.4.7 节)  

  显然 $k$-范数是置换不变的.  
  换言之，对于任意 $x\in \mathbb C^n$ 和排列矩阵 $P\in \mathbb C^{n\times n}$，都有 $\|Px\|_{[k]} = \|x\|_{[k]}$ 成立.

- 给定 $\mathbb C^m$ 上的一个范数 $\|\cdot\|$ 和矩阵 $A\in \mathbb C^{m\times n}$   
  定义函数 $\|\cdot\|_A : \mathbb C^n\mapsto \mathbb R$ 为 $\|x\|_A:=\|Ax\|\ (\forall\ x\in \mathbb C^n)$  
  若 $A\in \mathbb C^{m\times n}$ 是列满秩的，则函数 $\|\cdot\|_A$ 是 $\mathbb C^n$ 上的范数.



### 2.1.3 Hölder 不等式

**(凸函数的 Jensen 不等式)**  
若 $f:\mathbb R^n\mapsto \mathbb R$ 是一个凸函数，  
则对于任意 $\alpha\in (0,1)$ 和不相等的 $x_1,x_2\in \text{dom}(f)$，  
我们都有 $f(\alpha x_1 + (1-\alpha)x_2) < \alpha f(x_1) + (1-\alpha) f(x_2)$ 成立.

很多著名的不等式都可以通过将 Jensen 不等式应用于具体的凸函数得到.  
事实上，凸函数和 Jensen 不等式可以构成不等式理论的基础.

****

**(Hölder 不等式)**    
若 $p,q>1$ 为共轭子标，满足 $1/p + 1/q=1$，  
则对于任意 $x,y\in \mathbb{C}^n$ 都有 $|\langle x,y\rangle| = |x^{\mathrm H}y|\leq \|x\|_p\|y\|_q$，  
当且仅当存在 $\theta\in \mathbb{R}$ 使得 $|\bar{x}_iy_i| = \mathrm{e}^{\mathrm{i}\theta} |x_i||y_i|\ (i=1,\dots,n)$ 且 $|x|^p,|y|^q$ 线性相关时取等，  
其中 $\langle \cdot,\cdot \rangle$ 代表 Euclid 内积.

- 在泛函分析中，我们可以将有限维复 Euclid 空间上的 Hölder 不等式推广至级数形式和积分形式.

**证明:**  
对凸函数 $f(z)=-\log(z)$ 应用 Jensen 不等式可得:  
$$
-\log(\alpha z_1+(1-\alpha) z_2) \leq -\alpha\log(z_1)-(1-\alpha) \log(z_2) = -\log(z_1^\alpha z_2^{1-\alpha})\ \ \ (\forall\ z_1,z_2>0).
$$
不等式两边同时取指数，则有:
$$
\alpha z_1+(1-\alpha)z_2 \geq z_1^\alpha z_2^{(1-\alpha)}\quad (\forall\ z_1,z_2>0).
$$
特殊地，当 $\alpha=1/2$ 时即为**算术\-几何平均不等式** $(z_1+z_2)/2 \geq \sqrt{z_1z_2}$. 

若 $x,y$ 至少有一个是零向量，则命题显然成立.  
任意给定 $x,y\in \mathbb{C}^n/\{0_n\}$ 和 $i=1,\dots,n$，取
$$
z_1 = \frac{|x_i|^p}{\sum_{j=1}^n|x_j|^p},\quad 
z_2 = \frac{|y_i|^q}{\sum_{j=1}^n|y_j|^q},\quad\alpha = \frac1p,
$$
则我们有:
$$
\left(\frac{|x_i|^p}{\sum_{j=1}^n|x_j|^p}\right)^\frac1p
\left(\frac{|y_i|^q}{\sum_{j=1}^n|y_j|^q}\right)^{1-\frac1p}
\leq
\frac1p \cdot \frac{|x_i|^p}{\sum_{j=1}^n|x_j|^p} + \left(1-\frac1p\right)\cdot\frac{|y_i|^q}{\sum_{j=1}^n|y_j|^q},
$$
左右两式同时对 $i=1,\dots,n$ 求和:
$$
\begin{align}
\text{LHS}
&=
\sum_{i=1}^n
\left(\frac{|x_i|^p}{\sum_{j=1}^n|x_j|^p}\right)^{\frac1p}
\left(\frac{|y_i|^q}{\sum_{j=1}^n|y_j|^q}\right)^{1-\frac1p}\quad (\text{note that }1-\frac{1}{p} = \frac1q)\\

&=
\sum_{i=1}^n 
\left(\frac{|x_i|^p}{\sum_{j=1}^n|x_j|^p}\right)^{\frac1p}
\left(\frac{|y_i|^q}{\sum_{j=1}^n|y_j|^q}\right)^{\frac1q}\\

&=
\frac{\sum_{i=1}^n |x_i||y_i|}{(\sum_{j=1}^n |x_j|^p)^{1/p}(\sum_{j=1}^n |y_j|^q)^{1/q}}\\
&=
\frac{|x|^{\mathrm{H}}|y|}{\|x\|_p \|y\|_q}\\

\text{RHS}
&=
\sum_{i=1}^n 
\left\{
\frac1p \cdot \frac{|x_i|^p}{\sum_{j=1}^n|x_j|^p} + \left(1-\frac1p\right)\cdot\frac{|y_i|^q}{\sum_{j=1}^n|y_j|^q}
\right\}\quad (\text{note that }1-\frac{1}{p} = \frac1q)\\

&=
\sum_{i=1}^n 
\left\{
\frac1p \cdot \frac{|x_i|^p}{\sum_{j=1}^n|x_j|^p} + \frac{1}{q}\cdot\frac{|y_i|^q}{\sum_{j=1}^n|y_j|^q}
\right\}\\

&=

\frac1p \cdot \frac{\sum_{i=1}^n |x_i|^p}{\sum_{j=1}^n|x_j|^p} + \frac{1}{q}\cdot\frac{\sum_{i=1}^n |y_i|^q}{\sum_{j=1}^n|y_j|^q}\\

&=
\frac1p + \frac1q \\
&=
1
\end{align}
$$
便得到: 
$$
\frac{|x|^{\mathrm{H}}|y|}{\|x\|_p \|y\|_q} = \text{LHS} \leq \text{RHS} = 1\\
\Updownarrow\\
|x|^{\mathrm{H}}|y| \leq \|x\|_p \|y\|_q,
$$
当且仅当 $|x|^p,|y|^q$ 线性相关时取等.  
于是对于任意 $x,y\in \mathbb{C}^n/\{0_n\}$，我们都有:
$$
\begin{align}
|x^{\mathrm H} y|
&=
\left|\sum_{i=1}^n \bar x_iy_i\right|\quad\ \  (\text{triangle inequality})\\
&\leq
\sum_{i=1}^n |\bar x_i y_i|\\
&= 
\sum_{i=1}^n |x_i||y_i|\\
&=
|x|^{\mathrm{H}}|y|\qquad\ \ \  (\text{use the conclusion above})\\
&\leq 
\|x\|_p \|y\|_q,
\end{align}
$$
当且仅当存在 $\theta\in \mathbb{R}$ 使得 $|\bar{x}_iy_i| = \mathrm{e}^{\mathrm{i}\theta} |x_i||y_i|\ (i=1,\dots,n)$ 且 $|x|^p,|y|^q$ 线性相关时取等.

****

**(反 Hölder 不等式)**  
若 $p\in (0,1)$ 和 $q<0$ 为共轭子标，满足 $1/p + 1/q=1$，  
则对于任意 $x,y\in \mathbb{C}^n$ 都有 $|x^{\mathrm H}y|\geq \|x\|_p\|y\|_q$ 成立.  
其中 $\|\cdot\|_p$ 是由 $\|x\|_p:= (\sum_{i=1}^n |x_i|^p )^{1/p}\ (\forall\ x\in \mathbb{C}^n)$ 定义的函数.

**证明:**  
定义 $p' = 1/p>1$ 和 $q'>0$，满足 $1/p' + 1/q'=1$.   
解得 $q' = 1/(1-p)$，因此 $pq' = p/(1-p) = -q$.

若 $x,y$ 至少有一个是零向量，则命题显然成立.  
任意给定 $x,y\in \mathbb{C}^n/\{0_n\}$ 和 $i=1,\dots,n$，我们都有:
$$
\begin{align}
\|x\|_p^p 
&=
\sum_{i=1}^n |x_i|^p \\
&=
\sum_{i=1}^n |\bar{x}_i|^p \\
&=
\sum_{i=1}^n |\bar{x}_i y_i|^{p} |y_i|^{-p}\quad (\text{Holder inequality})\\
&\leq
\left(\sum_{i=1}^n |\bar{x}_iy_i|^{pp'}\right)^{\frac1{p'}} 
\left(\sum_{i=1}^n |y_i|^{-pq'}\right)^{\frac{1}{q'}}\quad (\text{note that }p' = \frac1p,\ pq' = -q,\ q' = 1/(1-p) = -\frac{q}{p})\\
&=
\left(\sum_{i=1}^n |\bar{x}_iy_i|\right)^{p} 
\left(\sum_{i=1}^n |y_i|^{q}\right)^{-\frac{p}{q}}\\
&=
|x^{\mathrm{H}} y|^p \cdot \|y\|_q^{-p}.
\end{align}
$$
因此我们有:
$$
|x^{\mathrm{H}}y|^p \geq \|x\|_p^p \|y\|_q^p\\
\Updownarrow\\
|x^{\mathrm{H}}y| \geq \|x\|_p \|y\|_q\\
$$
命题得证.



### 2.1.4 Minkowski 不等式

**(Minkowski 不等式,  $l_p$ 范数的次可加性)**  
任意给定 $p\geq 1$ 和 $x,y\in \mathbb{C}^n$，   
我们都有 $\|x+y\|_p\leq \|x\|_p + \|y\|_p$ 成立.

- 在泛函分析中，我们可以将有限维复 Euclid 空间上的 Minkowski 不等式推广至级数形式和积分形式.

**证明:**  
当 $p=1$ 时，我们有:
$$
\begin{align}
\|x+y\|_1 
&= \sum_{i=1}^n |x_i + y_i|\\ 
&\leq \sum_{i=1}^n (|x_i|+|y_i|)\\ 
&= \sum_{i=1}^n |x_i| + \sum_{i=1}^n |y_i|\\ 
&= \|x\|_1 + \|y\|_1
\end{align}
$$
当 $p>1$ 时，令 $q>1$ 满足 $1/p+1/q=1$，解得 $q=p/(p-1)$.  
任意给定 $x,y\in \mathbb C^{n}$.  
若 $x,y$ 均为零向量，则命题显然成立.  
若 $x,y$ 至少有一个不是零向量，则我们有:
$$
\begin{align}
\sum_{i=1}^n (|x_i| + |y_i|)^p
&=
\sum_{i=1}^n |x_i|(|x_i| + |y_i|)^{p-1} + \sum_{i=1}^n |y_i|(|x_i| + |y_i|)^{p-1}\quad (\text{Holder inequality})\\
&\leq
\left(\sum_{i=1}^n |x_i|^p\right)^{\frac1p} \left(\sum_{i=1}^n (|x_i| + |y_i|)^{(p-1)q}\right)^{\frac1q}
+
\left(\sum_{i=1}^n |y_i|^p\right)^{\frac1p} \left(\sum_{i=1}^n (|x_i| + |y_i|)^{(p-1)q}\right)^{\frac1q}\\
&=
(\|x\|_p + \|y\|_p)
\left(\sum_{i=1}^n (|x_i| + |y_i|)^{(p-1)q}\right)^{\frac1q}\quad (\text{note that }(p-1)q=p\text{ and }\frac1q = 1-\frac1p)\\
&=
(\|x\|_p + \|y\|_p) \left(\sum_{i=1}^n (|x_i| + |y_i|)^{p}\right)^{1-\frac1p}
\end{align}
$$
于是我们有:
$$
\left(\sum_{i=1}^n (|x_i| + |y_i|)^p\right)^{\frac1p} \leq \|x\|_p + \|y\|_p,
$$
进而有:
$$
\begin{align}
\|x+y\|_p
&=
\left(\sum_{i=1}^n |x_i + y_i|^p\right)^{\frac1p}\\ 
&\leq 
\left(\sum_{i=1}^n (|x_i| + |y_i|)^p\right)^{\frac1p}\\ 
&\leq \|x\|_p + \|y\|_p
\end{align}
$$
综上所述，命题得证.

***

**(反 Minkowski 不等式)**  
给定 $0<p<1$ 和 $x,y\in \mathbb{C}^n$.  
若存在 $\lambda_1,\dots,\lambda_n\geq 0$ 使得 $y_i = \lambda_i x_i\ (\forall\ i=1,\dots,n)$，  
则我们有 $\|x+y\|_p \geq \|x\|_p + \|y\|_p$ 成立.  
这说明由 $\|x\|_p:= (\sum_{i=1}^n |x_i|^p )^{1/p}\ (\forall\ x\in \mathbb{C}^n)$ 定义的函数 $\|\cdot\|_p$ 在 $p\in (0,1)$ 时不是范数.

**证明:**    
任意给定 $x,y\in \mathbb{C}^n$.  
若 $x,y$ 均为零向量，则命题显然成立.   
若 $x,y$ 至少有一个不是零向量，则我们有:
$$
\begin{align}
\sum_{i=1}^n (|x_i| + |y_i|)^p
&=
\sum_{i=1}^n |x_i|(|x_i| + |y_i|)^{p-1} + \sum_{i=1}^n |y_i| (|x_i| + |y_i|)^{p-1}
\quad (\text{Reverse Holder inequality})\\
&\geq
\left(\sum_{i=1}^n |x_i|^p\right)^{\frac1p} 
\left(\sum_{i=1}^n (|x_i| + |y_i|)^{(p-1)q}\right)^{\frac1q}
+
\left(\sum_{i=1}^n |y_i|^p\right)^{\frac1p} 
\left(\sum_{i=1}^n (|x_i| + |y_i|)^{(p-1)q}\right)^{\frac1q}\\
&\quad (\text{note that }(p-1)q=p\text{ and }\frac1q = 1-\frac1p)\\

&=
(\|x\|_p + \|y\|_p) \left(\sum_{i=1}^n (|x_i| + |y_i|)^{p}\right)^{1-\frac1p}
\end{align}
$$

于是我们有:
$$
\left(\sum_{i=1}^n (|x_i| + |y_i|)^p\right)^{\frac1p} \geq \|x\|_p + \|y\|_p.
$$
回顾题设条件，由于存在 $\lambda_1,\dots,\lambda_n\geq 0$ 使得 $y_i = \lambda_i x_i\ (\forall\ i=1,\dots,n)$， 
故我们有 $|x_i + y_i| = |x_i| + |y_i|\ (\forall\ i=1,\dots,n)$ 成立.   
因此我们有:
$$
\begin{align}
\|x+y\|_p 
&= 
\left(\sum_{i=1}^n (|x_i + y_i|)^p\right)^{\frac1p}\\
&=
\left(\sum_{i=1}^n (|x_i| + |y_i|)^p\right)^{\frac1p}\\ 
&\geq \|x\|_p + \|y\|_p.
\end{align}
$$
命题得证.




### 2.1.5 范数的等价性

在实际应用中，用于建立理论的范数和在给定情形中最容易计算的范数可能并不相同.  
(例如有时我们会用 $l_1$ 范数或 $l_2$ 范数建立理论，而用 $l_\infty$ 范数进行计算)  

但这可能会带来某些问题:  
例如一个序列关于某个范数收敛，但关于另一个范数可能就发散了 (Matrix Analysis 例 $5.4.2$) 

> (Matrix Analysis 定义 $5.4.1$)
> 设 $\|\cdot\|$ 是域 $\mathbb F=\mathbb R\text{ or }\mathbb C$ 上的向量空间 $V$ (不必是有限维) 上的一个范数.  
> 我们称向量序列 $\{x^{(k)}\}$ 关于范数 $\|\cdot\|$ 收敛于 $x\in V$，当且仅当 $\underset{k\to\infty}{\lim} \|x^{(k)}-x\|=0$   
> 此时我们记 "关于范数 $\|\cdot\|$ 有 $\underset{k\to\infty}{\lim} x^{(k)}=x$" 

**(Matrix Analysis 推论 $5.4.6$)**  
幸运的是，在有限维赋范空间中，所有范数在某种加强的意义下都是 "等价的"  
换言之，有限维实或复向量空间中向量序列的收敛性与所采用的范数无关.

**(Matrix Analysis 引理 $5.4.3$)**  
设 $\|\cdot\|$ 是域 $\mathbb F=\mathbb R\text{ or }\mathbb C$ 上的向量空间 $V$ (不必是有限维) 上的一个范数，$m\geq 1$ 为给定的正整数.  
给定向量 $x^{(1)},\dots,x^{(m)}\in V$，并定义 $x(z):= z_1x^{(1)} + \dotsm + z_m x^{(m)}\ (\forall\ z\in \mathbb F^m)$   
则由 $g(z):= \|x(z)\|$ 定义的函数 $g:\mathbb F^m\mapsto \mathbb R$ 在 $\mathbb F^m$ 上关于 $l_2$ 范数一致连续.

- **证明:**  
  对于任意 $u,v\in \mathbb F^m$ 我们都有:  
  $$
  \begin{align}
  |g(u) - g(v)|
  &=
  |\|x(u)\| - \|x(v)\||\\
  &\leq
  \|x(u)-x(v)\|\\
  &=
  \left\|\sum_{i=1}^m u_i x^{(i)} - \sum_{i=1}^m v_i x^{(i)}\right\|\\
  &= 
  \left\|\sum_{i=1}^m (u_i-v_i) x^{(i)}\right\|\\
  &\leq
  \sum_{i=1}^m |u_i-v_i| \|x^{(i)}\|\quad (\text{Cauchy–Schwarz inequality})\\
  &\leq
  \left(\sum_{i=1}^m |u_i-v_i|^2\right)^{\frac12} 
  \left(\sum_{i=1}^m \|x^{(i)}\|^2\right)^{\frac12}\\
  &=
  C\|u-v\|_2
  \end{align}
  $$
  其中 $C=(\sum_{i=1}^m \|x^{(i)}\|^2)^{\frac12}$ 是仅与给定范数 $\|\cdot\|$ 和给定向量 $x^{(1)},\dots,x^{(m)}\in V$ 有关的常数.  
  若 $x^{(1)},\dots,x^{(m)}$ 均为零向量，则 $C=0$，从而 $g(z)\equiv 0$，显然一致连续.  
  若存在某个 $x^{(i)}$ 不为零向量，则 $C>0$   
  因此对于任意 $\varepsilon>0$，只要 $\|u-v\|_2<\frac{\varepsilon}{C}$ (这个界与 $u,v$ 无关) 就有 $|g(u) - g(v)|<\varepsilon$  
  这表明函数 $g:\mathbb F^m\mapsto \mathbb R$ 在 $\mathbb F^m$ 上关于 $l_2$ 范数一致连续.

****

上一个引理中赋范空间 $V$ 不一定是有限维的，但 $V$ 的有限维度对下面的定理很重要.  
**(Matrix Analysis 定理 $5.4.4$)**  
设 $f_1,f_2$ 是域 $\mathbb F=\mathbb R\text{ or }\mathbb C$ 上的 $n$ 维向量空间 $V$ 上的实值函数. 
若 $f_i\ (i=1,2)$ 满足:

- 非负性: $f_i(x)\geq 0\ (\forall\ x\in V)$ 
- 正定性: $f_i(x)=0$ 当且仅当 $x=0_V$ 
- 齐次性: $f_i(\alpha x) = |\alpha|f_i(x)\ (\forall\ x\in V,\alpha\in \mathbb F)$ 
- 连续性: $f_i(x)$ 在 $\mathbb F^n$ 上关于 Euclid 范数是连续的

(事实上我们称满足上述性质的 $f_1,f_2$ 称为**准范数** (pre-norm)，满足三角不等式的准范数就是范数)  
则存在有限的常数 $C_\min,C_\max>0$ 使得: 
$$
C_\min f_1(x) \leq f_2(x) \leq C_\max f_1(x)\ \ (\forall\ x\in V)
$$
> **(Weierstrass 定理, Nonlinear Programming 命题 A.8)**  
> 设 $\mathcal X$ 为 $\mathbb F=\mathbb R \text{ or }\mathbb C$ 上的有限维赋范空间 $V$ 的非空子集  
> 且 $f:\mathcal X\mapsto \mathbb R$ 在 $\mathcal X$ 处下半连续  
> 即对于满足 $\underset{k\to\infty}{\lim}x^{(k)} = x$ 的每一个序列 $\{x^{(k)}\}\subset\mathcal X$ 都有 $f(x)\leq \underset{k\to\infty}{\lim} \inf f(x^{(k)})$   
> (特殊地，连续函数一定下半连续)
>
> 若下列条件有一个成立：
>
> - ① $\mathcal X$ 是紧集 (即有界闭集)
> - ② $\mathcal X$ 是闭集，且 $f$ 在 $x\in\mathcal X$ 上是强制的 (coercive)  
>   即对于满足 $\underset{k\to\infty}{\lim}\|x^{(k)}\| = \infty$ 的每一个序列 $\{x^{(k)}\}\subset\mathcal X$ 都有 $\underset{k\to\infty}{\lim}f(x^{(k)}) = \infty$ 
> - ③ 存在 $\alpha\in\mathbb R$ 使得下水平集 $\{x\in\mathcal X:f(x)\leq \alpha\}$ 是紧集
>
> 则 $f$ 在 $\mathcal X$ 上的全局最小点的集合 $\arg\underset{x\in\mathcal X}{\min} f(x)$ 为非空紧集.

- **证明:**    
  在 Euclid 单位球面 $S:=\{x\in \mathbb F^n:\|x\|_2=1\}$ (它是一个紧集) 上定义 $h(x) = \frac{f_2(x)}{f_1(x)}$   
  根据连续性假设可知 $h$ 是紧集 $S$ 上的连续函数  
  根据 **Weierstrass 定理**可知 $h$ 在 $S$ 上存在有限的最小值 $C_{\min}$ 和最大值 $C_{\max}$   
  即对于任意 $x\in S$ (即 $x\in \mathbb F^n$ 且 $\|x\|_2=1$) 我们都有 $C_\min \leq h(x) = \frac{f_2(x)}{f_1(x)}\leq C_\max$    
  根据非负性和正定性假设，并结合 $0_V\notin S$ 可知 $C_{\min},C_{\max}>0$ 

  根据齐次性假设我们可知: 对于任意非零向量 $x\in V$ 都有: 
  $$
  C_\min \leq h\left(\frac{x}{\|x\|_2}\right)
  =\frac{f_2(\frac{x}{\|x\|_2})}{f_1(\frac{x}{\|x\|_2})} 
  = \frac{\frac{1}{\|x\|_2}f_2(x)}{\frac{1}{\|x\|_2}f_1(x)} 
  = \frac{f_2(x)}{f_1(x)}\leq C_\max\\
  
  \Updownarrow\\
  
  C_\min f_1(x) \leq f_2(x) \leq C_\max f_1(x)
  $$
  显然 $x=0_V$ 也满足不等式 $C_\min f_1(x) \leq f_2(x) \leq C_\max f_1(x)$  
  因此我们有:  
  $$
  C_\min f_1(x) \leq f_2(x) \leq C_\max f_1(x)\ \ (\forall\ x\in V)
  $$
  命题得证.

- **(Matrix Analysis 推论 $5.4.5$)**  
  若 $\|\cdot\|_\alpha$ 和 $\|\cdot\|_\beta$ 是有限维实或复向量空间 $V$ 上的范数，  
  则存在有限的 $C_\min,C_\max>0$ 使得:  
  $$
  C_\min \|x\|_\alpha \leq \|x\|_\beta \leq C_\max \|x\|_\alpha
  $$

上述推论表明:  
**有限维实或复向量空间中向量序列的收敛性与所采用的范数无关.**  
**(Matrix Analysis 推论 $5.4.6$)**  
设 $\{x^{(k)}\}$ 为有限维实或复向量空间 $V$ 上的给定序列.  
若 $\|\cdot\|_\alpha$ 和 $\|\cdot\|_\beta$ 是 $V$ 上的范数， 
则关于 $\|\cdot\|_\alpha$ 有 $\underset{k\to\infty}{\lim} x^{(k)}=x$ 当且仅当关于 $\|\cdot\|_\beta$ 有 $\underset{k\to\infty}{\lim} x^{(k)}=x$   
换言之，$\underset{k\to\infty}{\lim} \|x^{(k)}-x\|_\alpha=0$ 当且仅当 $\underset{k\to\infty}{\lim} \|x^{(k)}-x\|_\beta=0$

- **证明:**  
  根据 Matrix Analysis 推论 $5.4.5$ 可知:  
  存在有限的 $C_\min,C_\max>0$ 使得:  
  $$
  C_\min \|x^{(k)}-x\|_\alpha \leq \|x^{(k)}-x\|_\beta \leq C_\max \|x^{(k)}-x\|_\alpha\ \ (\forall\ k=1,2,\dots)
  $$
  因此$\underset{k\to\infty}{\lim} \|x^{(k)}-x\|_\alpha=0$ 当且仅当 $\underset{k\to\infty}{\lim} \|x^{(k)}-x\|_\beta=0$   
  命题得证.

- 我们称实或复向量空间 $V$ (不一定有限维) 上的两个范数为**等价的** (equivalent)  
  如果只要 $V$ 上的一个向量序列 $\{x^{(k)}\}$ 依其中一个范数收敛于 $x\in V$，它就依另一个范数收敛于 $x$ 

  因此 Matrix Analysis 推论 $5.4.6$ 表明:  
  **有限维实或复向量空间中的所有范数都是等价的.**  

  但对于无限维空间来说，情况是非常不同的:  
  一个序列可能关于某个范数收敛，但关于另一个范数可能就发散了 (Matrix Analysis 例 $5.4.2$) 

****

根据有限维实或复向量空间上范数的等价性我们还能推出三个重要的事实:

- **向量序列收敛 (依范数收敛) 等价于依坐标收敛:**  
  由于有限维实或复赋范空间 (记其维数为 $n$) 上所有的范数都等价于 $\|\cdot\|_\infty$  
  故向量序列 $\{x^{(k)}\}$ 收敛于 $x$，即依某个 (事实上是任意) 范数 $\|\cdot\|$ 有 $\lim_{k\to\infty} \|x^{(k)}-x\|=0$ 成立，   
  当且仅当对于任意 $i=1,\dots,n$ 都有分量序列 $\{x_i^{(k)}\}$ 收敛于 $x_i$. 

- **有限维实或复赋范空间上的范数都是一致连续函数:**  
  设 $\|\cdot\|$ 是 $n$ 维实或复赋范空间 $V$ 上的范数，$\{e^{(1)},\dots,e^{(n)}\}$ 是 $V$ 的一组给定的基.  
  对于任意 $x\in V$，我们定义 $x$ 在基 $\{e^{(1)},\dots,e^{(n)}\}$ 下的坐标为 $(x_1,\dots,x_n)$，  
  则对于任意 $x,\Delta x\in V$ 我们都有:
  $$
  \begin{align}
  |\|x+\Delta x\|- \|x\|| 
  &\leq \|\Delta x\|\\
  &= \|\Delta x_1\cdot e^{(1)} +\dotsm + \Delta x_n\cdot e^{(n)}\|\\
  &\leq \sum_{i=1}^n |\Delta x_i|\|e^{(i)}\|\\
  &\leq \left(\max_{1\leq i\leq n}|\Delta x_i|\right) \sum_{i=1}^n \|e^{(i)}\|\\
  &= \left(\sum_{i=1}^n \|e^{(i)}\|\right)\cdot \|\Delta x\|_\infty\quad (\text{invoke norm equivalence})\\
  &\leq C\left(\sum_{i=1}^n \|e^{(i)}\|\right)\cdot \|\Delta x\|.\\
  \end{align}
  $$
  注意到 $C$ 和 $\sum_{i=1}^n \|e^{(i)}\|$ 都是与 $x,\Delta x$ 无关的常数，  
  因此当 $\Delta x\to 0_V$ (即 $\|\Delta x\|\to 0$) 时我们有 $|\|x+\Delta x\|- \|x\||\to 0$，而且收敛速度与 $x$ 无关.  
  这表明 $\|\cdot\|$ 在 $V$ 上是一致连续的.
  
- **(Matrix Analysis 推论 $5.4.8$)**  
  设 $f$ 为域 $\mathbb F=\mathbb R\text{ or }\mathbb C$ 上的有限维向量空间 $V$ 上的一个**准范数**，即满足:

  - 非负性: $f(x)\geq 0\ (\forall\ x\in V)$ 
  - 正定性: $f(x)=0$ 当且仅当 $x=0_V$ 
  - 齐次性: $f(\alpha x) = |\alpha|f(x)\ (\forall\ x\in V,\alpha\in \mathbb F)$ 
  - 连续性: $f(x)$ 在 $\mathbb F^n$ 上是连续的

  则对于任意给定的 $\alpha\in \mathbb F$，下水平集 $\{x:f(x)\leq \alpha\}$ 和水平集 $\{x:f(x)=\alpha\}$ 都是紧集.  
  (集合的紧性是指该集合上的任意序列都有收敛的子列)

  **证明思路:**  
  我们只需证明命题对 Euclid 范数 $\|\cdot\|_2$ 成立即可 (具体步骤从略)  
  结合 Matrix Analysis 定理 $5.4.4$ 可知命题对有限维实或复向量空间上的任意准范数都成立.



### 2.1.6 赋范空间的完备性

**(Matrix Analysis 定义 $5.4.9$)** 
赋范空间 $V$ 中的一个序列 $\{x^{(k)}\}$ 称为是关于范数 $\|\cdot\|$ 的 **Cauchy 序列**，  
如果对于任意 $\varepsilon>0$，都存在一个正整数 $N$ 使得  
$$
\|x^{(k_1)}-x^{(k_2)}\| \leq \varepsilon\ \ (\forall\ k_1,k_2\geq N).
$$
**(Matrix Analysis 定理 $5.4.10$)**     
设 $\|\cdot\|$ 是有限维实或复赋范空间 $V$ 上任意给定的范数 (回忆起有限维实或复向量空间中的所有范数都是等价的)  
若 $\{x^{(k)}\}$ 为 $V$ 中给定的向量序列，  
则当且仅当它关于范数 $\|\cdot\|$ 是一个 Cauchy 序列时，它收敛于 $V$ 中的某个向量.

- 这个定理的证明需要用到实数域 $\mathbb R$ 和复数域 $\mathbb C$ 的**完备性**:  
  数域 $\mathbb F = \mathbb R\text{ or }\mathbb C$ 上的数列 $\{x_k\}$ 收敛到 $\mathbb F$ 中的某个标量，当且仅当它是 **Cauchy 数列**  
  即对于任意 $\varepsilon>0$，都存在一个正整数 $N$ 使得 $|x_{k_1}-x_{k_2}|\leq \varepsilon\ (\forall\ k_1,k_2\geq N)$ 成立.

- **(Matrix Analysis 定义 $5.4.11$)**  
  一个赋范空间 $V$ 称为关于其范数 $\|\cdot\|$ 是**完备的** (complete)，  
  如果 $V$ 中任意一个关于 $\|\cdot\|$ 的 Cauchy 序列都关于 $\|\cdot\|$ 收敛于 $V$ 中的一个点.

  因此上述定理实际上说明了实或复数域的完备性可以延拓到任意一个有限维实或复赋范空间.  
  不幸的是，无限维赋范空间可能没有完备性 (参见 Matrix Analysis 定义 $5.4.11$ 后的习题).

<img src="Matrix Analysis 5.4.11.jpeg" style="zoom:50%;" />



### 2.1.7 对偶范数

> **(Matrix Analysis 推论 $5.4.8$)**  
> 设 $f$ 为域 $\mathbb F=\mathbb R\text{ or }\mathbb C$ 上的有限维向量空间 $V$ 上的一个**准范数**，即满足:
>
> - 非负性: $f(x)\geq 0\ (\forall\ x\in V)$ 
> - 正定性: $f(x)=0$ 当且仅当 $x=0_V$ 
> - 齐次性: $f(\alpha x) = |\alpha|f(x)\ (\forall\ x\in V,\alpha\in \mathbb F)$ 
> - 连续性: $f(x)$ 在 $\mathbb F^n$ 上是连续的
>
> 则对于任意给定的 $\alpha\in \mathbb F$，下水平集 $\{x:f(x)\leq \alpha\}$ 和水平集 $\{x:f(x)=\alpha\}$ 都是紧集.  
> (集合的紧性是指该集合上的任意序列都有收敛的子列)

回忆起 "有限维实或复向量空间上的准范数的单位球都是紧集" 这一事实，  
我们可以利用 Euclid 内积从给定的准范数定义一个对偶范数.  
**(Matrix Analysis 定义 $5.4.12$)**    
设 $f(\cdot)$ 为 $V=\mathbb F^n$ (其中 $\mathbb F=\mathbb R\text{ or }\mathbb C$) 上的一个准范数   
我们定义 $f$ 的**对偶范数** (dual norm) 为 $f^D(y):= \underset{f(x)=1}{\sup} \text{Re}(\langle x,y\rangle_2) = \underset{f(x)=1}{\sup} \text{Re}(y^{\mathrm H}x)$ 

- 首先 $f^D$ 是 $V$ 上的定义良好的函数.  
  因为对于任意固定的 $y\in V$，$\text{Re}(y^{\mathrm H}x)$ 都是 $x$ 的连续函数，且准范数单位球 $\{x:f(x)=1\}$ 是紧集.  
  Weierstrass 定理确保了 $\text{Re}(y^{\mathrm H}x)$ 在 $\{x:f(x)=1\}$ 存在有限的最大值 (即上确界可以取到)  
  故我们可以将 $f^D$ 的定义改写为 $f^D(y):= \underset{f(x)=1}{\max} \text{Re}(y^{\mathrm H}x)$

- 其次，准范数 $f$ 的齐次性告诉我们:  
  $$
  \begin{align}
  \max_{f(x)=1} |y^{\mathrm H}x|
  &=
  \max_{f(x)=1} \max_{|\alpha|=1} \text{Re}(\alpha y^{\mathrm H}x)\\
  &=
  \max_{f(x)=1} \max_{|\alpha|=1} \text{Re}(y^{\mathrm H}(\alpha x))\\
  &=
  \max_{|\alpha|=1}\max_{f(x/\alpha)=1} \text{Re}(y^{\mathrm H}x)\\
  &=
  \max_{f(x)=1} \text{Re}(y^{\mathrm H}x)\\
  &=
  f^D(y)
  \end{align}\quad (\forall\ y\in V)
  $$
  因此我们可以将 $f^D$ 的定义改写为 $f^D(y):= \underset{f(x)=1}{\max} |y^{\mathrm H}x| = \underset{x\neq 0_V}{\max}\frac{|y^{\mathrm H}x|}{f(x)}$ 

- 最后，即使准范数 $f$ 不是一个范数 (即不满足三角不等式)，$f^D$ 也一定是范数.    
  $f^D(y):= \underset{f(x)=1}{\max} |y^{\mathrm H}x|$ 的非负性、正定性和齐次性都是显然的.  
  下面证明即使准范数 $f$ 不满足三角不等式，$f^D$ 也满足三角不等式:  
  $$
  \begin{align}
  f^D(y+z)
  &=
  \max_{f(x)=1} |(y+z)^{\mathrm H}x|\\
  &\leq
  \max_{f(x)=1} (|y^{\mathrm H}x| + |z^{\mathrm H}x|)\\
  &\leq 
  \max_{f(x)=1} |y^{\mathrm H}x| + \max_{f(x)=1} |z^{\mathrm H}x|\\
  &= f^D(y) + f^D(z)
  \end{align}\quad (y,z\in V)
  $$
  因此准范数 $f$ 的对偶 $f^D$ 一定是一个范数.   

  特别地，由于范数是准范数的特例，故范数 $\|\cdot\|$ 的对偶一定是一个范数，记为 $\|\cdot\|_*$   
  $$
  \|y\|^D := \max_{\|x\|=1} |y^{\mathrm H}x| = \underset{x\neq 0_V}{\max}\frac{|y^{\mathrm H}x|}{\|x\|}
  $$

****

> **(Cauchy–Schwarz 不等式, Matrix Analysis 定理 $5.1.4$)**  
> 设 $V$ 是建立在域 $\mathbb F = \mathbb R\ \text{or }\mathbb C$ 上的线性空间  
> 若函数 $\langle\cdot,\cdot\rangle:V\times V\mapsto \mathbb F$ 是一个内积，则我们有:  
> $$
> |\langle x,y\rangle|^2 \leq \langle x,x\rangle \langle y,y\rangle \ \ (\forall\ x,y\in V)
> $$
> 当且仅当 $x,y$ 线性相关时取等.

作为 Cauchy–Schwarz 不等式的一个自然的推广  
下面的引理给出了对偶范数的一个简单的不等式:  
**(Matrix Analysis 引理 $5.4.13$)**  
若 $f(\cdot)$ 为 $V=\mathbb F^n$ (其中 $\mathbb F=\mathbb R\text{ or }\mathbb C$) 上的一个准范数  
则对于任意 $x,y\in V$ 我们都有:  
$$
|y^{\mathrm H}x| \leq f(x) f^D(y)\\
|y^{\mathrm H}x| \leq f^D(x) f(y)
$$
其中对偶范数的定义为 $f^D(y):= \underset{f(x)=1}{\max} |y^{\mathrm H}x| = \underset{x\neq 0_V}{\max}\frac{|y^{\mathrm H}x|}{f(x)}$ 

- **证明:**   
  注意到第二个不等式可由第一个不等式推出 (因为 $|x^{\mathrm H}y|=|y^{\mathrm H}x|$)，因此我们只需证明第一个不等式即可.

  当 $x=0_V$ 时，对于任意 $y\in V$ 第一个不等式显然成立.  
  当 $x\neq 0_V$ 时 (注意到此时有 $f(x)>0$ 成立)，对于任意 $y\in V$ 我们都有: 
  $$
  \frac{1}{f(x)}|y^{\mathrm H}x| = \left|y^{\mathrm H}\frac{x}{f(x)}\right| \leq \max_{f(x)=1} |y^{\mathrm H}z| = f^D(y)\\
  \Updownarrow\\
  |y^{\mathrm H}x| \leq f(x) f^D(y)
  $$
  命题得证.

***

**常用的对偶关系:**

- > 给定 $\mathbb C^m$ 上的一个范数 $\|\cdot\|$ 和矩阵 $A\in \mathbb C^{m\times n}$   
  > 定义函数 $\|\cdot\|_A : \mathbb C^n\mapsto \mathbb R$ 为 $\|x\|_A:=\|Ax\|\ (\forall\ x\in \mathbb C^n)$  
  > 若 $A\in \mathbb C^{m\times n}$ 是列满秩的，则函数 $\|\cdot\|_A$ 是 $\mathbb C^n$ 上的范数.

  特殊地，给定 $\mathbb C^n$ 上的一个范数 $\|\cdot\|$ 和非奇异阵 $A\in \mathbb C^{n\times n}$   
  若定义范数 $\|\cdot\|_A : \mathbb C^n\mapsto \mathbb R$ 为 $\|x\|_A:=\|Ax\|\ (\forall\ x\in \mathbb C^n)$   
  则对于任意 $y\in \mathbb C^n$ 我们都有:  
  $$
  \begin{align}
  (\|y\|_A)^D
  &=
  \max_{x\neq 0_n}
  \frac{|y^{\mathrm H}x|}{\|x\|_A}\\
  &=
  \max_{x\neq 0_n}
  \frac{|y^{\mathrm H}x|}{\|Ax\|}\\
  &=
  \max_{z\neq 0_n}
  \frac{|y^{\mathrm H} A^{-1} z|}{\|z\|}\quad (\text{denote }z:= Ax)\\
  &=
  \max_{z\neq 0_n}
  \frac{|(A^{-H}y)^{\mathrm H}z|}{\|z\|}\\
  &=
  \|A^{-\mathrm{H}}y\|^D\\
  &=
  (\|y\|^D)_{A^{-\mathrm{H}}}
  \end{align}
  $$
  这表明 $(\|\cdot\|_A)^D=(\|\cdot\|^D)_{A^{-\mathrm{H}}}$

- 对于任意 $x,y \in \mathbb C^n$ 我们都有:  
  $$
  |y^{\mathrm H}x| 
  =
  \left|\sum_{i=1}^n \bar y_i x_i\right|
  \leq
  \sum_{i=1}^n |\bar y_i x_i|
  
  \leq
  
  \begin{cases}
  \left(\underset{1\leq i\leq n}{\max}|y_i|\right) \sum_{i=1}^n |x_i| = \|y\|_\infty \|x\|_1\\
  \left(\underset{1\leq i\leq n}{\max}|x_i|\right) \sum_{i=1}^n |y_i| = \|x\|_\infty \|y\|_1
  \end{cases}
  $$
  注意到上述两个不等式都是可取等的:  
  前者的取等条件是 $y\neq 0_n$ 且 $x$ 对单独一个满足 $|y_i|=\|y\|_\infty$ 的 $i$ 有 $x_i=1$，而对所有 $j\neq i$ 都有 $x_i=0$   
  后者的取等条件是 $x\neq 0_n$ 且 $y$ 对单独一个满足 $|x_i|=\|x\|_\infty$ 的 $i$ 有 $y_i=1$，而对所有 $j\neq i$ 都有 $y_i=0$   
  于是我们有:  
  $$
  \|y\|_1^D = \max_{\|x\|_1 = 1}|y^{\mathrm H}x| = \max_{\|x\|_1=1}\|y\|_\infty \|x\|_1 = \|y\|_\infty\\
  \|y\|_\infty^D = \max_{\|x\|_\infty = 1}|y^{\mathrm H}x| = \max_{\|x\|_\infty=1}\|x\|_\infty \|y\|_1 = \|y\|_1
  $$

- > **Hölder 不等式:** $|x^{\mathrm H}y|\leq \|x\|_p\|y\|_q\ \ (\forall\ x,y\in \mathbb C^n)$  
  > 当且仅当 $|x|^p,|y|^q$ 线性相关时取等.  
  > 其中 $p,q>1$ 为共轭子标，满足 $\frac1p+\frac1q =1$  

  对于任意 $p>1$，取共轭子标 $q>1$ (即满足 $\frac{1}{p}+\frac1q = 1$)  
  根据 Hölder 不等式我们有:  
  $$
  |x^{\mathrm H}y|\leq \|x\|_p\|y\|_q
  $$
  取等条件是 $|x|^p,|y|^q$ 线性相关:

  - 当 $y=0_n$ 时，不等号对任意 $x\in \mathbb C^n$ 都取等 
  - 当 $y\neq 0_n$ 时，不等号对由 $x_i := \begin{cases}
    0 & \text{if }y_i=0\\
    \frac{|y_i|^q}{\bar y_i \|y\|_q^{q-1}} & \text{if }y_i\neq 0\end{cases}$ 定义的 $x=[x_i]$ 取等 

  于是我们有:    
  $$
  \|y\|_p^D = \max_{\|x\|_p = 1}|y^{\mathrm H}x| = \max_{\|x\|_p=1}\|x\|_p \|y\|_q = \|y\|_q\\
  $$
  从而 $\|\cdot\|_p^D = \|\cdot\|_q$   
  特殊地，$l_2$ 范数的对偶范数就是它自身，即 $\|\cdot\|_2^D = \|\cdot\|_2$   
  事实上，$l_2$ 范数是 $\mathbb C^n$ 上仅有的自对偶范数，这并非是偶然的.

***

**(Matrix Analysis 引理 $5.4.16$)**  
设 $f(\cdot)$ 和 $g(\cdot)$ 是 $V=\mathbb F^n$ (其中 $\mathbb F=\mathbb R\text{ or }\mathbb C$) 上的准范数，给定正实数 $c>0$  
则我们有:

- $cf(\cdot)$ 也是 $V$ 上的准范数，且其对偶范数是 $c^{-1}f^D(\cdot)$
- 若对于任意 $x\in V$ 都有 $f(x)\leq g(x)$ 成立，则我们有 $f^D(y)\geq g^D(y)\ \ (y\in V)$ 

**(Matrix Analysis 定理 $5.4.17$)**  
设 $\|\cdot\|$ 是 $V=\mathbb F^n$ (其中 $\mathbb F=\mathbb R\text{ or }\mathbb C$) 上的范数，给定正实数 $c>0$  
则当且仅当 $\|x\| = \sqrt{c}\|x\|_2\ (\forall\ x\in V)$ 时我们有 $\|x\|=c\|x\|^D\ (\forall x\in V)$ 成立.  
特别地，当且仅当 $\|\cdot\|$ 为 $l_2$ 范数时我们有 $\|\cdot\|^D=\|\cdot\|$

**(对偶定理, Matrix Analysis 定理 $5.5.9$)**  
设 $f(\cdot)$ 是 $V=\mathbb F^n$ (其中 $\mathbb F=\mathbb R\text{ or }\mathbb C$) 上的准范数.  
记 $f^D$ 为 $f$ 的对偶范数，$f^{DD}$ 为 $f^D$ 的对偶范数.  
记 $S=\{x\in V:f(x)\leq 1\}$ 和 $S^{DD} = \{x\in V:f^{DD}(x)\leq 1\}$  
则我们有:

- ① 对任意 $x\in V$ 都有 $f^{(DD)}(x)\leq f(x)$ 成立，因而有 $S\subseteq S^{DD}$ 
- ② $S^{DD} = \text{cl}(\text{conv}(S))$ (即 $S$ 的凸包的闭包)
- ③ 若 $f$ 是 $V$ 上的范数，则我们有 $f^{DD}\equiv f$，因而有 $S^{DD}=S$   
  (这表明任何范数都是其对偶范数的对偶)
- ④ 若 $f$ 是范数且给定 $x^{(0)}\in V$，  
  则存在某个 $z\in V$ (不一定唯一) 使得 $f^D(z)=1$ (即对于任意 $x\in V$ 都有 $|z^{\mathrm H}x|\leq f(x)$) 且 $f(x^{(0)})=z^{\mathrm H}x^{(0)}$



### 2.1.8 绝对性 & 单调性

> $k$-范数是 $\mathbb C^n$ 上的一个重要的离散范数族，它填补了 $l_1$ 范数和 $l_\infty$ 范数之间的空隙.  
> 对于任意 $k=1,\dots,n$，向量 $x\in \mathbb C^n$ 的 $k$-范数是 $x$ 的前 $k$ 大分量模长的和:  
> $$
> \|x\|_{[k]}:= |x_{\pi(1)}| + \dotsm + |x_{\pi(k)}|\text{where }\pi\text{ such that }|x_{\pi(1)}|\geq \dotsm \geq |x_{\pi(n)}|
> $$
> 它们在酉不变相容范数的理论中起着重要作用 (Matrix Analysis 7.4.7 节)  
>
> 显然 $k$-范数是置换不变的  
> 换言之，对于任意 $x\in \mathbb C^n$ 和排列矩阵 $P\in \mathbb C^{n\times n}$，都有 $\|Px\|_{[k]} = \|x\|_{[k]}$ 成立.

$\mathbb R^n$ 或 $\mathbb C^n$ 上的每一个 $k$-范数和每一个 $l_p$ 范数都有如下的性质:  
向量的范数值仅与其元素的模长有关 (绝对性)，且是关于元素模长的单调递增函数 (单调性).

**(Matrix Analysis 定义 $5.4.18$)**    
设 $V=\mathbb F^n$ (其中 $\mathbb F=\mathbb R\text{ or }\mathbb C$)   
记 $|x|$ 为 $x\in V$ 逐个元素取模得到的向量.  
记号 $x \leq y$ 表示对所有 $i=1,\dots,n$ 都有 $x_i\leq y_i$.   
我们称 $V$ 上的范数 $\|\cdot\|$ 是:

- ① **单调的** (monotone)，如果对于任意满足 $|x|\leq|y|$ 的 $x,y\in V$ 都有 $\|x\|\leq \|y\|$ 成立.
- ② **绝对的** (absolute)，如果 $\||x|\| = \|x\|\ (\forall\ x\in V)$

**(Matrix Analysis 定理 $5.4.19$)**  
设 $\|\cdot\|$ 是 $V=\mathbb F^n$ (其中 $\mathbb F=\mathbb R\text{ or }\mathbb C$) 上的范数.

- ① 若 $\|\cdot\|$ 是绝对的，则有 $\|y\|^D= \underset{x\neq 0_n}{\max} \frac{|y|^{\mathrm T} |x|}{\|x\|}\ (\forall\ y\in V)$
- ② 若 $\|\cdot\|$ 是绝对的，则对偶范数 $\|\cdot\|^D$ 是绝对且单调的
- ③ 范数 $\|\cdot\|$ 是绝对的，当且仅当它是单调的

****

**(Matrix Analysis 定理 $5.6.36$)**  
设 $\|\cdot\|$ 是 $V=\mathbb F^n$ (其中 $\mathbb F=\mathbb R\text{ or }\mathbb C$) 上的范数，而 $\|\cdot\|$ 是由它诱导的 $\mathbb F^{n\times n}$ 上的矩阵范数.  
则下列命题等价:

- ① $\|\cdot\|$ 是 $\mathbb F^n$ 上的绝对范数
- ② $\|\cdot\|$ 是 $\mathbb F^n$ 上的单调范数
- ③ 对于任意对角阵 $\Lambda=\text{diag}\{\lambda_1,\dots,\lambda_n\}$，都有诱导范数 $\|\Lambda\|=\max_{1\leq i\leq n}|\lambda_i|$ 成立.




## 2.2 内积空间

### 2.2.1 内积

设 $V$ 是建立在域 $\mathbb F = \mathbb R\ \text{or }\mathbb C$ 上的线性空间.   
函数 $\langle\cdot,\cdot\rangle:V\times V\mapsto \mathbb F$ 称为一个**内积** (inner product)，如果它满足下列五条公理:  
对于任意 $x,y,z\in V$ 和 $\alpha\in \mathbb F$ 

- ① 非负性: $\langle x,x\rangle \geq 0$
- ② 正定性: $\langle x,x\rangle = 0$ 当且仅当 $x=0_V$ 
- ③ 线性性: $\langle x+y,z\rangle = \langle x,z\rangle + \langle y,z\rangle$ 
- ④ 齐次性: $\langle \alpha x,y\rangle = \alpha \langle x,y\rangle$
- ⑤ 共轭对称性: $\overline{\langle x,y\rangle} = \langle y,x\rangle$

若函数 $\langle\cdot,\cdot\rangle:V\times V\mapsto \mathbb F$ 不满足正定性，但满足其他四个性质，  
则称其为**半内积** (semi-inner product).  
非零向量的半范数可能为零.

一个实的或复的线性空间 $V$ 与一个给定的内积 $\langle\cdot,\cdot\rangle:V\times V\mapsto \mathbb F$ 合在一起，  
就称为一个**内积空间** (inner product space).   
内积空间一定是赋范空间，因为由内积 $\langle \cdot,\cdot\rangle$ 可以诱导出范数 $\|\cdot\|$:  
$$
\|x\|:= \sqrt{\langle x,x\rangle}\ (\forall\ x\in X).
$$
我们称完备的内积空间为 **Hilbert 空间**.    
(即这个内积空间在其内积 $\langle \cdot,\cdot\rangle$ 诱导出的范数 $\|\cdot\|$ 意义下的所有 Cauchy 序列都收敛，且极限属于该空间)    
特殊地，有限维内积空间一定是 Hilbert 空间.  

***

根据定义可知内积具有以下基本性质: 
对于任意 $x,y,z,w \in V,a,b,c,d \in \mathbb{F}=\mathbb R \text{ or }\mathbb C$ 都有: 

* 对第二个向量的共轭齐次性: $\langle x,\alpha y \rangle = \bar\alpha \langle x,y\rangle$   
  据此可知 $\langle x,\langle x, y \rangle y \rangle = |\langle x,y\rangle|^2$ 
* 对第一个向量线性可加，对第二个向量共轭线性可加:  
  $\langle ax+by, cz+dw \rangle = a\bar c\langle x,z\rangle +  a\bar d\langle x,w\rangle + b\bar c\langle y,z\rangle +  b\bar d\langle y,w\rangle $
* 当且仅当 $x=0_V$ 时有 $\langle x,v\rangle = 0\ (\forall\ v\in V)$ 成立.



### 2.2.2 Cauchy–Schwarz 不等式

**(Cauchy–Schwarz 不等式, Matrix Analysis 定理 $5.1.4$)**  
设 $V$ 是建立在域 $\mathbb F = \mathbb R\ \text{or }\mathbb C$ 上的线性空间  
若函数 $\langle\cdot,\cdot\rangle:V\times V\mapsto \mathbb F$ 是一个内积，则我们有:  
$$
|\langle x,y\rangle|^2 \leq \langle x,x\rangle \langle y,y\rangle \ \ (\forall\ x,y\in V),
$$
当且仅当 $x,y$ 线性相关时取等.

- **(Matrix Analysis 推论 $5.1.7$)**   
  若函数 $\langle\cdot,\cdot\rangle:X\times X\mapsto \mathbb F$ 是一个内积，  
  则定义为 $\|x\|:= \sqrt{\langle x,x\rangle}$ 的函数 $\|\cdot\|:X\mapsto [0,+\infty)$ 一定是一个范数.  
  其中正定性和齐次性都是显然的，而次可加性可由 **Cauchy–Schwarz 不等式**推出:  
  $$
  \begin{align}
  \|x+y\|^2 
  &=
  \langle x+y,x+y\rangle\\
  &=
  \langle x,x\rangle + 2\langle x,y\rangle + \langle y,y\rangle\\
  &\leq
  \langle x,x\rangle + 2|\langle x,y\rangle| + \langle y,y\rangle\quad (\text{Cauchy–Schwarz})\\
  &\leq
  \langle x,x\rangle + 2\sqrt{\langle x,x\rangle \langle y,y\rangle} + \langle y,y\rangle\\
  &=
  \|x\|^2 + 2\|x\|\|y\| + \|y\|^2\\
  &=
  (\|x\| + \|y\|)^2
  \end{align}\Rightarrow \|x+y\| \leq \|x\| + \|y\|\ (\forall\ x,y\in V)
  $$
  因此**内积空间一定是赋范空间**，但反过来不成立.
  
- **(Matrix Analysis 定理 $5.1.8$)**   
  事实上，半内积也满足 Cauchy–Schwarz 不等式 (但取等条件不再是 "$x,y$​ 线性相关").    
  因此若函数 $\langle\cdot,\cdot\rangle:V\times V\mapsto \mathbb F$ 是一个半内积，  
  则定义为 $\|x\|:= \sqrt{\langle x,x\rangle}$ 的函数 $\|\cdot\|:V\mapsto [0,+\infty)$ 一定是一个半范数 (可根据定义验证).


**证明:**  
当 $x=y=0_V$ 时，命题显然成立.  
因此我们不妨假设 $y\neq 0_V$，于是有:   
(其中 $\langle y,y \rangle x - \langle x,y\rangle y$ 的构造源自于 Gram–Schmidt 过程)
$$
\begin{align}
0 
&\leq  \langle \langle y,y \rangle x - \langle x, y\rangle y,  \langle y,y \rangle x - \langle x, y\rangle y\rangle\\

&=
\langle y, y\rangle \overline{\langle y,y \rangle} \cdot \langle x, x\rangle
-
\langle y,y \rangle \overline{\langle x,y \rangle} \cdot\langle x,y \rangle
-
\langle x,y \rangle \overline{\langle y,y \rangle}\cdot \langle y, x\rangle 
+
\langle x,y \rangle \overline{\langle x,y \rangle} \langle y,y \rangle\\

&=
\langle y,y \rangle^2 \langle x,x \rangle 
- 
\langle y,y \rangle \langle y,x \rangle \langle x,y \rangle
-
\langle x,y \rangle \langle y,y \rangle \langle y,x \rangle
+
\langle x,y \rangle \langle y,x \rangle \langle y,y \rangle\\

&=
\langle y,y \rangle^2 \langle x,x \rangle 
- 
\langle y,y \rangle \langle y,x \rangle \langle x,y \rangle\\

&=
\langle y,y \rangle (\langle y,y \rangle \langle x,x \rangle - \langle y,x \rangle \langle x,y \rangle)\\

&=
\langle y,y \rangle (\langle x,x \rangle \langle y,y \rangle - \overline{\langle y,x \rangle} \langle x,y \rangle)\\

&=
\langle y,y \rangle (\langle x,x \rangle \langle y,y \rangle - |\langle x,y \rangle|^2)\\

\end{align}
$$
由于 $y\neq 0_V$，故 $\langle y,y\rangle > 0$，表明 $\langle x,x \rangle \langle y,y \rangle - |\langle x,y \rangle|^2\leq 0$，即有:  
$$
|\langle x,y\rangle|^2 \leq \langle x,x\rangle \langle y,y\rangle \ \ (\forall\ x,y\in V)
$$
当且仅当 $\langle y,y \rangle x - \langle x, y\rangle y=0_V$ 时取等，即当且仅当 $x,y$ 线性相关时取等.

***

**邵老师提供的证明: (实际上是上面证明的简化)**  
任意给定 $x,y\in \mathbb C^n$.  
考虑实值函数 $f(\mu) = \|x\mu + y\|^2 = \langle x \mu + y,x\mu + y\rangle = \|x\|^2 \mu^2 + 2\text{Re}(\langle x,y\rangle) \mu + \|y\|^2$.   
显然对于任意 $\mu\in \mathbb R$ 均有 $f(\mu)\geq 0$ 成立.  
因此判别式 $\Delta = 4\text{Re}(\langle x,y\rangle)^2 - 4\|x\|^2 \|y\|^2 \leq 0$.  
于是我们有 $|\text{Re}(\langle x,y\rangle)|^2 \leq \|x\|^2\|y\|^2 = \langle x,x\rangle \langle y, y\rangle$ 成立.

根据 $x,y$ 的任意性可知:  
我们只需将 $x$ 旋转至与 $y$ 共线 (即 $|\langle x,y\rangle| = |\text{Re}(\langle x,y\rangle)|$) 时，  
就有 $|\langle x,y\rangle|^2 \leq \|x\|^2\|y\|^2 = \langle x,x\rangle \langle y, y\rangle$ 成立.

更深刻地，我们可以将 $\mu\in \mathbb R$ 的取值范围拓展为 $\mu\in \mathbb C$，并将 $f(\mu)$ 改写为:  
$$
\begin{align}
f(\mu)
&=
\|x\mu + y\|^2\\
&=
\left\langle 
[x,y]
\begin{bmatrix}
\mu\\
1
\end{bmatrix},
[x,y]
\begin{bmatrix}
\mu\\
1
\end{bmatrix}
\right\rangle\\

&=
\begin{bmatrix}
\mu\\
1
\end{bmatrix}^{\mathrm H}
\begin{bmatrix}
\langle x,x\rangle  & \langle y,x\rangle\\
\langle x,y\rangle & \langle y,y\rangle
\end{bmatrix}
\begin{bmatrix}
\mu\\
1
\end{bmatrix}.
\end{align}
$$
显然对于任意 $\mu\in \mathbb C$ 均有 $f(\mu)\geq 0$ 成立.  
因此我们有:  
$$
{\begin{bmatrix}
\langle x,x\rangle  & \langle y,x\rangle\\
\langle x,y\rangle & \langle y,y\rangle
\end{bmatrix}\succeq 0}\\

\Updownarrow\\

\det(\begin{bmatrix}
\langle x,x\rangle  & \langle y,x\rangle\\
\langle x,y\rangle & \langle y,y\rangle
\end{bmatrix}) \geq 0\\

\Updownarrow\\

\langle y,x\rangle\langle x,y\rangle = |\langle x,y\rangle|^2 \leq \langle x,x\rangle\langle y,y\rangle
$$




### 2.2.3 内积的连续性

**内积是二元连续函数.**  
设 $(V , \langle\cdot,\cdot\rangle)$ 为内积空间，$\|\cdot\|$ 为 $\langle \cdot,\cdot \rangle$ 诱导出的范数，$\{x_n\},\{y_n\}$ 为 $V$ 中的两个序列.  
定义函数 $f:V\times V\mapsto \mathbb F=\mathbb R\text{ or }\mathbb C$ 为 $f(x,y) := \langle x,y\rangle\ (\forall\ x,y\in V)$.

若 $\begin{cases}
\underset{n\to\infty}{\lim} x_n = x\\
\underset{n\to\infty}{\lim} y_n = y\end{cases}$ (即 $\begin{cases} 
\underset{n\rightarrow \infty}{\lim} \|x_n-x\| = 0\\  
\underset{n\rightarrow \infty}{\lim} \|y_n-y\|=0 \end{cases}$)，则我们有:
$$
\begin{align}
|f(x_n, y_n) - f(x, y)| 
&= |\langle x_n, y_n \rangle - \langle x, y \rangle|\\
&= |\langle x_n, y_n \rangle - \langle x_n, y \rangle + \langle x_n, y \rangle - \langle x, y \rangle|\\
&= |\langle x_n, y_n - y \rangle + \langle x_n - x, y \rangle|\\
&\leq |\langle x_n, y_n - y \rangle| + |\langle x_n - x, y \rangle|\quad (\text{triangle inequality})\\
&\leq  \|x_n\| \cdot \|y_n - y\| + \|y\| \cdot \|x_n - x\| \quad (\text{Cauchy–Schwarz inequality})
\end{align}
$$
由于 $\{x_n\}$ 作为收敛序列一定有界 (即 $\|x_n\|$ 有界)，因此当 $n\rightarrow \infty$ 时右式趋近于 $0$   
表明 $f(x,y) = \langle x,y\rangle$ 关于 $x,y$ 连续.



### 2.2.4 极化恒等式

**极化恒等式 (Polarization Identity)** 告诉我们如何在已知范数的条件下构造形式上的内积.  
设 $(V , \|\cdot\|)$ 为建立在 $\mathbb{F}=\mathbb R\text{ or }\mathbb C$ 上的赋范空间，  
记 $\langle\cdot,\cdot\rangle$ 为 $\|\cdot\|$ 对应的形式上的内积，满足 $\langle x ,x\rangle = \|x\|^2$.   
我们希望导出 $\langle x,y\rangle$ 的表达式:

* **(实内积空间上的极化恒等式)**  
  若 $\mathbb{F} =\mathbb R $，则 $\begin{cases} \langle y,x\rangle = \langle x,y\rangle\\ \langle x,\alpha  y\rangle = \alpha \langle x,y\rangle\\ \end{cases}$   
  于是有 $\|x\pm y\|^2 =\langle x\pm y,x\pm y\rangle = \|x\|^2 +\|y\|^2 \pm 2\langle x,y\rangle$   
  相减消去平方项后，有 $\langle x,y\rangle = \frac14(\|x+y\|^2 - \|x-y\|^2)$ 
* **(复内积空间上的极化恒等式)**  
  若 $\mathbb F = \mathbb C$，则 $\begin{cases} \langle y,x\rangle = \overline {\langle x,y\rangle}\\ \langle x,\alpha  y\rangle = \bar \alpha \langle x,y\rangle \end{cases}$   
  于是有 $\begin{cases} \|x\pm y\|^2 =\langle x\pm y,x\pm y\rangle = \|x\|^2 +\|y\|^2 \pm (\langle y,x\rangle + \langle x,y \rangle)\\ 
  \|x\pm \mathrm{i}y\|^2 =\langle x\pm \mathrm{i} y,x\pm \mathrm{i}y\rangle = \|x\|^2 +\|y\|^2 \pm \mathrm{i}(\langle y,x\rangle - \langle x,y\rangle ) \end{cases}$   
  进而有 $\begin{cases} \langle x,y\rangle + \langle y,x\rangle = \frac12(\|x+y\|^2 - \|x-y\|^2)\\ \langle x,y\rangle - \langle y,x\rangle = \frac12(\mathrm{i}\|x+\mathrm{i}y\|^2 - \mathrm{i}\|x-\mathrm{i}y\|^2) \end{cases}$   
  最终得到 $\langle x,y\rangle  = \frac14 (\|x+y\|^2 - \|x-y\|^2 + \mathrm{i}\|x+\mathrm{i}y\|^2 - \mathrm{i}\|x-\mathrm{i}y\|^2 )$.

这样我们就对实/复赋范空间 $(V , \|\cdot\|)$ 导出了形式上的内积 $\langle\cdot,\cdot\rangle$.   
换言之，若 $||\cdot||$ 可由内积导出，则该内积一定具有极化恒等式给出的形式.  
随后我们只要逐一验证形式上的内积 $\langle\cdot,\cdot\rangle$ 是否满足内积定义中的四条性质即可.

由于内积空间的几何性质比较多，  
故我们拿到一个赋范空间先要看看它的范数能不能找到相应的内积，  
寻找的过程就是用极化恒等式构造形式上的内积，再验证这个内积是否满足内积的定义.  
从这里也可看出，我们之所以经常使用 $l_2$ 范数，  
不仅仅是因为计算方便 (实际上 $l_1,l_\infty$ 范数计算更方便)，其实还有几何上的考虑.

从另一个角度出发，我们可以看出：  
由范数 $\|\cdot\|$ 构造形式上的内积 $\langle\cdot,\cdot\rangle$ 实际上从平等中创造不平等，从对称中创造不对称.  
$x$ 的范数 $\|x\|$ 相当于 $\langle x,x\rangle$，参与内积的两个向量是平等的 (对称的)  
借助极化恒等式，我们可以用 $\|x\pm y\|$ 和 $\|x\pm iy\|$ 表示出 $\langle x,y\rangle$   
这样使得参与内积的两个向量不相等 (非对称)，从而适用于更一般的情况.  

- 具体地，我们可以考虑这样一个例子：  
  定义泛函 $f(x) := x^{\mathrm T}Ax\ (\forall\ x\in \mathbb R^n)$ (其中 $A\in \mathbb R^{n\times n}$ 是对称阵)  
  则通过 $(u\pm v)^{\mathrm T}A(u\pm v) = u^{\mathrm T}Au + v^{\mathrm T} A v \pm 2u^{\mathrm T}Av$   
  我们可以得到 $u^{\mathrm T}Av = \frac14 (u+v)^{\mathrm T} A(u+v) - \frac14 (u-v)^{\mathrm T} A (u-v) = \frac14 f(u+v) - \frac14 f(u-v)$   
  因此我们没必要直接研究非对称的二次型 $u^{\mathrm T} A v$，只需要研究对称的二次型 $x^{\mathrm T}Ax$ 即可.  
  这里就用到了极化恒等式的思想.



### 2.2.5 平行四边形恒等式

**平行四边形恒等式 (Parallelogram Identity)**  
设 $(V , \langle\cdot ,\cdot\rangle)$ 为建立在 $\mathbb{F}=\mathbb R\text{ or }\mathbb C$ 上的内积空间，$\|\cdot\|$ 为 $\langle\cdot,\cdot \rangle$ 诱导出的范数，则我们有:  
$$
\|x+y\|^2 +\|x-y\|^2 = 2\|x\|^2 + 2\|y\|^2\ \ (\forall\ x,y\in V).
$$
展开消去交叉项即可证明:    
$$
\begin{align}
\|x+y\|^2 +\|x-y\|^2 
&= \langle x+y ,x+y\rangle + \langle x-y ,x-y\rangle\\
&= \langle x ,x\rangle+\langle x ,y\rangle + \langle y ,x\rangle + \langle y ,y\rangle + \langle x ,x\rangle - \langle x ,y\rangle -\langle y ,x\rangle + \langle y ,y\rangle \\ 
&=  2\langle x ,x\rangle + 2\langle y ,y\rangle \\
&=  2\|x\|^2 + 2\|y\|^2
\end{align}.
$$

- 上式在 Euclid 几何中的意义即 "平行四边形对角线长度的平方和等于四条边长度的平方和"，  
  因此得名 "平行四边形恒等式".
  
- 实际上，平行四边形恒等式是判断一个范数 $\|\cdot\|$ 是否可以由内积导出的**充要条件**.  
  (必要性可由极化恒等式相加证明，而充分性可通过验证极化恒等式构造的形式内积满足内积定义来证明)  
  有些范数 (例如 $\|\cdot\|_1$ 和 $\|\cdot\|_\infty$) 不满足平行四边形恒等式，因此不能被某个内积诱导出来.  
  **也就是说，平行四边形恒等式刻画了什么样的范数可由内积导出**，  
  这为区分内积空间与一般的赋范空间提供了一个重要判据.
  
- 常见的能由内积导出的范数是 $l_2$ 范数和 Frobenius 范数.  
  前者可由 Euclid 内积 $\langle x,y\rangle_2 = y^{\mathrm H}x = \sum_{i=1}^n x_i \bar y_i$ 导出，    
  后者可由 Frobenius 内积 $\langle A,B\rangle_{\mathrm F} = \tr(B^{\mathrm H} A) = \sum_{i=1}^m \sum_{j=1}^n a_{ij} \bar {b}_{ij}$ 导出.  
  事实上 $l_2$ 范数是 Frobenius 范数的特例.  
  而 $l_1$ 范数和 $l_\infty$ 范数不满足平行四边形恒等式 (举反例即可)，因此不能由内积导出.
  
  **(待补充: Matrix Analysis 问题 $5.2\ \text{P}5 \& \text{P}7$)**



### 2.2.6 正交性

设 $(X,\langle \cdot,\cdot\rangle)$ 是内积空间，集合 $S_1,S_2\subseteq X$   

- 我们定义 $x,y\in X$ 的**夹角**为:  
  $$
  \theta := \arccos\left( \frac{|\langle x,y\rangle|}{\|x\|\|y\|}\right)\in [0,\pi]
  $$

- 若 $\langle x,y\rangle = 0$，则我们称 $x,y\in X$ 是正交的，记作 $x\ \bot\ y$

- 若对于任意 $y\in S_2$ 都有 $x\ \bot\ y$ 成立，则我们称 $x\in X$ 和 $S_2\subseteq X$ 正交，记作 $x\ \bot\ S_2$ 

- 若对于任意 $x\in S_1,y\in S_2$ 都有 $x\ \bot\ y$ 成立，则我们称 $S_1,S_2\subseteq X$ 正交，记作 $S_1\ \bot\ S_2$ 

**(勾股定理, 应用泛函分析 定理 $4.3.1$)**   
若 $x^{(1)},\dots,x^{(m)}$ 在内积空间 $(X,\langle \cdot,\cdot\rangle)$ 中两两正交，则我们有:  
$$
\left\|\sum_{k=1}^m x^{(k)}\right\|^2 = \sum_{k=1}^m \|x^{(k)}\|^2.
$$

****

我们定义 $S\subseteq X$ 的**正交补**为:
$$
S^\bot := \{x\in X:x\ \bot\ S\} = \{x\in X:x\ \bot\ y\text{ for all }y\in S\}
$$
**(工科泛函分析基础 定理 $4.2.3$, 泛函分析讲义 定理 $4.2.1$)**  
设 $(X,\langle \cdot,\cdot\rangle)$ 是内积空间.  
$X$ 的任意子集 $S\subseteq X$ 的正交补 $S^\bot$ 都是 $X$ 线性闭子空间.

**证明:**

- **① 首先证明 $S^\bot$ 是 $X$ 的线性子空间:**  
  $$
  \begin{align}
  \langle \alpha x + \beta y,z\rangle
  &=
  \alpha \langle x,z\rangle  + \beta\langle y,z\rangle\\
  &=
  \alpha\cdot 0 + \beta \cdot 0\\
  &= 
  0
  \end{align}\quad (\forall\ x,y\in S^\bot,z\in S,\alpha,\beta\in \mathbb F)
  $$

- **② 其次证明 $S^\bot$ 是 $X$ 中的闭集:** 
  考虑 $S^\bot$ 中的收敛序列 $\{x_n\}$，设其满足 $x_n\to x\ (n\to\infty)$ (其中 $x\in X$)  
  根据内积的连续性可知，对于任意 $z\in S$ 我们都有:  
  $$
  \begin{align}
  \langle x,z\rangle
  &=
  \left\langle \lim_{n\to\infty} x_n,z\right\rangle\\
  &=
  \lim_{n\to\infty} \langle x_n,z\rangle\quad (\text{note that }x_n\in S^\bot\text{ so that }\langle x_n,z\rangle = 0)\\
  &=
  \lim_{n\to\infty} 0\\
  &=
  0
  \end{align}
  $$
  于是 $x\ \bot\ z\ (\forall\ z\in S)$，表明 $x\ \bot\ S^\bot$.  
  因此 $S^\bot$ 是 $X$ 中的闭集.

**正交补空间还具有如下性质:**  
设 $(X,\langle \cdot,\cdot\rangle)$ 是内积空间，记子集 $S\subseteq X$ 的正交补空间为 $S^\bot$ 

- $S^\bot \cap S \subseteq \{0_X\}$ (取等当且仅当 $0_X\in S$) 
- $S^\bot = (\text{cl}(S))^\bot = (\text{span}(S))^\bot$ 
- $S \subseteq (S^\bot)^\bot = \text{cl}(S)=\text{span}(S)$ 
- 若 $S\subseteq M$，则 $M^\bot \subseteq S^\bot$
- **(应用泛函分析, 定理 $4.3.3$)**  
  设 $(X,\langle \cdot,\cdot\rangle)$ 是内积空间，$S\subseteq X$ 是 $X$ 的子集.  
  若 $\text{cl}(S)=X$ (即 $S$ 是 $X$ 的稠密子集)，则 $S^\bot = (\text{cl}(S))^\bot = X^\bot = \{0_X\}$.

****

**(Matrix Analysis 定理 $7.2.10$)**   
设 $(V, \langle \cdot,\cdot\rangle)$ 为域 $\mathbb C$ 上的内积空间，$v_1,\dots,v_n$ 是 $V$ 中的一组向量.  
若定义 $G:=[\langle v_j,v_i\rangle]_{i,j=1}^n \in \mathbb C^{n\times n}$，则有下列命题成立:

- ① $G$ 一定是 Hermite 半正定阵
- ② $G$ 是 Hermite 正定阵当且仅当 $v_1,\dots,v_n$ 线性无关
- ③ $\rank(G)= \dim(\text{span}\{v_1,\dots,v_n\})$.

我们称 $G:=[\langle v_j,v_i\rangle]_{i,j=1}^n \in \mathbb C^{n\times n}$ 为 $v_1,\dots,v_n$ 的**度量矩阵** (Gram matrix).

**证明:**  

- ① 记 $\langle \cdot,\cdot\rangle$ 诱导出的范数为 $\|\cdot\|$.  
  根据内积的共轭对称性可知 $G=[\langle v_j,v_i\rangle]_{i,j=1}^n$ 一定是 Hermite 阵.  
  同时对于任意 $x=[x_i]\in \mathbb C^n$ 我们都有:  
  $$
  \begin{align}
  x^{\mathrm H}Gx
  &=
  \sum_{i,j=1}^n \langle v_j,v_i\rangle \bar x_i x_j\\
  &=
  \sum_{i,j=1}^n \langle x_jv_j,x_iv_i\rangle\\
  &=
  \left\langle \sum_{j=1}^n x_jv_j, \sum_{i=1}^n x_iv_i\right\rangle\\
  &=
  \left\| \sum_{i=1}^n x_iv_i\right\|^2\\
  &\geq 0.
  \end{align}
  $$
  因此 $G$ 是 Hermite 半正定阵.

- ② 注意到 ① 中的不等式 $x^{\mathrm H}Gx \geq 0$ 取等当且仅当 $\sum_{i=1}^n x_iv_i=0$   
  因此 $G$ Hermite 正定等价说明 $\sum_{i=1}^n x_iv_i=0$ 当且仅当 $x=0_n$，进而等价于 $v_1,\dots,v_n$ 线性无关.  

- ③ 记 $r=\rank(G)$ 和 $d=\dim(\text{span}\{v_1,\dots,v_n\})$，显然有 $r,d\geq 1$. 

  - 一方面，$G$ 一定存在 $r$ 阶的非奇异 (从而 Hermite 正定的) 主子阵.  
    注意到这个主子阵是 $v_1,\dots,v_n$ 中 $r$ 个向量的度量矩阵，  
    根据 ② 可知这 $r$ 个向量线性无关，于是我们有 $r\leq d$.
  - 另一方面，$v_1,\dots,v_n$ 中有 $d$ 个向量是线性无关的，  
    根据 ② 可知这 $d$ 个向量的度量矩阵是 Hermite 正定的.  
    注意到这 $d$ 个向量的度量矩阵是原度量矩阵的 $d$ 阶主子阵，  
    于是我们有 $d\leq r$。
  
  因此我们有 $r=d$，即 $\rank(G)= \dim(\text{span}\{v_1,\dots,v_n\})$.

**(应用举例)**  
试证明 Hilbert 矩阵 $H:=[\frac{1}{i+j-1}]_{i,j=1}^n$ 是 Hermite 正定阵.  

我们只需证明 $H$ 可以表示某个内积空间中一组线性无关的向量的度量矩阵即可.  
考虑闭区间 $[0,1]$ 上的实系数多项式函数空间 $P([0,1])$   
它常用的内积是 $\langle x,y\rangle := \int_0^1 x(t)y(t)\mathrm{d}t\ (\forall\ x,y\in P([0,1]))$  
但我们这里定义内积为 $\langle x,y\rangle := \int_0^1 tx(t)y(t)\mathrm{d}t\ (\forall\ x,y\in P([0,1]))$  
容易验证 Hilbert 矩阵 $H:=[\frac{1}{i+j-1}]_{i,j=1}^n$ 是线性无关的向量组 $1,t,\dots,t^{n-1}$ 的度量矩阵.  
根据 **Matrix Analysis 定理 $7.2.10$** 的结论可知 $H$ 为 Hermite 正定阵.

*****

若 $q_1,\dots,q_n$ 是有限维内积空间 $V$ 的一组基，  
且 $\langle q_i,q_j\rangle = \delta _{ij} = \begin{cases}
1 & i=j\\
0 & i\neq j\end{cases}$ (即度量矩阵 $G=[\langle q_j,q_i\rangle]_{i,j=1}^n$ 是单位矩阵) 
则我们称 $q_1,\dots,q_n$ 为 $V$ 的一组**标准正交基** (orthonormal basis)

有限维内积空间 (一定是 Hilbert 空间) 一定具有标准正交基 (无限维的情况要小心)  
一组普通的基可通过 Gram–Schmidt 正交化得到标准正交基.  

- 设 $v_1,\dots,v_n$ 是 $V$ 的任意一组基.   
  我们可以使用**经典 Gram–Schmidt 过程**将 $v_1,\dots,v_n$ 标准正交化得到 $q_1,\dots,q_n$:   
  (由于 $v_1,\dots,v_n$ 线性无关，故这个过程可以完整进行)
  $$
  v_1 = q_1 r_{11}\\
  \Downarrow\\
  \begin{cases}
  r_{11} = \|v_1\|_2\\
  q_1 = \frac{v_1}{\|v_1\|_2} = \frac{v_1}{r_{11}}
  \end{cases}\\
  v_k 
  = \sum_{i=1}^k q_i r_{ik} 
  = \sum_{i=1}^k q_i\langle q_i,v_k\rangle 
  = q_k r_{kk} + \sum_{i=1}^{k-1} q_i\langle q_i, v_k\rangle
  \ (k=2,\dots,n)\\
  \Downarrow\\
  \text{for }k=2,\dots,n\ \begin{cases}
  r_{ik} = \langle q_i,v_k\rangle\ (i=1,\dots,k-1)\\
  r_{kk} = \|v_k - \sum_{i=1}^{k-1} q_i r_{ik}\|_2\\
  q_k = \frac{1}{r_{kk}} (v_k - \sum_{i=1}^{k-1} q_i r_{ik})
  \end{cases}
  $$
  可以证明 $q_1,\dots,q_n$ 是 $V$ 的一组基 (而且是标准正交的)  
  否则必然存在某个 $q_i=0_V$，这与 $\|q_i\| = \langle q_i,q_i\rangle =  1$ 的事实矛盾. 

*****

上述 Gram–Schmidt 过程的矩阵形式即满秩方阵的 $\text{QR}$ 分解:  
$$
V  = [v_1,v_2,\dots,v_n]
= [q_1,q_2,\dots,q_n]
\begin{bmatrix}
r_{11} & r_{12} & \dotsm & r_{1n}\\
& r_{22} & \dotsm & r_{2n}\\
& & \ddots & \vdots\\
& & & r_{nn}
\end{bmatrix} = QR.
$$
更一般地，我们有:  
**(QR 分解, Matrix Analysis 定理 $2.1.14$)**   
给定 $A\in \mathbb{C}^{m\times n}$，我们有下列命题成立:  
(若 $A$ 退化为实矩阵，则下列命题中的 $Q,R$ 均可取为实矩阵)

* ① 若 $m\geq n$，则存在一个列标准正交的复矩阵 $Q\in \mathbb{C}^{m\times n}$ (特殊地，当 $m=n$ 时为酉矩阵)  
  和一个对角元均为非负实数的上三角矩阵 $R \in \mathbb{C}^{n\times n}$ 使得 $A=QR$   
  这样的 $Q,R$ 可能是不唯一的.  
  其中 $R$ 的 $0$ 对角元所在行的元素全置为 $0$，对应的 $Q$ 的列向量是随意填入的 (只需保证 $Q$ 列标准正交即可)  

  > 一个具体的例子:  
  > 假设 $v_1,v_2,v_3,v_4,v_5$ 的秩是 $4$，而 $v_3$ 是 $v_1,v_2$ 的线性组合，则我们有:  
  > $$
  > [v_1,v_2,v_3,v_4,v_5] = [q_1,q_2,q_3,q_4,q_5] 
  > \begin{bmatrix}
  > r_{11} & r_{12} & r_{13} & r_{14} & r_{15}\\
  > & r_{22} & r_{23} & r_{24} & r_{25}\\
  > & & 0 & 0 &0\\
  > & & & r_{44} & r_{45}\\
  > & & & & r_{55}
  > \end{bmatrix}
  > $$
  > 其中 $\text{span}\{q_1,q_2,q_4,q_5\}=\text{span}\{v_1,v_2,v_3,v_4,v_5\}$  
  > 而 $q_3$ 是随意填入的 (只需保证 $q_1,q_2,q_3,q_4,q_5$ 标准正交即可)
  
  当 $m> n$ 时，我们还可将 $Q\in \mathbb C^{m\times n}$ 补成一个 $m$ 阶酉矩阵 $[Q,Q_\bot]\in \mathbb C^{m\times m}$:  
  $$
  A = QR = \begin{bmatrix} Q & Q_\bot\end{bmatrix}
  \begin{bmatrix}
  R\\
  0_{(m-n)\times n}
  \end{bmatrix}
  $$
  
* ② 若 $m \geq n$ ，记 $r=\rank(A)$，则我们有精简 $\text{QR}$ 分解 $A=QR$   
  其中 $Q\in \mathbb C^{m\times r}$ 列标准正交，而 $R=[R_1, R_2]\in \mathbb C^{r\times n}$   
  ($R_1\in \mathbb C^{r\times r}$ 是具有正实数对角元的上三角阵，而 $R_2\in \mathbb C^{r\times (n-r)}$)

  特殊地，若 $\rank(A) = n$ (即 $A$ 列满秩)  
  则第一个命题中的 $Q,R$ 是唯一的，且 $R$ 的对角元均为正实数 (此时它就是精简 $\text{QR}$ 分解)

***

一个重要的几何事实是:   
任意两个有相同个数的标准正交向量组都可通过酉变换联系在一起.  
**(Matrix Analysis 定理 $2.1.18$)**  
设 $X= [x_1,x_2,\dots,x_k] \in \mathbb{C}^{n\times k}$ 和 $Y = [y_1,y_2,\dots,y_k] \in \mathbb{C}^{n\times k}$ 的列向量标准正交，  
则存在一个酉矩阵 $U\in \mathbb C^{n\times n}$ 使得 $Y=UX$.  
(若 $X,Y$ 都是实矩阵，则 $U$ 可以取成实正交阵)

- **证明:**  
  分别将 $X,Y$ 的列向量组扩充为 $\mathbb C^n$ 的一组标准正交基，记为 $\widetilde X=[X,X_\bot]$ 和 $\widetilde Y=[Y,Y_\bot]$.   
  取 $U:=\widetilde Y \widetilde X^{\mathrm H}$，注意到:
  $$
  U^{\mathrm H}U = (\widetilde Y \widetilde X^{\mathrm H})^{\mathrm H}\widetilde Y \widetilde X^{\mathrm H} = \widetilde X\widetilde Y^{\mathrm H}\widetilde Y \widetilde X^{\mathrm H} = I_n\\
  UX= \widetilde Y \widetilde X^{\mathrm H} X = [Y,Y_\bot]
  \begin{bmatrix}
  X^{\mathrm H}X\\
  X_\bot^{\mathrm H}X
  \end{bmatrix}
  =
  [Y,Y_\bot]
  \begin{bmatrix}
  I_k\\
  0_{(n-k)\times (n-k)}
  \end{bmatrix} =
  Y
  $$
  因此 $U$ 是一个酉矩阵，且满足 $Y=UX$.



## 2.3 矩阵范数

### 2.3.1 基本定义

由于 $\mathbb C^{m\times n}$ 和 $\mathbb C^{mn}$ 是同构的，故矩阵范数的定义应等价于向量范数的定义.  
函数 $\|\cdot\|:\mathbb C^{m\times n}\mapsto \mathbb R$ 称为一个**矩阵范数**，如果它满足如下四条公理:  
任意给定 $A,B\in \mathbb C^{m\times n}$ 和 $\alpha\in \mathbb C$

- 非负性: $\|A\|\geq 0$ 
- 正定性: $\|A\|=0$ 当且仅当 $A=0_{m\times n}$ 
- 齐次性: $\|\alpha A\|=|\alpha|\|A\|$
- 次可加性 (三角不等式): $\|A+B\|\leq \|A\|+\|B\|$ 

值得注意的是，上述定义的是**一族矩阵范数**.  

我们称 $\mathbb C^{m\times q},\mathbb C^{m\times n},\mathbb C^{n\times q}$ 上的范数 $f_1,f_2,f_3$ 是**相容的** (compatible)，  
如果它们对于任意 $A\in \mathbb C^{m\times n}$ 和 $B\in \mathbb C^{n\times q}$ 满足:

- 次可积性 (相容性): $f_1(AB)\leq f_2(A)f_3(B)$ 

如果一族矩阵范数相互之间满足相容性，则我们称其为**相容范数族**.   
特别地，若 $\mathbb C^{n\times n}$ 上的范数 $\|\cdot\|$ 和自身满足相容性，则我们称其为**(自)相容的**.  
根据定义我们知道:

- 对于 $\mathbb C^{n\times n}$ 上的任何相容范数 $\|\cdot\|$ 我们都有 $\|A^2\|\leq \|A\|^2$.  
  (事实上，对于任意 $k=1,2,\dots$ 都有 $\|A^k\|\leq \|A\|^k$)  
  因此对于任何满足 $A^2 = A$ 的矩阵，我们都有 $\|A\|\geq 1$.   
  特别地，我们有 $\|I_n\|\geq 1$.  
  我们称满足 $\|I_n\|=1$ 的矩阵范数为**单位的** (unital).
- 若 $A\in \mathbb{C}^{n\times n}$ 是非奇异的，  
  则对于 $\mathbb C^{n\times n}$ 上的任何相容范数 $\|\cdot\|$ 我们都有 $\|I_n\| = \|AA^{-1}\| = \|A\|\|A^{-1}\|$ 成立，  
  即有 $\|A^{-1}\|$ 的下界估计 $\|A^{-1}\|\geq \|I_n\|/\|A\|$.

****

可以证明: 通过向任何一个 $\mathbb C^{n\times n}$ 上的相容范数中插入一个固定的相似变换，就可以导出新的相容范数.  
**(Matrix Analysis 定理 $5.6.7$)**  
若 $\|\cdot\|$ 是 $\mathbb C^{n\times n}$ 上的一个相容范数，且 $S\in \mathbb C^{n\times n}$ 是非奇异的， 
则由 $\|A\|_S:= \|SAS^{-1}\|\ (\forall\ A\in \mathbb C^{n\times n})$ 定义的函数 $\|\cdot\|_S$ 都是一个相容范数.   
特别地，若相容范数 $\|\cdot\|$ 是 $\mathbb C^n$ 上的范数 $\|\cdot\|$ 诱导的，则相容范数 $\|\cdot\|_S$ 是 $\mathbb C^n$ 上的范数 $\|\cdot\|_S$ 所诱导的.  
(其中范数 $\|\cdot\|_S : \mathbb C^n\mapsto \mathbb R$ 的定义为 $\|x\|_S:=\|Sx\|\ (\forall\ x\in \mathbb C^n)$)

- **证明:**  
  非负性、正定性、齐次性和次可加性都是显然的.  
  下证次可积性:  
  $$
  \begin{align}
  \|AB\|_S 
  &=
  \|SABS^{-1}\|\\
  &=
  \|(SAS^{-1})(SBS^{-1})\|\\
  &\leq
  \|SAS^{-1}\|\|SBS^{-1}\|\\
  &=
  \|A\|_S\|B\|_S
  \end{align}\ \ (\forall\ A,B\in \mathbb C^{n\times n})
  $$
  下证最后一个结论:  
  $$
  \begin{align}
  \max_{\|x\|_S=1}\|Ax\|_S
  &=
  \max_{\|Sx\|=1} \|SAx\|\\
  &=
  \max_{\|y\|=1}\|SAS^{-1}y\|\\
  &= 
  \|SAS^{-1}\|\\
  &=
  \|A\|_S
  \end{align}
  $$
  表明由 $\mathbb C^n$ 上的范数 $\|x\|_S:= \|Sx\|\ (\forall\ x\in \mathbb C^n)$ 所诱导的矩阵范数就是 $\|A\|_S:= \|SAS^{-1}\|\ (\forall\ A\in \mathbb C^{n\times n})$ 



### 2.3.2 将 $l_p$ 范数直接推广到矩阵空间

为避免本小节的记号与诱导范数冲突，我们做以下约定:  
用 $\|\cdot\|_{l_p}$ 表示对矩阵直接应用 $l_p$ 范数，而用 $\|\cdot\|_p$ 代表由 $l_p$ 范数诱导的矩阵范数.

- **(对矩阵空间直接应用 $l_1$ 范数)**  
  我们定义 $\mathbb C^{m\times n}$ 上的 $l_1$ 范数为 $\|A\|_{l_1}:= \sum_{i,j=1}^{m,n}|a_{ij}|$.  
  可以验证它满足次可乘性，因而是相容范数:  
  $$
  \begin{align}
  \|AB\|_{l_1}
  &=
  \sum_{i,j=1}^{m,n} \left|\sum_{k=1}^q a_{ik}b_{kj}\right|\\
  &\leq
  \sum_{i,j,k=1}^{m,n,q} |a_{ik}b_{kj}| \quad (\text{triangle inequality})\\
  &\leq 
  \sum_{i,j,k,l=1}^{m,n,q,q} |a_{ik}b_{lj}|\quad (\text{adding non-negative terms})\\
  &=
  \left(\sum_{i,k=1}^{m,q} |a_{ik}|\right)\left(\sum_{j,l=1}^{n,q} |b_{lj}|\right)\\
  &=
  \|A\|_{l_1} \|B\|_{l_1}.
  \end{align}\quad (\forall\ A\in \mathbb C^{m\times q},B\in \mathbb C^{q\times n})
  $$
  尽管 $\|\cdot\|_{l_1}$ 是相容范数族，但后面我们不会再使用它，因此弃用这个记号.
  
- **(对矩阵空间直接应用 $l_\infty$ 范数)**   
  我们定义 $\mathbb C^{m\times n}$ 上的 $l_\infty$ 范数为 $\|A\|_{l_\infty}:= \max_{1\leq i\leq m,1\leq j\leq n}|a_{ij}|$  
  可以举例说明它不满足次可乘性，因而不是相容范数:
  $$
  A = \begin{bmatrix}
  1 & 1\\
  1 & 1
  \end{bmatrix}
  \quad A^2 = \begin{bmatrix}
  2 & 2\\
  2 & 2
  \end{bmatrix}\\
  
  \|A^2\|_{l_\infty} = 2 > 1 = \|A\|_{l_\infty}^2
  $$
  然而我们可以定义 $\mathbb C^{n\times n}$ 上的函数 $N(A):= n\|A\|_{l_\infty}$，显然它仍是一个范数.  
  可以验证它满足次可乘性，因而是一个相容范数 (更多讨论参见 **Matrix Analysis $5.7.11$ 节**):  
  $$
  \begin{align}
  N(AB)
  &=
  n \max_{1\leq i,j\leq n} \left|\sum_{k=1}^n a_{ik}b_{kj}\right|\\
  &\leq
  n \max_{1\leq i,j\leq n} \sum_{k=1}^n |a_{ik}||b_{kj}|\\
  &\leq 
  n \max_{i\leq i,j\leq n} \{n\|A\|_{l_\infty} \|B\|_{l_\infty}\}\\
  &=
  n \|A\|_{l_\infty} \cdot n\|B\|_{l_\infty}\\
  &=
  N(A)N(B)
  \end{align}\quad (\forall\ A,B\in \mathbb C^{n\times n})
  $$
  尽管 $N(\cdot)$ 是相容范数，但后面我们不会再使用它，因此弃用这个记号.
  
- **(对矩阵空间直接应用 $l_2$ 范数)**  
  我们定义 $\mathbb C^{m\times n}$ 上的 $l_2$ 范数为 $\|A\|_{l_2} := \|\text{vec}(A)\|_2 =  (\sum_{i,j=1}^{m,n}|a_{ij}|^2)^{1/2}$.  
  可以验证它满足次可乘性，因而是相容范数:     
  $$
  \begin{align}
  \|AB\|_{l_2} 
  &=
  \left(\sum_{i,j=1}^{m,n} \left|\sum_{k=1}^q a_{ik}b_{kj}\right|^2\right)^\frac12\\
  &\leq
  \left(\sum_{i,j=1}^{m,n} \left(\sum_{k=1}^q |a_{ik}||b_{kj}|\right)^2\right)^\frac12\\
  &\leq 
  \left(\sum_{i,j=1}^{m,n} \left(\left(\sum_{k=1}^q |a_{ik}|^2\right)\left(\sum_{l=1}^q|b_{lj}|^2\right)\right)\right)^\frac12\quad (\text{Cauchy–Schwarz inquality})\\
  &=
  \left(\sum_{i,k=1}^{m,q} |a_{ik}|^2\right)^\frac12 \left(\sum_{j,l=1}^{n,q} |b_{lj}|^2\right)^\frac12\\
  &=
  \|A\|_{l_2} \|B\|_{l_2}\quad (\forall\ A\in \mathbb C^{m\times q},B\in \mathbb C^{q\times n})
  \end{align}
  $$
  我们称 $\|\cdot\|_{l_2}$ 为 **Frobenius 范数** (又称 Schur 范数或 Hilbert–Schimidt 范数)，并更改记号为 $\|\cdot\|_{\mathrm F}$.
  
  ****
  
  下面我们从另一个角度引入 Frobenius 范数 $\|\cdot\|_{\mathrm F}$.  
  定义 Frobenius 内积 $\langle \cdot ,\cdot\rangle_{\mathrm F}:\mathbb C^{m\times n}\times \mathbb C^{m\times n}\mapsto \mathbb C$ 为:  
  $$
  \langle A,B\rangle_{\mathrm F} := \tr(B^{\mathrm H}A) \ \ (\forall\ A,B\in \mathbb C^{m\times n})
  $$
  则我们可以定义 Frobenius 范数 $\|\cdot\|_{\mathrm F}:\mathbb C^{m\times n}\mapsto \mathbb R$ 为:     
  $$
  \|A\|_{\mathrm F}:= \sqrt{\langle A,A\rangle_{\mathrm F}}\ \ (\forall\ A\in \mathbb C^{m\times n})
  $$
  **Frobenius 范数的性质:**
  
  * **计算公式:** $\|A\|_{\mathrm F} = \sqrt{\tr(A^{\mathrm H}A)} = \sqrt{\sum_{i=1}^n \lambda_i(A^{\mathrm H}A)} =\sqrt{\sum^{\min\{m,n\}}_{i=1} \sigma^2_i(A)} = \|\text{vec}(A)\|_2$ 
  
  * **酉不变性:** 对任意酉矩阵 $U\in \mathbb C^{m\times m}$ 和 $V\in \mathbb C^{n\times n}$ 都有 $\|UAV\|_{\mathrm F} = \|A\|_{\mathrm F}$   
    $$
    \begin{align}
    \|UAV\|_{\mathrm F}^2
    &=
    \tr[(UAV)^{\mathrm H}(UAV)]\\
    &=
    \tr(V^{\mathrm H}A^{\mathrm H}U^{\mathrm H} UAV)\\
    &=
    \tr(V^{\mathrm H}A^{\mathrm H}AV)\\
    &=
    \tr(A^{\mathrm H}AVV^{\mathrm H})\\
    &=
    \tr(A^{\mathrm H}A)
    \end{align}
    $$
    
  * 当 $\min\{m,n\}>1$ 时，Frobenius 范数不是诱导范数，参见 **Homework 04 Problem 06**.
    
  * 对于任意 $A\in \mathbb C^{m\times n}$ 我们都有:   
    $$
    \begin{align}
    \|A\|_2 
    &= \sqrt{\rho(A^{\mathrm H}A)}\\
    &\leq \|A\|_{\mathrm F}\\
    &= \sqrt{\sum_{i=1}^{\min\{m,n\}} \sigma^2_i (A)}\\
    &= \sqrt{\sum_{i=1}^{\min\{m,n\}} \lambda_i (A^{\mathrm H}A)}\\
    &\leq \sqrt{\min\{m,n\}\rho(A^{\mathrm H}A)}\\
    &= \sqrt{\min\{m,n\}}\|A\|_2
    \end{align}
    $$
    其中 $\|\cdot\|_2$ 代表谱范数，参见 $2.3.4(3)$ 的内容.




### 2.3.3 诱导范数

设 $\|\cdot\|$ 为有限维复 Euclid 空间上的向量范数族  
我们定义 $\mathbb C^{m\times n}$ 上相应的**诱导范数** (induced norm) $\|\cdot\|:\mathbb C^{m\times n}\mapsto \mathbb R$ 为:  
$$
\|A\|:= \sup_{\|x\|=1}\|Ax\|=\max_{\|x\|=1} \|Ax\|\ \ (\forall\ A\in \mathbb C^{m\times n})
$$
注意到向量范数 $\|\cdot\|$ 具有连续性，且范数单位球 $\{x\in \mathbb C^n:\|x\|_2=1\}$ 是一个紧集.  
由于紧集上的连续函数一定可以取到上确界，且上确界是个有限数，  
故在有限维空间上我们可以放心大胆地把诱导范数定义中的 $\sup$ 写成 $\max$   
但在无限维空间中，诱导范数定义中的上确界和最大值还是要加以区分的.  

可以证明上述定义的诱导范数可以按照下面的任意一种方式计算:  
$$
\begin{align}
\|A\|
&= \max_{\|x\|=1} \|Ax\|\\
&= \max_{\|x\|\leq 1}\|Ax\|\\
&= \max_{\|x\|\neq 0}\frac{\|Ax\|}{\|x\|}\\
&= \max_{\|x\|_\alpha = 1} \frac{\|Ax\|}{\|x\|}\\
&(\text{where }\|\cdot\|\text{ denotes a given family of norm on finite-dimensional complex Euclidean Space})
\end{align}
$$
**(Matrix Analysis 定理 $5.6.2$)**  
设 $\|\cdot\|$ 为有限维复 Euclid 空间上的向量范数族，  
则由 $\|A\|:= \max_{\|x\|=1} \|Ax\|\ \ (\forall\ A\in \mathbb C^{m\times n},m,n\in \mathbb Z_+)$ 定义的诱导范数族具有如下性质:  

- ① $\|I_n\|=1$   
- ② 对于任意 $A\in \mathbb C^{m\times n}$ 和 $y\in \mathbb C^n$ 我们都有 $\|A y\|\leq \|A\|\|y\|$ 成立，  
  这表明向量范数与其诱导范数是**相容的** (compatible).
- ③ 诱导范数族 $\|\cdot\|$ 是相容范数族
- ④ $\|A\| = \max_{\|x\|=\|y\|^D = 1} |y^{\mathrm H}Ax|$

**证明:**

- ① $\|I_n\| = \max_{\|x\|=1} \|I_n x\| = \max_{\|x\|=1} \|x\| =1$ 

- ② 当 $y=0_n$ 时命题显然成立;  
  当 $y\neq 0_n$ 时，不妨考虑单位向量 $y/\|y\|$，我们有:  
  $$
  \|A\| = \max_{\|x\|=1} \|Ax\| \geq \left\|A \frac{y}{\|y\|}\right\| = \frac{\|Ay\|}{\|y\|}\\
  \Updownarrow\\
  \|A y\|\leq \|A\|\|y\|
  $$
  结论 ② 得证.

- ③ 我们逐一验证相容范数定义的五条公理:

  - **非负性:**   
    $\|A\|= \max_{\|x\|=1} \|Ax\|$ 作为非负函数的最大值，故它是非负的.

  - **正定性:**   
    当 $A=0_{m\times n}$ 时显然有 $\|A\|=0$.  
    当 $A\neq 0_{m\times n}$ 时，必然存在一个单位向量 $x^{(0)}\in \mathbb C^n$ 使得 $Ax^{(0)}\neq 0_m$.  
    于是有 $\|A\|= \max_{\|x\|=1} \|Ax\| \geq \|Ax^{(0)}\| >0$.

  - **齐次性:**
    $$
    \begin{align}
    \|c A\|
    &=
    \max_{\|x\|=1} \|cAx\|\\
    &=
    \max_{\|x\|=1} \{|c| \|Ax\|\}\\
    &=
    |c| \max_{\|x\|=1} \|Ax\|\\
    &=
    |c|\|A\|
    \end{align}\quad(\forall\ A\in \mathbb C^{m\times n},c\in \mathbb C)
    $$

  - **次可加性:**  
    $$
    \begin{align}
    \|A+B\|
    &=
    \max_{\|x\|=1}\|(A+B)x\|\\
    &\leq
    \max_{\|x\|=1} \{\|Ax\|+\|Bx\|\}\\
    &=
    \max_{\|x\|=1}\|Ax\| + \max_{\|x\|=1}\|Bx\|\\
    &= 
    \|A\| + \|B\|
    \end{align}\quad (\forall\ A,B\in \mathbb C^{m\times n})
    $$

  - **次可乘性:**  
    $$
    \begin{align}
    \|AB\|
    &=
    \max_{\|x\|=1} \|ABx\|\\
    &=
    \max_{\|x\|=1} \|A(Bx)\|\\
    &\leq 
    \max_{\|x\|=1} \{\|A\|\|Bx\|\} \quad (\text{note that the induced norm is compatible with the norm that induced it})\\
    &=
    \|A\| \max_{\|x\|=1} \|Bx\|\\
    &=
    \|A\| \|B\|
    \end{align}
    $$

  综上所述，诱导范数族 $\|\cdot\|$ 是相容范数族，而且它是单位的 (即满足 $\|I_n\|=1$)，  
  我们也称它为与向量范数 $\|\cdot\|$ 相伴的**最小上界范数** (least upper bound norm).

  这样一来，证明矩阵空间上某个非负函数族是相容范数族的一种方法就是证明它是由某个向量范数族诱导出来的.  
  不过并非所有相容范数族都是从某个向量范数族诱导出来的，例如 Frobenius 范数族.

- ④ 注意到范数总是其对偶范数的对偶 (即 $\|\cdot\|^{DD}:=(\|\cdot\|^D)^D = \|\cdot\|$)，故我们有:  
  $$
  \begin{align}
  \max_{\|x\|=\|y\|^D = 1} |y^{\mathrm H}Ax|
  &=
  \max_{\|x\|=1} \left\{\max_{\|y\|^D = 1}|y^{\mathrm H}Ax|\right\}\\
  &=
  \max_{\|x\|=1} \|Ax\|^{DD}\\
  &=
  \max_{\|x\|=1}\|Ax\|\\
  &=
  \|A\|
  \end{align}
  $$



### 2.3.4 由 $l_p$ 范数诱导的矩阵范数

#### (1) 列和范数 $\|\cdot\|_1$

任意给定矩阵 $A=[a_1,\dots,a_n]\in \mathbb C^{m\times n}$  
对于任意满足 $\|x\|_1=1$ 的 $x\in \mathbb C^n$，我们都有:  
$$
\begin{align}
\|Ax\|_1 
&=
\sum_{i=1}^m \left|\sum_{j=1}^n a_{ij}x_j\right|\\
&\leq
\sum_{i=1}^m \sum_{j=1}^n |a_{ij}||x_j|\\
&=
\sum_{j=1}^n \left\{\left(\sum_{i=1}^m |a_{ij}|\right)\cdot |x_j|\right\}\\
&=
\sum_{j=1}^n\{\|a_j\|_1 |x_j|\}\\
&\leq
\left(\max_{1\leq j\leq n} \|a_j\|_1\right) \sum_{j=1}^n |x_j|\\
&=
\left(\max_{1\leq j\leq n} \|a_j\|_1\right) \cdot\|x\|_1\\
&=
\left(\max_{1\leq j\leq n} \|a_j\|_1\right) \cdot 1\\
&=
\max_{1\leq j\leq n} \|a_j\|_1
\end{align}

\tag{1}
$$
接下来我们取一个特殊的 $x_0\in \mathbb C^n$，来说明这个上界是可以取到的:  

- 设 $a_{k_0}$ 是 $a_1,a_2,...,a_n$ 中 $l_1$ 范数最大的列向量，即 $\|a_{k_0}\|_1 = \max_{1\leq j\leq n} \|a_j\|_1$     
  取 $x_0 = e_{k_0}$ (其中 $e_{k_0}$ 代表 $\mathbb C^n$ 的标准单位基向量)，它满足:  
  $$
  \|x_0\|_1 =\|e_{k_0}\|_1 = 1\\
  \|Ax_0\|_1 = \|Ae_{k_0}\|_1 = \|a_{k_0}\|_1 = \max_{1\leq j\leq n} \|a_j\|_1
  $$

因此式 $(1)$ 中 $\|Ax\|_1 \leq \underset{1\leq j\leq n}{\max} \|a_j\|_1\ \ (\forall\ x\in \mathbb C^n\text{ such that }\|x\|_1=1)$ 的不等号是可以取等的. 
于是我们有:  
$$
\|A\|_1  = \max_{\|x\|_1=1} \|Ax\|_1 = \max_{1\leq j\leq n} \|a_j\|_1 = \max_{1\leq j\leq n}\left\{\sum_{i=1}^n |a_{ij}|\right\}
$$
这样我们就得到了矩阵范数 $\|\cdot\|_1$，它称为**最大绝对列和矩阵范数**，简称**列和范数**.



#### (2) 行和范数 $\|\cdot\|_\infty$

任意给定矩阵
$$
A=\begin{bmatrix}
a_1^{\mathrm T}\\
\vdots\\
a_m^{\mathrm T}\end{bmatrix}\in \mathbb C^{m\times n}.
$$
对于任意满足 $\|x\|_\infty=1$ 的 $x\in \mathbb C^n$，我们都有:   
$$
\begin{align}
\|Ax\|_\infty 
&= 
\left\| 
\begin{bmatrix}
a_1^{\mathrm T}x\\
\vdots\\
a_m^{\mathrm T}x
\end{bmatrix}
\right\|_\infty\\
&=
\max_{1\leq i\leq m} |a_i^{\mathrm T}x|\\
&=
\max_{1\leq i\leq m} \left|\sum_{j=1}^n a_{ij} x_j\right|\\
&\leq
\max_{1\leq i\leq m} \left\{\sum_{j=1}^n |a_{ij} ||x_j|\right\}\\
&\leq
\|x\|_\infty \cdot \max_{1\leq i\leq m}\sum_{j=1}^n |a_{ij}|\\
&=
1\cdot \max_{1\leq i\leq m}\sum_{j=1}^n |a_{ij}|\\
&=
\max_{1\leq i\leq m}\sum_{j=1}^n |a_{ij}|.
\end{align}

\tag{2}
$$
接下来我们取一个特殊的 $x_0$，来说明这个上界是可以取到的:

- 若 $A$ 是全零矩阵，则没什么要证明的;  
  因此我们可以假设 $A\neq 0_{m\times n}$  
  设 $a_{k_0}^{\mathrm T}$ 是 $a_1^{\mathrm T},a_2^{\mathrm T},...,a_m^{\mathrm T}$ 中 $l_1$ 范数最大的行向量，即 $\|a_{k_0}\|_1 = \max_{1\leq i\leq m} \|a_i\|_1$   
  取 $x_0 = [x^{(0)}_i]\in \mathbb C^n$，它满足: 
  $$
  x^{(0)}_{j} := \begin{cases}
  \frac{\overline{a_{k_0,j}}}{|a_{k_0,j}|} & \text{if }a_{k_0,j}\neq 0\\
  0 & \text{if }a_{k_0,j}=0\end{cases}\\
  \|x_0\|_\infty = 1\\
  \|Ax_0\|_\infty = \max_{1\leq i\leq m} \left|\sum_{j=1}^n a_{ij} x^{(0)}_j\right| \geq 
  \left|\sum_{j=1}^n a_{k_0,j} x^{(0)}_j\right| 
  = 
  \left|\sum_{j=1}^n |a_{k_0,j}|\right| 
  = 
  \sum_{j=1}^n |a_{k_0,j}|= \max_{1\leq i\leq m}\sum_{j=1}^n |a_{ij}|
  $$

因此式 $(2)$ 中 $\|Ax\|_\infty \leq \max_{1\leq i\leq m} \sum_{j=1}^n|a_{ij}|\ \ (\forall\ x\in \mathbb C^n\text{ such that }\|x\|_\infty=1)$ 的不等号是可以取等的.    
于是我们有:
$$
\|A\|_\infty  = \max_{\|x\|_\infty=1} \|Ax\|_\infty = \max_{1\leq i\leq m}\sum_{j=1}^n |a_{ij}|.
$$
这样我们就得到了矩阵范数 $\|\cdot\|_\infty$，它称为**最大绝对行和矩阵范数**，简称**行和范数**.  
容易看出: $\|A\|_\infty = \|A^{\mathrm T}\|_1 = \|A^{\mathrm H}\|_1$.   
(即最大行和矩阵范数等于 (共轭) 转置后的最大列和矩阵范数)



#### (3) 谱范数 $\|\cdot\|_2$

任意给定矩阵 $A\in \mathbb C^{m\times n}$ (不妨假设 $m>n$).  
设 $A$ 的奇异值分解为  
$$
A = U\widetilde \Sigma V^{\mathrm H} = U\begin{bmatrix}
\Sigma\\
0_{(m-n)\times n}\end{bmatrix}V^{\mathrm H},
$$
其中 $U\in \mathbb C^{m\times m},V\in \mathbb C^{n\times n}$ 是酉矩阵，$\Sigma=\text{diag}(\sigma_1,\dots,\sigma_n)$ 且 $\sigma_1\geq \dots \geq \sigma_n\geq 0$.

对于任意满足 $\|x\|_2=1$ 的 $x\in \mathbb C^n$，利用 $l_2$ 范数的酉不变性以及单调性可知:
$$
\begin{align}
\|A\|_2
&=
\max_{\|x\|_2=1}\|Ax\|_2\\
&=
\max_{\|x\|_2=1}\|U \widetilde \Sigma V^{\mathrm H}x\|_2\\
&=
\max_{\|x\|_2=1}\|\widetilde \Sigma V^{\mathrm H} x\|_2\quad (\text{replace }x \text{ with }Vy)\\
&=
\max_{\|Vy\|_2=1} \|\widetilde \Sigma V^{\mathrm H} Vy\|_2\quad (\text{note that }l_2\text{ norm is unitarily invariant})\\
&=
\max_{\|y\|_2 =1} \|\widetilde \Sigma y\|_2\qquad\quad\ \ \  (\text{note that }l_2\text{ norm is monotone})\\
&\leq
\max_{\|y\|_2 =1} \|\sigma_1 y\|_2\\
&=
|\sigma_1| \cdot 1\\
&=
\sigma_1 
\end{align}
$$
注意到上述不等号是可以取等的 (取 $y=e_1\in \mathbb C^n$ 即有 $\|y\|_2=1$ 且 $\|\widetilde \Sigma y\|_2 = \sigma_1$).   
于是我们有:
$$
\|A\|_2 = \sigma_\max(A) = \sqrt{\rho(A^{\mathrm H}A)} = \sqrt{\rho(AA^{\mathrm H})}.
$$
这样我们就得到了矩阵范数 $\|\cdot\|_2$，它称为**谱范数** (spectral norm).

* **酉不变性:** 对于任意酉矩阵 $U\in \mathbb C^{m\times m}$ 和 $V\in \mathbb C^{n\times n}$ 都有 $\|UAV\|_2 = \|A\|_2$ 成立.   
  (这是显然的: 左乘或右乘酉矩阵不改变矩阵的奇异值)

* **对称性:** $\|A\|_2 = \|A^{\mathrm T}\|_2 =\|A^{\mathrm H}\|_2 = \sqrt{\|A^{\mathrm H}A\|_2}$   
  这是显然的: $A,A^{\mathrm T},A^{\mathrm H}$ 具有相同的奇异值，而 $A^{\mathrm H}A$ 的奇异值是 $A$ 对应奇异值的平方.  
  (值得注意的是，酉矩阵 $U$ 的转置 $U^{\mathrm T}$ 仍是酉矩阵，因为 $U^{\mathrm T} (U^{\mathrm T})^{\mathrm H} = [U^{\mathrm H}U]^{\mathrm T} = I^{\mathrm T} = I$) 
  
* **相容性:** 一般的酉不变范数都对谱范数相容，Frobenius 范数也不例外.
  $$
  \begin{align}
  \|AB\|_{\mathrm F}
  &\leq 
  \min\{\|A\|_2 \|B\|_{\mathrm F}, \|A\|_{\mathrm F}\|B\|_2\}\\
  &\leq 
  \max\{\|A\|_2 \|B\|_{\mathrm F}, \|A\|_{\mathrm F}\|B\|_2\}\\
  &\leq 
  \|A\|_{\mathrm F} \|B\|_{\mathrm F}
  \end{align}
  $$
  证明参见 **Homework 03 Problem 05**.
  
* **范数不等式:**   
  **证明:** $\|A\|_2^2 \leq \|A\|_1\|A\|_\infty$ (提前使用谱半径定理)  
  $$
  \begin{align}
  \|A\|_2^2
  &= \rho(A^{\mathrm H}A)\quad (\text{use spectral radius theorem})\\
  &\leq \|A^{\mathrm H}A\|_1\\
  &\leq \|A^{\mathrm H}\|_1\|A\|_1\\
  &= \|A\|_\infty\|A\|_1
  \end{align}
  $$
  **证明: **$\|A\|_2\leq \|A\|_{\mathrm F} \leq \sqrt{\min\{m,n\}}\|A\|_2$ 
  $$
  \begin{align}
  \|A\|_2 
  &\leq \sqrt{\rho(A^{\mathrm H}A)}\\
  &\leq \|A\|_{\mathrm F}\\
  &= \sqrt{\sum_{i=1}^{\min\{m,n\}} \sigma^2_i (A)}\\
  &= \sqrt{\sum_{i=1}^{\min\{m,n\}} \lambda_i (A^{\mathrm H}A)}\\
  &\leq \sqrt{\min\{m,n\}\rho(A^{\mathrm H}A)}\\
  &= \sqrt{\min\{m,n\}}\|A\|_2
  \end{align}
  $$

- **(范数转换的界)** **(存疑, 需要列明来源)**  
  设 $1<p< q <+\infty$，对于任意 $A\in \mathbb C^{m\times n}$，   
  有 $n^{\frac1q-\frac1p}\|A\|_q \leq \|A\|_p \leq m^{\frac1p-\frac1q}\|A\|_q$，而且这是最好的界.

  对于三种常用的 $l_p$ 范数诱导的矩阵范数，有 $\begin{cases} n^{-\frac12}\|A\|_2 \ \leq \|A\|_1 \leq m^{\frac12}\|A\|_2\\ n^{-1}\|A\|_\infty \leq \|A\|_1 \leq m\|A\|_\infty\\ n^{-\frac12}\|A\|_\infty \leq \|A\|_2 \leq m^\frac12\|A\|_\infty \end{cases}$

* 当 $A$ 为正规矩阵 (即满足 $AA^{\mathrm H}=A^{\mathrm H}A$) 时，我们有 $\|A\|_2 = \rho(A)$ 成立.  
  若 $A,B$ 均为正规矩阵，则我们有 $\rho(AB)\leq \|AB\|_2 \leq \|A\|_2\|B\|_2 = \rho(A)\rho(B)$ 成立.  
  (注意 $\rho(AB)\leq \rho(A)\rho(B)$ 对一般的 $A,B\in \mathbb C^{n\times n}$ 并不成立)



### 2.3.5 谱半径定理及其推论

#### (1) 谱半径定理

相容范数的一个重要应用就是为矩阵的谱半径提供界限.  

设 $\lambda$ 是 $A\in \mathbb C^{n\times n}$ 的任意一个特征值，$x$ 是 $A$ 关于 $\lambda$ 的特征向量.  
考虑秩一矩阵 $X=x1_n^{\mathrm T} = [x,\dotsm, x]\in \mathbb C^{n\times n}$ (实际上定义 $X=xx^{\mathrm H}$ 也能做)，注意到 $AX=\lambda X$   
那么对于任意相容范数 $\mathbb C^{n\times n}$ 都有:  
$$
|\lambda|\|X\| = \|\lambda X\| = \|AX\| \leq \|A\| \|X\|\\
\Updownarrow\\
|\lambda|\leq \|A\|
$$
因此我们有 $\rho(A) = \max_{1\leq i\leq n} |\lambda| \leq \|A\|$ 成立.    
这表明相容范数总是**谱占优的** (spectrally dominant).

若 $A$ 是非奇异阵，则 $A^{-1}$ 存在且 $\lambda^{-1}$ 是 $A^{-1}$ 的一个特征值.  
因此我们有 $|\lambda^{-1}|\leq \|A^{-1}\|$，即有 $|\lambda|\geq 1/\|A^{-1}\|$.

****

上述讨论可以总结为以下定理:  
**(谱半径定理, Matrix Analysis 定理 $5.6.9$)**  
设 $\|\cdot\|$ 是 $\mathbb C^{n\times n}$ 上的相容范数.   
若 $\lambda$ 是 $A\in \mathbb C^{n\times n}$ 的一个特征值，则我们有 $|\lambda|\leq \rho(A)\leq \|A\|$ 成立.   
进一步，若 $A$ 是非奇异的，则我们有 $1/\|A^{-1}\|\leq|\lambda|\leq \rho(A)\leq \|A\|$ 成立.

- **($\mathbb C^{n\times n}$ 上的相容范数都存在与之相容的向量范数)**  
  设 $\|\cdot\|$ 为 $\mathbb C^{n\times n}$ 上的相容范数.  
  可以证明由 $\|x\|_\star:= \|x1_n^{\mathrm T}\|$ 定义的函数是 $\mathbb C^n$ 上的一个范数，  
  且有 $\|Ax\|_\star =\|Ax1_n^{\mathrm T}\| \leq \|A\|\|x1_n^{\mathrm T}\| = \|A\|\|x\|_\star\ \ (\forall\ A\in \mathbb C^{n\times n},x\in \mathbb C^n)$.

尽管谱半径函数 $\rho(\cdot)$ 并不是 $\mathbb C^{n\times n}$ 上的相容范数，  
但对于任意 $A\in \mathbb C^{n\times n}$，$\rho(A)$ 都是所有相容范数取值的下确界.  
**(Matrix Analysis 引理 $5.6.10$)**  
任意给定 $A\in\mathbb C^{n\times n}$ 和 $\varepsilon>0$，都存在一个 $\mathbb C^{n\times n}$ 上的相容范数 $\|\cdot\|_\varepsilon$ 使得 $\rho(A)\leq \|A\|_{\varepsilon} \leq \rho(A)+\varepsilon$.   
利用这个结果我们可以证明 $\rho(A)= \inf\{\|A\|:\|\cdot\|\text{ is an induced matrix norm on }\mathbb C^{n\times n}\}$.

- **Note:** 通过向任何相容范数中插入一个固定的相似变换，就可以导出新的相容范数.  
  
  > **(Matrix Analysis 定理 $5.6.7$)**  
  > 若 $\|\cdot\|$ 是 $\mathbb C^{n\times n}$ 上的一个相容范数，且 $S\in \mathbb C^{n\times n}$ 是非奇异的， 
  > 则由 $\|A\|_S:= \|SAS^{-1}\|\ (\forall\ A\in \mathbb C^{n\times n})$ 定义的函数 $\|\cdot\|_S$ 都是一个相容范数.   
  > 特别地，若相容范数 $\|\cdot\|$ 是 $\mathbb C^n$ 上的范数 $\|\cdot\|$ 诱导的，则相容范数 $\|\cdot\|_S$ 是 $\mathbb C^n$ 上的范数 $\|\cdot\|_S$ 所诱导的.  
  > (其中范数 $\|\cdot\|_S : \mathbb C^n\mapsto \mathbb R$ 的定义为 $\|x\|_S:=\|Sx\|\ (\forall\ x\in \mathbb C^n)$)
  
- **证明:**  
  Schur 分解定理 (Matrix Analysis 定理 $2.3.1$) 保证了，  
  存在一个酉矩阵 $U\in \mathbb C^{n\times n}$ 和一个上三角阵 $T\in \mathbb C^{n\times n}$ 使得 $A=UTU^{\mathrm H}$，  
  其中 $T$ 的对角元为 $A$ 的特征值 $\lambda_1,\dots,\lambda_n$.
  
  令 $D_s = \text{diag}\{s,\dots,s^n\}$，则我们有:  
  $$
  D_s T D_s^{-1} = \begin{bmatrix}
  \lambda_1 & s^{-1} t_{12} & s^{-2}t_{13} & \dotsm & s^{-n+1} t_{1n}\\
   & \lambda_2 & s^{-1}t_{23} & \dotsm & s^{-n+2}t_{2n}\\
  & & \lambda_3 & \ddots &  \vdots\\
  & & & \ddots & s^{-1 }t_{n-1,n}\\
  &&&& \lambda_n 
  \end{bmatrix}
  $$
  因此我们可取足够大的 $t>0$ 使得 $D_sTD_s^{-1}$ 的严格上三角元的绝对值之和小于 $\varepsilon$  
  这样就保证了 $D_sTD_s^{-1}$ 的最大绝对列和小于 $\rho(A)+\varepsilon$，  
  即有 $\|D_sTD_s^{-1}\|_1 = \|(D_sU^{\mathrm H}) A (D_s U^{\mathrm H})^{\mathrm H}\|_1 \leq \rho(A)+\varepsilon$.
  
  根据 **Matrix Analysis 定理 $5.6.7$** 可知:  
  由 $\|A\|_{(D_sU^{\mathrm H})}:=\|(D_sU^{\mathrm H}) A (D_s U^{\mathrm H})^{\mathrm H}\|_1$ 定义的函数 $\|\cdot\|_{(D_sU^{\mathrm H})}$ 也是一个相容范数.  
  它满足 $\rho(A)\leq \|A\|_{(D_sU^{\mathrm H})}\leq \rho(A)+\varepsilon$.  
  命题得证.



#### (2) Gelfand 谱半径公式

**(Matrix Analysis 引理 $5.6.11$)**  
任意给定 $A\in\mathbb C^{n\times n}$.  
若存在一个相容范数 $\|\cdot\|$ 使得 $\|A\|<1$，则我们有 $\lim_{k\to \infty} A^k = 0_{n\times n}$.

**证明:**  
若 $\|A\|<1$，则我们有 $\|A^k\| \leq \|A\|^k \to 0$，表明 $\{A^k\}$ 依范数 $\|\cdot\|$ 收敛于 $0_{n\times n}$.  
由于有限维赋范线性空间 $\mathbb C^{n\times n}$ 上的所有范数都是等价的，  
故 $\{A^k\}$ 依最大列和范数 $\|\cdot\|_\infty$ 收敛于 $0_{n\times n}$.   
这表明 $\{A^k\}$ 逐点收敛到 $0_{n\times n}$.

尽管从序列 $\{A^k\}$ 的收敛性来说，$n^2$ 维赋范线性空间 $\mathbb C^{n\times n}$ 上的所有范数都是等价的.  
但给定一个矩阵 $A$，是可能存在两个相容范数 $\|\cdot\|_\alpha$ 和 $\|\cdot\|_\beta$ 使得 $\|A\|_\alpha<1$ 而 $\|A\|_\beta>1$ 的.    
考虑一个简单的例子:
$$
A=\begin{bmatrix}
\frac12 & \frac58\\
0 & \frac12\end{bmatrix}
$$
我们有 $\|A\|_1 = \|A\|_\infty = \frac{9}{8}>1$ 而 $\|A\|_{\mathrm F}=\frac{\sqrt {57}}{8}<1$.    
不过 $\|A\|_{\mathrm F}<1$ 表明 $\|A^k\|_{\mathrm F}\to 0$，这等价地说明有 $\|A^k\|_1 \to 0$ 和 $\|A^k\|_\infty \to 0$ 成立.

***

满足 $\lim_{k\to \infty} A^k = 0_{n\times n}$ 的矩阵 $A\in\mathbb C^{n\times n}$ 称为**收敛的** (convergent)，它们在迭代过程分析中相当重要.  
幸运的是，它们的特征可由谱半径不等式加以刻画:  
**(Matrix Analysis 定理 $5.6.12$)**  
任意给定 $A\in\mathbb C^{n\times n}$，则 $\lim_{k\to \infty} A^k = 0_{n\times n}$ 当且仅当 $\rho(A)<1$.

**证明:**

- **充分性:**  
  若 $\rho(A)<1$，则 **Matrix Analysis 引理 $5.6.10$** 保证了存在某个相容范数 $\|\cdot\|$ 使得 $\|A\|<1$.   
  从而根据 **Matrix Analysis 引理 $5.6.11$** 可知 $\lim_{k\to \infty} A^k = 0_{n\times n}$.
- **必要性:**  
  若 $\lim_{k\to \infty} A^k = 0_{n\times n}$，  
  则对于 $A$ 的任意特征对 $(\lambda,x)$，我们都有 $\lambda^k x =A^kx\to 0_n\ (k\to \infty)$ 成立.  
  这表明 $\lambda^k\to 0\ (k\to\infty)$，因而有 $|\lambda|<1$ 成立.  
  由于 $\lambda$ 可以是 $A$ 的任意特征值，故我们有 $\rho(A)<1$ 成立.

**(Matrix Analysis 推论 $5.6.13$)**  
任意给定 $A\in\mathbb C^{n\times n}$ 和 $\varepsilon>0$，都存在一个常数 $C(A,\varepsilon)$ 使得对于任意 $k=1,2\dots$ 都有:  
$$
|(A^{k})_{ij}| \leq C(\rho(A) + \varepsilon)^k\ \ (\forall\ i,j=1,\dots,n).
$$
**证明:**  
考虑矩阵 $\widetilde A = \frac{1}{\rho(A)+\varepsilon}A$，其谱半径 $\rho(\widetilde A)<1$，因而有 $\lim_{k\to \infty} \widetilde A^k = 0_{n\times n}$.  
于是序列 $\{\widetilde A^k\}$ 是有界的，即存在某个有限的 $C>0$，  
使得对于任意 $k=1,2\dots$ 都有 $|(\widetilde A^k)_{ij}|\leq C\ (\forall\ i,j=1,\dots,n)$.

****

尽管说当 $k\to\infty$ 时 $A^k$ 的单个元素的性状与 $\rho(A)^k$ 的性状相仿是不够准确的，  
但对于任意相容范数 $\|\cdot\|$，序列 $\{\|A^k\|\}$ 的确都有这个渐近性质.  
**(Gelfand 公式, Matrix Analysis 推论 $5.6.14$)**  
若 $\|\cdot\|$ 是 $\mathbb C^{n\times n}$ 上的一个相容范数，则对于任意 $A\in \mathbb C^{n\times n}$ 我们都有:  
$$
\rho(A) = \lim_{k\to\infty} \|A^k\|^{\frac1k}.
$$
这表明极限情况下特征值决定一切.

- **Note:** 根据有限维赋范空间上的范数等价性可知，Gelfand 公式对不相容的矩阵范数也成立.

**证明:**

- 一方面，注意到对于任意 $k=1,2,\dots$ 我们都有 $(\rho(A))^k = \rho(A^k) \leq \|A^k\|$ 成立.  
  因而有 $\rho(A)\leq \|A^k\|^{\frac1k}$ 成立，这说明:
  $$
  \rho(A) \leq \underset{k\to\infty}{\lim \inf} \|A\|^{\frac1k}.
  $$
  
- 另一方面，对于任意 $\varepsilon>0$，注意到矩阵 $\widetilde A = \frac{1}{\rho(A)+\varepsilon}A$ 的谱半径 $\rho(\widetilde A)<1$，  
  因而有 $\lim_{k\to\infty} \widetilde A^k = 0_{n\times n}$ 成立.  
  于是存在 $N(\varepsilon,A)$ 使得对于任意 $k\geq N(\varepsilon,A)$ 都有 $\|\widetilde A^k\|\leq 1$，即有 $\|A^k\|\leq (\rho(A)+\varepsilon)^k$ 成立.  
  因而有 $\|A^k\|^\frac1k \leq \rho(A)+\varepsilon$ 成立，这说明:
  $$
  \underset{k\to\infty}{\lim \sup} \|A\|^{\frac1k} \leq \rho(A).
  $$

综上所述，我们有:
$$
\rho(A) \leq \underset{k\to\infty}{\lim \inf} \|A\|^{\frac1k} \leq \underset{k\to\infty}{\lim \sup} \|A\|^{\frac1k} \leq \rho(A).
$$
因此极限 $\lim_{k\to\infty} \|A^k\|^{\frac1k}$ 存在，且有 $\lim_{k\to\infty} \|A^k\|^{\frac1k} = \rho(A)$ 成立.



#### (3) Neumann 级数

有关矩阵序列和级数的收敛性的许多问题都可利用范数来刻画.  

- 设 $\{A_k\}\subset \mathbb C^{n\times n}$ 是给定的矩阵序列.  
  如果存在 $\mathbb C^{n\times n}$ 上的某个范数 $\|\cdot\|$ 使得正项级数 $\sum_{k=1}^\infty \|A_k\|$ 的部分和 $S_n:= \sum_{k=1}^n \|A_k\|$ 有上界，  
  那么我们可以证明部分和序列 $\{S_n\}$ 构成一个 Cauchy 序列，因而级数 $\sum_{k=1}^\infty \|A_k\|$ 收敛于 $\mathbb C^{n\times n}$ 中的某个矩阵.

相容范数特别适用于处理矩阵的幂级数.  
复的纯量幂级数 $\sum_{k=1}^\infty a_k z^k$ 的收敛半径 $R=(\lim\underset{k\to\infty}{\sup} \sqrt[k]{|a_k|})^{-1}$  
若这个极限存在，则它等于 $\underset{k\to\infty}{\lim} \frac{|a_k|}{|a_{k+1}|}$ (比值判别法)   
当 $|z|<R$ 时该级数绝对收敛，当 $|z|>R$ 时它发散，当 $|z|=R$ 时收敛或发散都可能发生.

注意到对于任意 $A\in \mathbb C^{n\times n}$ 和相容范数 $\|\cdot\|$ 我们都有:  
$$
\begin{align}
\left\|\sum_{k=1}^\infty a_k A^k\right\|
&\leq
\sum_{k=1}^\infty \|a_k A^k\|\\
&=
\sum_{k=1}^\infty |a_k| \|A^k\|\\
&\leq
\sum_{k=1}^\infty |a_k|\|A\|^k.
\end{align}
$$
这表明矩阵幂级数 $\sum_{k=1}^\infty a_k A^k$ 的范数值被复的纯量幂级数 $\sum_{k=1}^\infty |a_k| z^k$ (记其收敛半径为 $R$) 控制着.  
因此只要 $\|A\|<R$，矩阵幂级数 $\sum_{k=1}^\infty a_k A^k$ 就收敛.

- **注:** 我们暂时没有将 $k=0$ 的情况并入进去，我们可以给出一个并入 $k=0$ 的例子:
  $$
  \sum_{k=0}^\infty \|A\|^k = \|I_n\|\frac{1}{1-\|A\|}\quad (\|A\|<1)
  $$
  这个等比数列求和的首项 $\|I_n\|$ 就是一个容易出错的点.  
  尽管相容范数满足 $\| I_n\|\geq 1$，但当且仅当 $\|\cdot\|$ 为诱导范数时才有 $\|I_n\|=1$ 成立.   
  例如 Frobenius 范数就有 $\|I_n\|_{\mathrm F} = \sqrt n \neq 1\ (\forall\ n\geq 2)$.

> **(Matrix Analysis 引理 $5.6.10$)**  
> 任意给定 $A\in\mathbb C^{n\times n}$ 和 $\varepsilon>0$，都存在一个 $\mathbb C^{n\times n}$ 上的相容范数使得 $\rho(A)\leq \|A\|\leq \rho(A)+\varepsilon$   
> 利用这个结果我们可以证明 $\rho(A)= \inf\{\|A\|:\|\cdot\|\text{ is an induced matrix norm on }\mathbb C^{n\times n}\}$ 

注意到上述 $\|\cdot\|$ 可以是任意相容范数  
而 Matrix Analysis 引理 $5.6.10$ 表明只要 $\rho(A)<R$，这样一个相容范数 $\|\cdot\|$ 就是存在的.  
我们将上述结果总结成如下定理:  
**(Matrix Analysis 定理 $5.6.15$)**  
设复的纯量幂级数 $\sum_{k=1}^\infty |a_k| z^k$ 的收敛半径为 $R$，任意给定 $A\in \mathbb C^{n\times n}$  
若 $\rho(A)<R$ (只要有一个相容范数 $\|\cdot\|$ 使得 $\|A\|<R$, 这个条件就满足)， 则矩阵幂级数 $\sum_{k=1}^\infty a_k A^k$ 收敛.   

- 注意到标量指数函数 $e^z = \sum_{k=0}^\infty \frac{1}{k!} z^k$ 的幂级数的收敛半径是 $\infty$   
  故由矩阵幂级数 $e^A:= \sum_{k=0}^\infty \frac{1}{k!}A^k$ 定义的矩阵指数函数 $e^A$ 对于任意 $A\in \mathbb C^{n\times n}$ 都有良好的定义.

  标量余弦函数和标量正弦函数的幂级数的收敛半径也都是 $\infty$，我们同理有:  
  $$
  \begin{align}
  
  \cos(z) = \sum_{k=0}^\infty \frac{(-1)^k}{(2k)!} z^{2k}\ (\forall\ z\in \mathbb C)\ &\Rightarrow\ \cos(A) = \sum_{k=0}^\infty \frac{(-1)^k}{(2k)!} A^{2k}\ (\forall\ A\in \mathbb C^{n\times n})\\
  
  \sin(z) = \sum_{k=0}^\infty \frac{(-1)^k}{(2k+1)!} z^{2k+1}\ (\forall\ z\in \mathbb C)\ &\Rightarrow\ \sin(A) = \sum_{k=0}^\infty \frac{(-1)^k}{(2k+1)!} A^{2k+1}\ (\forall\ A\in \mathbb C^{n\times n})
  
  \end{align}
  $$
  标量对数函数 $\log(1-z)$ 可由级数 $-\sum_{k=0}^\infty \frac{z^k}{k}$ 表示，这个级数的收敛半径是 $1$ (即当 $|z|<1$ 时级数收敛)  
  推广到矩阵形式，我们定义 $\log(I-A):= -\sum_{k=0}^\infty \frac{z^k}{k}$，它只有当 $\rho(A)<1$ 时才有良好的定义.

- 一般地，设 $A\in \mathbb C^{n\times n}$ 可对角化，$A=S\Lambda S^{-1}$，$\Lambda = \text{diag}(\lambda_1,\dots,\lambda_n)$   
  给定一个定义域包含 $\lambda_1,\dots,\lambda_n$ 的复值函数 $f$   
  我们定义**初等矩阵函数**为: $f(A) := Sf(\Lambda)S^{-1} = S\text{diag}\{f(\lambda_1),\dots,f(\lambda_n)\}S^{-1}$   
  可以证明这个定义与相似矩阵 $S$ 的选择是无关的 (即不同的 $S$ 定义出的 $f(A)$ 是一致的)  
  这表明可对角化的矩阵的初等矩阵函数具有良好的定义.

  **(存疑)** 特殊地，如果 $f$ 在包含 $\lambda_1,\dots,\lambda_n$ 的某个开域解析，则我们有:  
  $$
  \begin{align}
  f(A)
  &:= Sf(\Lambda)S^{-1}\\
  &= S\text{diag}\{f(\lambda_1),\dots,f(\lambda_n)\}S^{-1}\\
  &= S\left\{\frac{1}{2\pi i}\oint_{\Gamma} f(\lambda)(\lambda I-\Lambda)^{-1}d\lambda\right\} S^{-1}\\
  &= \frac{1}{2\pi i}\oint_{\Gamma} f(\lambda)S(\lambda I-\Lambda)^{-1} S^{-1}d\lambda\\
  &= \frac{1}{2\pi i}\oint_{\Gamma} f(\lambda)(\lambda I-A)^{-1}d\lambda
  \end{align}
  $$
  其中 $\Gamma$ 是一个包含 $A$ 的所有特征值的正向围道.
  
  在 $f(A)$ 作为初等矩阵函数 (而不是作为幂级数) 的定义中，我们对函数 $f$ 的要求很少 (它不必是解析的)  
  但我们对矩阵要求的更多 (它必须是可对角化的)  
  不可对角化的矩阵的初等矩阵函数可以定义，但我们必须要求函数 $f$ 的解析性.  
  此时我们有 $f(A):=\frac{1}{2\pi i}\oint_{\Gamma} f(\lambda)(\lambda I-A)^{-1}d\lambda$   
  其中 $\Gamma$ 是一个包含 $A$ 的所有特征值的正向围道.
  
  若 $A\in \mathbb C^{n\times n}$ 可以对角化，且解析函数 $f(z)= \sum_{k=0}^\infty a_k z^k$ 是由一个收敛半径大于 $\rho(A)$ 的幂级数所定义的，  
  则可以证明 $f(A)$ 的初等矩阵函数定义与其幂级数定义是一致的.  
  具体来说，设 $A=S\Lambda S^{-1}$，则我们有:  
  $$
  \begin{align}
  \sum_{k=0}^\infty a_k A^k 
  &= \sum_{k=0}^\infty a_k (S\Lambda S^{-1})^k\\ 
  &= S \left(\sum_{k=0}^\infty a_k \Lambda^k\right) S^{-1}\\ 
  &= S\text{diag}\left\{\sum_{k=0}^\infty a_k \lambda_1^k ,\dots, \sum_{k=0}^\infty a_k \lambda_n^k\right\} S^{-1}\\ 
  &= S\text{diag}\{f(\lambda_1),\dots,f(\lambda_n)\}S^{-1}
  \end{align}
  $$

*****

**(Neumann 级数, Matrix Analysis 推论 $5.6.16$)**  
设 $A\in \mathbb C^{n\times n}$ 为非奇异矩阵.  
若存在一个相容范数 $\|\cdot\|$ 使得 $\|I-A\|<1$，则 $A^{-1}=\sum_{k=0}^\infty (I-A)^k$.

- 上述定理的等价形式为:  
  若存在一个相容范数 $\|\cdot\|$ 使得 $\|A\|<1$，则 $I-A$ 非奇异，且 $(I-A)^{-1}=\sum_{k=0}^\infty A^k$ 

  结合 Matrix Analysis 引理 $5.6.10$ 可知上述命题进一步等价于:  
  当且仅当 $\rho(A)<1$ 时有 $I-A$ 非奇异，且 $(I-A)^{-1}=\sum_{k=0}^\infty A^k$ 

- **证明:**  
  注意到级数 $\sum_{k=0}^\infty z^k$ 的收敛半径为 $1$  
  因此若 $\|I-A\|<1$，则级数 $\sum_{k=0}^\infty (I-A)^k$ 就收敛到 $\mathbb C^{n\times n}$ 中的某个矩阵.  
  又注意到:  
  $$
  \begin{align}
  A \sum_{k=0}^N (I-A)^k 
  &= (I-(I-A)) \sum_{k=0}^N (I-A)^k\\
  &= \sum_{k=0}^N (I-A)^k - \sum_{k=0}^N (I-A)^{k+1}\\ 
  &= I-(I-A)^{N+1}\to I\ \ (N\to \infty)
  \end{align}
  $$
  因此我们断定有 $\sum_{k=0}^\infty (I-A)^k=A^{-1}$.
  
- 有关非奇异性的一个有用的充分条件:  
  **(Levy-Desplanques 定理, Matrix Analysis 推论 $5.6.17$)**  
  设 $A=[a_{ij}]\in \mathbb C^{n\times n}$.  
  若 $A$ **严格对角占优** (strictly diagonally dominant)，即 $|a_{ii}|>\sum_{j\neq i}|a_{ij}|\ (\forall \ i=1,\dots,n)$，则 $A$ 非奇异.

  **证明:**  
  严格对角占优性保证了 $A$ 的主对角元均不为零，因此 $D:=\text{diag}(a_{11},\dots,a_{nn})$ 非奇异.  
  注意到 $D^{-1}A$ 的主对角元均为 $1$，因此 $B:=I-D^{-1}A$ 的主对角元均为零  
  且对于任意 $i\neq j$ 都有 $b_{ij} = -\frac{a_{ij}}{a_{ii}}$.

  考虑最大行和矩阵范数 $\|\cdot\|_\infty$，根据严格对角占优性可以证明:  
  $$
  \|B\|_\infty = \max_{1\leq i\leq n}\sum_{j=1}^n |b_{ij}| = \max_{1\leq i\leq n}
  \left\{\frac{1}{a_{ii}}\sum_{j\neq i}|a_{ij}|\right\} < 1
  $$
  根据 **Matrix Analysis 推论 $5.6.16$** 的等价形式可知 $I-B=D^{-1}A$ 是非奇异阵  
  且 $(I-B)^{-1}=A^{-1}D=\sum_{k=0}^\infty B^k = \sum_{k=0}^\infty (I-D^{-1}A)^k$   
  因此 $A$ 是非奇异阵，且 $A^{-1}= [\sum_{k=0}^\infty (I-D^{-1}A)^k]D^{-1}$ 
  
- **(数值线性代数, 推论 $2.1.1$)**  
  设 $\|\cdot\|$ 为 $\mathbb C^{n\times n}$ 上的相容范数 (由于它满足次可乘性，故有 $\|I\|\geq 1$ 成立)  
  若 $A\in \mathbb C^{n\times n}$ 满足 $\|A\|<1$，则我们有:  
  $$
  \frac{\|I\|}{\|I\|+\|A\|} \leq \|(I-A)^{-1}\| \leq \frac{\|I\| - (\|I\|-1)\|A\|}{1-\|A\|}
  $$
  特殊地，若 $\|\cdot\|$ 还满足 $\|I\|=1$ (即它是单位相容范数; 如果它是诱导范数，就会是这种情形)，则结论转化为:  
  $$
  \frac{1}{1+\|A\|} \leq \|(I-A)^{-1}\| \leq \frac{1}{1-\|A\|}
  $$
  **证明:**   
  首先考虑证明上界:  
  $$
  \begin{align}
  \|(I-A)^{-1}\| 
  &= \left\|\sum_{k=0}^\infty A^k\right\|\\
  &\leq \sum_{k=0}^\infty \|A^k\| \\
  &\leq \|I\| + \sum_{k=1}^\infty \|A\|^k\\
  &= \|I\| + \|A\|\cdot \frac{1}{1-\|A\|}\\
  &= \frac{\|I\| - (\|I\|-1)\|A\|}{1-\|A\|}
  \end{align}
  $$
  其次考虑证明下界:    
  注意到对于任意 $B\in \mathbb C^{n\times n}$ 我们都有 $\|I\| = \|BB^{-1}\|\leq \|B\|\|B^{-1}\|$，  
  故我们有 $\|B^{-1}\|\geq \|I\|/\|B\|\ (\forall\ B\in \mathbb C^{n\times n})$.   
  于是我们有:
  $$
  \begin{align}
  \|(I-A)^{-1}\|
  &\geq
  \frac{\|I\|}{\|I-A\|}\qquad (\text{note that }\|B^{-1}\|\geq \frac{\|I\|}{\|B\|}\ (\forall\ B\in \mathbb C^{n\times n}))\\
  &\geq
  \frac{\|I\|}{\|I\|+\|A\|} \quad (\text{triangle inequality})
  \end{align}
  $$
  命题得证.



### 2.3.6 诱导范数的极小性

现在我们专注于讨论诱导范数，它们有一个重要的极小性.  
我们常常通过判别条件 $\|A\|<1$ 来确定给定矩阵 $A$ 是收敛的 (即 $\lim_{k\to\infty} A^k = 0_{n\times n}$)  
因此一个自然的想法就是采用尽可能一致地小的相容范数.  
所有诱导范数都有这个所希望的性质，这个性质正是对诱导范数的特征的刻画.

下面的定理指出了**诱导范数的对称性**:  
若对于任意 $A\in \mathbb C^{n\times n}$ 都有 $\|A\|_\alpha \leq C\|A\|_\beta$，则对于任意 $A\in \mathbb C^{n\times n}$ 都有 $\|A\|_\beta \leq C\|A\|_\alpha$，即有:  
$$
\frac1C\|A\|_\beta \leq \|A\|_\alpha \leq C\|A\|_\beta\ \ (\forall\ A\in \mathbb C^{n\times n}).
$$
**(诱导范数的对称性, Matrix Analysis 定理 $5.6.18$)**  
若 $\|\cdot\|_\alpha$ 和 $\|\cdot\|_\beta$ 是 $\mathbb C^n$ 上给定的范数，则我们有:
$$
\max_{A\neq 0_{n\times n}}\frac{\|A\|_\alpha}{\|A\|_\beta} 
=
\max_{A\neq 0_{n\times n}}\frac{\|A\|_\beta}{\|A\|_\alpha}
=
R_{\alpha \beta} R_{\beta \alpha}
\text{ where }
\begin{cases}
R_{\alpha\beta}:= \max_{x\neq 0_n} \frac{\|x\|_\alpha}{\|x\|_\beta}\\
R_{\beta\alpha}:= \max_{x\neq 0_n} \frac{\|x\|_\beta}{\|x\|_\alpha}
\end{cases}\\

\Downarrow\\

\frac1{R_{\alpha \beta} R_{\beta \alpha}}\|A\|_\beta \leq \|A\|_\alpha \leq R_{\alpha \beta} R_{\beta \alpha}\|A\|_\beta\ \ (\forall\ A\in \mathbb C^{n\times n})
$$

*****

什么时候 $\mathbb C^n$ 上的两个给定的范数诱导出 $\mathbb C^{n\times n}$ 上同一个相容范数?  
答案是当其中一个是另一个的纯量倍数的时候.  
**(Matrix Analysis 引理 $5.6.23$)**  
若 $\|\cdot\|_\alpha$ 和 $\|\cdot\|_\beta$ 是 $\mathbb C^n$ 上给定的范数，记 
$$
\begin{cases}
R_{\alpha\beta}:= \max_{x\neq 0_n} \|x\|_\alpha/\|x\|_\beta\\
R_{\beta\alpha}:= \max_{x\neq 0_n} \|x\|_\beta/\|x\|_\alpha
\end{cases}
$$
则我们有 $R_{\alpha\beta} R_{\beta \alpha}\geq 1$.   
此外，下列结论是等价的:

- ① $R_{\alpha\beta}=R_{\beta\alpha}$ 
- ② 存在某个 $c>0$ 使得 $\|x\|_\alpha = c\|x\|_\beta\ (\forall\ x\in \mathbb C^n)$ 
- ③ 它们诱导的相容范数 $\|\cdot\|_\alpha$ 和 $\|\cdot\|_\beta$ 相同，即 $\|A\|_\alpha = c\|A\|_\beta\ (\forall\ A\in \mathbb C^{n\times n})$

下面的推论说的是，没有哪个诱导范数能够在 $\mathbb C^{n\times n}$ 上一致地小于与它不同的诱导范数.  
**(Matrix Analysis 推论 $5.6.25$)**  
设 $\|\cdot\|_\alpha$ 和 $\|\cdot\|_\beta$ 是 $\mathbb C^{n\times n}$ 上的诱导范数，  
则 $\|A\|_\alpha \leq \|A\|_\beta\ (\forall\ A\in \mathbb C^{n\times n})$ 成立的充要条件是 $\|A\|_\alpha = c\|A\|_\beta\ (\forall\ A\in \mathbb C^{n\times n})$。

下面的定理说的要更多一些:  
没有哪个相容范数能够在 $\mathbb C^{n\times n}$ 上一致地小于与它不同的诱导范数.  
**(Matrix Analysis 定理 $5.6.26$)**  
设 $\|\cdot\|$ 是 $\mathbb C^{n\times n}$ 上的相容范数，$\|\cdot\|_{\alpha}$ 是 $\mathbb C^{n\times n}$ 上的诱导范数.  
给定非零向量 $z\in \mathbb C^n$ 并定义 $\|\cdot\|_z$ 为 $\|x\|_z := \|xz^{\mathrm H}\|\ (\forall\ x\in \mathbb C^n)$.   
则我们有:

- $\|\cdot\|_z$ 是 $\mathbb C^n$ 上的范数
- $\|\cdot\|_z$ 的诱导范数 $N_z(\cdot)$ 满足 $N_z(A)\leq \|A\|\ (\forall\ A\in \mathbb C^{n\times n})$
- $\|A\|\leq \|A\|_\alpha\ (\forall\ A\in \mathbb C^{n\times n})$ 当且仅当 $N_z(A)=\|A\| = \|A\|_\alpha\ (\forall\ A\in \mathbb C^{n\times n})$ 

**根据上述定理可以导出结论:**  
对于任意给定非零向量 $z\in \mathbb C^n$ 和 $\mathbb C^{n\times n}$ 上的诱导范数 $\|\cdot\|$，  
定义 $\|\cdot\|_z$ 为 $\|x\|_z := \|xz^{\mathrm H}\|\ (\forall\ x\in \mathbb C^n)$ (上述定理表明 $\|\cdot\|_z$ 是是 $\mathbb C^n$ 上的范数)  
则 $\|\cdot\|_z$ 的诱导范数 $N_z(\cdot)$ 总是与诱导范数 $\|\cdot\|$ 相同，即 $N_z(A)=\|A\|\ (\forall\ A\in \mathbb C^{n\times n})$ 

- **上述结果还可用富有教益的不同方式进行处理:**  
  任意给定 $\mathbb C^{n\times n}$ 上的诱导范数 $\|\cdot\|$，我们都有:  
  $$
  \begin{align}
  \|Axz^{\mathrm H}\|
  &=
  \max_{\|\xi\|=\|\eta\|^D=1} |\eta^{\mathrm H} Axz^{\mathrm H}\xi|\\
  &=
  \max_{\|\eta\|^D=1} |\eta^{\mathrm H} Ax| \max_{\|\xi\|=1} |\xi^{\mathrm H}z|\\
  &=
  \|Ax\|^{DD} \|z\|^D\\
  &=
  \|Ax\|\|z\|^D
  \end{align}\quad (\forall\ x,z\in \mathbb C^n,A\in \mathbb C^{n\times n})
  $$
  取 $A=I$ 可知 $\|x\|_z=\|xz^{\mathrm H}\| = \|x\|\|z\|^D$   
  
  若 $z\neq 0_n$，我们就有:  
  
  $$
  \begin{align}
  N_z(A)
  &=
  \max_{x\neq 0_n}\frac{\|Axz^{\mathrm H}\|}{\|xz^{\mathrm H}\|}\\
  &=
  \max_{x\neq 0_n} \frac{\|Ax\|\|z\|^D}{\|x\|\|x\|^D}\\
  &=
  \max_{x\neq 0_n} \frac{\|Ax\|}{\|x\|}\\
  &=
  \|A\|
  \end{align}\quad (\forall\ A\in \mathbb C^{n\times n})
  $$

上面的结果启发我们考虑极小矩阵范数的定义:  
我们称 $\mathbb C^{n\times n}$ 上的相容范数 $\|\cdot\|$ 为一个**极小相容范数** (minimal compatible matrix norm)，  
如果 $\mathbb C^{n\times n}$ 上满足 $N(A)\leq \|A\|\ (\forall\ A\in \mathbb C^{n\times n})$ 的仅有的相容范数 $N(\cdot) = \|\cdot\|$

**(Matrix Analysis 定理 $5.6.32$)**  
设 $\|\cdot\|$ 是 $\mathbb C^{n\times n}$ 上的一个相容范数.  
对于每个非零的 $z\in \mathbb C^n$ 都根据 $N_z(A)=\underset{x\neq 0_n}{\max}\frac{\|Axz^{\mathrm H}\|}{\|xz^{\mathrm H}\|}$ 定义诱导范数 $N_z(\cdot)$  
则下列命题等价:  

- ① $\|\cdot\|$ 是诱导范数
- ② $\|\cdot\|$ 是极小相容范数
- ③ 对于某个非零的 $z\in \mathbb C^n$ 有 $N_z(\cdot)=\|\cdot\|$
- ④ 对于所有非零的 $z\in \mathbb C^n$ 都有 $N_z(\cdot)=\|\cdot\|$



### 2.3.7 酉不变的矩阵范数

$\mathbb C^{m\times n}$ 上的一个矩阵范数 $\|\cdot\|$ (不一定是相容范数) 称为**酉不变的** (unitarily invariant)，  
如果对于任意 $A\in \mathbb C^{m\times n}$ 和任意酉矩阵 $U\in \mathbb C^{m\times m},V\in \mathbb C^{n\times n}$ 都有 $\|A\|=\|UAV\|$.

**(Matrix Analysis 定理 $5.6.34$)**  
设 $\|\cdot\|$ 是 $\mathbb C^{n\times n}$ 上的一个酉不变的相容范数，$z\in \mathbb C^n$ 为非零向量，则我们有:

- ① 由 $\|x\|_z := \|xz^{\mathrm H}\|\ (\forall\ x\in \mathbb C^n)$ 定义的向量范数 $\|\cdot\|_z$ 是 $\mathbb C^n$ 上的酉不变范数
- ② $\|\cdot\|_z$ 是 $l_2$ 范数 $\|\cdot\|_2$ 的纯量倍数
- ③ 由 $N_z(A)= \underset{x\neq 0_n}{\max} \frac{\|Ax\|_z}{\|x\|_z}=\underset{x\neq 0_n}{\max}\frac{\|Axz^{\mathrm H}\|}{\|xz^{\mathrm H}\|}$ 定义的诱导范数 $N_z(\cdot)$ 就是谱范数 $\|\cdot\|_2$
- ④ $\|A\|_2\leq \|A\|\ \ (\forall \ A\in \mathbb C^{n\times n})$
- ⑤ 若酉不变相容范数 $\|\cdot\|$ 同时还是一个诱导范数，则它就是谱范数 $\|\cdot\|_2$

**证明:**

- ① 对于任意酉矩阵 $U\in \mathbb C^{n\times n}$ 我们都有:  
  $$
  \|Ux\|_z = \|Uxz^{\mathrm H}\| = \|xz^{\mathrm H}\| = \|x\|_z
  $$

- ② 对于任意 $x\in \mathbb C^n$，都存在一个酉矩阵 $U\in \mathbb C^{n\times n}$ (实际上是 Householder 变换) 使得 $Ux = \|x\|_2 e_1$   
  于是我们有:  
  $$
  \|x\|_z = \|Ux\|_z = \|Uxz^{\mathrm H}\| = \|(\|x\|_2e_1) z^{\mathrm H}\| = \|e_1z^{\mathrm H}\|\|x\|_2
  $$
  我们记 $c_z =\|e_1 z^{\mathrm H}\|$ 即有 $\|x\|_z = c_z \|x\|_2\ \ (\forall\ x\in \mathbb C^n)$  
  
- ③ 对于任意 $z\in \mathbb C^n$ 我们都有:  
  $$
  N_z(A) = \max_{x\neq 0_n} \frac{\|Ax\|_z}{\|x\|_z} = \max_{x\neq 0_n} \frac{c_z \|Ax\|_2}{c_z\|x\|_2} = \max_{x\neq 0_n} \frac{\|Ax\|_2}{\|x\|_2} = \|A\|_2
  $$

- ④⑤ 根据 Matrix Analysis 定理 $5.6.26$ 可得

****

设 $\|\cdot\|$ 是 $\mathbb C^{n\times n}$ 上的一个范数 (不一定是相容范数)  
可以证明由 $\|A\|':= \|A^{\mathrm H}\|$ 所定义的函数 $\|\cdot\|'$ 是 $\mathbb C^{n\times n}$ 上的一个范数，且 $(\|A'\|)' = \|A\|$   
我们称范数 $\|\cdot\|'$ 是 $\|\cdot\|$ 的**伴随范数** (adjoint norm) 

- 可以证明相容范数的伴随范数也是一个相容范数.  
  例如最大列和矩阵范数 $\|\cdot\|_1$ 的伴随范数就是最大行和矩阵范数 $\|\cdot\|_\infty$

满足 $\|A\|'=\|A\|\ (\forall\ A\in \mathbb C^{n\times n})$ 的范数称为**自伴随的** (self-adjoint)  

- 可以证明 $\mathbb C^{n\times n}$ 上的酉不变范数都是自伴随的.  
  例如 Frobenius 范数和谱范数都是自伴随的.

作为仅有的自伴随的诱导范数，谱范数是极其重要的.  
**(Matrix Analysis 定理 $5.6.35$)**  
设 $\|\cdot\|$ 是 $\mathbb C^{n\times n}$ 上的相容范数，且它是由 $\mathbb C^n$ 上的一个范数 $\|\cdot\|$ 诱导的.  
则我们有:

- 诱导范数 $\|\cdot\|$ 的伴随范数 $\|\cdot\|'$ 是由范数 $\|\cdot\|^D$ 诱导的
- 若诱导范数 $\|\cdot\|$ 同时还是自伴随的，则它就是谱范数

****

**(Von Neumann 定理, [Some Metric Inequalities in the Space of Matrices (Ky Fan & A. J. Hoffman)](https://www.jstor.org/stable/2032662?seq=1))**  
$\mathbb C^{m\times n}$ 上的任意一个矩阵范数 $\|\cdot\|$ 是酉不变的，当且仅当它可以表示为对称奇异值函数.  
即存在一个 $\mathbb R^{\min\{m,n\}}\mapsto \mathbb R_+$ 的**对称度规函数** (symmetric gauge function) $\phi$，  
使得对于任意 $A\in \mathbb C^{m\times n}$，都有 $\|A\|=\phi(\sigma_1(A),\dots,\sigma_{\min\{m,n\}}(A))$ 成立.  

- 谱范数 $\|A\|_2 =\sigma_\max(A)$ 对应的对称度规函数为 $\phi(x_1,\dots,x_{\min\{m,n\}}) = \max\{|x_1|,\dots,|x_{\min\{m,n\}}|\}$ 
- Frobenius 范数 $\|A\|_{\mathrm F} = (\sum_{i=1}^{\min\{m,n\}} \sigma_{i}^2(A))^{\frac12}$ 对应的对称度规函数为 $\phi(x_1,\dots,x_{\min\{m,n\}}) = (\sum_{i=1}^{\min\{m,n\}} |x_i|^2)^{\frac12}$
- 核范数 $\|A\|_* = \sum_{i=1}^{\min\{m,n\}}\sigma_i(A)$ 对应的对称度规函数为 $\phi(x_1,\dots,x_{\min\{m,n\}}) = \sum_{i=1}^{\min\{m,n\}} |x_i|$

换言之，酉不变范数和对称度规函数之间存在一一对应关系.  
其中对称度规函数就是一个绝对范数，且对元素置换具有不变性.  
我们称 $f:\mathbb R^r\mapsto \mathbb R_+$ 是一个对称度规函数，如果它满足:

- ① $f$ 是 $\mathbb R^r$ 上的一个范数
- ② $f(|x|)=f(x)\ (\forall\ x\in \mathbb R^r)$ (其中 $|x|$ 是对 $x$ 逐元素取绝对值得到的向量)
- ③ 对于任意给定的置换矩阵 $P\in \mathbb R^{r\times r}$ 都有 $f(Px) = f(x)\ (\forall\ x\in \mathbb R^r)$ 成立.



### 2.3.8 应用: 条件数

#### (1) 矩阵求逆

首先考虑求解非奇异矩阵 $A\in \mathbb C^{n\times n}$ 的逆矩阵.  
我们自然要问: 矩阵 $A$ 中的扰动和计算中引入的舍入误差会怎样影响 $A^{-1}$ 的计算解呢?

给定 $\mathbb C^{n\times n}$ 上的一个相容范数 $\|\cdot\|$  
设 $A$ 经扰动得到的矩阵为 $B=A+\Delta A = A(I-A^{-1}\Delta A)$   
我们假设 $\|A^{-1}\Delta A\|\leq 1$ 以确保 $B$ 是非奇异的   
(这是因为它保证了 $\rho(A^{-1}\Delta A)\leq \|A^{-1}\Delta A\|\leq 1$，因而 $I-A^{-1}\Delta A$ 不可能有零特征值，从而确保 $B$ 非奇异)

注意到 $A^{-1}-B^{-1} = A^{-1}(B-A) B^{-1} = A^{-1}\Delta A B^{-1}$，因此我们有:  
$$
\|A^{-1}-B^{-1}\| = \|A^{-1}\Delta A B^{-1}\| \leq \|A^{-1}\|\|\Delta A\|\|B^{-1}\|
\tag{2.3.8(1)}
$$
又注意到 $B^{-1} = A^{-1}-A^{-1}\Delta A B^{-1}$，因此我们有:  
$$
\|B^{-1}\| = \|A^{-1}-A^{-1}\Delta A B^{-1}\| \leq \|A^{-1}\| + \|A^{-1}\Delta A\|\|B^{-1}\|\\

\Updownarrow\\

\|B^{-1}\| \leq \frac{\|A^{-1}\|}{1-\|A^{-1}\Delta A\|}

\tag{2.3.8(2)}
$$
将 $(2.3.8(2))$ 代入 $(2.3.8(1))$ 中便得到:  
$$
\begin{align}
\|A^{-1}-B^{-1}\| 
&\leq \|A^{-1}\|\|\Delta A\|\|B^{-1}\|\\
&\leq \|A^{-1}\|\|\Delta A\| \frac{\|A^{-1}\|}{1-\|A^{-1}\Delta A\|}\\
&= \|A^{-1}\| \frac{\|A^{-1}\|\|A\|}{1-\|A^{-1}\Delta A\|} \frac{\|\Delta A\|}{\|A\|}
\end{align}
$$
因此计算逆矩阵时的相对误差上界为:  
$$
\frac{\|A^{-1}-(A+\Delta A)^{-1}\|}{\|A^{-1}\|} 
\leq 
\frac{\|A^{-1}\|\|A\|}{1-\|A^{-1}\Delta A\|} \frac{\|\Delta A\|}{\|A\|}
=
\frac{\kappa(A)}{1-\|A^{-1}\Delta A\|} \frac{\|\Delta A\|}{\|A\|}
$$
我们定义 $\kappa(A) = \begin{cases}
\|A^{-1}\|\|A\| & \text{if }A\text{ is non-singular}\\
\infty & \text{if }A\text{ is singular}\end{cases}$ 为**矩阵逆关于相容范数 $\|\cdot\|$ 的条件数**.  
(值得注意的是，关于任何相容范数的条件数都有 $\kappa(A)= \|A^{-1}\|\|A\|\geq \|A^{-1}A\|=\|I\|\geq 1$)   

- 若我们将假设条件 $\|A^{-1}\Delta A\|\leq 1$ 加强为 $\|A^{-1}\|\|\Delta A\|<1$，  
  则我们有 $1-\|A^{-1}\Delta A\| \geq 1-\|A^{-1}\|\|\Delta A\| = 1-\kappa(A)\frac{\|\Delta A\|}{\|A\|} >0$   
  于是我们有:    
  $$
  \frac{\|A^{-1}-(A+\Delta A)^{-1}\|}{\|A^{-1}\|} \leq \frac{\kappa(A)}{1-\|A^{-1}\Delta A\|} \frac{\|\Delta A\|}{\|A\|} \leq \frac{\kappa(A)}{1-\kappa(A)\frac{\|\Delta A\|}{\|A\|}} \frac{\|\Delta A\|}{\|A\|}
  $$
  上述相对误差上界称为**先验界** (priori bound)，因为它只与计算前已知的数据有关.

- 若 $\|A^{-1}\|\|\Delta A\|=\kappa(A)\frac{\|\Delta A\|}{\|A\|}\ll 1$，则相对误差上界就约等于 $\kappa(A)\frac{\|\Delta A\|}{\|A\|}$    
  我们有充分理由相信:   
  只要条件数 $\kappa(A)$ 不大，那么矩阵逆的相对误差就与数据的相对误差 $\frac{\|\Delta A\|}{\|A\|}$ 有相同的阶.

- 若 $\kappa(A)\gg 1$，则我们称 $A$ 是**病态的** (ill-conditioned)    
  若 $\kappa(A)$ 接近于 $1$，则我们称 $A$ 是**良态的** (well-conditioned)  
  (当然，上述有关态质的表述都是相对于一个指定的相容范数 $\|\cdot\|$ 来说的)



#### (2) 求解线性方程组

现在考虑求解线性方程组 $Ax=b$  
其中 $A\in \mathbb C^{n\times n}$ 非奇异，且 $b\in \mathbb C^n$ 是非零向量.  
由于数据误差和计算中的舍入误差的存在，我们实际上是在精确地求解摄动方程组 $(A+\Delta A)\widetilde x = b+\Delta b$ 

给定 $\mathbb C^{n\times n}$ 上的一个相容范数 $\|\cdot\|$ 以及 $\mathbb C^n$ 上的一个相容的向量范数 $\|\cdot\|$   

> **回忆起 ($\mathbb C^{n\times n}$ 上任意的相容范数都存在 $\mathbb C^n$ 上与之相容的范数)**  
> 设 $\|\cdot\|$ 为 $\mathbb C^{n\times n}$ 上的一个相容范数  
> 可以证明由 $\|x\|:= \|x1_n^{\mathrm T}\|$ 定义的函数是 $\mathbb C^n$ 上的一个范数  
> 且它与矩阵函数 $\|\cdot\|$ 相容，即 $\|Ax\|\leq \|A\|\|x\|\ \ (\forall\ A\in \mathbb C^{n\times n},x\in \mathbb C^n)$ 

我们假设 $\|A^{-1}\Delta A\|\leq 1$ 以确保 $A+\Delta A$ 是非奇异的:   
这是因为它保证了 $\rho(A^{-1}\Delta A)\leq \|A^{-1}\Delta A\|\leq 1$，因而 $I-A^{-1}\Delta A$ 不可能有零特征值  
从而确保了 $A+\Delta A=A(I-A^{-1}\Delta A)$ 非奇异.

令 $\widetilde x = x+\Delta x$，代入摄动方程组 $(A+\Delta A)\widetilde x = b+\Delta b$ 可得:  
$$
\begin{align}
(A+\Delta A)\widetilde x
&=
(A+\Delta A)(x+\Delta x)\\
&=
Ax + \Delta A x + (A + \Delta A) \Delta x\\
&=
b + \Delta A x + (A + \Delta A) \Delta x\\
&=
b + \Delta b
\end{align}
$$
因此 $\Delta A x + (A + \Delta A) \Delta x=\Delta b$，即有 $\Delta x=(A+\Delta A)^{-1}(\Delta b - \Delta Ax)$   
于是我们有:  
$$
\begin{align}
\|\Delta x\|
&=
\|(A+\Delta A)^{-1}(\Delta b - \Delta Ax)\|\\
&\leq
\|(A+\Delta A)^{-1}\| (\|\Delta b\|+ \|\Delta Ax\|)\quad (\text{note that }\|(A+\Delta A)^{-1}\| \leq \frac{\|A^{-1}\|}{1-\|A^{-1}\Delta A\|}\text{ and }\|\Delta Ax\|\leq \|\Delta A\|\|x\|)\\
&\leq
\frac{\|A^{-1}\|}{1-\|A^{-1}\Delta A\|} (\|\Delta b\|+ \|\Delta A\|\|x\|)
\end{align}
$$
进而有:  
$$
\begin{align}
\frac{\|\Delta x\|}{\|x\|}
&\leq 
\frac{\|A^{-1}\|\|A\|}{1-\|A^{-1}\Delta A\|} 
\left(\frac{\|\Delta b\|}{\|A\|\|x\|}+ \frac{\|\Delta A\|}{\|A\|}\right)\quad (\text{note that }\kappa(A)=\|A^{-1}\|\|A\|\text{ and }\|b\|=\|Ax\|\leq \|A\|\|x\|)\\
&\leq
\frac{\kappa(A)}{1-\|A^{-1}\Delta A\|} 
\left(\frac{\|\Delta b\|}{\|b\|}+ \frac{\|\Delta A\|}{\|A\|}\right)
\end{align}
$$

- 若我们将假设条件 $\|A^{-1}\Delta A\|\leq 1$ 加强为 $\|A^{-1}\|\|\Delta A\|<1$，  
  则我们有 $1-\|A^{-1}\Delta A\| \geq 1-\|A^{-1}\|\|\Delta A\| = 1-\kappa(A)\frac{\|\Delta A\|}{\|A\|} >0$   
  于是我们有:    
  $$
  \begin{align}
  \frac{\|\Delta x\|}{\|x\|}
  &\leq
  \frac{\kappa(A)}{1-\|A^{-1}\Delta A\|} 
  \left(\frac{\|\Delta b\|}{\|b\|}+ \frac{\|\Delta A\|}{\|A\|}\right)\\
  &\leq
  \frac{\kappa(A)}{1-\kappa(A)\frac{\|\Delta A\|}{\|A\|}} 
  \left(\frac{\|\Delta b\|}{\|b\|}+ \frac{\|\Delta A\|}{\|A\|}\right)
  \end{align}
  $$
  上述相对误差上界同样是**先验界** (priori bound)，因为它至于计算前已知的数据有关.
  
- 若线性方程组 $Ax=b$ 中的系数矩阵 $A$ 是良态的 (即 $\kappa(A)$ 接近于 $1$)，  
  则关于解的相对误差与关于数据的相对误差具有相同的阶.

*****

若线性方程组 $Ax=b$ 有一个现成的计算解 $\hat x$，则可以将其用于**后验界** (posteriori bound) 中.    
考虑残差向量 $r = b-A\hat x$   
根据 $A^{-1}r = A^{-1}(b-A\hat x) = A^{-1}b -\hat x = x-\hat x$ 可知:  
$$
\begin{align}
\|x-\hat x\|
&=
\|A^{-1}r\|\\
&\leq 
\|A^{-1}\|\|r\|\quad (\text{note that }\|b\|=\|Ax\|\leq \|A\|\|x\|\Rightarrow 1\leq \frac{\|A\|\|x\|}{\|b\|})\\
&\leq
\|A^{-1}\|\|r\|\cdot \frac{\|A\|\|x\|}{\|b\|}\\
&=
\|A\|\|A^{-1}\|\frac{\|r\|}{\|b\|}\|x\|\\
&=
\kappa(A)\frac{\|r\|}{\|b\|}
\end{align}
$$
于是我们有 $\frac{\|x-\hat x\|}{\|x\|}\leq \kappa(A)\frac{\|r\|}{\|b\|}$ 成立.  

因此若线性方程组 $Ax=b$ 中的系数矩阵 $A$ 是良态的 (即 $\kappa(A)$ 接近于 $1$)，  
则关于解的相对误差与关于数据的相对误差具有相同的阶.  
然而，若系数矩阵 $A$ 是病态的 (即 $\kappa(A)\gg 1$)， 
则即使残差向量的范数 $\|r\|$ 很小，计算解 $\hat x$ 和精确解 $x$ 仍有可能相差甚远.

****

相容范数误差界限的一个共同特征是它们的保守性:  
即使实际误差很小，相容范数给出的误差上界也可能很大.

然而，如果一个有中等大小元素的中等大小的矩阵 $A$ 有很大的条件数，  
那么其逆矩阵 $A^{-1}$ 必定有一些大的元素.  
这会导致 $x=A^{-1}b$ 不可避免地对 $A$ 以及 $b$ 的某些元素的摄动非常敏感.

- 考虑非齐次线性方程组 $Ax=\begin{bmatrix} 10 & 7 & 8 & 7 \\ 7 & 5 & 6 & 5 \\ 8 & 6 & 10 & 9 \\ 7 & 5 & 9 & 10 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix} = \begin{bmatrix} 32 \\ 23 \\ 33 \\ 31 \end{bmatrix}=b$，易验证它的唯一解是 $x = \begin{bmatrix}
  1\\ 1\\ 1\\ 1\end{bmatrix}$ 

  如果我们对 $b$ 施加微小的扰动 (扰动 $\Delta b$ 的相对量级 $\frac{\|\Delta b\|_\infty}{\|b\|_\infty}$ 约为 $\frac1{200}$)，得到一个新的线性系统:

  $$
  A(x+\Delta x)=\begin{bmatrix} 10 & 7 & 8 & 7 \\ 7 & 5 & 6 & 5 \\ 8 & 6 & 10 & 9 \\ 7 & 5 & 9 & 10 \end{bmatrix} \begin{bmatrix} x_1 + \Delta x_1 \\ x_2 + \Delta x_2 \\ x_3 + \Delta x_3 \\ x_4 + \Delta x_4 \end{bmatrix} = \begin{bmatrix} 32.1 \\ 22.9 \\ 33.1 \\ 30.9 \end{bmatrix}=b+\Delta b
  
  \ \ \Rightarrow\ \ 
  
  \hat x =x + \Delta x=
  \begin{bmatrix}
  9.2\\
  -12.6\\
  4.5\\
  -1.1
  \end{bmatrix}
  $$
  计算解的误差 $\Delta x$ 的相对量级 $\frac{\|\Delta x\|_\infty}{\|x\|_\infty}$ 约为 $\frac{10}{1}$，是数据扰动 $\Delta b$ 的相对量级 $\frac{\|\Delta b\|_\infty}{\|b\|_\infty}$ 的 $2000$ 倍.
  
  现在我们对矩阵 $A$ 施加微小的扰动 (扰动 $\Delta A$ 的相对量级 $\frac{\|\Delta A\|_\infty}{\|A\|_\infty}$ 约为 $\frac1{100}$)，得到一个新的线性系统:
  
  $$
  (A+\Delta A)(x+\Delta x)=\begin{bmatrix} 10 & 7 & 8.1 & 7.2 \\ 7.08 & 5.04 & 6 & 5 \\ 8 & 5.98 & 9.98 & 9 \\ 6.99 & 4.99 & 9 & 9.98 \\ \end{bmatrix} \begin{bmatrix} x_1 + \Delta x_1 \\ x_2 + \Delta x_2 \\ x_3 + \Delta x_3 \\ x_4 + \Delta x_4 \\ \end{bmatrix} = \begin{bmatrix} 32 \\ 23 \\ 33 \\ 31 \\ \end{bmatrix}
  = b
  \ \ \Rightarrow\ \ 
  \hat x = x+\Delta x = \begin{bmatrix}
  -5.8\\
  12.0\\
  -1.6\\
  2.6
  \end{bmatrix}
  $$
  计算解的误差 $\Delta x$ 的相对量级 $\frac{\|\Delta x\|_\infty}{\|x\|_\infty}$ 约为 $\frac{10}{1}$，是数据扰动 $\Delta A$ 的相对量级 $\frac{\|\Delta A\|_\infty}{\|A\|_\infty}$ 的 $1000$ 倍.    
  因此对于这个线性系统来说，数据的微小扰动可能会导致解的剧烈变动. 
  
  注意到 $A$ 是一个对称阵，特征值为 $0.0102,0.8431,3.8581,30.2887$  
  故 $\kappa_2(A) = \|A\|_2\|A^{-1}\|_2 = \frac{\sigma_\max(A)}{\sigma_\min(A)} = \frac{\lambda_\max(A)}{\lambda_\min(A)}  = \frac{30.2287}{0.0102} = 2969.5$  
  这与我们观察的相对误差放大现象相符.



#### (3) 条件数的性质

关于 $\mathbb C^{n\times n}$ 上相容范数 $\|\cdot\|$ 的条件数 $\kappa(A)=\|A\|\|A^{-1}\|$ 具有以下性质:

- 一般地，对于任意 $A\in \mathbb C^{n\times n}$ 我们都有:  
  $$
  \kappa(A)= \|A^{-1}\|\|A\|\geq \|A^{-1}A\|=\|I\|\geq 1\\
  \kappa(A^{-1}) = \kappa(A)\\
  \kappa(\alpha A) = \kappa(A)\ \ (\forall\ \alpha\in \mathbb C\backslash\{0\})
  $$

- 关于酉不变范数的条件数也是酉不变的，即对于任意酉矩阵 $U,V\in \mathbb C^{n\times n}$ 都满足 $\kappa(UAV^{\mathrm H})= \kappa(A)$   
  这个结论是数值线性代数中许多算法的数值稳定性的基础.

  可以证明: 对于任意 $A,B\in \mathbb C^{n\times n}$，关于任何相容范数都有 $\kappa(AB)\leq \kappa(A)\kappa(B)$   
  因此我们对于经受一系列变换的矩阵的条件数的增长有一个上界.  
  当这些变换都是酉变换时，关于酉不变范数的条件数是不变的.

- 根据谱范数定义的 $\kappa_2(A)=\|A\|_2\|A^{-1}\|_2 = \frac{\sigma_{\max}(A)}{\sigma_\min(A)}$    
  特殊地，当 $A$ 是正规矩阵 (即满足 $AA^{\mathrm H}=A^{\mathrm H}A$) 时，我们有 $\kappa_2(A)=\frac{|\lambda_\max(A)|}{|\lambda_\min(A)|}$   
  更特殊地，当 $A$ 是 Hermite 阵 (即满足 $A^{\mathrm H}=A$) 时，我们有 $\kappa_2(A)=\frac{\lambda_\max(A)}{\lambda_\min(A)}$   
  此外，若 $A^{-1}$ 已知，  
  则我们可以利用范数不等式 $\|A\|_2\leq \|A\|_{\mathrm F}\leq \sqrt{n}\|A\|_2$ 方便地估算 $\kappa_2(A)=\|A\|_2\|A^{-1}\|_2$ 

  可以证明: $\kappa_2(A)=1$ 当且仅当 $A$ 是某个酉矩阵的纯量倍数

- 关于谱范数的条件数 $\kappa_2(A)$ 的几何意义:   
  记 $u,v$ 为空间 $\mathbb C^n$ 中的任意一对正交向量，$\theta(A)$ 为所有 $Au,Av$ 中夹角的最小值，则 $\kappa_2(A) = \cot(\frac12 \theta(A))$    
  当 $A$ 接近奇异时，存在正交向量 $u,v\in C^n$ 使得 $Au,Av$ 接近平行   
  因此 $\theta(A)$ 会很接近 $0$，使得 $\kappa_2(A) = \cot(\frac12{\theta(A)})$ 很大.



#### (4) Kahan 公式

对于任意非奇异方阵 $A\in \mathbb C^{n\times n}$，能使得 $A$ 变成奇异矩阵的扰动 $\Delta A$ 的谱范数最小值等于 $\frac{1}{||A^{-1}||_2}$  
即 $\underset{\det(A+\Delta A)=0}{\min}\|\Delta A\|_2= \frac{1}{\|A^{-1}\|_2}$ (有趣的是，约束条件 $\det(A+\Delta A)=0$ 是一个流形)

设非奇异矩阵 $A\in \mathbb C^{n\times n}$ 的奇异值分解为 $A = U\Sigma V^{\mathrm H} = \sum_{i=1}^n u_i\sigma_iv_i^{\mathrm H}$   
其中 $U,V\in \mathbb C^{n\times n}$ 为酉矩阵，而 $\begin{cases} \Sigma = \text{diag}\{\sigma_1,\dots,\sigma_n\}\\ \sigma_1 \geq \dots\geq \sigma_n \geq 0 \end{cases}$   

- 这里我们提供一个粗糙的证明思路:    
  Eckart-Young-Mirsky 定理指出，$\Delta A_0 = -u_n \sigma_n v_n^{\mathrm H}$ 是能让 $A$ 降秩的秩一扰动中谱范数最小的那个.  
  **(存疑)** 即我们有 $\underset{\det(A+\Delta A)=0}{\min}\|\Delta A\|_2 = \|\Delta A_0\|_2 =\|-u_n \sigma_n v_n^{\mathrm H}\|_2 = \sigma_n = \frac{1}{\|A^{-1}\|_2}$ 

  > 根据类似的思想我们有:  
  > **(复矩阵的最佳低秩近似)**  
  > 给定复矩阵 $A\in\mathbb{C}^{m\times n}$，记 $q= \min\{m,n\}$，$r = \rank(A)$，显然有 $r\leq q$.  
  > 设 $A$ 的奇异值分解 $A=U\Sigma V^{\mathrm H}$  
  > 其中 $U \in\mathbb C^{m\times m}$ 和 $V \in\mathbb C^{n\times n}$为酉矩阵  
  > 而 $\Sigma = \begin{cases} \Sigma_q & \text{if }m=n\\ \begin{bmatrix} \Sigma_q &0 \end{bmatrix} & \text{if }m < n\\ \begin{bmatrix} \Sigma_q \\0 \end{bmatrix} & \text{if }m > n\\ \end{cases}$  且 $\Sigma _q = \text{diag}\{\sigma_1,\sigma_2,\dots,\sigma_q\}$，$\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_r >0 = \sigma_{r+1} = \dots = \sigma_q$ 
  >
  > 记 $U,V$的前 $q$ 个列向量分别为 $u_1,u_2,\dots,u_q \in \mathbb C^{m}$，$v_1,v_2,\dots,v_q \in \mathbb C^{n}$ 
  > 于是我们有 $A=U\Sigma V^{\mathrm H} =  \sum_{i=1}^q u_i \sigma_i v_i^{\mathrm H} = \sum_{i=1}^r u_i \sigma_i v_i^{\mathrm H}$  
  >
  > 给定正整数 $k$ 满足 $1\leq k< r$     
  > 定义 $A$ 的一个秩 $k$ 低秩近似为 $A_k = \sum_{i=1}^k u_i \sigma_i v_i^{\mathrm H}$   
  > **Eckart-Young-Mirsky 定理指出: $A_k$ 是 $A$ 的一个最佳秩 $k$ 低秩近似**  
  > 具体来说，对于任意酉不变范数 $\|\cdot\|$ (例如谱范数 $\|\cdot\|_2$ 和 Frobenius 范数 $\|\cdot\|_{\mathrm F}$)  
  > 我们都有 $\underset{\rank(B)\leq k}{\min} \|A-B\| = \|A-A_k\|$，即有 $A_k = \underset{\rank(B)\leq k}{\arg\min} \|A-B\|$    

- 由于上面的推理有些过于大胆了，我们提供一种更严谨的证明思路:  
  首先 $\Delta A_0 = -u_n \sigma_n v_n^{\mathrm H}$ 可以让 $A+\Delta A$ 为奇异矩阵，  
  因此我们有 $\underset{\det(A+\Delta A)=0}\inf\|\Delta A\|_2 \leq \|\Delta A_0\|_2 = \sigma_n = \frac{1}{\|A^{-1}\|_2}$ 

  对于任意满足 $\|\Delta A\|_2 < \sigma_n$ 的 $\Delta A$，我们都有:
  $$
  \begin{align}
  \rho(A^{-1}\Delta A)
  &\leq
  \|A^{-1}\Delta A\|_2\quad (\text{spectral radius theorem})\\
  &\leq
  \|A^{-1}\|_2 \|\Delta A\|_2\\
  &<
  \|A^{-1}\|_2 \cdot \sigma_n\\
  &=
  \frac1{\sigma_n} \sigma_n\\
  &=
  1
  \end{align}
  \Rightarrow \det(I_n+A^{-1}\Delta A)\neq 0
  $$
  因此我们有 $\det(A+\Delta A)=\det(A)\det(I_n+A^{-1}\Delta A) \neq 0$   
  这表明任意满足 $\|\Delta A\|_2 < \sigma_n$ 的 $\Delta A$ 都不能使 $\det(A+\Delta A)=0$   
  于是我们有 $\underset{\det(A+\Delta A)=0}\inf\|\Delta A\|_2\geq \sigma_n = \frac{1}{\|A^{-1}\|_2}$ 

  而显然这个下确界是可以取到的，例如 $\Delta A=\Delta A_0 = -u_n \sigma_n v_n^{\mathrm H}$ 时  
  因此我们把 $\inf$ 替换为 $\min$，即有 Kahan 公式:  
  $$
  \underset{\det(A+\Delta A)=0}{\min}\|\Delta A\|_2= \frac{1}{\|A^{-1}\|_2}
  $$

**The End**
