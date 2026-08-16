# FDU 数值算法 2. 稠密线性最小二乘问题的解法

本文参考以下教材:

- 数值线性代数 (第二版) 徐树方, 高立, 张平文 第 $3$ 章

欢迎批评指正!

## 2.1 线性最小二乘问题

### 2.1.1 基本定义

最小二乘问题多产生于数据拟合问题.

给定 $m$ 个点 $t_1,\dots,t_n$ 和观测数据 $y_1,\dots, y_m$ 以及 $n$ 个已知函数 $\phi_1(\cdot),\dots,\phi_n(\cdot)$   
考虑 $\phi_i$ 的线性组合 $f(x;t) = \underset{k=1}{\overset{n}\sum} x_k \phi_k(t)$，  
我们希望它在点 $t_1,\dots,t_n$ 上能最佳地逼近 $y_1,\dots,y_m$   
为此我们定义残差 $r_i(x) = y_i - f(x;t_i) = y_i - \underset{k=1}{\overset{n}\sum} x_k \phi_k(t_i)\ \ (i=1,\dots,m)$，并定义以下记号:
$$
A=[\phi_1(t),\dots,\phi_n(t)] = 
\begin{bmatrix} 
\phi_1(t_1) & \dotsm & \phi_n(t_1)\\
\vdots & & \vdots\\
\phi_1(t_m) & \dotsm & \phi_n(t_m)
\end{bmatrix}
\qquad
b = 
\begin{bmatrix}
y_1\\
\vdots\\
y_m\end{bmatrix}
\qquad 
x = 
\begin{bmatrix}
x_1\\
\vdots\\
x_n\end{bmatrix}
\qquad
r(x) = 
\begin{bmatrix}
r_1(x)\\
\vdots\\
r_m(x)\end{bmatrix}
$$
因此 $r(x) = b-Ax$ 

- 当 $m=n$ 时，我们通常可以要求 $r(x)=0$，即等价于求解线性方程组 $Ax=b$ 
- 当 $m>n$ 时，$r(x)=0$ 通常是无解的 (它称为**超定方程组**或**矛盾方程组**)  
  此时我们只能要求残差向量 $r(x)$ 在某种范数意义下最小， 
  而线性最小二乘问题就是要求残差向量 $r(x)$ 在 $2$ 范数意义下最小.

给定 $\begin{cases}
A\in \mathbb R^{m\times n}\\
b\in \mathbb R^m\end{cases}$ 我们称 $\underset{x\in \mathbb R^n}{\min} \|r(x)\|_2 = \underset{x\in \mathbb R^n}{\min} \|b-Ax\|_2$ 为**最小二乘问题**.   
最小二乘问题 $\underset{x\in \mathbb R^n}{\min} \|Ax-b\|_2$ 的解 $x_{\text{ls}}$ 称为线性方程组 $Ax=b$ 的**最小二乘解**.  

- 除非另有规定，本课程中我们**默认假设** $\rank(A) = n <m$.  
  后面我们可以看到这个假设保证了最小二乘解的唯一性.

- 在我们的假设中，残差向量 $r(x)$ 线性地依赖于 $x$，因此称为**线性最小二乘问题**. 
  更一般地，若 $r(x)$ 非线性地依赖于 $x$，则称其为非线性最小二乘问题.



### 2.1.2 最小二乘解

我们首先介绍一下线性方程组 $Ax=b$ 的基本性质.  
**(数值线性代数, 定理 $3.1.1$)**  
给定 $\begin{cases}
A\in \mathbb R^{m\times n}\\
b\in \mathbb R^m\end{cases}$ 记 $A=[a_1,\dots,a_n]$   
$Ax=b$ 有解的充要条件是 $\rank([A,b])=\rank(A)$，即 $b\in \text{Range}(A)$ 

- **必要性:**   
  若 $Ax=b$ 存在解 $x_0$，则 $b\in \text{Range}(A)$，说明 $\rank([A,b])=\rank(A)$  
- **充分性:**    
  若 $\rank([A,b])=\rank(A)$，则 $b\in \text{Range}(A)$   
  即存在 $c_1,\dots,c_n\in \mathbb R$ 使得 $b = \underset{i=1}{\overset{n}\sum} c_i a_i$  
  令 $x_0 = (c_1,\dots,c_n)^T$，即有 $b= Ax$ 成立，表明 $Ax=b$ 有解.

**(数值线性代数, 定理 $3.1.2$)**   
给定 $\begin{cases}
A\in \mathbb R^{m\times n}\\
b\in \mathbb R^m\end{cases}$ 记 $A=[a_1,\dots,a_n]$  
若 $x_0$ 是 $Ax=b$ 的任意给定的解，则 $Ax+b$ 的解集为 $x_0 + \text{Null}(A)=\{x_0+x: Ax=0_m\}$ 

- 这个定理表明:  
  只要知道了 $Ax=b$ 的一个解，便可以对 $\text{Null}(A)$ 位移得到 $Ax=b$ 的全部解.   
  因此 $Ax=b$ 存在唯一解，当且仅当 $\begin{cases}
  b\in \text{Range}(A)\\
  \text{Null}(A) = \{0_n\}\end{cases}$  

- **证明:**   
  一方面，对于任意 $y\in x_0+ \text{Null}(A)$，  
  我们有 $Ay=A(x_0 + y-x_0) = Ax_0 + A(y-x_0) = b+0_m = b$，  
  表明 $y$ 是 $Ax=b$ 的解.

  另一方面，对于 $Ax=b$ 的任意解 $y$，  
  我们有 $A(y-x_0) = Ay - Ax_0 = b-b = 0_m$，  
  说明 $y-x_0\in \text{Null}(A)$，即有 $y\in x_0+ \text{Null}(A)$ 

  综上所述，$Ax+b$ 的解集为 $x_0+ \text{Null}(A)$

****

下面我们讨论最小二乘解的存在性和唯一性问题.   

给定 $\begin{cases}
A\in \mathbb R^{m\times n}\\
b\in \mathbb R^m\end{cases}$   
向量 $b\in \mathbb R^{m}$ 可以唯一地分解为 $\begin{cases}
b=b_1+b_2\\
b_1\in \text{Range}(A)\\
b_2\in \text{Range}(A)^{\bot}\end{cases}(\mathbb R^m = \text{Range}(A) \oplus \text{Range}(A)^{\bot})$    
当 $x$ 取遍  $\mathbb R^n$ 时，$y = Ax$ 会取遍整个 $\text{Range}(A)$   
当残差向量 $r=b-y = b-Ax$ 垂直于 $\text{Range}(A)$，即 $r=b_2\in \text{Range}(A)^{\bot}$ 时，$\|b-y\|_2$ 达到极小.  
此时 $y_{\min} = b_1$，对应的残差向量 $b_2 = b-y_{\min}$   
有了 $y_\min$，只要求解 $Ax=y_{\min}$ 即可得到最小二乘解 $x_{\text{ls}}$ 

**(数值线性代数, 定理 $3.1.3$)**  
线性最小二乘问题 $\underset{x\in \mathbb R^n}{\min} \|Ax-b\|_2$ 的解总是存在的.  
最小二乘解唯一的充要条件是 $\text{Null}(A)=\{0_n\}$ 

- 记最小二乘解集 $\mathcal X_{\text{ls}}:= \arg \underset{x\in \mathbb R^n}{\min} \|b-Ax\|_2^2$，由上述定理可知 $\mathcal X_{\text{ls}}$ 总是非空的.  
  除非另有规定，本课程中我们**默认假设** $\rank(A) = n <m$.   
  这个假设保证了 $\begin{cases}
  \text{Range}(A) = \mathbb R^m\\
  \text{Null}(A) = \{0_n\}\end{cases}$，即保证了 $\mathcal X_{\text{ls}}$ 只有唯一的元素.  
  在此假设下，我们记这个唯一的最小二乘解为 $x_{\text{ls}}$ 

- **证明:**  
  向量 $b\in \mathbb R^{m}$​ 可以唯一地分解为 $\begin{cases}
  b=b_1+b_2\\
  b_1\in \text{Range}(A)\\
  b_2\in \text{Range}(A)^{\bot}\end{cases}(\mathbb R^m = \text{Range}(A) \oplus \text{Range}(A)^{\bot})$   
  于是对于任意 $x\in \mathbb R^{n}$，都有 $b_1-Ax \in \text{Range}(A)$，从而有:
  $$
  \begin{align}
  \|r(x)\|^2_2 
  &= \|b-Ax\|_2^2\\
  &= \|(b_1-Ax) +b_2\|_2^2\quad (b_1-Ax\ \bot\ b_2)\\
  &= \|b_1-Ax\|_2^2 + \|b_2\|_2^2
  \end{align}
  $$
  根据 $b_1 \in \text{Range}(A)$ 可知 $Ax=b_1$ 是可解的  
  因此 $\|b_1-Ax\|_2^2\geq 0$，当且仅当 $x$ 是 $Ax=b_1$ 的解时取等.  
  于是我们有 $\|r(x)\|_2^2 \geq \|b_2\|_2^2$，当且仅当 $x$ 是 $Ax=b_1$ 的解时取等.  
  所以最小二乘解总是存在的，其唯一的充要条件是 $\text{Null}(A)=\{0_n\}$.  

****

**(数值线性代数, 定理 $3.1.4$)**  
给定 $\begin{cases}
A\in \mathbb R^{m\times n}\\
b\in \mathbb R^m\end{cases}$ 考虑线性最小二乘问题 $\underset{x\in \mathbb R^n}{\min} \|Ax-b\|_2$  
$x\in \mathcal X_{\text{ls}}$ 当且仅当 $A^TAx=A^Tb$  

- 向量 $b\in \mathbb R^{m}$ 可以唯一地分解为 $\begin{cases}
  b=b_1+b_2\\
  b_1\in \text{Range}(A)\\
  b_2\in \text{Range}(A)^{\bot}\end{cases}(\mathbb R^m = \text{Range}(A) \oplus \text{Range}(A)^{\bot})$ 
- **必要性证明:**   
  若 $x\in \mathcal X_{\text{ls}}$，则我们有 $Ax=b_1$ 成立，于是 $r(x)=b-Ax=b-b_1 = b_2\in \text{Range}(A)^{\bot}=\text{Null}(A^T)$   
  因此 $A^Tr(x) = A^T(b-Ax) = 0$，即有 $A^TAx = A^Tb$ 
- **充分性证明:**  
  若 $A^TAx = A^Tb$，则对于任意 $y\in \mathbb R^n$ 有:  
  $\begin{align}
  \|b-A(x+y)\|_2^2 
  &= \|b-Ax\|_2^2 - 2y^T A^T(b-Ax) + \|Ay\|_2^2\\
  &= \|b-Ax\|_2^2 + \|Ay\|_2^2\\
  &\geq \|b-Ax\|_2^2\end{align}$     
  即有 $x\in \mathcal X_{\text{ls}}$ 成立.

方程组 $A^TA x=A^Tb$ 称为最小二乘问题的**正则化方程组**或**法方程组**  
它是一个含有 $n$ 个变量和 $n$ 个方程的线性方程组.  
在 $A$ 列线性无关 (即 $\text{Range}(A)=n$) 的条件下，$A^TA$ 是正定阵，  
因此我们可以根据 **Cholesky 分解 (平方根法)** 求解方程组 $A^TA x=A^Tb$   
这样我们就得到了求解最小二乘问题最古老的算法——**正则化方法**.

- 计算 $\begin{cases}
  C=A^TA\\
  d= A^Tb\end{cases}$   
  值得注意的是，在 $C=A^TA$ 的计算中，若不使用足够的精度，则矩阵 $A$ 的一些信息可能会丧失.
- 使用平方根法计算 $C$ 的 Cholesky 分解 $C=LL^T$ 
- 求解三角方程组 $\begin{cases}
  Ly = d\\
  L^Tx=y\end{cases}$

注意正则化方程组 $A^TA x=A^Tb$ 的解 $x$ 可以表示为 $x=(A^TA)^{-1} A^Tb$  
我们定义 $A^{\dagger} = (A^TA)^{-1} A^T\in \mathbb R^{n\times m}$ (它是 $A$ 的 **Moore-Penrose 广义逆**的特殊情况)  
在 $A$ 列线性无关 (即 $\text{Range}(A)=n$) 的条件下，$A^{\dagger}$ 满足 Penrose 方程组 $\begin{cases}
AXA=A\\
XAX=X\\
(AX)^T = AX\\
(XA)^T = XA\end{cases}$ 
于是这个解又可以记为 $x=A^{\dagger} b$ 

***

下面的定理给出了 $b$ 的扰动引起的最小二乘解 $x$ 的相对误差的界.     
($A,b$ 的同时扰动对最小二乘解 $x$ 的影响是一个非常复杂的问题，我们不进行讨论)

**(数值线性代数, 定理 $3.1.5$)**   
给定 $\begin{cases}
A\in \mathbb R^{m\times n}\\
b\in \mathbb R^m\end{cases}$ 考虑线性最小二乘问题 $\underset{x\in \mathbb R^n}{\min} \|Ax-b\|_2^2$ (默认假设 $\rank(A) = n <m$)  
假定 $b$ 有扰动 $\delta b$ 且 $\begin{cases}
x = \arg \underset{x\in \mathbb R^n}{\min} \|b-Ax\|_2^2\\
x+\delta x = \arg \underset{x\in \mathbb R^n}{\min} \| (b+\delta b)-Ax\|_2^2\end{cases}$  
即有 $\begin{cases}
x = A^{\dagger} b\\
x+\delta x = A^{\dagger} (b + \delta b)\end{cases}$ 

设 $b_1$ 和 $\tilde b_1$ 分别是 $b$ 和 $b+\delta b$ 在 $\text{Range}(A)$ 上的正交投影.  
若 $b_1\neq 0_m$，则 $\frac{\|\delta x\|_2}{\|x\|_2} \leq \kappa_2(A) \frac{\|b_1-\tilde b_1\|_2}{\|b_1\|_2}$ (其中 $\kappa_2(A) = \|A\|_2 \|A^\dagger\|_2$ 是广义条件数) 

- **证明:**     
  在 $\rank(A) = n<m$ 的假设下，$A^TA$ 正定.  
  因此 $\text{Null}(A^{\dagger})=\text{Null}((A^TA)^{-1}A^T) = \text{Null}(A^T) = \text{Range}(A)^{\bot}$   
  于是我们有 $\begin{cases}
  A^{\dagger} b = A^{\dagger} b_1\\
  A^{\dagger} (b+\delta b) = A^{\dagger} \tilde b_1\end{cases}$   
  从而有: 
  $$
  \begin{align}
  \|\delta x\|_2 
  &= 
  \|A^{\dagger} b -A^{\dagger} (b+\delta b)\|_2\\
  &=
  \|A^{\dagger} b_1 - A^{\dagger} \tilde b_1\|_2\\
  &\leq 
  \|A^{\dagger}\|_2 \|b_1-\tilde b_1\|_2\end{align}
  $$
  若 $x$ 是最小二乘解，则我们有 $Ax=b_1$ 成立，即有 $\|b_1\|_2 = \|Ax\|_2 \leq \|A\|_2 \|x\|_2$ 

  联立 $\begin{cases}
  \|\delta x\|_2 \leq \|A^\dagger\|_2 \|b_1- \tilde b_1\|_2\\
  \|x\|_2 \geq \frac{\|b_1\|_2}{\|A\|_2}\end{cases}$ 即得 $\frac{\|\delta x\|_2}{\|x\|_2} \leq \|A\|_2 \|A^\dagger\|_2 \frac{\|b_1-\tilde b_1\|_2}{\|b_1\|_2}$ 

- 这个定理表明:  
  只有 $b$ 在 $\text{Range}(A)$ 上的投影 $b_1$ 的变化会对最小二乘解产生影响.  
  此外，该影响的敏感性依赖于**最小二乘问题的条件数** $\kappa_2(A) = \|A\|_2 \|A^\dagger\|_2$.  
  若 $\kappa_2(A)$ 很小，则我们称该最小二乘问题是良态的; 否则称为病态的.

下面的定理给出了 $\kappa_2(A) = \|A\|_2 \|A^\dagger\|_2$ 与 $\kappa_2(A^TA)$ 之间的关系.  
**(数值线性代数, 定理 $3.1.6$)**  
若 $A$ 列线性无关 (这保证了 $A^{\dagger} = (A^TA)^{-1}A^T$)，则 $(\kappa_2(A))^2 = \kappa_2(A^TA)$ 

- **证明:**  
  根据谱范数的性质，我们有 $\begin{cases}
  \|A\|_2^2 = \|A^TA\|_2\\
  \|A^{\dagger}\|_2^2 = \|A^{\dagger}(A^{\dagger})^T\|_2 = \|(A^TA)^{-1}\|_2\end{cases}$  
  于是 $(\kappa_2(A))^2 = \|A\|_2^2 \|A^{\dagger}\|_2^2 = \|A^TA\|_2 \|(A^TA)^{-1}\|_2 = \kappa_2(A^TA)$ 
- 这个定理表明:  
  最小二乘问题 $\underset{x\in \mathbb R^n}{\min} \|Ax-b\|_2^2$ 化为正则化方程 $(A^TA)x = A^Tb$ 会导致条件数变为原来的平方.  
  这增加了求解过程对舍入误差的敏感性.   
  因此在使用正则化方法，要特别注意这一点.



## 2.2 初等正交变换

为给出求解最小二乘问题的更实用的算法，我们来介绍两个最基本的初等正交变换.  
它们是数值线性代数中许多重要算法的基础.

邵老师提到的一些有趣的事实:  

- 行列式等于 $1$ 的就是旋转变换，行列式等于 $-1$ 的就是镜像变换 (偶数次镜像变换的复合就是旋转变换)
- 一次空间旋转 (由行列式为 $1$ 的 $3$ 阶方阵表示) 可以分解为三次平面旋转 (由行列式为 $1$ 的 $2$ 阶方阵表示)
- 旋转变换的本身可以不断逼近单位阵 (而镜像变换做不到)，这在矩阵序列收敛中非常有用.  

### 2.2.1 Householder 变换

回顾 Gauss 变换: 
$$
l_k = \frac{1}{x_k} \begin{bmatrix}
0\\
\vdots\\
0\\
x_{k+1}\\
\vdots\\
x_{n}\end{bmatrix}\qquad 
L_k = I - l_k e_k^T = \begin{bmatrix}
1 & & & & & \\
& \ddots &  & & & \\
& & 1 & & &\\
& & -\frac{x_{k+1}}{x_k} & 1 & & \\
& & \vdots & & \ddots & \\
&& -\frac{x_n}{x_k} &&& 1\end{bmatrix}
\quad\Rightarrow\quad
L_k x =  \begin{bmatrix}
x_1\\
\vdots\\
x_k\\
0\\
\vdots\\
0\end{bmatrix}
$$
初等下三角阵 $L_k$ 可将 $x$ 的第 $k+1$ 个至第 $n$ 个分量置为零 (前提是 $x_k\neq 0$)   

现在我们来讨论如何求一个初等正交矩阵，来代替 Gauss 变换矩阵的功能.  
这样，一个矩阵的上三角化任务便可以由一系列的初等正交变换来完成.

**(Householder 变换)**  
设 $w\in \mathbb R^n$ 为单位向量 $(\|w\|_2=1)$，  
我们定义对应的 Householder 变换 $H:= I - 2ww^T\in \mathbb R^{n\times n}$ (又称**初等反射矩阵**或**镜像变换**)

**(Householder 变换的基本性质, 数值线性代数, 定理 $3.2.1$)**  
设 $H=I-2ww^T$ 是单位向量 $w$ 对应的 Householder 变换，则我们有:

- **(对称性)** $H^T=H$   
  $H^T = (I—2ww^T)^T = I-2ww^T = H$ 

- **(正交性)** $H^TH = I$  
  $H^TH = H^2 = (I-2ww^T)^2 = I - 4ww^T + 4ww^T ww^T = I-4ww^T + 4 \cdot 1\cdot ww^T = I$ 

- **(对合性)** $H^2 = I$  
  $H^2 = H^TH = I$ 

- **(反射性)** 对于任意 $x\in \mathbb R^n$，$Hx$ 是 $x$ 关于 $w$ 的垂直超平面 $\text{span}\{w\}^{\bot}=\{x\in \mathbb R^n:w^Tx=0\}$ 的镜像.  
  $x$ 可分解为 $\begin{cases}
  x=x_1 + x_2\\
  x_1 \in \text{span}(w)\\
  x_2 \in \text{span}(w)^{\bot}\end{cases}$ (于是有 $\begin{cases}
  \exist\ \alpha\in \mathbb R\text{ such that } x_1 = \alpha w\\
  w^Tx_2=0\\
  x = x_1 + x_2 =\alpha w + x_2\end{cases}$)    
  我们有:
  $$
  \begin{align}
  Hx 
  &= (I-2ww^T)(x_1 + x_2)\\
  &= (I-2ww^T)(\alpha w + x_2)\\
  &= \alpha w - 2ww^T\cdot \alpha w + x_2 - 2ww^Tx_2\\
  &= \alpha w- 2\alpha w\cdot 1 + x_2 -2w\cdot 0\\
  &= -\alpha w + x_2\\
  &= -x_1 + x_2
  \end{align}
  $$
  因此 $Hx$ 是 $x$ 关于 $w$ 的垂直超平面 $\text{span}\{w\}^{\bot}=\{x\in \mathbb R^n:w^Tx=0\}$ 的镜像.
  
  取 $\begin{cases}
  P_w := ww^T\\
  P_{w^\bot}:= I-ww^T\end{cases}$ 则我们可以将 $H=I-2ww^T$ 分解为 $H=-P_w + P_{w^\bot}$   
  其中 $P_{w^\bot}$ 是从 $\mathbb R^n$ 向 $\text{span}\{w\}^{\bot}$ 的投影算子，而 $P_w=I-P_{w^\bot}$ 可提取向量中垂直于 $\text{span}\{w\}^{\bot}$ 的部分.  
  $$
  \begin{align}
  x
  &= (ww^T)x + (I-ww^T)x\\
  &= P_w x + P_{w^\bot}x\\
  &= x_1 + x_2 
  \end{align}
  $$

****

Householder 变换的主要用途在于，它和 Gauss 变换一样，  
可以通过适当选取单位向量 $w$，将给定向量的若干个指定分量置为零.

**(数值线性代数, 定理 $3.2.2$)**  
若 $x\neq 0_n\in \mathbb R^n$，则可构造 $\begin{cases}
\alpha = \pm \|x\|_2\\
w = \frac{x-\alpha e_1}{\|x-\alpha e_1\|_2}\\
H = I - 2ww^T\end{cases}$ 使得 $Hx = \alpha e_1$ (其中 $e_1$ 是 $\mathbb R^n$ 的第 $1$ 个标准单位基向量)

- **证明:**   
  记 $\alpha = \pm\|x\|_2$  
  为使 $Hx = (I-2ww^T) x = x - 2(w^Tx) w = \alpha e_1$，我们有 $w=\frac{x-\alpha e_1}{\|x-\alpha e_1\|_2}$ 

****

一个自然的问题是，实际计算中，$\alpha$ 应当取 $\|x\|_2$ 还是 $-\|x\|_2$?  
我们通常取 $\alpha = \|x\|_2$，但这可能造成**相消**的问题:  
当 $x$ 的第 $1$ 个分量为正值 $(x_1>0)$ 且占主导地位 (即 $\|x\|_2 \approx x_1$) 时，  
$x-\|x\|_2 e_1$ 的第 $1$ 个分量便会出现 $x_1 - \|x\|_2$ 这种两个相近的数相减的情况，从而严重地损失有效数字.

但幸运的是，我们可通过等价变形来规避相消问题的出现:  
$$
x_1 - \|x\|_2 = \frac{x_1^2 - \|x\|_2^2}{x_1 + \|x\|_2} = -\frac{x_2^2 + \dots + x_n^2}{x_1 + \|x\|_2^2}
$$
当 $x_1>0$ 时，上述等价变形便可规避相消问题的出现.  
(对于 $x_1\leq 0$ 的情况，相消问题并不会出现，)

其次，记 $\begin{cases}
v = x-\alpha e_1\\
w= \frac{v}{\|v\|_2}\\
\beta = \frac{2}{v^Tv}\end{cases}$，我们有:
$$
H = I - 2ww^T = I - \frac{2}{v^Tv} vv^T = I - \beta vv^T
$$
因此我们没必要求出 $w$，只需求出 $v$ 和 $\beta$ 即可.  
在实际运算中，我们可以将 $v$ 的第一个分量规格化为 $1$ (这样就无需储存了)  
然后将 $v$ 的后 $n-1$ 个分量保存在 $x$ 的后 $n-1$ 个置为 $0$ 的分量上.

最后，$v^T v$ 的上溢和下溢也是计算中需要考虑的问题.  
为避免溢出，我们可用 $\frac{x}{\|x\|_\infty}$ 代替 $x$ 来构造 $v$   
因为理论上，正数乘是不影响向量单位化结果的，  
即对于任意 $\alpha>0$，向量 $\alpha v$ 和 $v$ 的单位化结果是相同的.

基于上述讨论，我们得到如下算法:  
**(计算 Householder 变换, 数值线性代数, 算法 $3.2.1$)**  
$$
\begin{align}
&\text{function: } [v,\beta] = \text{Householder}(x)\\
&\qquad n = \text{length}(x)\\
&\qquad x = \frac{x}{\|x\|_\infty}\\
&\qquad v(2:n) = x(2:n)\\
&\qquad (下面确定\ x_1\ 和\ \beta)\\ 
&\qquad \sigma = x(2:n)^T x(2:n)\\
&\qquad \text{if } \sigma =0\\
&\qquad\qquad \beta = 0\\
&\qquad \text{else}\\
&\qquad\qquad \alpha = \sqrt{x(1)^2 + \sigma}\\
&\qquad\qquad \text{if }x(1)>0\quad (规避相消)\\
&\qquad\qquad\qquad v(1) = -\frac{\sigma}{x(1) + \alpha}\\
&\qquad\qquad \text{else}\quad (x(1)\leq 0\ 时无需规避相消)\\
&\qquad\qquad\qquad v(1) = x(1) - \alpha\\
&\qquad\qquad\text{end}\\
&\qquad\qquad \beta = \frac{2 v(1)^2}{v(1)^2 + \sigma}\\
&\qquad\qquad v = \frac{v}{v(1)}\\
&\qquad \text{end}
\end{align}
$$
上述算法是数值稳定的.  
假定计算结果是 $\tilde v$ 和 $\tilde \beta$，定义 $\tilde H = I -\tilde\beta \tilde v \tilde v^T$  
我们可以证明 $\|H-\tilde H\|_2 = O(\text{eps})$ 

***

实际上 Householder 变换可以将向量中任意若干个相邻的分量置为零.  
例如，欲将 $x\in \mathbb R^{n}$ 的第 $k+1$ 个分量至第 $j$ 个分量置为零，  
只需定义 $\begin{cases}
\alpha = \pm\sqrt{\underset{i=k}{\overset{j}\sum} x_i^2}\\
v = (0,\dots,0,x_k - \alpha, x_{k+1},\dots,x_j,0,\dots,0)\\
\beta = \frac{2}{v^T v}\\
H= I - 2\beta vv^T\end{cases}$ 

***

在使用 Householder 变换 $H=I-\beta v v^T\in \mathbb R^{m\times m}$ 转化给定矩阵 $A\in \mathbb R^{m\times n}$ 的过程中，  
主要的计算量不是确定 $v$ 和 $\beta$，而是计算矩阵乘积.  
在实际计算时，$H$ 无需显式给出，而是根据如下的公式计算矩阵乘积 $HA$:
$$
HA = (I-\beta v v^T) A = A - \beta v (A^Tv)^T = A - v\cdot (\beta A^T v)^T
$$

- 确定 $v$ 和 $\beta$
- 计算 $u = \beta A^T v$ 
- 计算 $B = A - vu^T$ ($B$ 即为所求的乘积 $HA$) 

总浮点运算量为 $O(4mn)$ 



### 2.2.2 Givens 变换

若要将一个向量中许多相邻的分量置为零，则可以使用 Householder 变换.  
若只要将其中一个分量置为零，则应使用 Givens 变换.

**(Givens 变换)**  
$$
G(i,k,\theta)
=
I + \sin(\theta) (e_i e_k^T - e_k e_i^T) + (\cos(\theta)-1)(e_ie_i^T + e_k e_k^T)
=
\begin{array}{cl}
\begin{bmatrix}
  1 &  &  &  &  &  &  \\
   & \ddots &  &  &  &  &  \\
   &  & \cos(\theta) & \cdots & \sin(\theta) &  &  \\
   &  & \vdots & \ddots & \vdots &  &  \\
   &  & -\sin(\theta) & \cdots & \cos(\theta) &  &  \\
   &  &  &  &  & \ddots &  \\
   &  &  &  &  &  & 1 \\
\end{bmatrix} 
&
\begin{array}{}
\\
\\
i\\
\\
k\\
\\
\\
\end{array} \\
\begin{array}{}
&& \ \ i &\qquad& k &&\\
\end{array}
\end{array}
$$
易证 $G(i,k,\theta)$ 总是一个正交阵.  
从几何上看，$G(i,k,\theta)x$ 是在 $(i,k)$ 坐标平面 $\text{span}\{e_i,e_k\}$ 内将 $x$ 按顺时针方向做了 $\theta$ 的旋转，  
因此 Givens 变换又称为**平面旋转变换**.

设 $x\in \mathbb R^n$，记 $\begin{cases}
y=G(i,k,\theta) x\\
c = \cos(\theta)\\
s = \sin(\theta)\end{cases}$，则有 $\begin{cases}
y_i = cx_i + sx_k\\
y_k = -s x_i + c x_k\\
y_j = x_j & (\forall\ j\neq i,k)\end{cases}$   

若要令 $y_k = 0$，则只需取 $\begin{cases}
c = \cos(\theta) = \frac{x_i}{\sqrt{x_i^2 + x_k^2}}\\
s = \sin(\theta) = \frac{x_k}{\sqrt{x_i^2 + x_k^2}}\end{cases}$ 即有 $\begin{cases}
y_i = \sqrt{x_i^2 + x_k^2}\\
y_k = 0\end{cases}$ 

若用 $G(i,k,\theta)$ 变换左乘 (右乘) 矩阵 $A\in \mathbb R^{n\times n}$，  
则它只改变 $A$ 的第 $i,k$ 行 (列) 的元素，其余元素保持不变.

****

直接使用 $\begin{cases}
c = \cos(\theta) = \frac{x_i}{\sqrt{x_i^2 + x_k^2}}\\
s = \sin(\theta) = \frac{x_k}{\sqrt{x_i^2 + x_k^2}}\end{cases}$ 计算 $c$ 和 $s$，可能会发生下溢.

为简化记号，考虑 $\begin{cases}
\begin{bmatrix}
c & s\\
-s & c\end{bmatrix}
\begin{bmatrix}
a\\
b\end{bmatrix} = \begin{bmatrix}
r\\
0\end{bmatrix}\\
r = \sqrt{a^2 + b^2}\end{cases}$   
为避免下溢，我们不直接使用 $\begin{cases}
c = \frac{a}{r}\\
s = \frac{b}{r}\end{cases}$，而是使用以下算法:  
**(计算 Givens 变换, 数值线性代数, 算法 $3.2.2$)**  
$$
\begin{align}
&\text{function: }[c,s] = \text{Givens}(a,b)\\
&\qquad \text{if }b=0\\
&\qquad\qquad c=1;\ s=0\\
&\qquad \text{else}\\
&\qquad\qquad \text{if } |b| > |a|\\
&\qquad\qquad\qquad t = \frac{a}{b};\ \ s= \frac{1}{\sqrt{1+ t^2}};\ \ c=st\\
&\qquad\qquad \text{else}\\
&\qquad\qquad\qquad t = \frac{b}{a};\ \ c= \frac{1}{\sqrt{1+ t^2}};\ \ s=ct\\
&\qquad\qquad \text{end}\\
&\qquad \text{end}
\end{align}
$$
上述算法是数值稳定的.  
假定计算结果是 $\tilde c$ 和 $\tilde s$，我们可以证明 $\begin{cases}
\tilde c = c(1+\delta_c) & (\delta_c = O(\text{eps}))\\
\tilde s = s(1+\delta_s) & (\delta_s = O(\text{eps}))\end{cases}$ 

****

若 $a,b$ 为复数，记 $\begin{cases}
a=a_0 e^{i\theta_1}\\
b= b_0 e^{i\theta_2}\end{cases}$ 则我们可以利用一个对角的酉变换将 $a,b$ 转为实数 $a_0,b_0$:  
$$
\begin{bmatrix}
e^{-i\theta_1} & \\
& e^{-i\theta_2}
\end{bmatrix}
\begin{bmatrix}
a\\
b
\end{bmatrix}
=
\begin{bmatrix}
e^{-i\theta_1} \cdot a_0 e^{i\theta_1}\\
e^{-i\theta_2} \cdot b_0 e^{i\theta_2}
\end{bmatrix}
=
\begin{bmatrix}
a_0\\
b_0
\end{bmatrix}
$$
然后对实向量 $\begin{bmatrix} a_0\\ b_0 \end{bmatrix}$ 进行 Givens 变换:
$$
[c,s] = \text{Givens} (a_0,b_0)\\
\begin{bmatrix}
c & s\\
-s & c\end{bmatrix}
\begin{bmatrix}
a_0\\
b_0\end{bmatrix} = \begin{bmatrix}
r\\
0\end{bmatrix}
$$
最终得到复向量 $\begin{bmatrix} a\\ b \end{bmatrix}$ 的 Givens 变换:
$$
\begin{bmatrix}
c & s\\
-s & c\end{bmatrix}
\begin{bmatrix}
e^{-i\theta_1} & \\
& e^{-i\theta_2}
\end{bmatrix}
\begin{bmatrix}
a\\
b
\end{bmatrix} =
\begin{bmatrix}
r\\
0\end{bmatrix}
$$
(实际上复数的 Givens 变换可以有更快的实现，使用 $\begin{bmatrix}
c & s\\
-\bar s & c\end{bmatrix}$，其中 $c$ 为实数，$s$ 为复数，但我们不做过多讨论)



## 2.3 正交变换法

谱范数具有酉不变性，即对于任意酉矩阵 $U\in \mathbb C^n$ (满足 $U^HU=UU^H = I$) 都有 $\|UA\|_2 = \|A\|_2$    

给定 $\begin{cases}
A\in \mathbb R^{m\times n}\\
b\in \mathbb R^m\end{cases}$ 考虑线性最小二乘问题 $\underset{x\in \mathbb R^n}{\min} \|Ax-b\|_2^2$   
对于任意正交矩阵 $Q\in \mathbb R^n$ (满足 $Q^TQ = QQ^T = I$)，  
最小二乘问题 $\underset{x\in \mathbb R^n}{\min} \| Q^T(Ax-b)\|_2^2$ 都等价于原问题 $\underset{x\in \mathbb R^n}{\min} \|Ax-b\|_2^2$   
我们可以通过适当选取正交矩阵 $Q$ 使得 $\underset{x\in \mathbb R^n}{\min} \| Q^T(Ax-b)\|_2^2$ 是较容易求解的形式.  
这便是**正交变换法**的基本思想.

### 2.3.1 QR 分解

**($\text{QR}$ 分解定理, 数值线性代数, 定理 $3.3.1$)**  
若 $A\in \mathbb R^{m\times n}\ (m\geq n)$，则 $A$ 具有 $\text{QR}$ 分解 $A=Q\begin{bmatrix}
R\\
0_{(m-n)\times n}\end{bmatrix}$   
其中 $Q\in \mathbb R^{m\times m}$ 是正交矩阵，$R\in \mathbb R^{n\times n}$ 是具有非负对角元的上三角阵.  
特殊地，当 $m=n$ 且 $A$ 非奇异时，上述分解是唯一的. 

- **存在性证明:**  
  对 $n$ 使用数学归纳法.  
  当 $n=1$ 时，命题自然成立.  
  现假设命题对所有 $p\times (n-1)$ 矩阵成立 (其中 $p\geq n-1$) 

  设 $A$ 的第一列为 $a_1$，根据 Householder 变换的性质可知，  
  存在正交矩阵 $Q_1\in \mathbb R^{m\times m}$ 使得 $Q_1^T a_1 = \|a_1\|_2\cdot e_1$   
  我们记 $Q_1^T A = \begin{bmatrix}
  \|a_1\|_2 & v^T\\
  0_{m-1} & A_1 \end{bmatrix}$ 

  对 $A_1\in \mathbb R^{(m-1)\times (n-1)}$ 应用归纳假设可知 $A_1$ 具有 $\text{QR}$ 分解 $A_1 =Q_2\begin{bmatrix}
  R_2\\
  0_{(m-n)\times (n-1)}\end{bmatrix}$  
  其中 $Q_2\in \mathbb R^{(m-1)\times (m-1)}$ 是正交矩阵，$R_2\in \mathbb R^{(n-1)\times (n-1)}$ 是具有非负对角元的上三角阵.    
  则我们有:
  $$
  Q=Q_1 \begin{bmatrix}
  1 & \\
  & Q_2\end{bmatrix}
  \qquad 
  R=\begin{bmatrix} 
  \|a_1\|_2 & v^T\\
  0_{n-1} & R_2\\
  0_{m-n} & 0_{(m-n)\times (n-1)}\end{bmatrix}
  \qquad 
  A=Q\begin{bmatrix}
  R\\
  0_{(m-n)\times n}\end{bmatrix}
  $$
  因此 $Q,R$ 满足命题的要求.  
  根据数学归纳原理，存在性得证.

- **唯一性证明:**  
  当 $m=n$ 且 $A$ 非奇异时，假设 $A$ 具有 $\text{QR}$ 分解 $A=Q_1R_1= Q_2R_2$   
  其中 $Q_1,Q_2\in \mathbb R^{m\times m}$ 是正交矩阵，$R_1,R_2\in \mathbb R^{n\times n}$ 是具有非负对角元的上三角阵.   

  由于 $A$ 非奇异 (即满秩)，故 $R_1,R_2$ 的对角元均为正数 (即满秩)，因而可逆.  
  我们有 $Q_2^T Q_1 = R_2 R_1^{-1}$   
  这表明正交阵 $Q_2^T Q_1$ 等于一个对角元均为正数的上三角阵 $R_2R_1^{-1}$，  
  因而只能是单位阵，即 $Q_2^TQ_1 = I$  
  从而有 $\begin{cases}
  Q_1 = Q_2\\
  R_1 = R_2\end{cases}$，即 $A$ 具有唯一的 $\text{QR}$ 分解.

***

利用 $\text{QR}$ 分解，我们就可以实现**正交变换法**.  

给定 $\begin{cases}
A\in \mathbb R^{m\times n}\\
b\in \mathbb R^m\end{cases}$ 考虑线性最小二乘问题 $\underset{x\in \mathbb R^n}{\min} \|Ax-b\|_2^2$ (假设 $\rank(A) = n \leq m$)    
设 $A$ 的 $\text{QR}$ 分解是 $A=Q\begin{bmatrix}
R\\
0_{(m-n)\times n}\end{bmatrix}$，并分块为:
$$
Q= \begin{bmatrix} Q_1 & Q_2\end{bmatrix}
\qquad
c = \begin{bmatrix}
c_1\\
c_2 \end{bmatrix}
=
\begin{bmatrix} Q_1^Tb \\ Q_2^Tb\end{bmatrix}
=
Q^Tb
$$
 则我们有: 
$$
\begin{align}
\|Ax-b\|_2^2 
&= \|Q^T(Ax-b)\|_2^2\\
&= \|Q^TAx - Q^Tb\|_2^2\\
&= \|\begin{bmatrix}
R\\ 0_{m-n}
\end{bmatrix} x 
- 
\begin{bmatrix}
c_1\\
c_2
\end{bmatrix}\|_2^2\\
&= \|\begin{bmatrix}
Rx-c_1\\
c_2
\end{bmatrix}\|_2^2\\
&= \|Rx-c_1\|_2^2 + \|c_2\|_2^2\end{align}
$$
因此 $x$ 是 $\underset{x\in \mathbb R^n}{\min} \|Ax-b\|_2^2 = \underset{x\in \mathbb R^n}{\min} \|Q^T(Ax-b)\|_2^2$ 当且仅当 $x$ 是 $Rx=c_1$ 的解.   
综上所述，正交变换法的基本步骤为: 

- 计算 $A$​ 的 $\text{QR}$​ 分解 $A=Q\begin{bmatrix}
  R\\
  0_{(m-n)\times n}\end{bmatrix}$ 
- 计算 $c_1 = Q_1^Tb$
- 使用回代法求解上三角方程组 $Rx=c_1$ 

由此可知，实现正交变换法的关键是实现矩阵 $A$ 的 $\text{QR}$ 分解.



### 2.3.2 Householder 方法

下面我们使用 **Householder 方法**计算矩阵 $A$ 的 $\text{QR}$ 分解.  
它与 Gauss 消去法计算 $\text{LU}$ 分解 (利用 Gauss 变换逐步将 $A$ 转换为上三角阵 $U$) 类似，  
就是利用 Householder 变换逐步将 $A$ 转换为上三角阵 $R$ (具体来说是 $\begin{bmatrix}
R\\
0_{(m-n)\times n}\end{bmatrix}$)  
以 $\begin{cases}
m=6\\
n=5\end{cases}$ 的情况为例:
$$
A = 
\begin{bmatrix}
\times & \times & \times & \times  & \times \\
\times & \times & \times & \times  & \times \\
\times & \times & \times & \times  & \times \\
\times & \times & \times & \times  & \times \\
\times & \times & \times & \times  & \times \\
\times & \times & \times & \times  & \times
\end{bmatrix}\\

H_1 A = 
\begin{bmatrix}
\times & \times & \times & \times  & \times \\
& + & \times & \times &\times\\
& + & \times & \times &\times\\
& + & \times & \times &\times\\
& + & \times & \times &\times\\
& + & \times & \times &\times
\end{bmatrix}\\

H_2H_1 A = 
\begin{bmatrix}
\times & \times & \times & \times  & \times \\
&\times & \times &\times & \times \\
&& + & \times &\times\\
&& + & \times &\times\\
&& + & \times &\times\\
&& + & \times &\times
\end{bmatrix}\\

H_3H_2H_1 A = 
\begin{bmatrix}
\times & \times & \times & \times  & \times \\
&\times & \times &\times & \times \\
&&\times & \times &\times\\
&&& + &\times\\
&&& + &\times\\
&&& + &\times
\end{bmatrix}\\

H_4H_3H_2H_1 A = 
\begin{bmatrix}
\times & \times & \times & \times  & \times \\
&\times & \times &\times & \times \\
&&\times & \times &\times\\
&&& \times &\times\\
&&&& + \\
&&&& +
\end{bmatrix}\\

H_5H_4H_3H_2H_1 A = 
\begin{bmatrix}
\times & \times & \times & \times  & \times \\
&\times & \times &\times & \times \\
&&\times & \times &\times\\
&&& \times &\times\\
&&&& \times \\
&&&& 0
\end{bmatrix}\\
$$
给定矩阵 $A\in \mathbb R^{m\times n}\ (m\geq n)$  
假定我们已进行了 $k-1$ 步，得到 Householder 变换 $H_1,\dots, H_{k-1}$，使得:  
$$
A_k = H_{k-1}\dotsm H_1 A = 
\begin{bmatrix}
A_{11}^{(k)} & A_{12}^{(k)}\\
& A_{22}^{(k)}
\end{bmatrix}
$$
其中 $A_{11}^{(k)} \in \mathbb R^{(k-1)\times (k-1)}$ 是具有非负对角元的上三角阵，记 $A_{22}^{(k)} = [u_k,\dots, u_n]\in \mathbb R^{(m-k+1)\times (n-k+1)}$   

第 $k$ 步是: 

- 确定 Householder 变换 $\begin{cases}
  r_{kk} = \|u_k\|_2\\
  v_k = u_k - r_{kk} e_1\\
  \beta_k = \frac{2}{v_k^Tv_k}\\
  \tilde H_k = I_{m-k+1} - \beta_k v_k v_k^T \in \mathbb R^{(m-k+1)\times (m-k+1)}\end{cases}$   使得 $\tilde H_k u_k = r_{kk} e_1$   
  在实际计算中，我们还会将 $v_k$ 的第一个分量调整为 $1$ (以方便存储)，并相应地调整 $\beta_k$  

- 计算 $\tilde H_k A_{22}^{(k)} = (I_{m-k+1} - \beta_k v_k v_k^T) A_{22}^{(k)} = A_{22}^{(k)} - v_k \cdot (\beta_k (A_{22}^{(k)})^T v_k)^T = \begin{bmatrix}
  r_{kk} & w_k^T\\
  & A_{22}^{(k+1)}\end{bmatrix}$    
  令 $H_k = \begin{bmatrix}
  I_{k-1} & \\
  & \tilde H_k \end{bmatrix}$ 并记 $A_{12}^{(k)} = \begin{bmatrix}
  z_k & \tilde A_{12}^{(k)}\end{bmatrix}$ 则我们有:
  $$
  \begin{align}
  A_{k+1} 
  &= H_k A_k\\
  &= 
  \begin{bmatrix}
  I_{k-1} & \\
  & \tilde H_k \end{bmatrix}
  \begin{bmatrix}
  A_{11}^{(k)} & A_{12}^{(k)}\\
  & A_{22}^{(k)}
  \end{bmatrix}\\
  &=
  \begin{bmatrix}
  A_{11}^{(k)} & A_{12}^{(k)}\\
  & \tilde H_kA_{22}^{(k)}
  \end{bmatrix}\\
  &=
  \left[\begin{array}{c|cc}
  A_{11}^{(k)} & z_k & \tilde A_{12}^{(k)}\\
  \hline& r_{kk} & w_k^T \\
  & & A_{22}^{(k+1)}
  \end{array}\right]\\
  &=
  \left[\begin{array}{cc|c}
  A_{11}^{(k)} & z_k & \tilde A_{12}^{(k)}\\
  & r_{kk} & w_k^T \\
  \hline
  & & A_{22}^{(k+1)}
  \end{array}\right]\\
  &=
  \begin{bmatrix}
  A_{11}^{(k+1)} & A_{12}^{(k+1)}\\
  & A_{22}^{(k+1)}
  \end{bmatrix}
  \end{align}
  $$
  其中我们记 $\begin{cases}
  A_{11}^{(k+1)} = \begin{bmatrix}
  A_{11}^{(k)} & z_k\\
  & r_{kk}\end{bmatrix}\in \mathbb R^{k\times k}\\
  A_{12}^{(k+1)} = \begin{bmatrix}
  \tilde A_{12}^{(k)}\\
  w_k^T\end{bmatrix}\in \mathbb R^{k\times (n-k)}\end{cases}$  

这样，从 $k=1$ 开始，对 $A$ 依次进行 $n$ 次 Householder 变换，  
我们就可将 $A$ 转化为上三角阵 $A^{(n)} = \begin{bmatrix} A_{11}^{(n)}\\ 0_{(m-n)\times n}\end{bmatrix} = H_n \dotsm H_1 A$.  
现在记 $\begin{cases}
R = A_{11}^{(n)}\\
Q = H_1\dotsm H_n\end{cases}$ 即有 $A= Q\begin{bmatrix}
R\\ 0_{(m-n)\times n}\end{bmatrix}$ (显然 $R$ 的对角元均非负)

***

下面考虑使用 Householder 方法计算 $A$ 的 $\text{QR}$ 分解的存储问题.  
当分解完成后，$A$ 通常不再被需要，可用它来存储 $Q$ 和 $R$.  

此外，我们通常不用将 $Q$ 显式存储，  
而只需存放构成它的 $n$ 个 $m$ 阶 Householder 变换 $H_k\ (k=1,\dots,n)$ 的 $v_k$ 和 $\beta_k$​ 即可.  

在实际计算中，我们通常将 $v_k$ 的第 $k$ 个分量调整为 $1$ (以方便存储)，并相应地调整 $\beta_k$    
于是 $v_k$ 形如 $v_k = (0,\dots,0,1,v_{k+1}^{(k)},\dots, v_n^{(k)})^T$  
因此我们可将 $v_k$ 的第 $k+1$ 个分量至第 $m$ 个分量存储在 $A$ 的第 $k$ 个对角元以下的位置.

以 $\begin{cases}
m=4\\
n=3\end{cases}$ 的情况为例:  
$$
H_3 H_2 H_1 A = \begin{bmatrix}
R\\
0_{1\times 3}\end{bmatrix} 
= 
\left[\begin{array}{}
r_{11} & r_{12} & r_{13}\\
& r_{22} & r_{23}\\
&&r_{33}\\
\hline
0 & 0 & 0\end{array}\right]\\

H_1 = I_4 - \beta_1 v_1 v_1^T\qquad v_1 = (1,v^{(1)}_2,v_3^{(1)},v_4^{(1)})^T\\
H_2 = I_4 - \beta_2 v_2 v_2^T\qquad v_2 = (0,1,v_3^{(2)},v_4^{(2)})^T\\
H_3 = I_4 - \beta_3 v_3 v_3^T\qquad v_3 = (0,0,1,v_4^{(3)})^T\\

\text{Storage:} \begin{cases}
d := (\beta_1,\beta_2,\beta_3) \\
A:= \begin{bmatrix}
r_{11} & r_{12} & r_{13}\\
v_2^{(1)} & r_{22} & r_{23}\\
v_3^{(1)} & v_3^{(2)} & r_{33}\\
v_4^{(1)} & v_4^{(2)} & v_4^{(3)}
\end{bmatrix}\end{cases}
$$

****

综合上述讨论，可得如下算法:  
**(Householder 方法计算 $\text{QR}$ 分解, 数值线性代数, 算法 $3.3.1$)**  
$$
\begin{align}
&\text{function: }[Q,R] = \text{Householder\_QR}(A)\\
&\qquad \\
&\text{for }j=1:n\\
&\qquad \text{if }\ k<m\\
&\qquad\qquad [v,\beta] = \text{Householder}(A(k:m,k))\\
&\qquad\qquad A(k:m,k:n) = (I_{m-k+1} - \beta v v^T) A(k:m,k:n) = A(k:m,k:n) - \beta v(A(k:m,k:n) ^Tv)^T\\
&\qquad\qquad d(k) = \beta\\
&\qquad\qquad A(k+1:m,k) = v(2:m-k+1)\\
&\qquad \text{end}\\
&\text{end}
\end{align}
$$
该算法是数值稳定的.  
总浮点运算量为 $2n^2m -\frac{1}{3} n^3$.    
考虑求解最小二乘问题 $\underset{x\in \mathbb R^n}{\min} \|Ax-b\|_2^2$ (假设 $\rank(A)=n\leq m$)，  
与正则化方法相比，利用 Householder 方法计算 $\text{QR}$ 分解的正交变换法代价更大，但得到的计算解更加精确.  

****

Householder 方法并不是实现 $\text{QR}$ 分解的唯一方法.  
我们也可以利用 Givens 变换或 Gram-Schmidt 正交化来实现.  

- 一般来说，利用 Givens 变换来实现 $A$ 的 $\text{QR}$ 分解所需的运算量大约是 Householder 方法的二倍.  
  但如果 $A$ 有较多的零元素 (即较为稀疏)，则灵活地使用 Givens 变换往往会使运算量大为减少.
- Gram-Schmidt 正交化来实现 $A$ 的 $\text{QR}$ 分解数值不稳定.  
  在本课程的后续内容中，我们将对 Gram-Schmidt 正交化进行改进，以提高数值稳定性.

此外，$\text{QR}$ 分解不仅可用来求解最小二乘问题，  
它也是数值线性代数诸多重要算法的基础,，例如的求解特征值问题的 $\text{QR}$ 方法.    

我们亦可利用 $\text{QR}$ 分解求解线性方程组 $Ax=b$ (其中 $A\in \mathbb R^{n\times n}$):

- 计算 $A$ 的 $\text{QR}$ 分解 $A=QR$，于是 $Ax=b\ \Leftrightarrow\ QRx=b\ \Leftrightarrow\ Rx=Q^Tb$
- 使用回代法求解上三角方程组 $Rx=Q^Tb$ 得到 $x$

对于某些病态的线性方程组，使用 $\text{QR}$ 分解的计算结果往往比 $\text{LU}$ 分解的要好得多 (当然计算量也大得多)



### 2.3.3 Givens 方法

















### 2.3.4 Gram-Schmidt 正交化方法











