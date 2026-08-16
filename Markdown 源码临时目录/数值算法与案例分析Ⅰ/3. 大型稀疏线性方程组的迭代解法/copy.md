# FDU 数值算法 3. 大型稀疏线性方程组的迭代解法 (一)

本文参考以下教材:

- 数值线性代数 (第二版) 徐树方, 高立, 张平文 第 $4$ 章
- Applied Numerical Linear Algebra (J. Demmel) Chapter $6$
- 应用数值线性代数 (J. Demmel) 第 $6$ 章

欢迎批评指正!

## 3.0 Introduction

求解线性方程组 $Ax=b$ 的直接方法大多数都需要对系数矩阵 $A$ 进行分解，  
因而一般不能保持 $A$ 的稀疏性.  

> 考虑以下例子:  
> $$
> \begin{bmatrix}
> * & * & * & *\\
> * & * \\
> * & & * \\
> * & & & *
> \end{bmatrix}
> $$
> 一步 Gauss 消元，这种类型的稀疏矩阵就变成稠密矩阵了.  
> 不过也不是不能解决，我们可以将其行列重排为:  
> $$
> \begin{bmatrix}
> * & & & * \\
> & * & & *\\
> & & * & * \\
> * & * &*&*
> \end{bmatrix}
> $$
> 这样 Gauss 消元就能保持稀疏性了.

但在实际应用中，我们常常会遇到大型稀疏线性方程组的求解问题.  
目前发展起来的求解稀疏线性方程组的方法主要有两类:

- 一类是稀疏直接法.  
  它是直接法与某些稀疏矩阵技巧结合的产物，  
  充分利用了给定线性方程组的系数矩阵的零元素的分布特点 (即矩阵的结构)，   
  采用灵活的主元选取策略，使得分解出的矩阵因子尽可能地保持原有的稀疏性.  
  本课程不涉及这类方法.
- 另一类是迭代法，即按照某种规则构造向量序列 $\{x^{(k)}\}$，使其极限 $x^\star$ 是方程组 $Ax=b$ 的精确解.  
  我们将主要关注以下问题:
  - 如何构造迭代序列 $\{x^{(k)}\}$?
  - 构造的迭代序列是否收敛? 在什么情况下收敛?
  - 若迭代序列收敛，则收敛的速度如何?
  - 由于实际计算总是有限次的，故我们要讨论近似解的误差估计和迭代过程的中断处理等问题.

### 3.0.1 存储

维度超过百万级别的矩阵就称为大型矩阵.  
稀疏矩阵一般按坐标方式 (COO format) 存储: $(i,j,\text{value})$ (并且按列存储，因为列向量比行向量更常用)  
以如下矩阵为例:  
$$
\begin{align}
A&=\begin{bmatrix}
* \\
& * & * \\
& &  *\\
*& & & *
\end{bmatrix}
\end{align}
$$
其存储如下:  

| $(m,n,\text{nnz}) = (4,4,6)$ |
| :--------------------------: |
| $(i,j,\text{value})=(0,0,*)$ |
| $(i,j,\text{value})=(3,0,*)$ |
| $(i,j,\text{value})=(1,1,*)$ |
| $(i,j,\text{value})=(1,2,*)$ |
| $(i,j,\text{value})=(2,2,*)$ |
| $(i,j,\text{value})=(3,3,*)$ |

其中 $\text{nnz}$ 代表 number of non-zero elements，即非零元素的个数.  
但实际应用中是可以存储零元素的，因为要完全不存储零元素也是有代价的: 我们需要将零元素识别出来.  
存储复杂度为 $O(\text{nnz}+1)$   

从上述存储格式衍生出 CSC (Compressed Sparse Column) 格式，它包含以下三个数组:

1. **`values[]`**: 长度为 $\text{nnz}$，存储矩阵中的所有非零元素
2. **`row_indices[]`**: 长度为 $\text{nnz}$，存储与 `values[]` 中对应的每个非零元素所在的行索引
3. **`col_ptr[]`**: 长度为 $n+1$，存储每列的起始位置.  
   最后一个元素指向 `values[]` 中最后一个非零元素的后一个位置.  
   (这有助于通过两个指针值相减得到对应列的非零元素个数)

若稀疏矩阵 $A$ 按 CSC 格式存储，则我们很容易开发出 $(A,x)\mapsto Ax$ 和 $(A,x)\mapsto A^Tx$ 的程序.   
但其效率很糟糕，因为间接寻址是低效的.

*****

其中 $(A,x)\mapsto Ax$ 操作是最重要的.  
根据 Cayley-Hamilton 定理可知 $A^{-1}$ 可以表示为 $A$ 的 $n-1$ 次多项式 $p(A) = \sum_{i=0}^{n-1}c_iA^i$  
因此 $x=A^{-1}b$ 可以写成 $x=p(A)b = \sum_{i=0}^{n-1}c_iA^{i}b$ 

> 任意复方阵都满足其特征方程.  
> **(Cayley-Hamilton 定理, Matrix Analysis 定理 $2.4.3.2$)**    
> 设 $p_A(t):=\det(tI_n- A)$ 是 $A\in \mathbb C^{n\times n}$ 的特征多项式，则我们有 $p_A(A) = 0_{n\times n}$ 成立.
>
> - **(Matrix Analysis 推论 $2.4.3.4$)**  
>   设 $A\in \mathbb C^{n\times n}$ 非奇异  
>   记其特征多项式 $p_A(t) = \det(tI-A) = t^n + c_{n-1}t^{n-1} + \dotsm + c_1 t + c_0$    
>   根据 Cayley-Hamilton 定理我们有 $p_A(A) = A^n + c_{n-1}A^{n-1} + \dotsm + c_1 A + c_0 I_n = 0_{n\times n}$     
>   则 $A^{-1} = -\frac{1}{c_0}(A^{n-1} + c_{n-1} A^{n-2} + \dotsm + c_2 A + c_1)$

考虑如下稀疏矩阵:   
$$
A =  \begin{bmatrix}
0 & 1 & & &\\
1 & 0 & 1 & &\\
& \ddots & \ddots & \ddots & \\
& & 1 & 0 & 1\\
& & & 1 & 0
\end{bmatrix}_{n\times n}
$$
根据高等线性代数 Homework 07 Problem 04 的结果可知其特征值和单位特征向量为:  
$$
\lambda_k = 2\cos(\frac{k\pi}{n+1})\quad x^{(k)}=
\sqrt{\frac{2}{n+1}}\begin{bmatrix}
\sin(\frac{k\pi}{n+1})\\
\sin(\frac{2k\pi}{n+1})\\
\vdots\\
\sin(\frac{(n-1)k\pi}{n+1})\\
\sin(\frac{nk\pi}{n+1})\\
\end{bmatrix}\quad (k=1,\dots,n)
$$
显然 $A$ 的谱分解 $A=U\Lambda U^H$ 中特征矩阵 $U\in \mathbb C^{n\times n}$ 是稠密矩阵.  
因此直接对 $A$ 求谱分解是不可行的.  
但我们可以进行幂法迭代，收敛出特征值 **(存疑: 那特征向量呢?)**

****

常用的稀疏矩阵资源网站:

- [Matrix Market](https://math.nist.gov/MatrixMarket/)
- [SuiteSparse Matrix Collection (by Tim Devis)](https://sparse.tamu.edu/)



### 3.0.2 一维 Poisson 方程

考虑一维 Poisson 方程:  
$$
-\frac{d^2}{dx^2}u(x) = f(x)\quad (0<x<1)
$$
其中初值为 $u(0)=u(1)=0$ (即 Dirichlet 边界条件)   
我们使用有限差分去逼近上述方程:  
$$
\begin{align}
h &:= \frac{1}{n+1}\\
\frac{d}{dx}u(x){\Large|}_{x=(i-0.5)h} &\approx \frac{u_i-u_{i-1}}{h}\quad (i=1,\dots,n)\\
\frac{d}{dx}u(x){\Large|}_{x=(i+0.5)h} &\approx \frac{u_{i+1}-u_{i}}{h}\quad (i=1,\dots,n)\\
\end{align}
$$
将上述近似式相减并除以 $h$ 便得到中心差分近似:  
$$
-\frac{d^2}{dx^2}u(x){\Large |}_{x=x_i} = \frac{2u_i - u_{i-1} - u_{i+1}}{h^2} + O\left(h^2\cdot \left\|\frac{d^4}{dx^4}u(x) \right\|_\infty\right)\quad (i=1,\dots,n)
$$
为求解上述方程，我们丢弃截断误差，将其改写为:   
$$
-u_{i-1} + 2u_{i} - u_{i+1} = h^2 f_i\quad (i=1,\dots,n)\\
\Leftrightarrow\\

T_{n} u 
=
\begin{bmatrix}
2 & -1 & & \\
-1 & \ddots & \ddots & \\
& \ddots & \ddots & -1\\
& & -1 & 2
\end{bmatrix}
\begin{bmatrix}
u_1\\
u_2\\
\vdots\\
u_n
\end{bmatrix}
=
h^2 \begin{bmatrix}
f_1\\
f_2\\
\vdots\\
f_n
\end{bmatrix}
=
h^2 f
$$
**(应用数值线性代数, 引理 $6.1$)**  
对称正定三对角阵 $T_n$ 的特征值和单位特征向量为:  
$$
\lambda_k = 2\left(1-\cos(\frac{k}{n+1}\pi)\right)
\quad
x^{(k)} = \sqrt{\frac{2}{n+1}}
\begin{bmatrix}
\sin(\frac{k\pi}{n+1})\\
\sin(\frac{2k\pi}{n+1})\\
\vdots\\
\sin(\frac{(n-1)k\pi}{n+1})\\
\sin(\frac{nk\pi}{n+1})\\
\end{bmatrix}\quad (k=1,\dots,n)
$$
以 $n=21$ 的情况为例:

<img src="应用数值线性代数 图 6.1-6.2.png" style="zoom:40%;" />

**(应用数值线性代数, 习题 $6.2$)**    
$T_n$ 的 Cholesky 分解和 $\text{LU}$ 分解具有简单形式:

- $T_n$ 的 Cholesky 分解 $T_n:= L_n L_n^T$ 中的 $L_n$ 是下双对角阵:  
  $$
  \begin{align}
  L_n(i,i) &= \sqrt{\frac{i+1}{i}}\quad (i=1,\dots,n)\\
  L_n(i+1,i) &= \sqrt{\frac{i}{i+1}}\quad (i=1,\dots,n-1)
  \end{align}
  $$

- $T_n$ 的 Gauss 消去法的结果 $T_n := L_n U_n$ 中的 $L_n$ 是单位下双对角阵，而 $U_n$ 是上双对角阵:  
  $$
  \begin{align}
  L_n(i,i) &= 1\qquad\quad\ \ \  (i=1,\dots,n)\\
  L_n(i+1,i) &= -\frac{i}{i+1}\quad (i=1,\dots,n-1)\\
  \hline
  U_n(i,i) &= \frac{i+1}{i}\quad\ \ \ (i=1,\dots,n)\\
  U_n(i,i+1) &= -1 \qquad\quad (i=1,\dots,n-1) 
  \end{align}
  $$

 


### 3.0.3 二维 Poisson 方程

现在转向二维 Poisson 方程:  
$$
-\frac{\partial^2}{\partial x^2}u(x,y) - \frac{\partial^2}{\partial x^2}u(x,y)= f(x,y)\quad (0<x,y<1)
$$
其中初值为 $\begin{cases}
u(x,0) = u(x,1)= 0 & (\forall\ 0\leq x\leq 1)\\
u(0,y) = u(1,y)= 0 & (\forall\ 0\leq y\leq 1)\end{cases}$ (即 Dirichlet 边界条件)    

结合一维 Poisson 方程的结论，我们可以使用有限差分去逼近上述方程:  
$$
\begin{align}
h &:= \frac{1}{n+1}\\
-\frac{\partial^2}{\partial x^2}u(x,y){\Large|}_{x=x_i,y=y_j} &\approx \frac{2u_{ij}-u_{i-1,j} - u_{i+1,j}}{h^2}\quad (i,j=1,\dots,n)\\
-\frac{\partial^2}{\partial y^2}u(x,y){\Large|}_{x=x_i,y=y_j} &\approx \frac{2u_{ij}-u_{i,j-1} - u_{i,j+1}}{h^2}\quad (i,j=1,\dots,n)\\
\end{align}
$$
将上述近似相加得到:  
$$
-\left\{
\frac{\partial^2}{\partial x^2}u(x,y) + 
\frac{\partial^2}{\partial y^2}u(x,y)
\right\}{\Large|}_{x=x_i,y=y_j} \approx \frac{4u_{ij} - u_{i-1,j}-u_{i+1,j} - u_{i,j-1}-u_{i,j+1}}{h^2}
$$
其中截断误差仍然是 $O(h^2)$ 级别的.  
于是我们得到一组具有 $n$ 个未知量 $u_{ij}\ (1\leq i,j\leq n)$ 的 $n^2$ 维线性方程组:
$$
4u_{ij} - u_{i-1,j} - u_{i+1,j} - u_{i,j-1}- u_{i,j+1} = h^2 f_{ij}\quad (1\leq i,j\leq n)
$$
其对应的十字称为 Poisson 方程的**五点型板** (stencil)，如图所示:

<img src="应用数值线性代数 图 6.3.png" style="zoom:40%;" />

上述方程组具有如下紧凑形式:
$$
4u_{ij} - u_{i-1,j} - u_{i+1,j} - u_{i,j-1}- u_{i,j+1} = h^2 f_{ij}\quad (1\leq i,j\leq n)\\
\Leftrightarrow\\
T_n U + UT_n  = h^2 F\\
T_{n} 
=
\begin{bmatrix}
2 & -1 & & \\
-1 & \ddots & \ddots & \\
& \ddots & \ddots & -1\\
& & -1 & 2
\end{bmatrix} 
\qquad

U = \begin{bmatrix}
u_{1,1} & u_{1,2} &\dotsm  & u_{1,n}\\
u_{2,1} & u_{2,2} & \dotsm & u_{2,n}\\
\vdots & \vdots & & \vdots\\
u_{n,1} & u_{n,2} & \dotsm & u_{n,n}
\end{bmatrix}
\qquad

F = \begin{bmatrix}
f_{1,1} & f_{1,2} &\dotsm  & f_{1,n}\\
f_{2,1} & f_{2,2} & \dotsm & f_{2,n}\\
\vdots & \vdots & & \vdots\\
f_{n,1} & f_{n,2} & \dotsm & f_{n,n}
\end{bmatrix}\\
\Leftrightarrow\\
T_{n\times n} \text{vec}(U) = h^2 \text{vec}(F)\\

\text{where }T_{n\times n}:=(I_n \otimes T_n + T_n\otimes I_n) = 
\begin{bmatrix}
T_n\\
& \ddots\\
& & \ddots\\
& & & T_n
\end{bmatrix}
+
\begin{bmatrix}
2I_n & -I_n & & \\
-I_n & \ddots & \ddots & \\
& \ddots & \ddots & -I_n\\
& & -I_n & 2I_n
\end{bmatrix}
$$
**(Gauss-Seidel 迭代红黑序参见应用数值线性代数算法 $6.4$)**  
这类问题是 Gauss-Seidel 迭代更快，其每步迭代的效果相当于 Jacobi 迭代的两步.  
对于大多数经典问题来说，都是 Gauss-Seidel 迭代更快.



## 3.1 单步线性定常迭代法

在众多迭代法中，最简单最基本的就是单步线性定常迭代法.   
给定 $A\in \mathbb C^{n\times n}$，我们可将其拆分为:
$$
A = M-N
$$
其中 $M$ 非奇异 (我们希望 $M\approx A$ 且容易求逆)  

> 例如使用 Toepliz 分解:  
> $$
> A := H + K = \frac{A+A^H}{2} + \frac{A-A^H}{2}
> $$
> 我们记:  
> $$
> M := H + \alpha I\\
> N := -K + \alpha I\\
> A := M-N\\
> $$
> 其中 $\alpha>0$ 为一个常数，以保证 $M=H+\alpha I$ 可逆 (但它可能不容易求逆)

于是我们有:  
$$
Ax = (M-N)x = b\\
\Leftrightarrow\\
Mx = b+Nx\\
\Leftrightarrow\\
x = M^{-1}Nx + M^{-1}b
$$
我们可以据此写出单步线性定常迭代法的通用格式:
$$
x^{(k+1)} = M^{-1}N x^{(k)} + M^{-1}b\\
\Leftrightarrow\\
x^{(k+1)} = Bx^{(k)} + c\ (\text{where}\begin{cases}
B=M^{-1}N\\
c = M^{-1}b\end{cases})
$$
我们首先从最简单的 Jacobi 迭代法讲起.



### 3.1.1 Jacobi 迭代法

给定 $\begin{cases}
A\in \mathbb R^{n\times n}\\
b\in \mathbb R^{n}\end{cases}$ 考虑非奇异线性方程组 $Ax=b$   
我们将非奇异矩阵 $A = [a_{ij}]$ 分解为:
$$
A = D-L-U\\
D=
\begin{bmatrix}
a_{11} & & & \\
& a_{22} & & \\
&&\ddots &\\
&&& a_{nn}\end{bmatrix}
\qquad
L= 
\begin{bmatrix}
0 & & &\\
-a_{21} & 0 &&\\
\vdots & \ddots & \ddots &\\
-a_{n1} & \dotsm & -a_{n,n-1} & 0
\end{bmatrix}
\qquad
U= 
\begin{bmatrix}
0 & -a_{12} & \dotsm & -a_{1n}\\
& 0 &\ddots& \vdots\\
&&\ddots &-a_{n-1,n}\\
&&& 0 
\end{bmatrix}
$$
假设 $D$ 非奇异，即 $A$ 的对角元均非零，则 $Ax=b$ 可以改写为:
$$
Ax = (D-L-U)x = b\\
\Leftrightarrow\\
x = D^{-1}(L+U)x + D^{-1}b\\
\Leftrightarrow\\
x=Bx + c\ \ \text{where } 
\begin{cases}
B=D^{-1}(L+U)\\
c=D^{-1}b
\end{cases}
$$
给定初始向量 $x^{(0)} = (x_1^{(0)},\dots,x_n^{(0)})^T$   
我们将 $x^{(0)}$ 带入 $x=Bx+c$ 的右侧，得到 $x^{(1)}= Bx^{(0)} + c$   
再将 $x^{(1)}$ 带入 $x=Bx+c$ 的右侧，又可得到 $x^{(2)}= Bx^{(1)} + c$    
以此类推，我们有 $x^{(k)}=Bx^{(k-1)} + c\ \ (k=1,2,\dots)$   
这就是 **Jacobi 迭代法**，其中 $B$ 为迭代矩阵，$c$ 为常数项.  

Jacobi 迭代法中，$x^{(k)}$ 的所有分量都由 $x^{(k-1)}$ 确定，因此先计算哪个分量都一样.  



### 3.1.2 Gauss-Seidel 迭代法

现在假设不按 Jacobi 迭代格式.  
记    
$$
B=\begin{bmatrix}
b_1^T\\
\vdots\\
b_n^T\end{bmatrix} = D^{-1}(L+U) 
= \begin{bmatrix}
a_{11}^{-1} & &\\
&\ddots &\\
&& a_{nn}^{-1}\end{bmatrix}
\left(\begin{bmatrix}
l_1^T\\
\vdots\\
l_n^T\end{bmatrix} + \begin{bmatrix}
u_1^T\\
\vdots\\
u_n^T\end{bmatrix}\right)
$$
于是 $b_k^T = a_{kk}^{-1} (l_k^T + u_k^T)\ \ (k=1,\dots,n)$     
回顾之前的记号:
$$
A = D-L-U\\
D=
\begin{bmatrix}
a_{11} & & & \\
& a_{22} & & \\
&&\ddots &\\
&&& a_{nn}\end{bmatrix}
\qquad
L=
\begin{bmatrix}
l_1^T\\
l_2^T\\
\vdots\\
l_n^T\end{bmatrix}
= 
\begin{bmatrix}
0 & & &\\
-a_{21} & 0 &&\\
\vdots & \ddots & \ddots &\\
-a_{n1} & \dotsm & -a_{n,n-1} & 0
\end{bmatrix}
\qquad
U=
\begin{bmatrix}
u_1^T\\
u_2^T\\
\vdots\\
u_n^T\end{bmatrix}
=
\begin{bmatrix}
0 & -a_{12} & \dotsm & -a_{1n}\\
& 0 &\ddots& \vdots\\
&&\ddots &-a_{n-1,n}\\
&&& 0 
\end{bmatrix}
$$

- 计算第 $1$ 个分量:   
  $$
  \begin{align}
  x_1^{(k)} 
  &= b_1^T x^{(k-1)} + c_1 \\
  &= a_{11}^{-1} (l_1^T + u_1^T) x^{(k-1)} + c_1\\
  &= a_{11}^{-1} u_1^T x^{(k-1)} + c_1\quad (\text{note that }l_1=0_n)\\
  &= a_{11}^{-1} (l_1^T x^{(k)} + u_1^T x^{(k-1)}) + c_1\end{align}
  $$

- 在计算第 $2$​ 个分量 $x_2^{(k)}$​ 时，第 $1$​ 个分量 $x_1^{(k)}$​ 已经算出，于是我们可用 $x_1^{(k)}$​ 代替 $x_1^{(k-1)}$​ 
  $$
  \begin{align}
  x_2^{(k)} 
  &= 
  b_2^T \begin{bmatrix}
  x_1^{(k)}\\
  x_2^{(k-1)}\\
  \vdots\\
  x_n^{(k-1)}\end{bmatrix} + c_2 \\
  &=  
  a_{22}^{-1}
  (l_2^T + u_2^T)
  \begin{bmatrix}
  x_1^{(k)}\\
  x_2^{(k-1)}\\
  \vdots\\
  x_n^{(k-1)}\end{bmatrix} + c_2\\
  &=
  a_{22}^{-1} ([-a_{21},0,0,\dots,0] + [0,0,-a_{23},\dots,-a_{2n}]) \begin{bmatrix}
  x_1^{(k)}\\
  x_2^{(k-1)}\\
  \vdots\\
  x_n^{(k-1)}\end{bmatrix} + c_2\\
  &=
  a_{22}^{-1}\left([-a_{21},0,0,\dots,0] \begin{bmatrix}
  x_1^{(k)}\\
  x_2^{(k)}\\
  \vdots\\
  x_n^{(k)}\end{bmatrix} + [0,0,-a_{23},\dots,-a_{2n}]
  \begin{bmatrix}
  x_1^{(k-1)}\\
  x_2^{(k-1)}\\
  \vdots\\
  x_n^{(k-1)}\end{bmatrix}\right) + c_2\\
  &=
  a_{22}^{-1}(l_2^T x^{(k)} + u_2^T x^{(k-1)}) + c_2
  \end{align}
  $$

- 以此类推，在计算第 $m$ 个分量 $x_l^{(k)}$ 时，第 $1$ 个分量至第 $m-1$ 个分量 $x_1^{(k)},\dots,x_{m-1}^{(k)}$ 已经算出，  
  于是我们可用 $x_1^{(k)},\dots,x_{m-1}^{(k)}$ 代替 $x_1^{(k-1)},\dots,x_{m-1}^{(k-1)}$    
  $$
  \begin{align}
  x_m^{(k)} 
  &= 
  b_m^T \begin{bmatrix}
  x_1^{(k)}\\
  \vdots\\
  x_{m-1}^{(k)}\\
  x_m^{(k-1)}\\
  \vdots\\
  x_n^{(k-1)}\end{bmatrix} + c_m \\
  &=  
  a_{mm}^{-1}
  (l_m^T + u_m^T)
  \begin{bmatrix}
  x_1^{(k)}\\
  \vdots\\
  x_{m-1}^{(k)}\\
  x_m^{(k-1)}\\
  \vdots\\
  x_n^{(k-1)}\end{bmatrix} + c_m\\
  &=
  a_{mm}^{-1} ([-a_{m1},\dots,-a_{m,m-1},0,0,\dots,0] + [0,\dots,0,0,-a_{m,m+1},\dots,-a_{m,n}]) \begin{bmatrix}
  x_1^{(k)}\\
  \vdots\\
  x_{m-1}^{(k)}\\
  x_m^{(k-1)}\\
  \vdots\\
  x_n^{(k-1)}\end{bmatrix} + c_m\\
  &=
  a_{mm}^{-1}\left([-a_{m1},\dots,-a_{m,m-1},0,0,\dots,0]
  \begin{bmatrix}
  x_1^{(k)}\\
  \vdots\\
  x_{m-1}^{(k)}\\
  x_m^{(k)}\\
  \vdots\\
  x_n^{(k)}\end{bmatrix} 
  + [0,\dots,0,0,-a_{m,m+1},\dots,-a_{m,n}]
  \begin{bmatrix}
  x_1^{(k-1)}\\
  \vdots\\
  x_{m-1}^{(k-1)}\\
  x_m^{(k-1)}\\
  \vdots\\
  x_n^{(k-1)}\end{bmatrix}\right) + c_m\\
  &=
  a_{mm}^{-1}(l_m^T x^{(k)} + u_m^T x^{(k-1)}) + c_m
  \end{align}
  $$

综上所述，我们得到:  
$$
{\begin{cases}
x_1^{(k)} = a_{11}^{-1}(l_1^T x^{(k)} + u_1^T x^{(k-1)}) + c_1\\
\qquad\vdots\\
x_n^{(k)} = a_{nn}^{-1}(l_n^T x^{(k)} + u_n^T x^{(k-1)}) + c_n\\
\end{cases}}\\
\Leftrightarrow\\
x^{(k)} = D^{-1}(Lx^{(k)} + Ux^{(k-1)}) + c 
= D^{-1}(Lx^{(k)} + Ux^{(k-1)}) + D^{-1}b
$$
由于我们假设 $D$ 非奇异 (对角元均非零)，故 $D-L$ 也是非奇异的.  
因此上式可改写为 $x^{(k)} = (D-L)^{-1} U x^{(k-1)} + (D-L)^{-1} b\ \ (k=1,2,\dots)$     
这就是 **Gauss-Seidel 迭代法**.  
我们称 $(D-L)^{-1}U$ 为 $\text{G-S}$ 迭代法的迭代矩阵，而 $(D-L)^{-1}b$ 为常数项.



### 3.1.3 单步线性定常迭代法

考虑非奇异线性方程组 $Ax=b$，假设 $A$ 的对角元均非零.  
回顾之前的记号:
$$
A = D-L-U\\
D=
\begin{bmatrix}
a_{11} & & & \\
& a_{22} & & \\
&&\ddots &\\
&&& a_{nn}\end{bmatrix}
\qquad
L
= 
\begin{bmatrix}
0 & & &\\
-a_{21} & 0 &&\\
\vdots & \ddots & \ddots &\\
-a_{n1} & \dotsm & -a_{n,n-1} & 0
\end{bmatrix}
\qquad
U
=
\begin{bmatrix}
0 & -a_{12} & \dotsm & -a_{1n}\\
& 0 &\ddots& \vdots\\
&&\ddots &-a_{n-1,n}\\
&&& 0 
\end{bmatrix}
$$
对比 Jacobi 迭代格式和 Gauss-Seidel 迭代格式 $\begin{cases}
x^{(k)} = D^{-1}(L+U)x^{(k-1)} + D^{-1}b\\
x^{(k)} = (D-L)^{-1} U x^{(k-1)} + (D-L)^{-1} b \end{cases}$   
我们发现它们有一个共同点:   
**新近似解 $x^{(k)}$ 是已知近似解 $x^{(k-1)}$ 的线性函数，且仅与 $x^{(k-1)}$ 有关，与其他已知近似解无关.**   
即它们可以抽象成 $x^{(k)} = Bx^{(k-1)} + c$ 的形式，  
对于 Jacobi 迭代法有 $\begin{cases}
B=D^{-1}(L+U)\\
c=D^{-1}b
\end{cases}$，对于 Gauss-Seidel 迭代法有 $\begin{cases}
B=(D-L)^{-1}U\\
c=(D-L)^{-1}b
\end{cases}$ 

我们称形如 $x^{(k)} = Bx^{(k-1)} + c$ 的迭代法为**单步线性定常迭代法**.  
若对于任意初始向量 $x^{(0)}$，该迭代格式产生的向量序列 $\{x^{(k)}\}$ 都有极限，则称该迭代法收敛;  
否则我们称它是发散的.

- 若迭代法收敛，并记其极限为 $x^\star$，  
  则对 $x^{(k)} = Bx^{(k-1)} + c$ 两侧同时取极限，即得到 $x^\star = B x^{\star} + c$   
  表明 $x^\star$ 恰为方程组 $x=Bx+c$ 的解.  
  对于充分大的 $k$，$x^{(k)}$ 便可作为 $x=Bx+c$ 的近似解.
- 进一步，若方程组 $x=Bx+c$ 和原方程组 $Ax=b$ 等价，  
  即存在非奇异矩阵 $G\in \mathbb R^{n\times n}$ 使得 $\begin{cases}
  A=G(I_n-B)\\
  b= Gc\end{cases}$ (此时称迭代法与 $Ax=b$ **相容**)，  
  则迭代法的极限 $x^\star$ 也是非奇异线性方程组 $Ax=b$ 的唯一解.  
  此时，对于充分大的 $k$，$x^{(k)}$ 便可作为 $Ax=b$ 的近似解.

从实用角度来讲，我们只对相容的收敛迭代法感兴趣.  
显然对于任意一个满足 "$A$ 对角元均非零" 的非奇异线性方程组 $Ax=b$，  
Jacobi 迭代法和 Gauss-Seidel 迭代法都与之相容.  
但它们的收敛条件是什么呢?



## 3.2 收敛性理论

### 3.2.1 收敛的充要条件

设 $x^\star$ 是非奇异线性方程组 $Ax=b$ 的精确解.  
考虑与 $Ax=b$ 相容的单步线性定常迭代法 $x^{(k)} = Bx^{(k-1)} + c$ 产生的迭代序列 $\{x^{(k)}\}$  
若该迭代法收敛，则其极限 $x^\star$ 满足 $x^\star = Bx^\star + c$    

定义 $x^{(k)}$ 误差向量为 $e^{(k)} = x^{(k)}-x^\star$   
联立 $\begin{cases}
x^{(k)} = Bx^{(k-1)} + c\\
x^\star = B x^\star + c\\
e^{(k)} = x^{(k)}-x^\star\\
e^{(k-1)} = x^{(k-1)}-x^\star\end{cases}$ 可得 $e^{(k)} = B\cdot e^{(k-1)}\ \ (k=1,2,\dots)$   
于是我们有 $e^{(k)} = B^k e^{(0)}\ \ (k=1,2,\dots)$   
因此 $\underset{k\to\infty}{\lim} x^{(k)} = x^\star$ (即 $\underset{k\to\infty}{\lim} e^{(k)} = 0_n$) 的充要条件是 $\underset{k\to\infty}{\lim} B^{k} = O_{n\times n}$ 

- **(数值线性代数, 引理 $4.2.1$)**  
  单步线性定常迭代法 $x^{(k)} = Bx^{(k-1)} + c$ 收敛的充要条件是 $\underset{k\to\infty}{\lim} B^{k} = O_{n\times n}$ 
  
- 定义 $A\in \mathbb C^{n\times n}$ 的谱半径为 $\rho(A) := \underset{1\leq i\leq n}{\max} |\lambda_i (A)|$   
  **(数值线性代数, 定理 $2.1.7$)**  
  对于任意 $A\in \mathbb C^{n\times n}$，当且仅当 $\rho(A)<1$ 时 $\underset{k\to\infty}{\lim} A^{k} = O_{n\times n}$  
  也就是说，当且仅当 $\rho(A)<1$ 时 Neumann 级数 $\underset{k=0}{\overset{\infty}\sum} A^k$ 收敛 (且收敛到 $(I-A)^{-1}$) 

- 结合上述两个定理，我们有:  
  **(数值线性代数, 定理 $4.2.1$)**   
  单步线性定常迭代法 $x^{(k)} = Bx^{(k-1)} + c$ 收敛的充要条件是 $\rho(B)<1$    

  这表明迭代序列 $\{x^{(k)}\}$ 的收敛性仅取决于迭代矩阵 $B$ 的谱半径，  
  而与初始向量 $\{x^{(0)}\}$ 的选取和常数项 $c$ 无关.


****

考虑一个满足 "$A$ 对角元均非零" 的非奇异线性方程组 $Ax=b$   
回顾之前的记号:  
$$
A = D-L-U\\
D=
\begin{bmatrix}
a_{11} & & & \\
& a_{22} & & \\
&&\ddots &\\
&&& a_{nn}\end{bmatrix}
\qquad
L
= 
\begin{bmatrix}
0 & & &\\
-a_{21} & 0 &&\\
\vdots & \ddots & \ddots &\\
-a_{n1} & \dotsm & -a_{n,n-1} & 0
\end{bmatrix}
\qquad
U
=
\begin{bmatrix}
0 & -a_{12} & \dotsm & -a_{1n}\\
& 0 &\ddots& \vdots\\
&&\ddots &-a_{n-1,n}\\
&&& 0 
\end{bmatrix}
$$
Jacobi 迭代矩阵 $D^{-1}(L+U)$ 和 Gauss-Seidel 迭代矩阵 $(D-L)^{-1} U$ 的谱半径之间并无直接联系.  
因此，有时 Jacobi 迭代法收敛而 Gauss-Seidel 迭代法发散，  
有时 Gauss-Seidel 迭代法收敛而 Jacobi 迭代法发散，有时它们都收敛，有时它们都发散.

- 考虑 $A= \begin{bmatrix}
  1 & 2 & -2\\
  1 & 1 & 1\\
  2 & 2 & 1\end{bmatrix}$ 的情况，此时我们有:   
  $$
  D=\begin{bmatrix}
  1 &  & \\
   & 1 & \\
   &  & 1\end{bmatrix}
  \qquad
  L = 
  \begin{bmatrix}
   0 &  & \\
  -1 & 0 & \\
  -2 & -2 & 0\end{bmatrix}
  \qquad
  U =
  \begin{bmatrix}
  0 & -2 & 2\\
   & 0 & -1\\
   &  & 0\end{bmatrix}\\
  $$
  Jacobi 迭代矩阵 $D^{-1}(L+U) = \begin{bmatrix}
  1 &  & \\
   & 1 & \\
   &  & 1\end{bmatrix}^{-1} \begin{bmatrix}
  0 & -2 & 2\\
  -1 & 0 & -1\\
  -2 & -2 & 0\end{bmatrix} = \begin{bmatrix}
  0 & -2 & 2\\
  -1 & 0 & -1\\
  -2 & -2 & 0\end{bmatrix}$   
  其特征多项式为 $\lambda^3$，特征值为 $0,0,0$，谱半径为 $0<1$，因此 Jacobi 迭代法收敛.

  $\text{G-S}$ 迭代矩阵 $(D-L)^{-1} U = \begin{bmatrix}
  1 &  & \\
  1 & 1 & \\
  2 & 2 & 1\end{bmatrix}^{-1} \begin{bmatrix}
  0 & -2 & 2\\
   & 0 & -1\\
   &  & 0\end{bmatrix} 
  = \begin{bmatrix}
  1 &  & \\
  -1 & 1 & \\
   & -2 & 1\end{bmatrix} \begin{bmatrix}
  0 & -2 & 2\\
   & 0 & -1\\
   &  & 0\end{bmatrix} = \begin{bmatrix}
  0 & -2 & 2\\
   & 2 & -3 \\
   &  & 2\end{bmatrix}$   
  其特征多项式为 $\lambda(\lambda-2)^2$，特征值为 $0,2,2$，谱半径为 $2\geq 1$，因此 $\text{G-S}$ 迭代发散.

- 考虑 $A= \begin{bmatrix}
  2 & -1 & 1\\
  1 & 1 & 1\\
  1 & 1 & -2\end{bmatrix}$ 的情况，此时我们有:   
  $$
  D=\begin{bmatrix}
  2 &  & \\
   & 1 & \\
   &  & -2\end{bmatrix}
  \qquad
  L = 
  \begin{bmatrix}
   0 &  & \\
  -1 & 0 & \\
  -1 & -1 & 0\end{bmatrix}
  \qquad
  U =
  \begin{bmatrix}
  0 & 1 & -1\\
   & 0 & -1\\
   &  & 0\end{bmatrix}\\
  $$
  Jacobi 迭代矩阵 $D^{-1}(L+U) = \begin{bmatrix}
  2 &  & \\
   & 1 & \\
   &  & -2\end{bmatrix}^{-1} \begin{bmatrix}
  0 & 1 & -1\\
  -1 & 0 & -1\\
  -1 & -1 & 0\end{bmatrix} = \begin{bmatrix}
  0 & \frac12 & -\frac12\\
  -1 & 0 & -1\\
  \frac12 & \frac12 & 0\end{bmatrix}$   
  其特征多项式为 $\lambda (\lambda^2+\frac{5}{4})$，特征值为 $0,\pm\frac{\sqrt{5}}{2}i$，谱半径为 $\frac{\sqrt{5}}{2}\geq 1$，因此 Jacobi 迭代法发散.

  $\text{G-S}$ 迭代矩阵 $(D-L)^{-1} U = \begin{bmatrix}
  2 &  & \\
  1 & 1 & \\
  1 & 1 & -2\end{bmatrix}^{-1} 
  \begin{bmatrix}
  0 & 1 & -1\\
   & 0 & -1\\
   &  & 0\end{bmatrix} 
  = \begin{bmatrix}
  \frac12 &  & \\
  -\frac12 & 1 & \\
  0 & \frac12 & -\frac12\end{bmatrix} 
  \begin{bmatrix}
  0 & 1 & -1\\
   & 0 & -1\\
   &  & 0\end{bmatrix} = \begin{bmatrix}
  0 & \frac12 & -\frac12\\
   & -\frac12 & -\frac12 \\
   &  & -\frac12\end{bmatrix}$   
  其特征多项式为 $\lambda(\lambda+\frac12)^2$，特征值为 $0,\frac12,\frac12$，谱半径为 $\frac12 < 1$，因此 $\text{G-S}$ 迭代收敛.



### 3.2.2 误差估计

设 $x^\star$ 是非奇异线性方程组 $Ax=b$ 的精确解.  
考虑与 $Ax=b$ 相容的单步线性定常迭代法 $x^{(k)} = Bx^{(k-1)} + c$ 产生的迭代序列 $\{x^{(k)}\}$  

**(数值线性代数, 定理 $4.2.2$)**  
若迭代矩阵 $B$ 的范数 $\|B\|=q <1$ (根据**谱半径定理**，这保证了 $\rho(B)\leq \|B\|<1$，因而迭代法收敛)，  
则误差向量的范数 $\|e^{(k)}\| = \|x^{(k)}-x^\star\| \leq \frac{q^k}{1-q} \|x^{(1)} - x^{(0)}\|$ 

- 在实际计算中，定理中的相容范数 $\|\cdot\|$ 通常选用 $\|\cdot\|_1$ 或 $\|\cdot \|_\infty$ (因为它们容易计算)

- **证明:** 根据之前的结论，我们有 $e^{(k)} = B^k e^{(0)}$   
  两边取范数，即有 $\|e^{(k)}\| = \|B^k e^{(0)}\| \leq \|B\|^k\cdot \|e^{(0)}\| = q^k \|e^{(0)}\|$

  下面估计 $\|e^{(0)}\|$: 
  $$
  \begin{align}
  \|e^{(0)}\| 
  &=
  \|x^{(0)} - x^\star\|\\
  &=
  \|x^{(0)} - x^{(1)} + x^{(1)} - x^\star\|\\
  &\leq
  \|x^{(0)}-x^{(1)}\| + \|x^{(1)}-x^\star\|\\
  &=
  \|x^{(0)} - x^{(1)}\| + \|Be^{(0)}\|\quad(\text{note that }\|x^{(1)}-x^\star\| = \|e^{(1)}\| = \|Be^{(0)}\| \leq \|B\|\|e^{(0)}\| = q\|e^{(0)}\|)\\
  &\leq
  \|x^{(1)} - x^{(0)}\| + q\|e^{(0)}\|
  \end{align}
  $$
  于是有 $\|e^{(0)}\| \leq \frac1{1-q} \|x^{(1)}-x^{(0)}\|$   
  因此 $\|e^{(k)}\| \leq q^k \|e^{(0)}\|\leq \frac{q^k}{1-q} \|x^{(1)} - x^{(0)}\|$，命题得证.

上述定理给出的误差估计往往偏高，在实际计算时用它控制迭代次数的效果并不好.  
所以给出下面的定理:  
**(数值线性代数, 定理 $4.2.3$)**  
若迭代矩阵 $B$ 的范数 $\|B\|=q <1$ (根据**谱半径定理**，这保证了 $\rho(B)\leq \|B\|<1$，因而迭代法收敛)，  
则误差向量的范数 $\|e^{(k)}\| = \|x^{(k)}-x^\star\| \leq \frac{q}{1-q} \|x^{(k)} - x^{(k-1)}\|$ 

- **证明:**  
  $$
  \begin{align}
  \|e^{(k)}\| 
  &= \|B e^{(k-1)}\|\\
  &\leq 
  \|B\| \|e^{(k-1)}\|\\
  &= q \|e^{(k-1)}\|\\
  &= q\|x^{(k-1)}-x^\star\|\\
  &\leq q(\|x^{(k-1)}-x^{(k)}\| +\|x^{(k)}-x^\star\|)\\
  &= q\|x^{(k)}-x^{(k-1)}\| + q\|e^{(k)}\|
  \end{align}
  $$
  于是有 $\|e^{(k)}\| \leq \frac{q}{1-q} \|x^{(k)}-x^{(k-1)}\|$，命题得证.

- 上述定理表明: 在 $\|B\|=q <1$ 的前提条件下，  
  当相邻两次的迭代解 $x^{(k-1)}$ 和 $x^{(k)}$ 很接近时，$x^{(k)}$ 与精确解 $x^{(k)}$ 也很接近.  
  在实际计算中，这个误差估计用来控制迭代次数 (构造终止条件) 是非常方便的.



### 3.2.3 Jacobi 迭代法和 Gauss-Seidel 迭代法的收敛性

考虑一个满足 "$A$ 对角元均非零" 的非奇异线性方程组 $Ax=b$   
回顾之前的记号:  
$$
A = D-L-U\\
D=
\begin{bmatrix}
a_{11} & & & \\
& a_{22} & & \\
&&\ddots &\\
&&& a_{nn}\end{bmatrix}
\qquad
L
= 
\begin{bmatrix}
0 & & &\\
-a_{21} & 0 &&\\
\vdots & \ddots & \ddots &\\
-a_{n1} & \dotsm & -a_{n,n-1} & 0
\end{bmatrix}
\qquad
U
=
\begin{bmatrix}
0 & -a_{12} & \dotsm & -a_{1n}\\
& 0 &\ddots& \vdots\\
&&\ddots &-a_{n-1,n}\\
&&& 0 
\end{bmatrix}
$$
Jacobi 迭代矩阵 $D^{-1}(L+U)$ 和 Gauss-Seidel 迭代矩阵 $(D-L)^{-1} U$ 的谱半径之间并无直接联系，  
但是它们的范数之间有一些联系.  
考虑到 Jacobi 迭代矩阵 $D^{-1}(L+U)$ 要比 Gauss-Seidel 迭代矩阵 $(D-L)^{-1} U$ 更容易计算，  
我们可以利用 Jacobi 迭代矩阵构造 Gauss-Seidel 迭代法的收敛条件，从而规避更复杂的计算.

对于 Jacobi 迭代法记 $\begin{cases}
B^{(1)}=D^{-1}(L+U)\\
c^{(1)}=D^{-1}b
\end{cases}$，对于 Gauss-Seidel 迭代法记 $\begin{cases}
B^{(2)}=(D-L)^{-1}U\\
c^{(2)}=(D-L)^{-1}b
\end{cases}$ 

**(数值线性代数, 定理 $4.2.4$)**  
若 $\|B^{(1)}\|_\infty <1$ (这保证了 Jacobi 迭代法收敛)，  
则 $\|B^{(2)}\|_\infty <1$ (这说明了 Gauss-Seidel 迭代法也收敛)，  
且 Gauss-Seidel 迭代法的误差向量的范数 $\|e^{(k)}\|_\infty = \|x^{(k)}-x^\star\|_\infty \leq \frac{\mu^k}{1-\mu} \|x^{(1)}-x^{(0)}\|_\infty$   
其中 $\mu = \underset{1\leq i\leq n}{\max} \{\frac{\underset{j=i-1}{\overset{n}\sum} |b^{(1)}_{ij}|}{1-\underset{j=1}{\overset{i-1}\sum}|b_{ij}^{(1)}|}\} \leq \|B^{(1)}\|_\infty < 1$

**(数值线性代数, 定理 $4.2.5$)**  
若 $\|B^{(1)}\|_1 <1$ (这保证了 Jacobi 迭代法收敛)，   
则 $\rho(B^{(2)})<1$ (这说明了 Gauss-Seidel 迭代法也收敛)，   
且 Gauss-Seidel 迭代法的误差向量的范数 $\|e^{(k)}\|_1 = \|x^{(k)}-x^\star\|_1 \leq \frac{\mu^k}{(1-\mu)(1-s)} \|x^{(1)}-x^{(0)}\|_1$   
其中 $\begin{cases}
\mu = \underset{1\leq j\leq n}{\max} \{\frac{\underset{i=1}{\overset{j-1}\sum} |b^{(1)}_{ij}|}{1-\underset{i=j+1}{\overset{n}\sum}|b_{ij}^{(1)}|}\} \leq \|B^{(1)}\|_1 < 1\\
s= \underset{1\leq j\leq n}{\max} \{\underset{i=j+1}{\overset{n}\sum}|b_{ij}^{(1)}|\} \end{cases}$ 

***

考虑一个满足 "$A$ 对角元均非零" 的非奇异线性方程组 $Ax=b$    
若系数矩阵 $A$ 还满足某些性质，我们还能推出一些更好的结论.   
回顾之前的记号:  
$$
A = D-L-U\\
D=
\begin{bmatrix}
a_{11} & & & \\
& a_{22} & & \\
&&\ddots &\\
&&& a_{nn}\end{bmatrix}
\qquad
L
= 
\begin{bmatrix}
0 & & &\\
-a_{21} & 0 &&\\
\vdots & \ddots & \ddots &\\
-a_{n1} & \dotsm & -a_{n,n-1} & 0
\end{bmatrix}
\qquad
U
=
\begin{bmatrix}
0 & -a_{12} & \dotsm & -a_{1n}\\
& 0 &\ddots& \vdots\\
&&\ddots &-a_{n-1,n}\\
&&& 0 
\end{bmatrix}
$$
**(数值线性代数, 定理 $4.2.6$)**  
若 $A$ 是对称矩阵，且对角元均为正数 (即 $a_{ii}>0\ (i=1,\dots,n)$)，  
则当且仅当 $A$ 和 $2D-A$ 都正定时 Jacobi 迭代法收敛.

- **证明:** 记 Jacobi 迭代矩阵为 $B=D^{-1}(L+U) = D^{-1}(D-A)=I-D^{-1}A$   
  由于 $A$ 对角元均为正数，故 $D= \text{diag}(a_{11},\dots,a_{nn})$ 是一个正定的对角阵.  
  我们可以定义 $D^\frac12 := \text{diag}(\sqrt{a_{11}},\dots,\sqrt{a_{nn}})$   
  于是有 $B=I-D^{-1}A = D^{-\frac12}(I-D^{-\frac12}A D^{-\frac12})D^{\frac12}$   
  可以推出 $\begin{cases}
  I-B = D^{-\frac12}(D^{-\frac12}A D^{-\frac12})D^{\frac12}\\
  I+B = D^{-\frac12}(2I-D^{-\frac12}A D^{-\frac12})D^{\frac12}\end{cases}$ 

  因为 $A$ 是对称阵，所以 $B$ 也是对称阵 (因此 $B$ 的特征值均为实数)  
  我们知道 Jacobi 迭代法收敛，当且仅当迭代矩阵 $B$ 的谱半径 $\rho(B)<1$，  
  也就等价于 $|\lambda_i(B)| <1\ (i=1,\dots,n)$，  
  即等价于 $-1<\lambda_i(B) <1\ (i=1,\dots,n)$ (注意到 $B$ 的特征值均为实数)，  
  即等价于 $I-B$ 和 $I+B$ 均正定，  
  即等价于 $D^{-\frac12}A D^{-\frac12}$ 和 $2I-D^{-\frac12}A D^{-\frac12}$ 均正定 (相似变换不改变特征值)，  
  即等价于 $A$ 和 $2I-A$ 均正定 (合同变换不改变正定性).  
  命题得证.

**(数值线性代数, 定理 $4.2.7$)**  
若 $A$ 对称正定，则 Gauss-Seidel 迭代法收敛.

***

从上述两个定理可以看到，对 Jacobi迭代法和 Gauss-Seidel 迭代法收敛性的判别，  
已经从对迭代矩阵 $B$ 性质的研究转到对系数矩阵 $A$ 性质的研究上来了，这是一个很大的进展.  
然而矩阵正定性的判别并不很直观，所以要把对判别条件的研究再深入一步.  
为此我们引进两个定义:

- **(严格对角占优性)**  
  设 $A=[a_{ij}]\in \mathbb R^{n\times n}$  
  若对任意 $1\leq i\leq n$，都有 $|a_{ii}|\geq \underset{j\neq i}{\overset{n}\sum} |a_{ij}|$，且至少对一个 $i$ 有严格不等号成立，则称 $A$ 是**弱严格对角占优的**.  
  若对任意 $1\leq i\leq n$，都有 $|a_{ii}|> \underset{j\neq i}{\overset{n}\sum} |a_{ij}|$，则称 $A$ 是**严格对角占优的**.  

- **(不可约性)**  
  设 $A=[a_{ij}]\in \mathbb R^{n\times n}$   
  若存在排列方阵 $P\in \mathbb R^{n\times n}$ 使得 $\begin{cases}
  PAP^T = \begin{bmatrix}
  A_{11} & \\
  A_{12} & A_{22}\end{bmatrix}\\
  A_{11} \in \mathbb R^{r\times r}\\
  A_{22} \in \mathbb R^{(n-r)\times (n-r)}\end{cases}$，则称 $A$ 是**可约的**.  
  否则称 $A$ 是**不可约的**.   

  若 $A$ 可约，则可将 $Ax=b$ 化为 $PAP^TPx= Pb$    
  记 $\begin{cases}
  Px=y\\
  Pb= c\end{cases}$，即有: 
  $$
  (PAP^T)Px = \begin{bmatrix}
  A_{11} & \\
  A_{12} & A_{22}
  \end{bmatrix} 
  \begin{bmatrix}
  y_1\\
  y_2
  \end{bmatrix}
  =
  \begin{bmatrix}
  c_1\\
  c_2
  \end{bmatrix} 
  = Pb
  $$
  利用前代法的思想，我们可以先求解 $r$ 阶方程组 $A_{11}y_1 = c_1$ 得到 $y_1$  
  再代入 $A_{12}y_1 + A_{22}y_2 = c_2$ 解得 $y_2$.  
  这样就将求解一个高阶方程组的问题转化为求解两个低阶方程组.

  **等价定义:**  
  设 $A=[a_{ij}]\in \mathbb R^{n\times n}$   
  若存在 $\{1,\dots,n\}$ 的两个非空子集 $S_1,S_2$ 使得 $\begin{cases}
  S_1\cup S_2 = \{1,\dots,n\}\\
  S_1\cap S_2 = \emptyset\\
  a_{ij}=0\ \ (\forall\ i\in S_1,j\in S_2)\end{cases}$   
  则称 $A$ 是可约的; 否则称 $A$ 不可约.

- 设 $A=[a_{ij}]\in \mathbb R^{n\times n}$   
  若 $A$ 不可约，且是弱严格对角占优的，则称 $A$ **不可约对角占优**.  
  例如三对角阵 $T=\begin{bmatrix}
  2 & -1 &&\\
  -1 & 2 & -1 & \\
  &-1 & 2 &-1\\
  && -1 & 2\end{bmatrix}$ 就是不可约对角占优的.

****

**(数值线性代数, 定理 $4.2.8$)**  
设 $A=[a_{ij}]\in \mathbb R^{n\times n}$  
若矩阵 $A$ 严格对角占优，或不可约对角占优，则 $A$ 非奇异.

- **证明:**  
  设 $A$ **严格对角占优**.  
  **(反证法)** 假设 $A$ 非奇异，则 $Ax=0_n$ 有非零解 $x$.  
  不妨设 $|x_i| = \|x\|_\infty = 1$ (即 $x$ 分量的最大模长是 $1$，对应下标为 $i$)，则有:  
  $$
  \begin{align}
  |a_{ii}| 
  &= |a_{ii}x_i|\\
  &= |0-\underset{j\neq i}{\overset{n}\sum} a_{ij} x_j|\\
  &= |\underset{j\neq i}{\overset{n}\sum} a_{ij} x_j|\\
  &\leq \underset{j\neq i}{\overset{n}\sum} |a_{ij}| |x_j| \\
  &\leq \underset{j\neq i}{\overset{n}\sum} |a_{ij}|\cdot 1 \\
  &= \underset{j\neq i}{\overset{n}\sum} |a_{ij}|
  \end{align}
  $$
  这与严格对角占优性 $|a_{ii}|> \underset{j\neq i}{\overset{n}\sum} |a_{ij}|\ (i=1,\dots,n)$ 矛盾，因而 $A$ 非奇异.

  设 $A$ **不可约对角占优**.  
  **(反证法)** 假设 $A$ 非奇异，则 $Ax=0_n$ 有非零解 $x$ (不妨设满足 $\|x\|_\infty = 1$, 即分量最大模长为 $1$). 

  定义 $\begin{cases}
  S_1:= \{i:|x_i|=1\}\\
  S_2:= \{1,\dots,n\}\backslash S_1 = \{i:|x_i|<1\}\end{cases}$​ (显然满足 $\begin{cases}
  S_1\cup S_2 = \{1,\dots,n\}\\
  S_1\cap S_2 = \emptyset\end{cases}$)   

  - 显然 $S_1$ 不是空集.  
    而 $S_2$ 也不是空集，这是因为:  
    假设 $S_2$ 为空集，即 $x$ 的所有分量的模长都为 $1$.  
    则对于任意 $i=1,\dots,n$，都有: 
    $$
    \begin{align}
    |a_{ii}| 
    &= |a_{ii}x_i|\\
    &= |0-\underset{j\neq i}{\overset{n}\sum} a_{ij} x_j|\\
    &= |\underset{j\neq i}{\overset{n}\sum} a_{ij} x_j|\\
    &\leq \underset{j\neq i}{\overset{n}\sum} |a_{ij}| |x_j| \\
    &= \underset{j\neq i}{\overset{n}\sum} |a_{ij}|\cdot 1 \\
    &= \underset{j\neq i}{\overset{n}\sum} |a_{ij}|
    \end{align}
    $$
    即没有一个 $i$ 使得 $|a_{ii}|\geq \underset{j\neq i}{\overset{n}\sum} |a_{ij}|$ 严格成立，这与 $A$ 弱严格对角占优矛盾，因而 $S_2$ 不是空集.

  由于 $A$ 不可约，故必定存在 $\begin{cases}
  i \in S_1\\
  k\in S_2\end{cases}$ 使得 $a_{ik}\neq 0$，于是有:  
  (注意到 $|x_i|=1$ 和 $|x_k|<1$) 
  $$
  \begin{align}
  |a_{ii}| 
  &= |a_{ii}x_i|\\
  &= |0-\underset{j\neq i}{\overset{n}\sum} a_{ij} x_j|\\
  &= |\underset{j\neq i}{\overset{n}\sum} a_{ij} x_j|\\
  &\leq 
  |\underset{j\in S_1\backslash \{i\}}\sum a_{ij}x_j| + |\underset{j\in S_2\backslash \{k\}}\sum a_{ij}x_j|
  + |a_{ik} x_k|\\
  &\leq 
  \underset{j\in S_1\backslash \{i\}}\sum |a_{ij}| |x_j| + \underset{j\in S_2\backslash \{k\}}\sum |a_{ij}| |x_j|
  + |a_{ik}| |x_k|\\
  &< 
  \underset{j\in S_1\backslash \{i\}}\sum |a_{ij}|\cdot 1 + \underset{j\in S_2\backslash \{k\}}\sum |a_{ij}|\cdot 1
  + |a_{ik}| \cdot 1\quad (\text{note that the strict inequality stems from }
  \begin{cases}
  |a_{ik}|> 0\\
  |x_k| < 1\end{cases})\\
  &= 
  \underset{j\neq i}{\overset{n}\sum} |a_{ij}|
  \end{align}
  $$
  这与 $A$ 弱严格对角占优矛盾，因而 $A$ 非奇异.

**(数值线性代数, 推论 $4.2.1$)**  
设 $A=[a_{ij}]\in \mathbb R^{n\times n}$ 是对称阵 (即 $A^T = A$)，且对角元均为正数.  
若 $A$ 严格对角占优或不可约对角占优，则 $A$ 正定.

- **证明:** 对于任意 $\lambda\leq 0$，考虑矩阵 $A-\lambda I$   
  显然 $A-\lambda I$ 也是严格对角占优或不可约对角占优的，因此 $A-\lambda I$ 总是非奇异的.  
  表明 $A$ 的特征值均大于 $0$，即 $A$ 正定.
- **"对角元均为正数"** 的条件是有必要的，  
  因为严格对角占优或不可约对角占优只能保证对角元均非零，即 $|a_{ii}|>0\ (i=1,\dots,n)$ 

****

考虑非奇异线性方程组 $Ax=b$     
回顾之前的记号:  
$$
A = D-L-U\\
D=
\begin{bmatrix}
a_{11} & & & \\
& a_{22} & & \\
&&\ddots &\\
&&& a_{nn}\end{bmatrix}
\qquad
L
= 
\begin{bmatrix}
0 & & &\\
-a_{21} & 0 &&\\
\vdots & \ddots & \ddots &\\
-a_{n1} & \dotsm & -a_{n,n-1} & 0
\end{bmatrix}
\qquad
U
=
\begin{bmatrix}
0 & -a_{12} & \dotsm & -a_{1n}\\
& 0 &\ddots& \vdots\\
&&\ddots &-a_{n-1,n}\\
&&& 0 
\end{bmatrix}
$$
**(数值线性代数, 定理 $4.2.9$)**    
若 $A$ 严格对角占优或不可约对角占优，则 Jacobi 迭代法和 Gauss-Seidel 迭代法都收敛.  

- **(应用数值线性代数, 定理 $6.2$)**

**证明:**  
"$A$ 严格对角占优或不可约对角占优" 的条件保证了 $A$ 的对角元均非零，  
因而 $D$ 和 $D-L$ 都是可逆矩阵.  

- 首先考虑 Jacobi 迭代法，记 $\begin{cases}
  B=D^{-1}(L+U)\\
  c=D^{-1}b
  \end{cases}$
  **(反证法)** 假设 Jacobi 迭代法发散，即 $\rho(B)\geq 1$，即存在 $B$ 的某个特征值 $\lambda$ 满足 $|\lambda|\geq 1$  
  于是 $A+(\lambda-1)D = \lambda D - L -U$ 也是严格对角占优或不可约对角占优的，  
  因此 $\lambda D-L-U$ 非奇异.

  由 $\lambda I-B = \lambda I - D^{-1}(L+U) = D^{-1}(\lambda D-L-U)$ 可知 $\lambda I-B$ 也是非奇异的.  
  这与 "$\lambda$ 是 $B$ 的特征值" 矛盾，因而 Jacobi 迭代法收敛.

- 其次考虑 Gauss-Seidel 迭代法，记 $\begin{cases}
  B=(D-L)^{-1}U\\
  c=(D-L)^{-1}b
  \end{cases}$     
  
  **(反证法)** 假设 Gauss-Seidel 迭代法发散，即 $\rho(B)\geq 1$，即存在 $B$ 的某个特征值 $\lambda$ 满足 $|\lambda|\geq 1$  
  于是 $A+(\lambda-1)(D-L) = \lambda D - \lambda L -U$ 也是严格对角占优或不可约对角占优的，  
  因此 $\lambda D-L-U$ 非奇异.
  
  由 $\lambda I-B = \lambda I - (D-L)^{-1}U = (D-L)^{-1}(\lambda D-\lambda L-U)$ 可知 $\lambda I-B$ 也是非奇异的.  
  这与 "$\lambda$ 是 $B$ 的特征值" 矛盾，因而 Gauss-Seidel 迭代法收敛.

命题得证.

*****

**(邵老师的补充)**  
Hermite 正定矩阵 $A\in \mathbb C^{n\times n}$ 的 Gauss-Seidel 迭代法一定收敛.  
(从这个意义来说，Gauss-Seidel 迭代法要优于 Jacobi 迭代法)

- **证明:**   
  考虑 Hermite 正定线性方程组 $Ax=b$     
  回顾之前的记号:  
  $$
  A = D-L-U\\
  D=
  \begin{bmatrix}
  a_{11} & & & \\
  & a_{22} & & \\
  &&\ddots &\\
  &&& a_{nn}\end{bmatrix}
  \qquad
  L
  = 
  \begin{bmatrix}
  0 & & &\\
  -a_{21} & 0 &&\\
  \vdots & \ddots & \ddots &\\
  -a_{n1} & \dotsm & -a_{n,n-1} & 0
  \end{bmatrix}
  \qquad
  U
  =
  \begin{bmatrix}
  0 & -a_{12} & \dotsm & -a_{1n}\\
  & 0 &\ddots& \vdots\\
  &&\ddots &-a_{n-1,n}\\
  &&& 0 
  \end{bmatrix}
  $$
  $A$ 的 Hermite 性保证了 $U=L^H$  
  于是 $B = (D-L)^{-1}U = (D-L)^{-1}L^H$ 

   















## 3.3 收敛速度

现在给出单步线性定常迭代法收敛速度的定量刻画.

### 3.3.1 基本定义

设 $x^\star$ 是非奇异线性方程组 $Ax=b$ 的精确解.  
考虑与 $Ax=b$ 相容的单步线性定常迭代法 $x^{(k)} = Bx^{(k-1)} + c$ 产生的迭代序列 $\{x^{(k)}\}$ 

若该迭代法收敛，则其极限 $x^\star$ 满足 $x^\star = Bx^\star + c$    
定义 $x^{(k)}$ 误差向量为 $e^{(k)} = x^{(k)}-x^\star$   
联立 $\begin{cases}
x^{(k)} = Bx^{(k-1)} + c\\
x^\star = B x^\star + c\\
e^{(k)} = x^{(k)}-x^\star\\
e^{(k-1)} = x^{(k-1)}-x^\star\end{cases}$ 可得 $e^{(k)} = B\cdot e^{(k-1)}\ \ (k=1,2,\dots)$   
于是我们有 $e^{(k)} = B^k e^{(0)}\ \ (k=1,2,\dots)$     
因此 $\|e^{(k)}\|\leq \|B^k\|\|e^{(0)}\ \ (k=1,2,\dots)$

初始误差 $\|e^{(0)}\|$ 通常是未知的，因此通常用 $\|B^k\|$ 来刻画收敛速度.  
我们定义 $k$ 次迭代的**平均收敛速度** $R_k:=R_k(B) = -\frac{1}{k} \log(\|B^k\|)$

- 注意，若迭代法收敛，则 $\underset{k\to\infty}{\lim} \|B^k\| \to 0$   
  因此当 $k$ 充分大时，总有 $R_k >0$   
  我们记 $\mu = (\frac{\|e^{(k)}\|}{\|e^{(0)}\|})\approx \|B^k\|^{\frac1k} = e^{-R_k}$ 为**误差范数在 $k$ 次迭代中的平均缩减因子**.

- 一般情况下，$R_k(B)$ 既依赖于 $k$ 又依赖于 $B$，因此计算起来很复杂.  
  但如果 $B$ 是对称矩阵 (或 Hermite 阵, 或正规矩阵)，则有 $\|B^k\|_2 = (\rho(B))^k$   
  因此 $R_k(B) = -\frac{1}{k} \log(\|B^k\|_2) = -\frac1k\cdot k \log(\rho(B)) = -\log(\rho(B))$   
  此时 $R_k(B)$ 仅依赖于 $B$，不依赖于 $k$.

设有两个迭代法，对应的迭代矩阵分别为 $B^{(1)}$ 和 $B^{(2)}$.  
若 $R_k(B^{(1)}) > R_k(B^{(2)}) > 0$，  
则我们称前 $k$ 次迭代中 $B^{(1)}$ 对应的迭代法比 $B^{(2)}$ 对应的迭代法收敛速度慢.

但可能对于某些 $k$ 有 $R_k(B^{(1)}) > R_k(B^{(2)})$，而对另一些 $k$ 有 $R_k(B^{(1)}) < R_k(B^{(2)})$  
因此为了刻画整个迭代过程的收敛速度，  
我们自然考虑定义**渐近收敛速度** $R_\infty(B):= \underset{k\to\infty}{\lim} R_k(B) = -  \underset{k\to\infty}{\lim} \frac{1}{k} \log(\|B^k\|) = -  \underset{k\to\infty}{\lim} \log(\|B^k\|^{\frac1k})$  

**(数值线性代数, 定理 $4.3.1$)**  
给定 $\begin{cases}
A\in \mathbb R^{n\times n}\\
b\in \mathbb R^n\end{cases}$ (设 $A$ 非奇异)  
考虑与 $Ax=b$ 相容的单步线性定常迭代法 $x^{(k)} = Bx^{(k-1)} + c$   
我们有 $R_\infty(B) = -\log(\rho(B))$ 成立.

- **证明:** 只需证明 $ \underset{k\to\infty}{\lim} \|B^k\|^{\frac1k}=\rho(B)$ 即可.  

  一方面，因为 $(\rho(B))^k = \rho(B^k) \leq \|B^k\|\ \ (k=1,2,\dots)$，所以 $\rho(B)\leq \|B^k\|^\frac1k\ \ (k=1,2,\dots)$  

  另一方面，对于任意 $\varepsilon>0$，考虑矩阵 $B_\varepsilon = \frac{1}{\rho(B) + \varepsilon} B$   
  显然有 $\rho(B_\varepsilon)<1$，于是有 $\underset{k\to\infty}{\lim} B_\varepsilon^k = O_{n\times n}$ 成立  
  因此存在正整数 $K$ 使得对于任意 $k\geq K$ 都有 $\|B^k_\varepsilon\|\leq 1$ 成立，  
  即有 $\|B^k\|\leq (\rho(B)+\varepsilon)^k$ 成立，即有 $\|B^k\|^{\frac1k} \leq \rho(B) + \varepsilon$ 成立.

  综上所述，对于任意 $\varepsilon>0$ 和足够大的 $k$ 都有 $\rho(B)\leq \|B^k\|^\frac1k \leq \rho(B)+\varepsilon$ 成立.  
  因此 $ \underset{k\to\infty}{\lim} \|B^k\|^{\frac1k}=\rho(B)$，命题得证.



### 3.3.2 模型问题

为更具体地说明迭代法的收敛速度，我们来考虑模型问题.  
所谓模型问题是指用**五点差分格式**求解单位正方形区域上的二维 Poisson 方程第一边值问题:
$$
\begin{cases}
\Gamma = \text{bd}(\{(x,y)\in \mathbb R^2:0\leq x,y\leq 1\}) & (单位正方形区域的边界)\\
\Delta u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = -f(x,y) & (0<x,y<1)\\
u|_{\Gamma} = \phi & (边界值函数)
\end{cases}
$$
为将微分方程离散化，我们将正方形的每条边 $n$ 等分，记 $h=\frac1{n}$，得到 $n^2$ 个小正方形.  
记小正方形的顶点为 $(x_i,y_i)\ \ (i,j=0,1,\dots,n)$，记 $\begin{cases}
u(x_i,y_j)=u_{ij}\\
f(x_i,y_j) = f_{ij}\\
\phi(x_i,y_j) = \phi_{ij}\end{cases}$    
$n= 4$ 的情况如图所示:

<img src="数值线性代数 图 4.1.png" style="zoom:50%;" />

我们有二阶差商:
$$
\begin{cases}
\frac{\partial^2 u}{\partial x^2}|_{(x_i,y_j)} 
= 
\frac{1}{h^2}(u_{i-1,j} + u_{i+1,j} - 2u_{i,j}) + O(h^2)\\
\frac{\partial^2 u}{\partial y^2}|_{(x_i,y_j)} 
= 
\frac{1}{h^2}(u_{i,j-1} + u_{i,j+1} - 2u_{i,j}) + O(h^2)
\end{cases} \quad (i,j=0,1,\dots,n)
$$
带入原微分方程有:
$$
\begin{cases}
f_{i,j} = \frac{\partial^2 u}{\partial x^2}|_{(x_i,y_j)}+ \frac{\partial^2 u}{\partial y^2}|_{(x_i,y_j)} = \frac{1}{h^2}(u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4u_{i,j}) + O(h^2) & (i,j=0,1,\dots,n)\\
u_{i,0} = \phi_{i,0},\ u_{i,n} = \phi_{i,n} & (i=0,1,\dots,n)\\
u_{0,j} = \phi_{0,j},\ u_{n,j} = \phi_{n,j} & (j=0,1,\dots,n)
\end{cases}
$$
整理并略去误差项得到:
$$
\begin{cases}
4 u_{i,j} - (u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1}) = h^2 f_{i,j} & (i,j=0,1,\dots,n)\\
u_{i,0} = \phi_{i,0},\ u_{i,n} = \phi_{i,n} & (i=0,1,\dots,n)\\
u_{0,j} = \phi_{0,j},\ u_{n,j} = \phi_{n,j} & (j=0,1,\dots,n)
\end{cases}
$$

***

现假定边界值函数 $\phi\equiv 0$，即有:
$$
\begin{cases}
(2 u_{i,j} - u_{i-1,j} - u_{i+1,j}) + (2 u_{i,j} - u_{i,j-1} - u_{i,j+1}) = h^2 f_{i,j}\\
u_{i,0} = u_{i,n} = u_{0,j} = u_{n,j} = 0
\end{cases}\quad (i,j = 0,1,\dots, n)
$$
写成矩阵形式即为:
$$
T_{n-1} U + U T_{n-1} = h^2 F\\
T_{n-1} 
=
\begin{bmatrix}
2 & -1 & & \\
-1 & \ddots & \ddots & \\
& \ddots & \ddots & -1\\
& & -1 & 2
\end{bmatrix} 
\qquad

U = \begin{bmatrix}
u_{1,1} & u_{1,2} &\dotsm  & u_{1,n-1}\\
u_{2,1} & u_{2,2} & \dotsm & u_{2,n-1}\\
\vdots & \vdots & & \vdots\\
u_{n-1,1} & u_{n-1,2} & \dotsm & u_{n-1,n-1}
\end{bmatrix}
\qquad

F = \begin{bmatrix}
f_{1,1} & f_{1,2} &\dotsm  & f_{1,n-1}\\
f_{2,1} & f_{2,2} & \dotsm & f_{2,n-1}\\
\vdots & \vdots & & \vdots\\
f_{n-1,1} & f_{n-1,2} & \dotsm & f_{n-1,n-1}
\end{bmatrix}
$$

***

将正方形的顶点按 $(i,j) $ **自然顺序**排列，即先按 $j$ 由小到大排列，$j$ 相同的按 $i$ 由小到大排列.  
$n= 4$ 的情况如图所示:  

<img src="数值线性代数 图 4.2.png" style="zoom:50%;" />

按自然顺序排列后得到的线性方程组为:
$$
Au = h^2 f\\

A =
\begin{bmatrix}
T_{n-1} + 2I_{n-1} & -I_{n-1} &  &\\
-I_{n-1} & \ddots & \ddots &\\
&\ddots & \ddots & -I_{n-1}\\
& & -I_{n-1} & T_{n-1} + 2I_{n-1}
\end{bmatrix}

\qquad

u=
\begin{bmatrix}
u_{1,1}\\
\vdots\\
u_{1,n-1}\\
u_{2,1}\\
\vdots\\
u_{2,n-1}\\
\vdots\\
\vdots\\
u_{n-1,1}\\
\vdots\\
u_{n-1,n-1}
\end{bmatrix}

\qquad

f=
\begin{bmatrix}
f_{1,1}\\
\vdots\\
f_{1,n-1}\\
f_{2,1}\\
\vdots\\
f_{2,n-1}\\
\vdots\\
\vdots\\
f_{n-1,1}\\
\vdots\\
f_{n-1,n-1}
\end{bmatrix}
$$
显然 $A$ 是 $(n-1)^2$ 阶方阵，它具有以下性质:

- 若 $U$ 中顶点排列次序发生改变，则 $A$ 的性状也会发生改变，但可以证明 $A$ 的特征值仍旧不变.
- $A$ 是块三对角阵，若不看分块的话，共有五条对角线上有非零元素 (因此 $A$ 是稀疏的).
- $A$ 是不可约对角占优的.
- $A$ 是对称正定的.

$T_{n-1} 
=
\begin{bmatrix}
2 & -1 & & \\
-1 & \ddots & \ddots & \\
& \ddots & \ddots & -1\\
& & -1 & 2
\end{bmatrix} $ 的特征值为 $\lambda_k = 2(1-\cos\frac{k\pi}{n})$，单位特征向量为 $z_k = \sqrt{\frac2n} \begin{bmatrix}
\sin \frac{k\pi}{n}\\
\sin \frac{2k\pi}{n}\\
\vdots\\
\sin \frac{(n-1)k\pi}{n}\end{bmatrix}$   
(其中 $k = 1,\dots,n-1$) 

根据 $T_{n-1} U + U T_{n-1} = h^2 F$ 可知 $A$ 的特征值为 $\lambda_{p,q} = \lambda_p + \lambda_q = 2(2-\cos\frac{p\pi}{n} - \cos \frac{q\pi}{n})$   
对应的单位特征向量为 $v_{p,q}$ 是矩阵 $z_pz_q^T$ 按列 "拉直" 得到的 $(n-1)^2$ 维向量  
即有 $v_{p,q} = \sqrt{\frac2n} \begin{bmatrix}
\sin \frac{q\pi}{n}\cdot z_p\\
\sin \frac{2q\pi}{n}\cdot z_p\\
\vdots\\
\sin \frac{(n-1)q\pi}{n}\cdot z_p
\end{bmatrix}$



### 3.3.3 Jacobi 迭代法和 Gauss-Seidel 迭代法求解模型问题的渐近收敛速度

模型问题:
$$
Au = h^2 f\\

A =
\begin{bmatrix}
T_{n-1} + 2I_{n-1} & -I_{n-1} &  &\\
-I_{n-1} & \ddots & \ddots &\\
&\ddots & \ddots & -I_{n-1}\\
& & -I_{n-1} & T_{n-1} + 2I_{n-1}
\end{bmatrix}

\qquad

u=
\begin{bmatrix}
u_{1,1}\\
\vdots\\
u_{1,n-1}\\
u_{2,1}\\
\vdots\\
u_{2,n-1}\\
\vdots\\
\vdots\\
u_{n-1,1}\\
\vdots\\
u_{n-1,n-1}
\end{bmatrix}

\qquad

f=
\begin{bmatrix}
f_{1,1}\\
\vdots\\
f_{1,n-1}\\
f_{2,1}\\
\vdots\\
f_{2,n-1}\\
\vdots\\
\vdots\\
f_{n-1,1}\\
\vdots\\
f_{n-1,n-1}
\end{bmatrix}
$$
回顾之前的记号:  
$$
A = D-L-U\\
D=
\begin{bmatrix}
a_{11} & & & \\
& a_{22} & & \\
&&\ddots &\\
&&& a_{nn}\end{bmatrix}
\qquad
L
= 
\begin{bmatrix}
0 & & &\\
-a_{21} & 0 &&\\
\vdots & \ddots & \ddots &\\
-a_{n1} & \dotsm & -a_{n,n-1} & 0
\end{bmatrix}
\qquad
U
=
\begin{bmatrix}
0 & -a_{12} & \dotsm & -a_{1n}\\
& 0 &\ddots& \vdots\\
&&\ddots &-a_{n-1,n}\\
&&& 0 
\end{bmatrix}
$$
Jacobi 迭代法计算模型问题的迭代矩阵和常数项为: 
$$
\begin{cases}
B=D^{-1}(L+U) = D^{-1}(D-A) = I - D^{-1} A = I- (\frac14 I) A = I-\frac14 A\\
c=D^{-1}b = (\frac14 I) h^2 f = \frac{h^2}{4} f
\end{cases}
$$
考虑 $I-\frac14 A$ 的结构: 
$$
I_{(n-1)^2}-\frac14 A = 
\frac14\begin{bmatrix}
2I_{n-1}-T_{n-1} & I_{n-1} &  &\\
I_{n-1} & \ddots & \ddots &\\
&\ddots & \ddots & I_{n-1}\\
& & I_{n-1} & 2I_{n-1}-T_{n-1}
\end{bmatrix}\\
\text{where }
T_{n-1} 
=
\begin{bmatrix}
2 & -1 & & \\
-1 & \ddots & \ddots & \\
& \ddots & \ddots & -1\\
& & -1 & 2
\end{bmatrix}
\qquad 
2I_{n-1} - T_{n-1} = 
\begin{bmatrix}
0 & 1 & & \\
1 & \ddots & \ddots & \\
& \ddots & \ddots & 1\\
& & 1 & 0
\end{bmatrix}
$$
  于是其迭代格式 $u^{(k)} = Bu^{(k-1)} + c$ 可表示为:
$$
u^{(k)} = Bu^{(k-1)} + c = (I-\frac14 A) u^{(k-1)} + \frac{h^2}4 f\\
\Leftrightarrow\\
u_{ij}^{(k)} = \frac14 (u_{i-1,j}^{(k-1)} + u_{i,j-1}^{(k-1)} + u_{i+1,j}^{(k-1)} + u_{i,j+1}^{(k-1)}) + \frac{h^2}{4}f_{ij}\quad (i,j = 1,\dots,n-1)
$$
其中边界值 $u_{i,0} = u_{i,n} = u_{0,j} = u_{n,j} = 0\ \ (i,j = 1,\dots,n-1)$ 在迭代过程中不改变 (例如 $u_{i,0}^{(k)} \equiv u_{i,0} = 0$).    
总之，我们有迭代公式:
$$
\begin{cases}
u_{ij}^{(k)} = \frac14 (u_{i-1,j}^{(k-1)} + u_{i,j-1}^{(k-1)} + u_{i+1,j}^{(k-1)} + u_{i,j+1}^{(k-1)}) + \frac{h^2}{4}f_{ij}\\
u_{i,0}^{(k)} = u_{i,n}^{(k)} = u_{0,j}^{(k)} = u_{n,j}^{(k)} = 0 \end{cases}\quad (i,j = 1,\dots,n-1)
$$
迭代矩阵 $B=I-\frac14 A$ 的特征值为 $\lambda_{p,q}(B) = 1-\frac14 \lambda_{p,q}(A) = 1-\frac14\cdot 2(2-\cos\frac{p\pi}{n} - \cos \frac{q\pi}{n}) = \frac12 (\cos\frac{p\pi}{n} + \cos \frac{q\pi}{n})$   
因此谱半径为: 
$$
\begin{align}
\rho(B) 
&= \underset{1\leq p,q\leq n-1}{\max} |\lambda_{p,q}(B)|\\
&= \underset{1\leq p,q\leq n-1}{\max} |\frac12 (\cos\frac{p\pi}{n} + \cos \frac{q\pi}{n})|\\
&= \frac12 \{\underset{1\leq p\leq n-1}{\max} |\cos\frac{p\pi}{n}| + \underset{1\leq q\leq n-1}{\max} |\cos\frac{q\pi}{n}|\}\\
&= \frac12 (\cos\frac{\pi}{n} + \cos\frac{\pi}{n})\\
&= \cos\frac{\pi}{n}\\
&= \cos(h\pi)\quad (h=\frac{1}n) 
\end{align}
$$
因此 Jacobi 迭代法求解模型问题的渐近收敛速度为:
$$
\begin{align}
R_\infty(B) 
&= -\log(\rho(B))\\
&= -\log(\cos(h\pi))\\
&= -\log(\sqrt{1-(\sin(h\pi))^2})\\
&= -\frac12 \log(1-(\sin(h\pi))^2)\\
&\sim -\frac12 \cdot -(\sin(h\pi))^2\\
&\sim -\frac12 \cdot -(h\pi)^2\\
&=  \frac12 \pi^2 h^2\quad (h\to 0)
\end{align}
$$

***

Gauss-Seidel 迭代法的迭代格式为:
$$
u^{(k)} = (D-L)^{-1} U u^{(k-1)} + (D-L)^{-1} b\\
\Leftrightarrow\\
u^{(k)} = D^{-1} L u^{(k)} + D^{-1} U u^{(k-1)} + D^{-1}b
$$
我们计算得:
$$
D^{-1} L = (\frac14 I) L = \frac14 L 
= \frac14 
\begin{bmatrix}
L_{n-1} & & &\\
I_{n-1} & \ddots & & \\
&\ddots&\ddots & \\
&& I_{n-1} &L_{n-1}\end{bmatrix}
\text{ where }
L_{n-1} 
= 
\begin{bmatrix}
0 & &&\\
1&\ddots & &\\
&\ddots & \ddots & \\
&&1&0\end{bmatrix}\\

D^{-1} U = (\frac14 I) U = \frac14 U
=\frac14 
\begin{bmatrix}
U_{n-1} &I_{n-1} & &\\
&\ddots&\ddots&\\
&&\ddots& I_{n-1}\\
&&&U_{n-1}\end{bmatrix}
\text{ where }
 U_{n-1} =
\begin{bmatrix}
0 & 1 & &\\
& \ddots & \ddots &\\
&&\ddots &1\\
&&& 0
\end{bmatrix}\\
D^{-1} b = (\frac14 I) b = \frac14 b = \frac{h^2}{4} f
$$
因此 $u^{(k)} = D^{-1} L u^{(k)} + D^{-1} U u^{(k-1)} + D^{-1}b$ 可表示为:
$$
u_{ij}^{(k)} = \frac14 (u_{i-1,j}^{(k)} + u_{i,j-1}^{(k)} + u_{i+1,j}^{(k-1)} + u_{i,j+1}^{(k-1)}) + \frac{h^2}{4}f_{ij}\quad (i,j = 1,\dots,n-1)
$$

于是迭代矩阵 $B^{(2)}=(D-L)^{-1} U$ 的特征值 $\lambda$ 满足:  
$$
\begin{cases}
\lambda \xi_{i,j} = \frac14 (\lambda \xi_{i-1,j} + \lambda \xi_{i,j-1} + \xi_{i+1,j} + \xi_{i,j+1})\\
\xi_{i,0} = \xi_{i,n} = \xi_{0,j} = \xi_{n,j} = 0
\end{cases}
$$
设 $\lambda \neq 0$，作变换 $\xi_{i,j} = \lambda^{\frac{i+j}{2}} v_{i,j}$ 可得:
$$
\begin{align}
\lambda \xi_{i,j}
&= 
\lambda \cdot \lambda^{\frac{i+j}{2}} v_{i,j}\\
&=
\lambda^{1 + \frac{i+j}{2}} v_{i,j}\\
&=
\frac14 (\lambda \xi_{i-1,j} + \lambda \xi_{i,j-1} + \xi_{i+1,j} + \xi_{i,j+1})\\
&=
\frac14 (\lambda \cdot \lambda^{\frac{i-1+j}{2}}v_{i-1,j} + \lambda\cdot \lambda^{\frac{i+j-1}{2}} v_{i,j-1} + \lambda^{\frac{i+1+j}{2}}v_{i+1,j} + \lambda^{\frac{i+j+1}{2}}v_{i,j+1})\\
&=
\frac14 \lambda^{\frac{i+j+1}{2}}(v_{i-1,j} + v_{i,j-1} + v_{i+1,j} + v_{i,j+1})
\end{align}
$$
于是有:
$$
\lambda^{1 + \frac{i+j}{2}} v_{i,j} = \frac14 \lambda^{\frac{i+j+1}{2}}(v_{i-1,j} + v_{i,j-1} + v_{i+1,j} + v_{i,j+1})\\
\Leftrightarrow\\
\lambda^{\frac12}v_{i,j} = \frac14 (v_{i-1,j} + v_{i,j-1} + v_{i+1,j} + v_{i,j+1})
$$

> 回忆起 Jacobi 迭代法应用到模型问题得到的迭代公式:  
> $$
> \begin{cases}
> u_{ij}^{(k)} = \frac14 (u_{i-1,j}^{(k-1)} + u_{i,j-1}^{(k-1)} + u_{i+1,j}^{(k-1)} + u_{i,j+1}^{(k-1)}) + \frac{h^2}{4}f_{ij}\\
> u_{i,0}^{(k)} = u_{i,n}^{(k)} = u_{0,j}^{(k)} = u_{n,j}^{(k)} = 0 \end{cases}\quad (i,j = 1,\dots,n-1)
> $$
> 我们知道迭代矩阵 $B^{(1)}=D^{-1} (L+U)$ 的特征值 $\mu$ 满足:
> $$
> \begin{cases}
> \mu v_{i,j} = \frac14 (v_{i-1,j} + v_{i,j-1} + v_{i+1,j} + v_{i,j+1})\\
> v_{i,0} = v_{i,n} = v_{0,j} = v_{n,j} = 0
> \end{cases}
> $$

对比发现，$\lambda \neq 0$ 是 Gauss-Seidel 迭代矩阵 $B^{(2)}=(D-L)^{-1} U$ 的非零特征值，  
当且仅当 $\mu=\lambda^\frac12$ 是 Jacobi 迭代矩阵 $B^{(1)}=D^{-1} (L+U)$ 的非零特征值.

回忆起 Jacobi 迭代矩阵 $B^{(1)}=D^{-1} (L+U)$ 的谱半径 $\rho(B^{(1)}) = \cos\frac{\pi}{n} = \cos(h\pi)$   
因此 Gauss-Seidel 迭代法求解模型问题的渐近收敛速度为:
$$
R_\infty(B^{(2)}) = -\log(\rho{(B^{(2)})}) =-\log(\rho^2{(B^{(1)})}) = -2\log (\rho{(B^{(1)})})= - 2\log(\cos(h\pi)) \sim 2\cdot \frac12 \pi^2 h^2 = \pi^2 h^2\quad (h\to 0)
$$
这说明 Gauss-Seidel 迭代法求解模型问题的渐近收敛速度是 Jacobi 迭代法的两倍.




## 3.4 超松弛迭代法

超松弛迭代法是 Gauss-Seidel 迭代法的引申和推广，也可看作 Gauss-Seidel 迭代法的加速.

### 3.4.1 迭代格式

考虑一个满足 "$A$ 对角元均非零" 的非奇异线性方程组 $Ax=b$   
回顾之前的记号:  
$$
A = D-L-U\\
D=
\begin{bmatrix}
a_{11} & & & \\
& a_{22} & & \\
&&\ddots &\\
&&& a_{nn}\end{bmatrix}
\qquad
L
= 
\begin{bmatrix}
0 & & &\\
-a_{21} & 0 &&\\
\vdots & \ddots & \ddots &\\
-a_{n1} & \dotsm & -a_{n,n-1} & 0
\end{bmatrix}
\qquad
U
=
\begin{bmatrix}
0 & -a_{12} & \dotsm & -a_{1n}\\
& 0 &\ddots& \vdots\\
&&\ddots &-a_{n-1,n}\\
&&& 0 
\end{bmatrix}
$$
Gauss-Seidel 迭代法的迭代格式为:
$$
x^{(k)} = (D-L)^{-1} U x^{(k-1)} + (D-L)^{-1} b\\
\Leftrightarrow\\
x^{(k)} = D^{-1} L x^{(k)} + D^{-1} U x^{(k-1)} + D^{-1}b
$$

现令修正项 $\Delta x^{(k-1)} = x^{(k)} - x^{(k-1)}$，则我们有:
$$
\begin{align}
\Delta x^{(k-1)} 
&= x^{(k)} - x^{(k-1)}\\
&= (D^{-1} L x^{(k)} + D^{-1} U x^{(k-1)} + D^{-1}b) - x^{(k-1)}
\end{align}
$$
若我们在修正项 $\Delta x^{(k-1)}$ 的前面加上一个系数 $\omega>0$，便得到**松弛迭代法**的迭代格式:
$$
\begin{align}
x^{(k)} 
&= x^{(k-1)} + \omega \Delta x^{(k-1)}\\
&= x^{(k-1)} + \omega [(D^{-1} L x^{(k)} + D^{-1} U x^{(k-1)} + D^{-1}b) - x^{(k-1)}]\\
&= (1-\omega) x^{(k-1)} + \omega (D^{-1} L x^{(k)} + D^{-1} U x^{(k-1)} + D^{-1}b)
\end{align}
$$
我们称 $\omega$ 为**松弛因子**:

- 当 $\omega>1$ 时，称为**超松弛迭代法** (简称为 $\text{SOR}$ 迭代法)
- 当 $\omega=1$ 时，就是 **Gauss-Seidel 迭代法**
- 当 $\omega<1$ 时，称为**低松弛迭代法**

显然 $I-\omega D^{-1} L$ 是下三角阵，且对角元均非零 (前提条件是 $A$ 对角元均非零)，因此它是可逆的.  
于是我们可以改写**松弛迭代法**的迭代格式:
$$
x^{(k)} = (1-\omega) x^{(k-1)} + \omega (D^{-1} L x^{(k)} + D^{-1} U x^{(k-1)} + D^{-1}b)\\
\Leftrightarrow\\
x^{(k)} = (I-\omega D^{-1}L)^{-1} [(D^{-1}U +(1-\omega) I) x^{(k-1)} + \omega D^{-1} b] = (D-\omega L)^{-1}[(1-\omega)D+\omega U] x^{(k-1)} + \omega (D-\omega L)^{-1} b\\
\Leftrightarrow\\
x^{(k)} = B_\omega x^{(k-1)} + c_{\omega} \text{ where }
\begin{cases}
B_\omega = (D-\omega L)^{-1}[(1-\omega)D+\omega U]\\
c_\omega = \omega (D-\omega L)^{-1} b
\end{cases}
$$


### 3.4.2 收敛性分析

现在我们研究超松弛迭代法的收敛性判别和松弛因子 $\omega$ 的选取范围.  
回顾之前的记号:
$$
A = D-L-U\\
D=
\begin{bmatrix}
a_{11} & & & \\
& a_{22} & & \\
&&\ddots &\\
&&& a_{nn}\end{bmatrix}
\qquad
L
= 
\begin{bmatrix}
0 & & &\\
-a_{21} & 0 &&\\
\vdots & \ddots & \ddots &\\
-a_{n1} & \dotsm & -a_{n,n-1} & 0
\end{bmatrix}
\qquad
U
=
\begin{bmatrix}
0 & -a_{12} & \dotsm & -a_{1n}\\
& 0 &\ddots& \vdots\\
&&\ddots &-a_{n-1,n}\\
&&& 0 
\end{bmatrix}\\

x^{(k)} = B_\omega x^{(k-1)} + c_{\omega} \text{ where }
\begin{cases}
B_\omega = (D-\omega L)^{-1}[(1-\omega)D+\omega U]\\
c_\omega = \omega (D-\omega L)^{-1} b
\end{cases}
$$
**(超松弛迭代法收敛的必要条件, 数值线性代数, 定理 $4.4.2$)**  
超松弛迭代法收敛的必要条件是 $0<\omega < 2$ 

- **证明:**  
  设超松弛迭代法收敛，即等价于 $\rho(B_\omega)<1$   
  设 $\lambda_1,\dots,\lambda_n$ 是 $B_\omega$ 的 $n$ 个特征值，则我们必须有:
  $$
  \begin{align}
  |\det(B_w)| 
  &=
  |\det\{(D-\omega L)^{-1}[(1-\omega)D+\omega U]\}|\\
  &=
  |\det\{(I-\omega D^{-1}L)^{-1}[(1-\omega)I+\omega D^{-1}U]\}|\\
  &=
  |\det\{ (I-\omega D^{-1}L)^{-1}\}| |\det \{(1-\omega)I+\omega D^{-1}U\}|\\
  &=
  |1^n| \cdot |(1-\omega)^n|\\
  &=
  |(1-\omega)^n|\\
  &=
  |\lambda_1 \dotsm \lambda_n|\\
  &=
  |\lambda_1| \dotsm |\lambda_n|\\
  &< 1^n\\
  &= 1
  \end{align}
  $$
  根据 $|(1-\omega)^n|<1$ 可知 $|1-\omega|<1$，即有 $0<\omega<2$，命题得证.

**(超松弛迭代法收敛的充分条件, 数值线性代数, 定理 $4.4.3$)**  
若系数矩阵 $A$ 严格对角占优或不可约对角占优，且松弛因子 $\omega\in (0,1)$，则超松弛迭代法收敛.

- **证明:**  
  "$A$ 严格对角占优或不可约对角占优" 的条件保证了 $A$ 的对角元均非零，  
  于是下三角阵 $D-\omega L$ 的对角元也均非零，因而是可逆的.

  考虑超松弛迭代法 $\begin{cases}
  B_\omega = (D-\omega L)^{-1}[(1-\omega)D+\omega U]\\
  c_\omega = \omega (D-\omega L)^{-1} b
  \end{cases}$   
  **(反证法)** 假设超松弛迭代法发散，即 $\rho(B_\omega)\geq 1$，即存在 $B_\omega$ 的某个特征值 $\lambda$ 满足 $|\lambda|\geq 1$  
  于是 $\omega A + (\lambda-1)(D-\omega L) = (\omega + \lambda -1) D -\omega\lambda L - \omega U$ 也是严格对角占优或不可约对角占优的，  
  因此 $(\omega + \lambda -1) D -\omega\lambda L - \omega U$ 非奇异.

  由 $\lambda I- B_{\omega} = \lambda I - (D-\omega L)^{-1}[(1-\omega)D+\omega U] = (D-\omega L)^{-1} [(\omega + \lambda -1) D -\omega\lambda L - \omega U]$   
  可知 $\lambda I-B_\omega$ 也是非奇异的.  
  这与 "$\lambda$ 是 $B_\omega$ 的特征值" 矛盾，因而超松弛迭代法收敛.

**(正定情形下超松弛迭代法收敛的充要条件, 数值线性代数, 定理 $4.4.4$)**   
若系数矩阵 $A$ 对称正定，则当且仅当松弛因子 $\omega\in (0,2)$ 时超松弛迭代法收敛.

- **证明:**  
  由于系数矩阵 $A$ 对称，故 $U=L^T$   

  设 $\lambda$ 是 $B_\omega$ 的任一特征值 (不一定是实数)，$x$ 是对应的特征向量，则有:
  $$
  B_\omega x = (D-\omega L)^{-1}[(1-\omega)D+\omega U] x = \lambda x\\
  \Leftrightarrow\\
  [(1-\omega)D+\omega L^T] x= \lambda (D-\omega L) x\\
  \Leftrightarrow\\
  x^H[(1-\omega)D+\omega L^T] x= \lambda x^H(D-\omega L) x
  $$
  记 $\begin{cases}
  \gamma = x^HDx\\
  \alpha + i\beta = x^HLx \ \Rightarrow\ x^HL^Tx = \alpha -i \beta\end{cases}$  则我们有:
  $$
  (1-\omega)\gamma + \omega (\alpha- i\beta) = \lambda[\gamma - \omega(\alpha + i\beta)]\\
  \Leftrightarrow\\
  (1-\omega)\gamma + \omega\alpha - i\omega \beta = \lambda (\gamma - \omega \alpha - i\omega \beta)
  $$
  两边取模可得:
  $$
  |\lambda|^2 = \frac{[(1-\omega)\gamma + \omega \alpha]^2 + (\omega \beta)^2}{(\gamma-\omega \alpha)^2 + (\omega \beta)^2}
  $$
  分子减分母可知:
  $$
  \begin{align}
  &\{[(1-\omega)\gamma + \omega \alpha]^2 + (\omega \beta)^2\} - \{(\gamma-\omega \alpha)^2 + (\omega \beta)^2\}\\
  &=
  [(1-\omega)\gamma + \omega \alpha]^2 - (\gamma-\omega \alpha)^2\\
  &=
  [(1-\omega)\gamma + \omega\alpha - (\gamma-\omega\alpha)] \cdot [(1-\omega)\gamma + \omega\alpha + (\gamma-\omega\alpha)]\\
  &=
  (2\omega \alpha -\omega \gamma)(2-\omega)\gamma\\
  &=
  \gamma(\gamma - 2\alpha)\omega (\omega-2)
  \end{align}
  $$
  由于 $A$ 正定，故我们有 (注意到 $x$ 作为 $B_\omega$ 的特征向量，一定不是零向量):
  $$
  \begin{cases}
  \delta = x^H Dx > 0\\
  \delta - 2\alpha = \delta - (\alpha + i\beta) - (\alpha -i\beta) = x^H Dx - x^H L x - x^HL^Tx = x^H(D-L-U)x = x^HAx >0
  \end{cases}
  $$
  因此 $|\lambda|^2 <1$ (即 $|\lambda|<1$) 当且仅当 $\omega(\omega-2)<0$，即 $\omega\in(0,2)$   
  根据 $\lambda$ 是 $B_\omega$ 的任意特征值可知，$\rho(B_\omega)<1$ 当且仅当 $\omega\in (0,2)$   
  这表明超松弛迭代法收敛当且仅当 $\omega\in (0,2)$，命题得证.



### 3.4.3 最佳松弛因子

要研究收敛速度在 $\omega$ 为何值时最快，我们考虑特征值问题:
$$
B_\omega x = (D-\omega L)^{-1}[(1-\omega)D+\omega U] x = \lambda x\\
\Leftrightarrow\\
[(\lambda+\omega-1) D - \lambda \omega L - \omega U] x = 0_n\\
$$
设 $\lambda_1,\dots,\lambda_n$ 是 $B_\omega$ 的 $n$ 个特征值，则我们有:
$$
\begin{align}
\det(B_w)
&=
\det\{(D-\omega L)^{-1}[(1-\omega)D+\omega U]\}\\
&=
\det\{(I-\omega D^{-1}L)^{-1}[(1-\omega)I+\omega D^{-1}U]\}\\
&=
\det\{ (I-\omega D^{-1}L)^{-1}\}\det \{(1-\omega)I+\omega D^{-1}U\\
&=
1^n \cdot (1-\omega)^n\\
&=
(1-\omega)^n
\end{align}
$$
因此当 $\omega \neq 1$ 时，$B_\omega$ 无零特征值.

***

模型问题:
$$
Au = h^2 f\\

A =
\begin{bmatrix}
T_{n-1} + 2I_{n-1} & -I_{n-1} &  &\\
-I_{n-1} & \ddots & \ddots &\\
&\ddots & \ddots & -I_{n-1}\\
& & -I_{n-1} & T_{n-1} + 2I_{n-1}
\end{bmatrix}

\qquad

u=
\begin{bmatrix}
u_{1,1}\\
\vdots\\
u_{1,n-1}\\
u_{2,1}\\
\vdots\\
u_{2,n-1}\\
\vdots\\
\vdots\\
u_{n-1,1}\\
\vdots\\
u_{n-1,n-1}
\end{bmatrix}

\qquad

f=
\begin{bmatrix}
f_{1,1}\\
\vdots\\
f_{1,n-1}\\
f_{2,1}\\
\vdots\\
f_{2,n-1}\\
\vdots\\
\vdots\\
f_{n-1,1}\\
\vdots\\
f_{n-1,n-1}
\end{bmatrix}
$$
回顾之前的记号:  
$$
A = D-L-U\\
D=
\begin{bmatrix}
a_{11} & & & \\
& a_{22} & & \\
&&\ddots &\\
&&& a_{nn}\end{bmatrix}
\qquad
L
= 
\begin{bmatrix}
0 & & &\\
-a_{21} & 0 &&\\
\vdots & \ddots & \ddots &\\
-a_{n1} & \dotsm & -a_{n,n-1} & 0
\end{bmatrix}
\qquad
U
=
\begin{bmatrix}
0 & -a_{12} & \dotsm & -a_{1n}\\
& 0 &\ddots& \vdots\\
&&\ddots &-a_{n-1,n}\\
&&& 0 
\end{bmatrix}
$$
对于模型问题，在自然次序排列下，$[(\lambda+\omega-1) D - \lambda \omega L - \omega U] x = 0_n$ 可写成:
$$
\begin{cases}
(\lambda+\omega -1)u_{ij} - \frac{\omega}{4} (\lambda u_{i-1,j} + \lambda u_{i,j-1} + u_{i+1,j} + u_{i,j+1}) = 0\\
u_{i,0} = u_{i,n} = u_{0,j} = u_{n,j} = 0
\end{cases}\quad (i,j = 1,\dots,n-1)
$$
下面分两步讨论:

***

**第一步:** 找出 Jacobi 迭代矩阵 $B^{(1)} = D^{-1}(L+U)$ 与 $B_{\omega}=(D-\omega L)^{-1}[(1-\omega)D+\omega U]$ 特征值之间的关系.

假设 $\omega\neq 1$ (这保证了 $B_\omega$ 没有零特征值)  **(存疑: $\lambda$ 一定是正实数吗?)**
对 $B_\omega$ 的特征值 $\lambda$ 作变换 $u_{i,j} = (\pm \lambda^{\frac12})^{i+j} v_{i,j}$ 可得:
$$
\begin{align}
(\lambda+\omega - 1)u_{i,j}
&=
(\lambda+\omega - 1)\cdot (\pm \lambda^{\frac12})^{i+j} v_{i,j}\\
&=
\frac{\omega}{4} (\lambda u_{i-1,j} + \lambda u_{i,j-1} + u_{i+1,j} + u_{i,j+1})\\
&=
\frac{\omega}{4} [\lambda \cdot (\pm \lambda^{\frac12})^{i-1+j} v_{i-1,j}
+
\lambda \cdot (\pm \lambda^{\frac12})^{i+j-1} v_{i,j-1} 
+
(\pm \lambda^{\frac12})^{i+1+j} v_{i+1,j}
+
(\pm \lambda^{\frac12})^{i+j+1} v_{i,j+1}]\\
&=
\frac{\omega}{4}(\pm \lambda^\frac12)^{i+j+1} [ v_{i-1,j} + v_{i,j-1} + v_{i+1,j} + v_{i,j+1}]
\end{align}
$$
因此有: 
$$
\begin{cases}
\mu v_{i,j} = \frac{1}{4}[ v_{i-1,j} + v_{i,j-1} + v_{i+1,j} + v_{i,j+1}]\\
v_{i,0} = v_{i,n} = v_{0,j} = v_{n,j} = 0
\end{cases}
\quad (i,j=1,\dots,n)
\quad \text{where }\mu = \pm \frac{\lambda +\omega -1}{\omega \lambda^{\frac12}}
$$

> 回忆起 Jacobi 迭代法应用到模型问题得到的迭代公式:  
> $$
> \begin{cases}
> u_{ij}^{(k)} = \frac14 (u_{i-1,j}^{(k-1)} + u_{i,j-1}^{(k-1)} + u_{i+1,j}^{(k-1)} + u_{i,j+1}^{(k-1)}) + \frac{h^2}{4}f_{ij}\\
> u_{i,0}^{(k)} = u_{i,n}^{(k)} = u_{0,j}^{(k)} = u_{n,j}^{(k)} = 0 \end{cases}\quad (i,j = 1,\dots,n-1)
> $$
> 我们知道迭代矩阵 $B^{(1)}=D^{-1} (L+U)$ 的特征值 $\mu$ 满足:
> $$
> \begin{cases}
> \mu v_{i,j} = \frac14 (v_{i-1,j} + v_{i,j-1} + v_{i+1,j} + v_{i,j+1})\\
> v_{i,0} = v_{i,n} = v_{0,j} = v_{n,j} = 0
> \end{cases}
> $$

对比发现，在 $\omega\neq 1$ 的前提条件下，  
$\lambda$ 是松弛迭代矩阵 $B_{\omega}=(D-\omega L)^{-1}[(1-\omega)D+\omega U]$ 的特征值，   
当且仅当 $\mu = \pm \frac{\lambda +\omega -1}{\omega \lambda^{\frac12}}$ 是 Jacobi 迭代矩阵 $B^{(1)} = D^{-1}(L+U)$ 的特征值.  

- 实际上，前面关于 $\omega\neq 1$ 的限制是可以去掉的:  
  我们可以将 $\omega=1$ 的情况视作 $\omega\to 1$ 的极限情况，  
  此时 $\mu = \pm \frac{\lambda +\omega -1}{\omega \lambda^{\frac12}}$ 退化成 $\mu = \pm \lambda^{\frac12}$  
  可知 $B^{(1)}$ 的每对特征值 $\pm \mu_i$，都对应 $B_{\omega}=(D-\omega L)^{-1}[(1-\omega)D+\omega U]$ 的特征值 $0$ 和 $\mu_i^2$ 

因此对于任意 $\omega\in (0,2)$，我们都有:
$\lambda$ 是松弛迭代矩阵 $B_{\omega}=(D-\omega L)^{-1}[(1-\omega)D+\omega U]$ 的特征值，   
当且仅当 $\mu = \pm \frac{\lambda +\omega -1}{\omega \lambda^{\frac12}}$ 是 Jacobi 迭代矩阵 $B^{(1)} = D^{-1}(L+U)$ 的特征值. 

因此松弛迭代矩阵 $B_\omega$ 的特征值完全由松弛因子 $\omega$ 和 Jacobi 迭代矩阵 $B^{(1)}$ 的特征值 $\mu$ 确定.

****

**第二步:** 确定谱半径 $\rho(B_\omega)$ 随 $\omega$ 变化的情况

设 $0\leq \mu < 1$ 是 Jacobi 迭代矩阵 $B^{(1)}$ 的一个特征值，$\omega \in (0,2)$   
则由 $\mu = \pm \frac{\lambda +\omega -1}{\omega \lambda^{\frac12}}$ (即二次方程 $\lambda+\omega -1 = \pm \mu \omega \lambda^{\frac12}$) 可得松弛迭代矩阵 $B_\omega$ 的两个特征值分别为:
$$
\lambda_+(\omega,\mu) = [\frac{\mu \omega}{2} + \sqrt{(\frac{\mu \omega}{2})^2 - (\omega-1)}]^2\\
\lambda_-(\omega,\mu) = [\frac{\mu \omega}{2} - \sqrt{(\frac{\mu \omega}{2})^2 - (\omega-1)}]^2
$$
记这两个特征值模长的最大值为 $M(\omega,\mu) := \max\{|\lambda_+(\omega,\mu)|, |\lambda_-(\omega,\mu)|\}$   

记二次方程 $\lambda+\omega -1 = \pm \mu \omega \lambda^{\frac12}$ 的判别式为 $\Delta = (\mu\omega)^2 - 4(\omega-1)$   
判别式 $\Delta$ 在 $(0,2)$ 的唯一零点是 $\omega_\mu = \frac{2}{1+\sqrt{1-\mu^2}}\in (0,2)$ 

- 当 $0<\omega<\omega_\mu$ 时 $\Delta>0$，从而有 $\lambda_+(\omega,\mu)> \lambda_-(\omega,\mu)>0$ 
- 当 $\omega = \omega_\mu$ 时 $\Delta=0$，从而有 $\lambda_+(\omega,\mu)= \lambda_-(\omega,\mu)>0$
- 当 $\omega_\mu < \omega<2$ 时 $\Delta<0$，故 $\lambda_+(\omega,\mu)$ 和 $\lambda_-(\omega,\mu)$ 为一对共轭复数，且 $|\lambda_+(\omega,\mu)|=|\lambda_-(\omega,\mu)| = \omega-1$ 

综上所述 $M(\omega,\mu) = \max\{|\lambda_+(\omega,\mu)|, |\lambda_-(\omega,\mu)|\}=\begin{cases}
\lambda_+(\omega,\mu) & 0<\omega \leq \omega_\mu\\
\omega-1 & \omega_\mu < \omega < 2\end{cases}$

**固定 $0\leq \mu<1$，令 $\omega$ 从 $0$ 变到 $2$，考虑 $M(\omega,\mu)$ 的变化情况:**

- 首先我们有 $M(\omega,\mu) < 1$，这是因为:

  - 当 $0<\omega \leq \omega_\mu$ 时，我们有:  
    $$
    \begin{align}
    M(\omega,\mu) 
    &= \lambda_+(\omega,\mu)\\
    &= [\frac{\mu \omega}{2} + \sqrt{(\frac{\mu \omega}{2})^2 - (\omega-1)}]^2\\
    &\leq [\frac{\mu \omega}{2} + \sqrt{(\frac{\mu \omega}{2})^2 + 1 - 0}]^2\\
    &< [\frac{\mu \omega}{2} + \sqrt{(\frac{\mu \omega}{2})^2 + 1 - \mu\omega}]^2\\
    &= [\frac{\mu \omega}{2} + (1-\frac{\mu\omega}{2})]^2\\
    &=1
    \end{align}
    $$

  - 当 $\omega_\mu < \omega <2$ 时，我们有 $M(\omega,\mu) = \omega-1<2-1 = 1$ 

- 其次，当 $0<\omega\leq \omega_\mu$ 时，我们有:
  $$
  \begin{align}
  \frac{\partial M(\omega,\mu)}{\partial\omega}
  &=
  \frac{\partial}{\partial\omega}\{[\frac{\mu \omega}{2} + \sqrt{(\frac{\mu \omega}{2})^2 - (\omega-1)}]^2\}\\
  &=
  2[\frac{\mu \omega}{2} + \sqrt{(\frac{\mu \omega}{2})^2 - (\omega-1)}]\cdot [\frac{\mu}{2} + 
  \frac{\frac12 \omega\mu^2 -1}{2\sqrt{(\frac{\mu\omega}{2})^2 - (\omega-1)}}]\\
  &=
  M(\omega,\mu)^{\frac12} 
  \frac{\mu M(\omega,\mu)^\frac12 -1}{\sqrt{(\frac{\mu\omega}{2})^2 - (\omega-1)}}\\
  &<0
  \end{align}
  $$
  于是当 $\omega\to \omega_\mu -0$ 时，我们有 (注意到 $\omega_\mu$ 是方程 $(\mu\omega)^2 - 4(\omega-1)=0$ 的解):
  $$
  M(\omega,\mu) =[\frac{\mu \omega}{2} + \sqrt{(\frac{\mu \omega}{2})^2 - (\omega-1)}]^2 \to (\frac{\mu \omega_\mu}{2})^2 = \omega_\mu -1 = \frac{1-\sqrt{1-\mu^2}}{1+\sqrt{1-\mu^2}}\\
  \frac{\partial M(\omega,\mu)}{\partial\omega} = M(\omega,\mu)^{\frac12} 
  \frac{\mu M(\omega,\mu)^\frac12 -1}{\sqrt{(\frac{\mu\omega}{2})^2 - (\omega-1)}} \to \frac{\text{Negative Value}}{0} = -\infty
  $$
  而当 $\omega_\mu < \omega<2$ 时，$M(\omega,\mu)=\omega-1$ 关于 $\omega$ 严格单调递增.

综上所述，对于固定的 $0\leq \mu<1$，  
$M(\omega,\mu)$ 关于 $\omega$ 在 $(0,\omega_\mu)$ 上严格单调递减，在 $(\omega_\mu,2)$ 上严格单调递增，因此在 $\omega_\mu$ 处达到极小值 $\omega_\mu-1 = \frac{1-\sqrt{1-\mu^2}}{1+\sqrt{1-\mu^2}}$ 

**固定 $\omega\in (0,2)$，考虑 $M(\omega,\mu)$ 随 $0\leq \mu<1$ 的变化情况:**  
可以证明，若 $0<\mu_1<\mu_2<1$ 是 Jacobi 迭代矩阵 $B^{(1)}$ 的两个特征值，则有 $M(\omega,\mu_1)\leq M(\omega,\mu_2)$   
即 $M(\omega,\mu)$ 关于 $\mu$ 是单调递增的.

- 首先 $\omega_\mu = \frac{2}{1+\sqrt{1-\mu^2}}\in (0,2)$ 关于 $0\leq \mu<1$ 严格单调递增，因此 $\omega_{\mu_1}<\omega_{\mu_2}$

- 当 $\omega_{\mu_2} \leq \omega<2$ 时，我们有 $\begin{cases}
  M(\omega,\mu_1) = \omega-1\\
  M(\omega,\mu_2) = \omega-1\end{cases}\ \Rightarrow\ M(\omega,\mu_1) = M(\omega,\mu_2) = \omega-1$ 

- 当 $0<\omega\leq \omega_{\mu_1}$ 时，我们有 $\begin{cases}
  M(\omega,\mu_1) = [\frac{\mu_1 \omega}{2} + \sqrt{(\frac{\mu_1 \omega}{2})^2 - (\omega-1)}]^2\\
  M(\omega,\mu_2) = [\frac{\mu_2 \omega}{2} + \sqrt{(\frac{\mu_2 \omega}{2})^2 - (\omega-1)}]^2\end{cases}$  
  $$
  \begin{align}
  M(\omega,\mu_2) - M(\omega,\mu_1)
  &=
  [M(\omega,\mu_2)^{\frac12} + M(\omega,\mu_1)^{\frac12}]\cdot [M(\omega,\mu_2)^{\frac12} - M(\omega,\mu_1)^{\frac12}]\\
  &=
  [M(\omega,\mu_2)^{\frac12} + M(\omega,\mu_1)^{\frac12}]\cdot 
  [\frac{\mu_2 \omega}{2} + \sqrt{(\frac{\mu_2 \omega}{2})^2 - (\omega-1)} - \frac{\mu_1 \omega}{2} - \sqrt{(\frac{\mu_1 \omega}{2})^2 - (\omega-1)}]\\
  &=
  [M(\omega,\mu_2)^{\frac12} + M(\omega,\mu_1)^{\frac12}]\cdot 
  [\frac{\omega(\mu_2-\mu_1)}{2} + 
   \frac{\frac{\omega^2}{4}(\mu^2_2 -\mu_1^2)}{\sqrt{(\frac{\mu_2 \omega}{2})^2 - (\omega-1)} + \sqrt{(\frac{\mu_1 \omega}{2})^2 - (\omega-1)}}]\\
  &>0
  \end{align}
  $$

- 当 $\omega_{\mu_1}<\omega \leq \omega_{\mu_2}$​ 时，我们有 $\begin{cases}
  M(\omega,\mu_1) = \omega-1\\
  M(\omega,\mu_2) = [\frac{\mu_2 \omega}{2} + \sqrt{(\frac{\mu_2 \omega}{2})^2 - (\omega-1)}]^2\end{cases}$
  $$
  \begin{align}
  M(\omega,\mu_2)
  &\geq \omega_{\mu_2} -1\qquad (\text{Note that }\inf_{0<\omega<2}M(\omega,\mu_2) = \omega_{\mu_2}-1)\\
  &\geq \omega - 1\\
  &= M(\omega,\mu_1)
  \end{align}
  $$

综上所述，对于固定的 $0< \omega <2$，$M(\omega,\mu)$ 关于 $0\leq \mu<1$ 是单调递增的.  

***

对于 Jacobi 迭代矩阵 $B^{(1)}$ 的任意给定的特征值 $\mu$，我们都可绘制一条 $M(\omega,\mu)$ 关于 $\omega$ 的曲线.  
它关于 $\omega$ 在 $(0,\omega_\mu)$ 上严格单调递减，在 $(\omega_\mu,2)$ 上严格单调递增，因此在 $\omega_\mu$ 处达到极小值 $\omega_\mu-1 = \frac{1-\sqrt{1-\mu^2}}{1+\sqrt{1-\mu^2}}$ 

而对于确定的 $\omega$，$M(\omega,\mu)$ 又是 $\mu$ 的增函数，  
因此 Jacobi 迭代矩阵 $B^{(1)}$ 的较大特征值对应的曲线在较小特征值对应的曲线的上方.   
于是 Jacobi 迭代矩阵 $B^{(1)}$ 取到谱半径 $\rho(B^{(1)})$ 的特征值 $\mu_\max$ 对应的是最上方的曲线，同时也对应着 $\rho(B_\omega)$   
如图所示:

<img src="数值线性代数 图 4.3.png" style="zoom:33%;" />

根据最上方的曲线 $\rho(B_\omega)=\underset{0\leq \mu < 1}{\max} M(\omega,\mu) = M(\omega,\mu_\max)$，我们得到结论:  

- 当 $0<\omega \leq \omega_{\text{opt}} = \omega_{\mu_\max} = \frac{2}{1 + \sqrt{1-(\rho(B^{(1)}))^2}}$ 时 $\rho(B_\omega)= [\frac{\mu_\max \omega}{2} + \sqrt{(\frac{\mu_\max \omega}{2})^2 - (\omega-1)}]^2$ 严格单调递减
- 当 $\omega_{\text{opt}} < \omega <2$ 时 $\rho(B_\omega)=\omega -1$ 严格单调递增

因此 $\rho(B_\omega)$ 在 $\omega_{\text{opt}}$ 处取到极小值 $\omega_{\text{opt}}-1 = \frac{1-\sqrt{1-(\rho(B^{(1)}))^2}}{1+\sqrt{1-(\rho(B^{(1)}))^2}}$   
我们称 $\omega_{\text{opt}} = \frac{2}{1 + \sqrt{1-(\rho(B^{(1)}))^2}}$ 为**最佳松弛因子**  
对应的**最佳松弛迭代矩阵**记为 $B_{\omega_{\text{opt}}} = (D-\omega_{\text{opt}} L)^{-1}[(1-\omega_{\text{opt}})D+\omega_{\text{opt}} U]$   
(由于 $\omega_{\text{opt}}>1$，故对应的松弛迭代是**超松弛迭代**)

***

在实际计算中，我们有时无法确定 Jacobi 迭代矩阵的谱半径 $\rho(B^{(1)})$，因此最佳松弛因子 $\omega_{\text{opt}}$ 也无法确定.   
此时如何求近似的 $\omega_{\text{opt}}$ 呢?  
我们通常选取不同的松弛因子 $\omega\in (0,2)$，  
然后用相同的初始向量 $x^{(0)}$ 进行试算，迭代相同次数 (得到 $x^{(k_0)}$)，比较残差向量的范数 $\|b-Ax^{(k_0)}\|$   
最后选取使残差向量范数最小的 $\omega$ 作为松弛因子.

另外，根据松弛迭代矩阵 $B_\omega$ 的谱范数 $\rho(B_\omega)$ 随 $\omega$ 变化的曲线来看，  
当我们不能得到准确的最佳松弛因子时，宁可取得稍大一些，因为 $\omega_{\text{opt}}$ 左侧的曲线比右侧更加陡峭.



### 3.4.4 渐近收敛速度

最佳松弛因子 $\omega_{\text{opt}}$ 对应的 (超) 松弛迭代法的渐近收敛速度为:  
(回忆起 Jacobi 迭代矩阵的谱半径 $\rho(B^{(1)}) = \cos\frac{\pi}{n} = \cos(h\pi)$)
$$
\begin{align}
R_\infty(B_{\omega_{\text{opt}}}) 
&=
-\log({\rho(B_{\omega_{\text{opt}}})})\\
&=
-\log(\frac{1-\sqrt{1-(\rho(B^{(1)}))^2}}{1+\sqrt{1-(\rho(B^{(1)}))^2}})\\
&=
-\log(\frac{1-\sqrt{1-(\cos(h\pi))^2}}{1+\sqrt{1-(\cos(h\pi))^2}})\\
&=
-\log(\frac{1-\sin(h\pi)}{1+\sin(h\pi)})\\
&=
-\log(1-\sin(h\pi)) +\log(1+\sin(h\pi))\\
&\sim
-(-\sin(h\pi)) + \sin(h\pi)\\
&\sim
-(-h\pi) + h\pi\\
&= 2h\pi\quad (h\to 0)
\end{align}
$$
回忆起 $\begin{cases}
R_\infty(B^{(1)}) = R_\infty(D^{-1}(L+U)) = \frac12 \pi^2 h^2 \\
R_\infty(B^{(2)}) = R_\infty((D-L)^{-1}U) = \pi^2 h^2 \end{cases}\ \ (h\to 0)$  
可以看出，使用最佳松弛因子的超松弛迭代法的收敛速度比 Jacobi 迭代法和 Gauss-Seidel 迭代法快得多.  



### 3.4.5 超松弛理论的推广

前面的讨论都基于模型问题，事实上我们可以将其推广到更一般的情形.

**(数值线性代数, 定理 $4.4.5$)**    
设线性方程组 $Ax=b$ 的系数矩阵 $A$ 具有块三对角形式:
$$
A=
\begin{bmatrix}
D_1 & K_1 &  & & \\
H_1 & D_2 & K_2 & & \\
& H_2 & D_3 & \ddots &\\
&&\ddots&\ddots &K_{m-1}\\
&&&H_{m-1} & D_m
\end{bmatrix}\\
$$
其中 $D_i\ (i=1,\dots,m)$ 是非奇异的对角阵.  
我们记:
$$
A = D-L-U\\
D=
\begin{bmatrix}
a_{11} & & & \\
& a_{22} & & \\
&&\ddots &\\
&&& a_{nn}\end{bmatrix}
\qquad
L
= 
\begin{bmatrix}
0 & & &\\
-a_{21} & 0 &&\\
\vdots & \ddots & \ddots &\\
-a_{n1} & \dotsm & -a_{n,n-1} & 0
\end{bmatrix}
\qquad
U
=
\begin{bmatrix}
0 & -a_{12} & \dotsm & -a_{1n}\\
& 0 &\ddots& \vdots\\
&&\ddots &-a_{n-1,n}\\
&&& 0 
\end{bmatrix}
$$
设 $\begin{cases}
B^{(1)} = D^{-1}(L+U)\\
B^{(2)} = (D-L)^{-1} U\\
B_\omega = (D-\omega L)^{-1}[(1-\omega)D+\omega U]\end{cases}$ 分别为 Jacobi 迭代法、Gauss-Seidel 迭代法和超松弛迭代法的迭代矩阵.  
若 $B^{(1)}$ 的特征值均为实数，则我们有:

- **Jacobi 迭代矩阵和超松弛迭代矩阵的特征值之间的关系:**  
  若 $\mu\neq 0$ 是 Jacobi 迭代矩阵 $B^{(1)}$ 的特征值，  
  则由 $\mu = \pm \frac{\lambda +\omega -1}{\omega \lambda^{\frac12}}$ (即二次方程 $\lambda+\omega -1 = \pm \mu \omega \lambda^{\frac12}$) 可得松弛迭代矩阵 $B_\omega$ 的两个特征值分别为:  
  $$
  \lambda_+(\omega,\mu) = [\frac{\mu \omega}{2} + \sqrt{(\frac{\mu \omega}{2})^2 - (\omega-1)}]^2\\
  \lambda_-(\omega,\mu) = [\frac{\mu \omega}{2} - \sqrt{(\frac{\mu \omega}{2})^2 - (\omega-1)}]^2
  $$
  反过来，若 $\lambda\neq 0$ 是超松弛迭代迭代 $B_\omega$ 的特征值，  
  则 $\mu = \pm \frac{\lambda +\omega -1}{\omega \lambda^{\frac12}}$ 是 Jacobi 迭代矩阵 $B^{(1)}$ 的两个特征值.  
  (因此若 $\mu\neq 0$ 是 $B^{(1)}$ 的特征值，则 $-\mu$ 也是 $B^{(1)}$ 的特征值)

  **特殊地，当 $\omega = 1$​ 时，我们得到 Jacobi 迭代矩阵和 Gauss-Seidel 迭代矩阵的特征值之间的关系:**    
  $\lambda \neq 0$ 是 Gauss-Seidel 迭代矩阵 $B^{(2)}=(D-L)^{-1} U$ 的非零特征值，  
  当且仅当 $\mu=\lambda^\frac12$ 是 Jacobi 迭代矩阵 $B^{(1)}=D^{-1} (L+U)$ 的非零特征值.

  因此谱半径具有关系: $\rho(B^{(2)}) = (\rho(B^{(1)}))^2$   
  渐近收敛速度 $R_\infty(B^{(2)}) = -\log({\rho(B^{(2)})}) = -2\log({\rho(B^{(1)})}) = 2R_\infty(B^{(1)})$ 

- **最佳松弛因子的相关陈述:**  
  若 Jacobi 迭代矩阵的谱半径 $\rho(B^{(1)})<1$ 且松弛因子 $0<\omega<2$，  
  则超松弛迭代法收敛，且最佳松弛因子为 $\omega_{\text{opt}} = \frac{2}{1+\sqrt{1-(\rho(B^{(1)}))^2}}$   
  对应的谱半径 $\rho(B_{\omega_{\text{opt}}}) = \omega_{\text{opt}} -1 = \frac{1-\sqrt{1-(\rho(B^{(1)}))^2}}{1+\sqrt{1-(\rho(B^{(1)}))^2}}$ 

  当 $\rho(B^{(1)})$ 从下方趋近于 $1$ 时，我们有渐近收敛速度 $R_\infty(B_{\omega_{\text{opt}}}) \sim 2\sqrt{2 R_\infty(B^{(1)})}$ 

***

上述定理要求线性方程组的系数矩阵是块三对角阵，且对角块为非奇异对角阵.   
但是 "对角块为非奇异对角阵" 的要求太苛刻了.  
回忆起模型问题中，系数矩阵 $A$ 的对角块为 $T_{n-1} + 2I_{n-1} =
\begin{bmatrix}
4 & -1 & & \\
-1 & \ddots & \ddots & \\
& \ddots & \ddots & -1\\
& & -1 & 4
\end{bmatrix}$ 是三对角阵，而不是对角阵.  
然而我们成功对它建立了超松弛理论.  
实际上，超松弛理论可以适用于一类更广泛的矩阵.  
为此，我们引入相容性的概念.

**(具有相容次序的矩阵)**  
设 $A$ 是 $n$ 阶方阵.  
我们称 $A$ 是**具有相容次序的矩阵**，  
如果存在 $\{1,\dots,n\}$ 的一个划分 $S_1,\dots,S_m$ 使得每个非零非对角元 $a_{ij}\neq 0\ (i\neq j)$ 的下标 $i,j$ 满足:  
若 $i\in S_k$，则 $j\in \begin{cases}
S_{k-1} & j<i\\
S_{k+1} & j>i\end{cases}$ (其中 $k=1,\dots,m$) 

- 以模型问题的系数矩阵 $A$ 为例，取 $n=4$ 的情况. 
  $$
  A=
  \left[ 
  \begin{array}{c|c}
  \begin{matrix}
  4& -1 &\\
  -1&4 &-1\\
  &-1&4
  \end{matrix}
  &
  \begin{matrix}
  -1 & & \\
  & -1 & \\
  && -1
  \end{matrix}
  &
  \begin{matrix}
  \ \ \ \ \  & & \\
  & \ \ \ \ \ & \\
  && \ \ \ \ \ 
  \end{matrix}\\
  
  \hline
  \begin{matrix}
  -1 & & \\
  & -1 & \\
  && -1
  \end{matrix}
  &
  \begin{matrix}
  4& -1 &\\
  -1&4 &-1\\
  &-1&4
  \end{matrix}
  &
  \begin{matrix}
  -1 & & \\
  & -1 & \\
  && -1
  \end{matrix}\\
  
  \hline
  \begin{matrix}
  \ \ \ \ \  & & \\
  & \ \ \ \ \ & \\
  && \ \ \ \ \ 
  \end{matrix}
  &
  \begin{matrix}
  -1 & & \\
  & -1 & \\
  && -1
  \end{matrix}
  &
  \begin{matrix}
  4& -1 &\\
  -1&4 &-1\\
  &-1&4
  \end{matrix}
  
  \end{array}
  \right]
  $$
  我们取集合 $\{1,\dots,9\}$ 的划分为:  
  $$
  S_1 = \{1\}\quad S_2 = \{2,4\}\quad S_3 = \{3,5,7\}\quad S_4 = \{6,8\}\quad S_5 = \{9\}
  $$
  可以验证这个划分是满足定义的，也就是说上述矩阵具有相容性次序.

对于具有相容性次序且对角元非零的矩阵，亦可建立超松弛理论.    
这是因为它可以按照 "相容性次序" 定义中的 $\{1,\dots,n\}$ 的划分 $S_1,\dots,S_m$   
重排得到**数值线性代数 定理 $4.4.5$** 中描述的对角块均为非奇异对角阵的块三对角阵.

**(数值线性代数, 定理 $4.4.6$)**  
设线性方程组 $Ax=b$ 的系数矩阵 $A$ 具有相容性次序，且对角元均不为零.  
设 $\begin{cases}
B^{(1)} = D^{-1}(L+U)\\
B^{(2)} = (D-L)^{-1} U\\
B_\omega = (D-\omega L)^{-1}[(1-\omega)D+\omega U]\end{cases}$ 分别为 Jacobi 迭代法、Gauss-Seidel 迭代法和超松弛迭代法的迭代矩阵.  

若 $B^{(1)}$ 的特征值均为实数，则我们有:

- **Jacobi 迭代矩阵和超松弛迭代矩阵的特征值之间的关系:**  
  若 $\mu\neq 0$ 是 Jacobi 迭代矩阵 $B^{(1)}$ 的特征值，  
  则由 $\mu = \pm \frac{\lambda +\omega -1}{\omega \lambda^{\frac12}}$ (即二次方程 $\lambda+\omega -1 = \pm \mu \omega \lambda^{\frac12}$) 可得松弛迭代矩阵 $B_\omega$ 的两个特征值分别为:  
  $$
  \lambda_+(\omega,\mu) = [\frac{\mu \omega}{2} + \sqrt{(\frac{\mu \omega}{2})^2 - (\omega-1)}]^2\\
  \lambda_-(\omega,\mu) = [\frac{\mu \omega}{2} - \sqrt{(\frac{\mu \omega}{2})^2 - (\omega-1)}]^2
  $$
  反过来，若 $\lambda\neq 0$ 是超松弛迭代迭代 $B_\omega$ 的特征值，  
  则 $\mu = \pm \frac{\lambda +\omega -1}{\omega \lambda^{\frac12}}$ 是 Jacobi 迭代矩阵 $B^{(1)}$ 的两个特征值.   
  (因此若 $\mu\neq 0$ 是 $B^{(1)}$ 的特征值，则 $-\mu$ 也是 $B^{(1)}$ 的特征值)

  **特殊地，当 $\omega = 1$​ 时，我们得到 Jacobi 迭代矩阵和 Gauss-Seidel 迭代矩阵的特征值之间的关系:**    
  $\lambda \neq 0$ 是 Gauss-Seidel 迭代矩阵 $B^{(2)}=(D-L)^{-1} U$ 的非零特征值，  
  当且仅当 $\mu=\lambda^\frac12$ 是 Jacobi 迭代矩阵 $B^{(1)}=D^{-1} (L+U)$ 的非零特征值.

  因此谱半径具有关系: $\rho(B^{(2)}) = (\rho(B^{(1)}))^2$   
  渐近收敛速度 $R_\infty(B^{(2)}) = -\log({\rho(B^{(2)})}) = -2\log({\rho(B^{(1)})}) = 2R_\infty(B^{(1)})$ 

- **最佳松弛因子的相关陈述:**  
  若 Jacobi 迭代矩阵的谱半径 $\rho(B^{(1)})<1$ 且松弛因子 $0<\omega<2$，  
  则超松弛迭代法收敛，且最佳松弛因子为 $\omega_{\text{opt}} = \frac{2}{1+\sqrt{1-(\rho(B^{(1)}))^2}}$   
  对应的谱半径 $\rho(B_{\omega_{\text{opt}}}) = \omega_{\text{opt}} -1 = \frac{1-\sqrt{1-(\rho(B^{(1)}))^2}}{1+\sqrt{1-(\rho(B^{(1)}))^2}}$ 

  当 $\rho(B^{(1)})$ 从下方趋近于 $1$ 时，我们有渐近收敛速度 $R_\infty(B_{\omega_{\text{opt}}}) \sim 2\sqrt{2 R_\infty(B^{(1)})}$ 

***

最后我们给出如下定理:  
**(数值线性代数, 定理 $4.4.7$)** 
若线性方程组 $Ax=b$ 的系数矩阵 $A$ 对称正定，且具有相容性次序，  
则 Jacobi 迭代矩阵 $B^{(1)} = D^{-1}(L+U) = I-D^{-1}A$ 的特征值均为实数，且谱半径 $\rho(B^{(1)})<1$ 

- 回顾之前的记号:
  $$
  A = D-L-U\\
  D=
  \begin{bmatrix}
  a_{11} & & & \\
  & a_{22} & & \\
  &&\ddots &\\
  &&& a_{nn}\end{bmatrix}
  \qquad
  L
  = 
  \begin{bmatrix}
  0 & & &\\
  -a_{21} & 0 &&\\
  \vdots & \ddots & \ddots &\\
  -a_{n1} & \dotsm & -a_{n,n-1} & 0
  \end{bmatrix}
  \qquad
  U
  =
  \begin{bmatrix}
  0 & -a_{12} & \dotsm & -a_{1n}\\
  & 0 &\ddots& \vdots\\
  &&\ddots &-a_{n-1,n}\\
  &&& 0 
  \end{bmatrix}
  $$

- **证明:**  
  因为 $A$ 为正定阵，所以 $D=\text{diag}(a_{11},\dots,a_{nn})$ 为正定对角阵  
  于是 $B^{(1)} = I-D^{-1}A = D^{-\frac12}(I-D^{-\frac12}A D^{-\frac12}) D^{\frac12}$   
  表明 $B^{(1)}$ 与对称阵 $I-D^{-\frac12}A D^{-\frac12}$ 相似，因而 $B^{(1)}$ 的特征值均为实数.

  > **(数值线性代数, 定理 $4.2.7$)**  
  > 若 $A$ 对称正定，则 Gauss-Seidel 迭代法收敛.

  我们知道 Gauss-Seidel 迭代法收敛  
  因此 Gauss-Seidel 迭代矩阵 $B^{(2)} = (D-L)^{-1}U$ 的谱半径 $\rho(B^{(2)})<1$     
  而根据**数值线性代数 定理 $4.4.6$** 的结论，我们有 $\rho(B^{(1)}) = \sqrt{\rho(B^{(2)})} <1$   
  表明 Jacobi 迭代法也收敛，命题得证.

**The End**
