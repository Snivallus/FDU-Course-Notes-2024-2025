# FDU 高等线性代数 3. 特征值

本文根据邵美悦老师授课内容整理而成，并参考了以下教材:   

* Matrix Analysis (R. Horn & C. Johnson) Chapter $1,3,6$
* 矩阵分析 (R. Horn & C. Johnson) 第 $1,3,6$ 章

欢迎批评指正!

## 3.1 基础知识

### 3.1.1 代数基本定理

我们想要知道:   
一元 $n$ 次复系数方程 $\lambda^n +a_{n-1}\lambda^{n-1}+...+a_1\lambda+a_0=0$ 在复数域 $\mathbb C$ 上一定可解吗?  
如果可解，有多少个解?

**代数基本定理** (fundamental theorem of algebra) 回答了这一系列问题:   

> **(Matrix Analysis 附录 C) & (Complex Variables and Applications 第 58 节 定理 2)**   
> **任何一元 $n\ (n\geq 1)$ 次复系数方程组都至少在复数域 $\mathbb C$ 上有一个解.**

接下来，根据存在的这个复根，可以利用长除法 (多项式带余除法) 将原方程降一阶，变为一个 $n-1$ 次方程. 
而这个 $n-1$ 次方程又至少存在一个复根，因而可以继续降阶.  
这样的过程可以一直进行下去，直到我们找到 $n$ 个复根.   
因此我们可以推出: 


> **任何一元 $n\ (n\geq 1)$ 次复系数方程组在复数域 $\mathbb C$ 上都有且仅有 $n$ 个解 (按重数计算).**

由于求解 $n$ 阶复方阵 $A\in \mathbb C^{n\times n}$ 特征值的本质是求解一元 $n$ 次复系数方程 $\det(\lambda I-A)=0$  
故又能推出: 


> **任意** $n$ **阶复方阵都有且仅有** $n$ **个复特征值 (按重数计算)** 

***

引入复数的一个历史原因是一元实系数方程可能在实数域 $\mathbb R$ 上没有解 (例如 $x^2+1 = 0$).  
幸运的是，任何一元实系数方程的所有解都包含在复数域 $\mathbb C$ 中.  
事实上，复数域 $\mathbb C$ 是一个**代数封闭的域** (algebraically closed field):  

不存在这样的域 $\mathbb F$，使得 $\mathbb C$ 是 $\mathbb F$ 的子域，且存在一个系数属于 $\mathbb C$ 的一元方程，它有一个解在 $\mathbb F$ 中但不在 $\mathbb C$ 中.



### 3.1.2 特征方程

设 $A\in \mathbb C^{n\times n}$  
若纯量 $\lambda\in \mathbb C$ 和非零向量 $x\in \mathbb C^n$ 满足方程 $Ax=x \lambda $，  
则我们称 $\lambda$ 是 $A$ 的一个**特征值** (eigenvalue)，而 $x$ 称为 $A$ 的一个与 $\lambda$ 相伴的**特征向量** (eigenvector).  
我们称元素对 $(\lambda,x)$ 为 $A$ 的一个**特征对** (eigenpair).

- 定义中的关键点是: 特征向量永远不会是零向量.
- 方程 $Ax=x \lambda $ 可以改写为正方的齐次线性方程组 $(A-\lambda I)x=0_n$  
  如果这个方程组有非平凡的解，那么 $\lambda$ 就是 $A$ 的一个特征值，而矩阵 $A-\lambda I$ 就是奇异的.
- 若 $x$ 是 $A\in \mathbb C^{n\times n}$ 的与 $\lambda$ 相伴的特征向量，那么将它标准化为 $\xi=\frac{x}{\|x\|_2}$ 通常更为方便.  
  值得注意的是，标准化得到的 $\xi=\frac{x}{\|x\|_2}$ 并不是与 $\lambda$ 相伴的唯一的标准特征向量  
  无论如何，$e^{i\theta}\xi\ (\forall\ \theta\in \mathbb R)$ 都是 $A$ 的与 $\lambda$ 相伴的标准特征向量.

***

给定复系数 $k\ (k\geq 1)$ 次多项式 $p(t)=a_k t^k + \dotsm + a_1 t + a_0\ (a_k\neq 0)$   
若 $a_k=1$，则我们成 $p(t)$ 是**首一的** (monic)    
由于 $a_k\neq 0$，故 $a_k^{-1}p(t)$ 总是 $k$ 次首一多项式.  
为方便起见，后面我们总是假定多项式是首一的.

我们可以定义其在 $A\in \mathbb C^{n\times n}$ 处的值为 (约定 $A^0=I$):  
$$
p(A) = A^k + a_{k-1}A^{k-1} \dotsm + a_1 A + a_0 I
$$
代数基本定理确保了 $k$ 次首一多项式 $p(t)$ 可以写成 $k$ 个线性因子的乘积:  
$$
p(t) = (t-\alpha_1)\dotsm (t-\alpha_k)
$$
每个 $\alpha_j$ 都是方程 $p(t)=0$ 的一个**根** (root)，即多项式 $p(t)$ 的一个**零点** (zero)  
因子 $(t-\alpha_j)$ 重复的次数就是 $\alpha_j$ 作为 $p(t)$ 零点的**重数** (multiplicity)  
我们同样可以给出 $p(A)$ 的分解式:
$$
p(A) = (A-\alpha_1 I)\dotsm (A-\alpha_k I)
$$
这样我们就将 $p(A)$ 的特征值与 $A$ 的特征值以一种简单的方式联系在一起.  
**(Matrix Analysis 定理 $1.1.6$)**  
设 $p(t)$ 是给定的 $k\ (k\geq 1)$ 次 (首一) 多项式.  

- 若 $(\lambda,x)$ 是 $A\in \mathbb C^{n\times n}$ 的一个特征对，则 $(p(\lambda),x)$ 就是 $p(A)$ 的特征对.
- 反过来，若 $\mu$ 是 $p(A)$ 的一个特征值，则存在 $A$ 的某个特征值 $\lambda$ 使得 $p(\lambda)=\mu$ 

***

我们定义由 $A\in \mathbb C^{n\times n}$ 的 $n$ 个特征值构成的集合为 $A$ 的**谱** (spectrum)，记为 $\text{eig}(A)$   
(代数基本定理保证了复方阵 $A\in \mathbb C^{n\times n}$ 有且仅有 $n$ 个复特征值 (按重数计算))  
我们定义 $A\in \mathbb C^{n\times n}$ 的 $n$ 个特征值的模最大值为 $A$ 的**谱半径** (spectrum radius)，记为 $\rho(A):= \max\{|\lambda|:\lambda\in \text{eig}(A)\}$ 

- 注意到 $Ax=x \lambda \ \Leftrightarrow\ \bar A \bar x = \bar\lambda \bar x$，因此我们有 $\text{eig}(\bar A) = \overline{\text{eig}(A)}$ 
- $A\in \mathbb C^{n\times n}$ 是奇异的，当且仅当 $0\in \text{eig}(A)$ 
- 给定 $A\in \mathbb C^{n\times n}$ 和 $\lambda,\mu \in \mathbb C$，则 $\lambda\in \text{eig}(A)$ 当且仅当 $\lambda+\mu \in \text{eig}(A+\mu I)$ 
- **(Matrix Analysis 定理 $1.2.17$)**  
  设 $A\in \mathbb C^{n\times n}$，则存在某个 $\delta >0$ 使得对于任意 $0<|\varepsilon|<\delta$，矩阵 $A+\varepsilon I$ 都是非奇异的.  
  (这个结论是显然的，因为 $A$ 仅有有限个特征值，所以它们之间有空隙)



### 3.1.3 特征多项式

给定 $A\in \mathbb C^{n\times n}$  
将特征方程 $Ax=x \lambda \ (x\neq 0_n)$ 改写为 $(\lambda I-A)x=0_n\ (x\neq 0_n)$  
从而 $\lambda\in \text{eig}(A)$ 当且仅当 $\lambda I-A$ 是奇异的，即当且仅当 $\det(\lambda I-A) = 0$   
我们称 $p_A(t)=\det(t I-A)$ 为 $A\in \mathbb C^{n\times n}$ 的**特征多项式** (characteristic polynomial)  
我们称 $p_A(t)=0$ 为 $A\in \mathbb C^{n\times n}$ 的**特征方程** (characteristic equation)

$A$ 的特征值 $\lambda$ 的**代数重数** (algebraic multiplicity) 是指它作为特征多项式 $p_A(t)$ 的零点的重数.  
从现在开始，$A\in \mathbb C^{n\times n}$ 的特征值总是指这个特征值和相应的代数重数的合并称谓.

- **(Matrix Analysis 定理 $1.2.18$)**    
  设 $A\in \mathbb C^{n\times n}$ 并假设特征值 $\lambda$ 的代数重数为 $k$，则 $n-1\geq\rank(A-\lambda I)\geq n-k$   
  (显然其不等号都至少在 $k=1$ 时取等)  

***

**(Vieta 定理, Matrix Analysis 定理 $1.2.16$)**  
设复方阵 $A\in \mathbb C^{n\times n}$ 的特征值为 $\lambda_1,\dots,\lambda_n$   
则 $\lambda_1,\dots,\lambda_n$ 的任意 $k=1,\dots,n$ 次初等对称多项式等于 $A$ 的所有 $k$ 阶主子式 (这样的主子阵一共有 $\binom{n}{k}$ 个) 之和:  
$$
\sum_{1\leq i_1<\dotsm <i_k \leq n} \lambda_{i_1}\dotsm \lambda_{i_k} = \sum_{1\leq i_1<\dotsm <i_k \leq n} \det(A_{i_1,\dots,i_k})
$$
其中 $A_{i_1,\dots,i_k}$ 代表由 $A$ 的第 $i_1,\dots,i_k$ 行、列构成的 $k$ 阶主子阵.

- 对于 $k=1$ 的情况: $\underset{i=1}{\overset{n}\sum} \lambda_i = \tr(A)$
- 对于 $k=n-1$ 的情况: $\underset{i=1}{\overset{n}\sum} \underset{j\neq i}{\overset{n}\prod} \lambda_j = \tr(\text{adj}(A))$   
  (其中 $\text{adj}(A)=\{[(-1)^{i+j}\det(A[\{i\}^c,\{j\}^c])]_{i,j=1}^n\}^{\mathrm T}$ 为 $A$ 的伴随矩阵, 是 $A$ 所有代数余子式构成矩阵的转置)
- 对于 $k=n$ 的情况: $\underset{i=1}{\overset{n}\prod} \lambda_i = \det(A)$ 



### 3.1.4 Cayley-Hamilton 定理

**(Matrix Analysis 引理 $2.4.3.1$)**  
考虑形如 $R=\begin{bmatrix}
0_{k\times k} & R_{12}\\
0_{(n-k)\times k} & R_{22}\end{bmatrix},
\ T= \begin{bmatrix}
T_{11} & T_{12}\\
0_{(n-k)\times k} & T_{22}\end{bmatrix}$ 的上三角阵的乘积.  
其中分块 $R_{22},T_{11},T_{22}$ 都是上三角阵，且分块 $T_{22}$ 的左上角元素也为 $0$，即 $t_{k+1,k+1}=0$  
记 $T_{22} = [0_{n-k}\ \ \widetilde T_{22}]$，则我们有:
$$
\begin{align}
RT
&= 
\begin{bmatrix}
0_{k\times k} & R_{12}\\
0_{(n-k)\times k} & R_{22}\end{bmatrix}

\begin{bmatrix}
T_{11} & T_{12}\\
0_{(n-k)\times k} & T_{22}\end{bmatrix}\\

&=

\begin{bmatrix}
0_{k\times k} & R_{12} T_{22}\\
0_{(n-k)\times k} & R_{22} T_{22}
\end{bmatrix}\\

&=
\begin{bmatrix}
0_{k\times k} & R_{12} [0_{n-k}\ \ \widetilde T_{22}]\\
0_{(n-k)\times k} & R_{22} [0_{n-k}\ \ \widetilde T_{22}]
\end{bmatrix}\\

&=
\begin{bmatrix}
0_{k\times k} & 0_{k} & R_{12}\widetilde T_{22}\\
0_{(n-k)\times k} & 0_{n-k} & R_{22}\widetilde T_{22}
\end{bmatrix}\\

&=
\begin{bmatrix}
0_{(k+1)\times (k+1)} & R_{12}\widetilde T_{22}\\
0_{(n-k-1)\times (k+1)} & R_{22}\widetilde T_{22}
\end{bmatrix}\\

\end{align}
$$
即 $RT$ 在左上角有一个 $k+1$ 阶的全零子矩阵，比 $R$ 左上角的 $k$ 阶全零子矩阵高了一阶.

****

任意复方阵都满足其特征方程.  
**(Cayley-Hamilton 定理, Matrix Analysis 定理 $2.4.3.2$)**    
设 $p_A(t):=\det(tI_n- A)$ 是 $A\in \mathbb C^{n\times n}$ 的特征多项式，则我们有 $p_A(A) = 0_{n\times n}$ 成立.

- **证明:**  
  设 $A$ 的特征值为 $\lambda_1,\dots,\lambda_n$，则特征多项式可表示为 $p_A(t)= (t-\lambda_1)\dotsm (t-\lambda_n)$​     

  设 $A$ 的 Schur 分解是 $U^{\mathrm H} A U= T$   
  其中 $U\in \mathbb C^{n\times n}$ 是酉矩阵，$T$ 是对角元为 $A$ 的特征值 $\lambda_1,\dots,\lambda_n$ 的上三角阵.     
  由于 $p_A(A) = p_A(UTU^{\mathrm H}) = Up_A(T) U^{\mathrm H}$，故要证明 $p_A(A) = 0_{n\times n}$，只需证明 $p_A(T)=0_{n\times n}$ 即可.

  考虑 $p_A(T)=(T-\lambda_1 I)\dotsm (T-\lambda_n I)$   

  - 注意到 $T-\lambda_1 I$ 左上角的 $1\times 1$ 分块是 $0$，$T-\lambda_2 I$ 的 $(2,2)$ 位置是 $0$  
    根据 **Matrix Analysis 引理 $2.4.3.1$** 可知 $(T-\lambda_1 I)(T-\lambda_2 I)$ 左上角的 $2\times 2$ 分块是全零矩阵.
  - 注意到 $T-\lambda_3 I$ 的 $(3,3)$ 位置是 $0$   
    根据 **Matrix Analysis 引理 $2.4.3.1$** 可知 $(T-\lambda_1 I)(T-\lambda_2 I)(T-\lambda_3 I)$ 左上角的 $3\times 3$ 分块是全零矩阵.
  - **(归纳法)** 假定 $(T-\lambda_1 I)\dotsm (T-\lambda_k I)$ 左上角的 $k\times k$ 是全零矩阵  
    注意到 $T-\lambda_{k+1} I$ 的 $(k+1,k+1)$ 位置是 $0$   
    根据 **Matrix Analysis 引理 $2.4.3.1$** 可知 $(T-\lambda_1 I)\dotsm (T-\lambda_n I)$ 左上角的 $(k+1)\times (k+1)$ 分块是全零矩阵.

  根据归纳原理我们得出 $p_A(T)=(T-\lambda_1 I)\dotsm (T-\lambda_n I)$ 为全零矩阵 $0_{n\times n}$.  
  定理得证.
  
- **邵老师提供的证明:**  
  我们定义:  
  $$
  A(\varepsilon) := A + \varepsilon \Delta A\ (\forall\ \varepsilon \in (0,1))
  \text{ where }
  \begin{cases}
  A = QTQ^{\mathrm H}\\
  \Delta A := Q\text{diag}\{0,1,\dots,n-1\} Q^{\mathrm H} \cdot \delta\\ 
  \delta < \frac{1}{n} \min\{|\lambda_i -\lambda_j|: 1\leq i< j\leq n\}
  \end{cases}
  $$
  则对于任意 $\varepsilon\in (0,1)$，$\varepsilon \Delta A$ 可使 $A$ 的重特征值分裂，同时不相同的特征值碰不到一块.  
  因此 $A(\varepsilon) = A+\varepsilon \Delta A$ 的 $n$ 个特征值 $\lambda_1(\varepsilon),\dots, \lambda_n(\varepsilon)$ 互不相同，从而可以酉对角化  
  即存在酉矩阵 $U(\varepsilon)\in \mathbb C^{n\times n}$ 使得:
  $$
  A(\varepsilon) = U(\varepsilon) \Lambda(\varepsilon) U(\varepsilon)^{\mathrm H}\text{ where }\Lambda(\varepsilon) = 
  \text{diag}\{\lambda_1(\varepsilon),\dots,\lambda_n(\varepsilon)\}
  $$
  
  其特征多项式为: $p_{A(\varepsilon)}(t) = (t-\lambda_1(\varepsilon))\dotsm (t-\lambda_n(\varepsilon))$   
  它显然可以零化 $A(\varepsilon)$:  
  $$
  \begin{align}
  p_{A(\varepsilon)}(A(\varepsilon)) 
  &= (A(\varepsilon)-\lambda_1(\varepsilon)I_n)\dotsm (A(\varepsilon)- \lambda_n(\varepsilon) I_n)\\
  &= (U(\varepsilon) \Lambda(\varepsilon) U(\varepsilon)^{\mathrm H} - \lambda_1(\varepsilon) I_n) \dotsm 
  (U(\varepsilon)\Lambda(\varepsilon) U(\varepsilon)^{\mathrm H} - \lambda_n(\varepsilon) I_n)\\
  &=
  U(\varepsilon)[(\Lambda(\varepsilon)-\lambda_1(\varepsilon)I_n)\dotsm (\Lambda(\varepsilon)- \lambda_n(\varepsilon) I_n)] U(\varepsilon)^{\mathrm H}\\
  &=
  U(\varepsilon)\cdot 0_{n\times n}\cdot U(\varepsilon)^{\mathrm H}\\
  &=
  0_{n\times n}
  \end{align}
  $$
  令 $\varepsilon \to 0$ 即有:  
  $$
  p_A(A) = \lim_{\varepsilon\to 0}p_{A(\varepsilon)}(A(\varepsilon)) = 0_{n\times n}
  $$
  这就证明了 Cayley-Hamilton 定理.
  

****

Cayley-Hamilton 定理的重要应用是将 $A\in \mathbb C^{n\times n}$ 的高次幂 $A^k\ (k\geq n)$ 写成 $I,A,\dots,A^{n-1}$ 的线性组合.  

- 以 $A=\begin{bmatrix}
  3 & 1\\
  -2 & 0\end{bmatrix}$ 为例:  
  其特征多项式 $p_A(t) = \det(tI-A) = t^2 - 3t + 2$   
  根据 Cayley-Hamilton 定理我们有 $p_A(A) = A^2 -3A + 2I = 0_{2\times 2}$   
  我们可将 $A$ 的 $k\geq 2$ 次幂写成 $I,A$ 的线性组合:  
  $$
  A^2 = 3A-2I = 
  \begin{bmatrix}
  7 & 3\\
  -6 & -2
  \end{bmatrix}\\
  A^3 = A(A^2) = A(3A-2I) = 3A^2-2A = 3(3A-2I) - 2A = 7A-6I
  =\begin{bmatrix}
  15 & 7\\
  -14 & -6
  \end{bmatrix}\\
  A^4 = A(A^3) = A(7A-6I) = 7A^2 - 6A = 7(3A-2I) - 6A = 15 A - 14 I
  =
  \begin{bmatrix}
  31 & 15\\
  -30 & -14
  \end{bmatrix}
  $$

Cayley-Hamilton 定理还可将非奇异阵 $A\in \mathbb C^{n\times n}$ 的负幂 $A^k\ (k\leq -1)$ 写成 $I,A,\dots,A^{n-1}$ 的线性组合.    
**(Matrix Analysis 推论 $2.4.3.4$)**  
设 $A\in \mathbb C^{n\times n}$ 非奇异  
记其特征多项式 $p_A(t) = \det(tI-A) = t^n + c_{n-1}t^{n-1} + \dotsm + c_1 t + c_0$    
根据 Cayley-Hamilton 定理我们有 $p_A(A) = A^n + c_{n-1}A^{n-1} + \dotsm + c_1 A + c_0 I_n = 0_{n\times n}$     
则 $A^{-1} = -\frac{1}{c_0}(A^{n-1} + c_{n-1} A^{n-2} + \dotsm + c_2 A + c_1)$

- 以 $A=\begin{bmatrix}
  3 & 1\\
  -2 & 0\end{bmatrix}$ 为例:  
  其特征多项式 $p_A(t) = \det(tI-A) = t^2 - 3t + 2$   
  根据 Cayley-Hamilton 定理我们有 $p_A(A) = A^2 -3A + 2I = 0_{2\times 2}$     
  我们可将 $A$ 的 $k\leq -1$ 次幂写成 $I,A$ 的线性组合:    
  $$
  A^{-1} = -\frac12(A-3I) = 
  \begin{bmatrix}
  0 & -\frac12\\
  1 & \frac32\end{bmatrix}\\
  
  A^{-2} = (A^{-1})^2 = [-\frac12(A-3I)]^2 = \frac14(A^2 - 6A + 9I)
  =
  \frac14(3A-2I - 6A + 9I) 
  =\frac14(-3A + 7I) = 
  \frac14\begin{bmatrix}
  -2 & -3\\
  6 & 7\end{bmatrix}\\
  
  A^{-3} =  (A^{-2})A^{-1} = \frac14(-3A + 7I)\cdot -\frac12(A-3I)
  = -\frac18(-3A^2 + 16 A - 21 I) = -\frac18(7A -15 I) 
  = -\frac18 \begin{bmatrix}
   6 & 7\\
   -14 & -15
  \end{bmatrix}
  $$

***

我们再来看一个例子，它说明特征多项式并不一定是 $A$ 所能满足的最低次的多项式方程.  
考虑 $A = \begin{bmatrix}
1 & &\\
&1 & 1\\
&&1\end{bmatrix}$   
其特征多项式 $p_A(t) = \det(tI-A) = (t-1)^3$   
根据 Cayley-Hamilton 定理我们有 $p_A(A) = (A-I)^3 = 0_{3\times 3}$      
但是可以验证 $(A-I)^2= 0_{3\times 3}$，即 $A$ 满足 $2$ 阶多项式 $p_2(t) = (t-1)^2$  
不过 $A$ 不可能满足形如 $p_1(t) = t + c_0$ 的 $1$ 阶多项式，因为不存在 $c_0$ 使得 $A+c_0 I = 0_{3\times 3}$ 

*****

最后我们看一个例子，它说明复方阵 $A\in \mathbb C^{n\times n}$ 的伴随矩阵 $\text{adj}(A)$ 可以表示为 $A$ 的不超过 $n-1$ 次的多项式.  
考虑复方阵 $A\in \mathbb C^{n\times n}$   
记其特征多项式 $p_A(t) = \det(tI-A) = t^n + c_{n-1}t^{n-1} + \dotsm + c_1 t + c_0$    
根据 Cayley-Hamilton 定理我们有 $p_A(A) = A^n + c_{n-1}A^{n-1} + \dotsm + c_1 A + c_0 I_n = 0_{n\times n}$   

- ① 若 $A$ 非奇异，则根据 **Matrix Analysis 推论 $2.4.3.4$** 可知 $A^{-1}$ 可表示为 $A$ 的至多 $n-1$ 次的多项式.  
  具体来说是 $A^{-1} = -\frac{1}{c_0}(A^{n-1} + c_{n-1} A^{n-2} + \dotsm + c_2 A + c_1)$   
  则 $\text{adj}(A)=\det(A) A^{-1}$ 可以表示为 $A$ 的至多 $n-1$ 次的多项式.

- ② 若 $A$ 奇异，则可对其做任意小扰动 $\varepsilon I_n$ 得到非奇异阵 $A+\varepsilon I_n$ 
  根据 ① 可知存在至多 $n-1$ 次的多项式 $g_\varepsilon(t)$ 使得 $\text{adj}(A+\varepsilon I_n) = g_\varepsilon(A + \varepsilon I_n)$   
  我们断言 $g_\varepsilon(t)$ 的系数均为关于 $\varepsilon$ 的连续函数，  
  因此存在一个至多 $n-1$ 次的多项式 $g(t)$ 使得 $\lim_{\varepsilon\to 0} g_\varepsilon(t) = g(t)\ (\forall\ t\in \mathbb C)$ **(这个收敛是一致的?)**  
  注意到 $\text{adj}(A+\varepsilon I_n)$ 的所有元素也都是关于 $\varepsilon$ 的连续函数，故我们有 $\lim_{\varepsilon\to 0} \text{adj}(A+\varepsilon I_n) = \text{adj}(A)$  
  于是我们有:  
  $$
  \begin{align}
  \text{adj}(A) 
  &= 
  \lim_{\varepsilon\to 0} \text{adj}(A+\varepsilon I_n)\\
  &=
  \lim_{\varepsilon\to 0}g_\varepsilon(A + \varepsilon I_n)\\
  &=
  g(A)
  \end{align}
  $$
  **(存疑: 最后一步似乎有点悬)**

综上所述，命题得证.



### 3.1.5 相似性

每一个可逆矩阵都是一个可以变换基的矩阵，反之亦然.  
设 $\mathcal B$ 是域 $\mathbb F$ 上 $n$ 维向量空间 $V$ 的一组给定的基，$T$ 是 $V$ 上的线性变换，$A=[T]_{\mathcal B\to \mathcal B}$ 是 $T$ 在基 $\mathcal B$ 上的表示矩阵.   
设 $\mathcal B_1$ 是 $V$ 的另一组基，$A_1 = [T]_{\mathcal B_1\to \mathcal B_1}$ 是 $T$ 在基 $\mathcal B_1$ 上的表示矩阵，则我们有 $\begin{cases}
T(\mathcal B)=\mathcal B A\\
T(\mathcal B_1) = \mathcal B_1 A_1\end{cases}$   
设 $S=[I]_{\mathcal B\to \mathcal B_1}$ 是基 $\mathcal B$ 到基 $\mathcal B_1$ 的过渡矩阵，即满足 $\mathcal B S = \mathcal B_1$   
则我们有:
$$
\begin{align}
\mathcal B S A_1
&=
\mathcal B_1 A_1\\
&=
T(\mathcal B_1)\\
&=
T(\mathcal B S)\\
&=
T(\mathcal B)S\\
&=
(\mathcal B A)S\\
\end{align}\ \Rightarrow\ SA_1 = AS\ \Rightarrow\ A_1 = S^{-1}AS
$$
由于 $\mathcal B_1$ 可以是 $V$ 的任意一组基，故 $T$ 的所有可能的表示矩阵的集合为:  
$$
\begin{align}
\{A_1 = [T]_{\mathcal B_1\to \mathcal B_1}:\mathcal B_1\text{ is a basis of }V\}
&=
\{[I]_{\mathcal B_1\to \mathcal B} [T]_{\mathcal B\to \mathcal B} [I]_{\mathcal B\to \mathcal B_1}:\mathcal B_1\text{ is a basis of }V\} \\
&= 
\{S^{-1}AS:S\in \mathbb F^{n\times n}\text{ is non-singular}\}
\end{align}
$$
这恰好是所有与给定矩阵 $A$ 相似的矩阵的集合.  
这样一来，相似但不相等的矩阵都是单个线性变换在不同基上的表示矩阵.   
我们期待相似矩阵共享许多重要的性质 (至少是那些反映线性变换本质的性质).

***

给定 $A,B\in \mathbb C^{n\times n}$  
若存在一个非奇异阵 $S\in \mathbb C^{n\times n}$ 使得 $B=S^{-1}AS$，则我们称 $B$ **相似** (similar) 于 $A$，记为 $B\sim A$  
变换 $A\mapsto S^{-1}AS$ 称为由非奇异阵 $S$ 给出的相似变换.  
若存在一个排列矩阵 $P$ 使得 $B=P^{\mathrm T}AP$，则我们称 $B$ **排列相似** (permutation similar) 于 $A$.

相似关系是 $\mathbb C^{n\times n}$ 上的等价关系:

- 自反性: $A\sim A$
- 对称性: $A\sim B\ \Leftrightarrow\ B\sim A$ (因此以后我们将 "$B$ 相似于 $A$" 称为 "$A,B$ 相似")
- 传递性: $\begin{cases}
  A\sim B\\
  B\sim C\end{cases}\ \Rightarrow\ A\sim C$ 

因此相似关系将 $\mathbb C^{n\times n}$ 划分为不相交的等价类.  
一个相似等价类中所有的矩阵都是相似的，不同相似等价类中的矩阵都是不相似的.  
一个相似等价类中的矩阵共同享有许多重要的性质.

**(Matrix Analysis 定理 $1.3.3$ & 推论 $1.3.4$)**  
设 $A,B\in \mathbb C^{n\times n}$  
若 $A,B$ 相似 (即存在非奇异阵 $S\in \mathbb C^{n\times n}$ 使得 $A=S^{-1}BS$)，则我们有:

- ① $A,B$ 具有相同的特征多项式，即 $p_A(t)=\det(tI-A) = \det(tI-B) = p_B(t)$ 
  $$
  \begin{align}
  p_A(t)
  &=\det(t I-A)\\
  &= \det(tS^{-1}S - S^{-1}BS)\\
  &= \det(S^{-1}(tI-B)S)\\
  &= \det(S^{-1}) \det(t I-B) \det(S)\\
  &= \frac{1}{\det(S)}\det(t I-B)\det(S)\\
  &= \det(tI-B)\\
  &= p_B(t)
  \end{align}
  $$

- ② $A,B$ 具有相同的特征值 (包括重数)

- ③ 若 $A$ 是一个对角阵，则其主对角元就是它的特征值

- ④ $B=0_{n\times n}$ 当且仅当 $A=0_{n\times n}$ 

- ⑤ $B$ 为对角阵当且仅当 $A=I_n$ 

- ⑥ 对于任意 $\alpha\in \mathbb C$，$A+\alpha I_n$ 与 $B+\alpha I_n$ 也相似

- ⑦ 对于任意多项式 $q(t)$，$q(A)$ 与 $q(B)$ 也相似

- ⑧ $\begin{cases}
  \rank(B)=\rank(A)\\
  \tr(B)=\tr(A)\\
  \det(B)=\det(A)\end{cases}$ 这表明秩、迹和行列式都是**相似不变量** (similarity invariant)  
  实际上，根据 Vieta 定理可知 $A,B$ 的 $\binom{n}{k}$ 个 $k$ 阶主子式之和也是相似不变量.  
  我们后面会对相似不变量 (例如 Jordan 标准型) 进行完整描述.

****

**(Matrix Analysis 引理 $1.3.28$)**    
若将非奇异阵 $S\in \mathbb C^{n\times n}$ 拆分为 $S=S_1 + i S_2$ (其中 $S_1,S_2\in \mathbb R^{n\times n}$)  
则存在实数 $\tau\in \mathbb R$ 使得 $\widetilde S := S_1 + \tau S_2$ 是非奇异的.   
**证明:**  

- 若 $S_1$ 非奇异，则可取 $\tau = 0$  
- 若 $S_1$ 奇异，考虑多项式 $p(t)=\det(S_1+tS_2)$   
  注意到 $p(t)$ 不是常数多项式，因为 $\begin{cases}
  p(0) = \det(S_1) =0\\
  p(i) = \det(S_1+iS_2) = \det(S) \neq 0\end{cases}$   
  由于 $p(t)$ 在复平面上仅有有限个零点，故我们总能找到一个实数 $\tau\in \mathbb R$ 使得 $p(\tau) = \det(S_1 + \tau S_2)\neq 0$   
  这表明实矩阵 $\widetilde S := S_1 + \tau S_2$ 是非奇异的.

**(Matrix Analysis 定理 $1.3.29$)**   
两个在复数域上相似的实矩阵在实数域上也是相似的.  
**证明:**  
若 $A,B\in \mathbb{R}^{n\times n}$ 在复数域 $\mathbb{C}$ 上相似，  
则存在非奇异矩阵 $S\in \mathbb{C}^{n\times n}$，使得 $B=S^{-1}AS$，即 $AS=SB$.  
将 $S$ 拆分为 $S=S_1 + iS_2$，其中 $S_1,S_2\in \mathbb{R}^{n\times n}$  
则 $AS = A(S_1+iS_2) = (S_1+iS_2)B=SB$   
比较实部、虚部可得 $\begin{cases} AS_1 = S_1B\\ AS_2=S_2B \end{cases}$   

根据 Matrix Analysis 引理 $1.3.28$ 可知，存在实数 $\tau \in \mathbb{R}$ 使得实矩阵 $\widetilde S = S_1+\tau S_2$ 是非奇异矩阵.  
则 $A\widetilde S = A(S_1+\tau S_2) = AS_1+\tau AS_2 = S_1B+\tau S_2B = \widetilde SB$   
于是 $B=\widetilde S^{-1}A\widetilde S$，即 $A,B$ 在实数域 $\mathbb{R}$ 上相似.  
定理得证.

****

由于对角阵特别简单而且有很好的性质，  
故我们希望知道什么样的矩阵与对角阵相似，即**可对角化** (diagonalizable).    
**(Matrix Analysis 定理 $1.3.7$)**  
给定 $A\in \mathbb C^{n\times n}$ 

- ① 当且仅当 $A$ 具有 $k$ 个线性无关的特征向量 $x_1,\dots,x_k$ 时，$A$ 与一个形如 $\begin{bmatrix}
  \Lambda & B\\
  & C\end{bmatrix}$ 的矩阵相似  
  其中相似矩阵 $S=[x_1,\dots,x_k,s_{k+1},\dots,s_n]$ 的列向量是由 $x_1,\dots,x_k$ 扩充成的 $\mathbb C^n$ 的一组基  
  对角块 $\Lambda=\text{diag}\{\lambda_1,\dots,\lambda_k\}\ (k=1,\dots,n-1)$ (其中 $\lambda_1,\dots,\lambda_k$ 是 $A$ 的特征值)

  此时我们有 $p_A(t)=p_{\Lambda}(t)p_C(t)$ 成立.

- ② 当且仅当 $A$ 具有 $n$ 个线性无关的特征向量 $x_1,\dots,x_n$ 时，$A$ 可对角化  
  其中相似矩阵 $S=[x_1,\dots,x_n]$ 且 $S^{-1}AS = \Lambda =\text{diag}\{\lambda_1,\dots,\lambda_n\}$ (其中 $\lambda_1,\dots,\lambda_n$ 是 $A$ 的特征值)

  此时我们有 $p_A(t)=p_{\Lambda}(t)$ 成立.
  
  原则上讲，命题 ② 给出了对一个可对角化矩阵 $A$ 进行对角化的算法:  
  求出 $A$ 的 $n$ 个特征值，并求出 $n$ 个与之相伴且线性无关的特征向量 $x_1,\dots,x_n$  
  得到 $S=[x_1,\dots,x_n]$ 并计算 $S^{-1}$，最终得到 $\Lambda=S^{-1}AS$   
  不幸的是，这通常不是一种实用的算法.

**(Matrix Analysis 引理 $1.3.8$)**  
设 $\lambda_1,\dots,\lambda_k$ 是 $A\in \mathbb C^{n\times n}$ 的 $k\geq 2$ 个不同的特征值.  
若 $x^{(1)},\dots,x^{(k)}$ 分别是与 $\lambda_1,\dots,\lambda_k$ 相伴的特征向量，则它们是线性无关的.

**(可对角化的一个充分条件, Matrix Analysis 定理 $1.3.9$)**  
如果 $A\in \mathbb C^{n\times n}$ 所有特征值都不相同，则它一定可以对角化.

*****

若 $A\in \mathbb C^{n\times n}$ 可对角化，且 $A=S\Lambda S^{-1}$，则对于任意 $\alpha\neq 0$，$\alpha S$ 也都可以使 $A$ 对角化.  
因此对角化相似从来都不是唯一的.  
尽管如此，$A$ 与一个特殊的对角阵的每一个相似都可由一个给定的相似得出.

**(Matrix Analysis 定理 $1.3.27$)**  
设 $A\in \mathbb C^{n\times n}$ 可以对角化，且 $\mu_1,\dots,\mu_d$ 是它的不同的特征值，相应的重数分别为 $n_1,\dots,n_d$  
设 $S,T\in \mathbb C^{n\times n}$ 为非奇异阵，又假设 $A=S\Lambda S^{-1}$，其中 $\Lambda= \mu_1 I_{n_1}\oplus \dotsm \oplus \mu_d I_{n_d}$ 

- ① $A=T\Lambda T^{-1}$ 当且仅当存在 $d$ 个非奇异矩阵 $R_i\in \mathbb C^{n_i\times n_i}$ 使得 $T=S(R_1\oplus \dotsm \oplus R_d)$ 
- ② 若将 $S,T$ 与 $\Lambda$ 共形地划分为 $\begin{cases}
  S=[S_1,\dots,S_d]\\
  T=[T_1,\dots,T_d]\end{cases}$   
  则 $A=S\Lambda S^{-1}=T\Lambda T^{-1}$ 当且仅当对于每个 $i=1,\dots,d$，都有 $\text{Range}(S_i)=\text{Range}(T_i)$ (即列空间相同)
- ③ 若 $A$ 具有 $n$ 个不同的特征值 (即 $d=n$)，且划分 $S,T$ 为 $\begin{cases}
  S=[S_1,\dots,S_n]\\
  T=[T_1,\dots,T_n]\end{cases}$   
  则 $A=S\Lambda S^{-1}=T\Lambda T^{-1}$ 当且仅当存在一个非奇异的对角阵 $R=\text{diag}(r_1,\dots,r_n)$ 使得 $T=SR$，  
  即当且仅当存在 $n$ 个非零纯量 $r_1,\dots,r_n$ 使得对于每个 $i=1,\dots,n$ 都有列向量 $s_i$ 是列向量 $t_i$ 的 $r_i$ 倍

***

复方阵 $A\in \mathbb C^{n\times n}$ 的特征值与其主对角元素之间仅有的联系是它们的和相等，即 $\tr(A) = \sum_{i=1}^n \lambda_i$    
**(Mirsky 定理, Matrix Analysis 定理 $1.3.31$)**   
给定整数 $n\geq 2$ 和纯量 $\lambda_1,\dots,\lambda_n, d_1,\dots,d_n\in \mathbb C$ 

- ① 当且仅当 $\sum_{i=1}^n \lambda_i = \sum_{i=1}^n d_i$ 时存在复方阵 $A\in \mathbb C^{n\times n}$ 使得其特征值为 $\lambda_1,\dots,\lambda_n$ 同时主对角元为 $d_1,\dots,d_n$ 

- ② 若纯量 $\lambda_1,\dots,\lambda_n, d_1,\dots,d_n\in \mathbb R$，且 $\sum_{i=1}^n \lambda_i = \sum_{i=1}^n d_i$，  

  则存在实方阵 $A\in \mathbb R^{n\times n}$ 使得其特征值为 $\lambda_1,\dots,\lambda_n$ 同时主对角元为 $d_1,\dots,d_n$ 

****

任意给定矩阵 $A \in \mathbb{C}^{m\times n},B \in \mathbb{C}^{n\times m}$  
尽管 $AB,BA$ 不必相同 (当 $m\neq n$ 时它们的尺寸甚至都不相同)，  
但它们的谱是非常接近的，甚至当 $m = n$ 时，它们的谱是完全相同的.

**(矩阵乘积的谱不变性, Matrix Analysis 定理 $1.3.22$)**   
任意给定矩阵 $A \in \mathbb{C}^{m\times n},B \in \mathbb{C}^{n\times m}$ (其中 $m\geq n$)  
则我们有 $AB\in \mathbb{C}^{m\times m},BA \in \mathbb{C}^{n\times n}$，且 $\text{eig}(AB) = \text{eig}(BA) \cup \{\underbrace{0, \ldots, 0}_{m-n \text{ times}}\}$
即 $AB$ 的 $m$ 个特征值即 $BA$ 的 $n$ 个特征值附加上 $m-n$ 个零特征值.  
换句话说，二者的特征多项式满足: $p_{AB}(\lambda) = \lambda^{m-n} p_{BA}(\lambda)$   
**这意味着 $AB,BA$ 的非零特征值是完全相同的 (计代数重数)，而零特征值的个数相差 $m-n$ 个.**    

- 特殊地，当 $m=n$ 时，矩阵乘积 $AB,BA$ 的特征值完全相同 (计代数重数).  
  此时若 $A,B$ 至少有一个是非奇异阵 (不妨设 $A$ 非奇异)，则有 $AB = A(BA)A^{-1}$，表明 $AB$ 和 $BA$ 相似.   
  但 $A,B$ 均为非奇异阵时，$AB$ 不一定相似于 $BA$，例如:
  $$
  A = \begin{bmatrix} 0 & 1\\ 0 & 0\end{bmatrix}\quad
  B = \begin{bmatrix} 0 & 0\\ 0 & 1\end{bmatrix}\\
  AB = \begin{bmatrix} 0 & 1\\ 0 & 0\end{bmatrix}\quad 
  BA = \begin{bmatrix} 0 & 0\\ 0 & 0\end{bmatrix}
  $$
  这是因为特征值完全相同 (计代数重数) 并不能保证 Jordan 标准型相同.   
  更多内容可参考: [(On the similarity of AB and BA for normal and other matrices)](https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=1450&context=pomona_{\mathrm F}ac_pub)
  
- **证明: **  
  任意给定矩阵 $A \in \mathbb{C}^{m\times n},B \in \mathbb{C}^{n\times m}$  (其中 $m\geq n$)，我们都有:   
  $$
  \begin{bmatrix} I_n & -B\\ &I_m \end{bmatrix} \begin{bmatrix} BA & 0_{n\times m}\\ A&0_{m\times m} \end{bmatrix} \begin{bmatrix} I_n & B\\ &I_m \end{bmatrix} = \begin{bmatrix}   0_{n\times n} & 0_{n\times m}\\ A &AB \end{bmatrix}
  $$
  注意到 $\begin{bmatrix} I_n & -B\\ &I_m \end{bmatrix}$ 的逆矩阵即为 $\begin{bmatrix} I_n & B\\ &I_m \end{bmatrix}$ 

  记 $C_1 = \begin{bmatrix} BA & 0_{n\times m}\\ A&0_{m\times m} \end{bmatrix},C_2 =\begin{bmatrix} 0_{n\times n} & 0_{n\times m}\\ A &AB \end{bmatrix}$    
  则上述等式表明 $C_1,C_2$ 相似，于是 $C_1,C_2$ 的特征值完全相同   
  (即特征多项式 $\det(t I_{m+n}-C_1)=\det(t I_{m+n}-C_1)$)
  
  注意到 $C_1$ 的特征值由 $BA$ 的 $n$ 个特征值和 $m$ 个零特征值构成   
  (因为特征多项式 $\det(\lambda I_{m+n}-C_1) = \lambda^m \det(\lambda I_n- BA)$)  
  而 $C_2$ 的特征值由 $AB$ 的 $m$ 个特征值和 $n$ 个零特征值构成   
  (因为特征多项式 $\det(\lambda I_{m+n}-C_2) = \lambda^n \det(\lambda I_m- AB)$)  
  比较二者，即可知 $AB$ 的 $m$ 个特征值即 $BA$ 的 $n$ 个特征值附加上 $m-n$ 个零特征值.

上述定理有许多应用:  

- **(下秩矩阵的特征值, Matrix Analysis 例 $1.3.23$)**  
  设 $A\in \mathbb C^{n\times n}$ 的一个满秩分解为 $A=XY^{\mathrm T}$ (其中 $X,Y\in \mathbb C^{n\times r}$, $r<n$)  
  根据 Matrix Analysis 定理 $1.3.22$ 可知 $A$ 的特征值就是 $Y^{\mathrm T}X\in \mathbb C^{r\times r}$ 的特征值加上 $n-r$ 个零.  

  例如至多秩一的矩阵 $A=xy^{\mathrm T}$ 的特征值便是 $y^{\mathrm T}x$ 再加上 $n-1$ 个零.  
  而至多秩二的矩阵 $A=xy^{\mathrm T}+zw^{\mathrm T} = [x,z][y,w]^{\mathrm T}$ 的特征值  
  便是 $[y,w]^{\mathrm T}[x,z] =\begin{bmatrix}
  y^{\mathrm T}x & y^{\mathrm T}z\\
  w^{\mathrm T}x & w^{\mathrm T}z\end{bmatrix}$ 的 $2$ 个特征值再加上 $n-2$ 个零.  
  (计算实例参见 Matrix Analysis 例 $1.3.25$)

- **(Cauchy 行列式恒等式, Matrix Analysis 例 $1.3.24$)**   
  任意给定非奇异阵 $A \in \mathbb{C}^{n\times n}$ 和 $x,y\in \mathbb C^n$，我们都有:  
  $$
  \begin{align}
  \det(A+xy^{\mathrm T})
  &=
  \det(A) \det(I_n+A^{-1}xy^{\mathrm T})\\
  &=
  \det(A) \prod_{i=1}^n \lambda_i (I_n+A^{-1}xy^{\mathrm T})\\
  &=
  \det(A) \prod_{i=1}^n \{1+ \lambda_i (A^{-1}xy^{\mathrm T})\}\quad (\text{note that }\text{eig}(A^{-1}xy^{\mathrm T}) = \{y^{\mathrm T}A^{-1}x,\underset{n-1}{\underbrace{0,\dots,0}}\})\\
  &=
  \det(A) (1 + y^{\mathrm T}A^{-1}x)(1+0)^{n-1}\\
  &=
  \det(A) + \det(A) y^{\mathrm T}A^{-1}x\\
  &=
  \det(A) + y^{\mathrm T} \text{adj}(A) x
  \end{align}
  $$



### 3.1.6 几何重数

**(Matrix Analysis 定理 $1.4.1$)**  
任意给定复方阵 $A\in \mathbb C^{n\times n}$，我们都有:

- ① $A^{\mathrm T}$ 与 $A$ 具有相同的特征值
- ② $A^{\mathrm H}$ 的特征值是 $A$ 的特征值的复共轭

对于任意特征值 $\lambda\in \text{eig}(A)$  
与其相伴的特征值全体和零向量构成 $\mathbb C^n$ 的子空间 $\text{Ker}(A-\lambda I_n)$，称为与 $\lambda$ 相伴的**特征空间** (eigenspace).  
其维数为 $\text{null}(A-\lambda I)=n-\rank(A-\lambda I)$，称为 $\lambda$ 的**几何重数** (geometric multiplicity)  
它也是与 $\lambda$ 相伴的线性无关特征向量的最大个数.

- 一般来说，$\lambda$ 的代数重数一定大于等于其几何重数.
- 若 $\lambda$ 的代数重数是 $1$ (则其几何重数必定是 $1$)，则我们称 $\lambda$ 是**简单的** (simple)  
- 若 $\lambda$ 的代数重数等于其几何重数，则我们称 $\lambda$ 是**半简单的** (semi-simple)  
- 若 $A$ 的每个不同的特征值都至少是半简单的，则我们称 $A$ 是**无亏损的** (non-defective)  
  否则我们称 $A$ 是**有亏损的** (defective)  
  $A$ 可对角化当且仅当 $A$ 是无亏损的.
- 若 $A$ 的每个不同的特征值的几何重数都等于 $1$，则我们称 $A$ 是**非退化的** (non-derogatory)  
  否则我们称 $A$ 是**退化的** (derogatory)  
  $A$ 具有 $n$ 个不同的特征值当且仅当 $A$ 是非退化且无亏损的.

显然 $A$ 与特征值 $\lambda$ 相伴的特征空间 $\text{Ker}(A-\lambda I_n)$ 是一个 $A$-不变子空间，  
但一个$A$-不变子空间不一定就是 $A$ 的某个特征空间.  
此外，最小的 $A$-不变子空间是由 $A$ 的单独一个特征向量所张成的子空间.

****

设 $A\in \mathbb C^{n\times n}$  

- 若纯量 $\lambda\in \mathbb C$ 和非零向量 $x\in \mathbb C^n$ 满足方程 $Ax=x\lambda$，  
  则我们称 $\lambda$ 是 $A$ 的一个特征值，而 $x$ 称为 $A$ 的一个与 $\lambda$ 相伴的**右特征向量** (right eigenvector).  
  我们称元素对 $(\lambda,x)$ 为 $A$ 的一个**右特征对** (right eigenpair).

- 若纯量 $\lambda\in \mathbb C$ 和非零向量 $y\in \mathbb C^n$ 满足方程 $y^{\mathrm H}A=\lambda y^{\mathrm H}$，  
  则我们称 $\lambda$ 是 $A$ 的一个特征值，而 $y$ 称为 $A$ 的一个与 $\lambda$ 相伴的**左特征向量** (left eigenvector)    
  我们称元素对 $(\lambda,y)$ 为 $A$ 的一个**左特征对** (left eigenpair).

  请不要将左特征向量贬低为是在理论上与右特征向量等价的概念.  
  左特征向量可以传递出与右特征向量不同的信息，了解它们之间的关系是非常有用的.

**(Matrix Analysis 结论 $1.4.6$)**  

- ① 设 $A\in \mathbb C^{n\times n}$ 而 $x\in \mathbb R^n$ 是非零向量，若 $\begin{cases}
  Ax=x\lambda\\
  x^{\mathrm H}A = \mu x^{\mathrm H}\end{cases}$ 则 $\lambda = \mu$ (反之不一定成立)
- ② $A\in \mathbb C^{n\times n}$ 的与特征值 $\lambda$ 相伴的右特征向量 $x$ 的复共轭 $\bar x$ 是 $A^{\mathrm T}$ 的与特征值 $\lambda$ 相伴的左特征向量 
- ③ $A\in \mathbb C^{n\times n}$ 的与特征值 $\lambda$ 相伴的右特征向量 $x$ 也是 $A^{\mathrm H}$ 的与特征值 $\bar \lambda$ 相伴的左特征向量

**(双正交完备原理, Matrix Analysis 定理 $1.4.7$ & $2.4.11.1$)**  
给定复方阵 $A\in \mathbb C^{n\times n}$ 和单位向量 $x,y\in \mathbb C^n$ 以及 $\lambda,\mu\in \mathbb C$ 

- ① (双正交原理) 若 $\begin{cases}
  Ax=x \lambda \\
  y^{\mathrm H}A=\mu y^{\mathrm H}\\
  \lambda \neq \mu\end{cases}$ 则 $y^{\mathrm H}x=0$   
  设 $U = [x,y,u_3,\dotsm,u_n]$ 是酉矩阵，则我们有:  
  $$
  U^{\mathrm H}AU = \begin{bmatrix}
  \lambda & * & *\\
  & \mu & \\
  & * & A_{n-2}
  \end{bmatrix}\ \ (A_{n-2}\in \mathbb C^{(n-2)\times (n-2)})
  $$

- ② 若 $\begin{cases}
  Ax=x \lambda \\
  y^{\mathrm H}A=\lambda y^{\mathrm H}\end{cases}$ 且 $y^{\mathrm H}x=0$，则 $\lambda$ 的代数重数至少是 $2$  
  设 $U = [x,y,u_3,\dotsm,u_n]$ 是酉矩阵，则我们有:  
  $$
  U^{\mathrm H}AU = \begin{bmatrix}
  \lambda & * & *\\
  & \lambda & \\
  & * & A_{n-2}
  \end{bmatrix}\ \ (A_{n-2}\in \mathbb C^{(n-2)\times (n-2)})
  $$

- ③ 若 $\begin{cases}
  Ax=x \lambda \\
  y^{\mathrm H}A=\lambda y^{\mathrm H}\end{cases}$ 且 $y^{\mathrm H}x\neq 0$，则存在一个非奇异阵 $S=[x, S_1]\in \mathbb C^{n\times n}$ 使得:  
  $$
  S^{-1}AS = \begin{bmatrix}
  \lambda & \\
  & A_{n-1}
  \end{bmatrix}\ \ (A_{n-1}\in \mathbb C^{(n-1)\times (n-1)})
  $$
  其中 $S_1$ 的列是 $y$ 的正交补空间的任意一组基  
  (这保证了 $S$ 是非奇异的 (因为 $x\notin \text{span}(S_1)$)，且 $S^{-1}$ 的第一行是 $\frac{y^{\mathrm H}}{x^{\mathrm H}y}$)     
  如果 $\lambda$ 的几何重数是 $1$，则其代数重数也是 $1$.

  反过来，若 $A$​ 与形如 $\begin{bmatrix}
  \lambda & \\
  & A_{n-1}
  \end{bmatrix}$ 的矩阵相似，则它就有一对关于 $\lambda$ 的非正交的左右特征向量.

- ④ 若 $\begin{cases}
  Ax=x \lambda \\
  y^{\mathrm H}A=\lambda y^{\mathrm H}\end{cases}$ 且 $x=y$，则我们称 $x$​ 为**正规特征向量** (normal eigenvector)   
  设 $U = [x,u_2,u_3,\dotsm,u_n]$ 是酉矩阵，则我们有:  
  $$
  U^{\mathrm H}AU = \begin{bmatrix}
  \lambda & \\
  & A_{n-1}
  \end{bmatrix}\ \ (A_{n-1}\in \mathbb C^{(n-1)\times (n-1)})
  $$

*****

相似不改变方阵的特征值，其特征向量在相似变换下以一种简单的方式变换.  
**(Matrix Analysis 定理 $1.4.9$)**  
设 $A,B\in \mathbb C^{n\times n}$，并假设存在某个非奇异阵 $S\in \mathbb C^{n\times n}$ 使得 $B=S^{-1}AS$   

- 若 $x\in \mathbb C^n$ 是 $B$ 的与特征值 $\lambda$ 相伴的右特征向量，则 $Sx$ 就是 $A$ 的与特征值 $\lambda$ 相伴的右特征向量.    
  $$
  Bx = (S^{-1}AS)x = \lambda x\ \ \Leftrightarrow\ \ A(Sx)=\lambda (Sx)
  $$

- 若 $y\in \mathbb C^n$ 是 $B$ 的与特征值 $\lambda$ 相伴的左特征向量，则 $S^{-H}y$ 就是 $A$ 的与特征值 $\lambda$ 相伴的左特征向量.  
  $$
  y^{\mathrm H}B = y^{\mathrm H}(S^{-1}AS) = \lambda y^{\mathrm H}\ \ \Leftrightarrow\ \ (S^{-H}y)^{\mathrm H} A = \lambda (S^{-H}y)^{\mathrm H}
  $$

****

基于主子阵的特征值信息我们能知道什么?  
**(Matrix Analysis 定理 $1.4.10$)**  
设 $A\in \mathbb C^{n\times n}$，$\lambda\in \mathbb C$，$k\geq 1$，则下列命题中，① 包含 ②，而 ② 包含 ③: 

- ① $\lambda$ 是 $A$ 的特征值，且几何重数至少是 $k$ (即至少有 $k$ 个线性无关的相伴特征向量)
- ② $\lambda$ 是 $A$ 的每个 $m=n-k+1,\dots,n$ 阶主子阵的特征值
- ③ $\lambda$ 是 $A$ 的特征值，且代数重数至少是 $k$ 

特别地，特征值的代数重数至少与其几何重数一样大.

***

代数重数为 $1$ 的特征值 $\lambda$ 的几何重数一定为 $1$，与 $\lambda$ 相伴的左右特征向量不可能是正交的.  
几何重数为 $1$ 的特征值 $\lambda$ 可能拥有 $2$ 或更高的代数重数，但这只有在与 $\lambda$ 相伴的左右特征向量为正交时才可能发生.  
**(Matrix Analysis 定理 $1.4.12$)**  
设给定 $A\in \mathbb C^{n\times n}$，$\lambda\in \mathbb C$ 以及非零向量 $x,y\in \mathbb C^n$  
假设 $\lambda$ 是 $A$ 的一个特征值，$Ax=x \lambda $ 且 $y^{\mathrm H}A=\lambda y$ 

- ① 如果 $\lambda$ 的代数重数为 $1$，则 $y^{\mathrm H}x\neq 0$
- ② 如果 $\lambda$ 的几何重数为 $1$，则它的代数重数为 $1$ 当且仅当 $y^{\mathrm H}x\neq 0$ 



### 3.1.7 Gershgorin 圆盘定理

我们希望利用一系列容易刻画的有界集来估计一个复方阵的特征值.  
**谱半径定理** (Spectral Radius Theorem) 给出了这样一个有界集:  
对于 $\mathbb C^{n\times n}$ 上的任意相容范数 $\|\cdot\|$ 和复方阵 $A\in \mathbb{C}^{n\times n}$，我们都有 $\rho(A)\leq \|A\|$ 成立.  
这说明对 $A$ 任取一个相容范数，都可作为谱半径 $\rho(A)$ 的一个上界.  
因此 $A$ 的所有特征值都位于一个中心为复平面原点、半径为 $\|A\|$ 的圆盘上.  
但我们能否得到更精确的估计呢?

***

考虑将复方阵 $A = [a_{i,j}]\in \mathbb{C}^{n\times n}$ 分解为两个方阵的和: $A=A_D+A_N$  
其中 $A_D$ 保留 $A$ 的主对角元，$A_N$ 保留 $A$ 的非对角元.  
对于任意实数 $\varepsilon \in \mathbb R$，记 $A(\varepsilon) := A_D + \varepsilon A_N$   

- 当 $\varepsilon = 0$ 时， $A(0) = A_D$ 的主对角元 $a_{11},a_{22},\dots,a_{nn}$ 即为 $A(0)$ 的特征值；  
- 当 $\varepsilon > 0$ 时，根据特征值关于矩阵元素的连续性可知:  
  只要 $\varepsilon $ 足够小，$A(\varepsilon) = A_D + \varepsilon A_N$ 的特征值就一定落在以 $a_{11},a_{22},\dots,a_{nn}$ 为中心的一系列圆盘中.

将上述推理严格化，便得到:  
**(关于行的 Gershgorin 圆盘定理, Matrix Analysis 定理 $6.1.1$)**  
记 $A = [a_{i,j}]\in \mathbb C^{n\times n}$ 第 $i$ 行的**去心绝对行和** (deleted absolute row sums) 为 $\text{Row}'_i(A) = \sum_{j\neq i}^n |a_{i,j}|$   
记 $A$ 关于第 $i$ 行的 **Gershgorin 圆盘**为 $G_i(A):=\{z\in \mathbb C : |z-a_{i,i}| \leq \text{Row}'_i(A)\}$    
则我们有以下命题成立: 

- ① $A$ 的 $n$ 个特征值一定落在这 $n$ 个关于行的 Gershgorin 圆盘的并集 $G(A):= \bigcup_{i=1}^n G_i(A)$ 中.  

- ② 进一步，若 $S\subset \mathbb C$ 是 $A$ 的 $k$ 个关于行的 Gershgorin 圆盘的并集，  
  且与剩下的 $n-k$ 个关于行的 Gershgorin 圆盘不相交，  
  则 $S$ 恰好包含 $A$ 的 $k$ 个特征值 (按照它们的代数重数计算)

- ③ 特殊地，若某个关于行的 Gershgorin 圆盘与其余 $n-1$ 个关于行的 Gershgorin 圆盘不相交，  
  则它恰好包含 $A$ 的 $1$ 个特征值，且该特征值一定是单重的.

- ④ 考虑第 $i$ 行的 Gershgorin 圆盘 $G_i(A)$ 的元素到原点的最远距离  
  即为 $\text{Row}'_i(A) + |a_{i,i}| = \sum_{j\neq i}^n |a_{i,j}| + |a_{i,i}| = \sum_{j=1}^n |a_{i,j}|$，即 $A$ 第 $i$ 行的绝对行和.  
  注意到 $A$ 的模最大特征值落在 $G(A):= \bigcup_{i=1}^n G_i(A)$ 中，因此我们有:   
  $$
  \rho(A) = \max_{1\leq i\leq n} |\lambda_i(A)| \leq \max_{1\leq i\leq n} \sum_{j=1}^n |a_{i,j}| = \|A\|_\infty
  $$
  这与谱半径定理的结论 $\rho(A)\leq \|A\|_\infty$ 是吻合的.

邵老师: 同学们，小心一点，大多数教材上关于这个定理的证明都是错误的.

****

注意到 $A$ 与 $A^{\mathrm T}$ 有同样的特征值 (我们会在后文给出证明)，  
因此我们可以将关于行的 Gershgorin 圆盘定理应用于 $A^{\mathrm T}$ 而得到关于列的 Gershgorin 圆盘定理.  
**(关于列的 Gershgorin 圆盘定理, Matrix Analysis 推论 $6.1.3$)**    
记 $A = [a_{i,j}]\in \mathbb C^{n\times n}$ 第 $j$ 列的**去心绝对行和** (deleted absolute column sums) 为 $\text{Col}'_j(A) := \sum_{i\neq j}^n |a_{i,j}| = \text{Row}_j'(A^{\mathrm T})$   
记 $A$ 关于第 $j$ 列的 **Gershgorin 圆盘**为 $G_j(A^{\mathrm T}):=\{z\in \mathbb C : |z-a_{jj}| \leq \text{Col}'_j(A)\}=\{z\in \mathbb C : |z-a_{jj}| \leq \text{Row}'_j(A^{\mathrm T})\}$    
则我们有以下命题成立: 

- ① $A$ 的 $n$ 个特征值一定落在这 $n$ 个关于列的 Gershgorin 圆盘的并集 $G(A^{\mathrm T}):= \bigcup_{j=1}^n G_j(A^{\mathrm T})$ 中.  

- ② 进一步，若 $S\subset \mathbb C$ 是 $A$ 的 $k$ 个关于列的 Gershgorin 圆盘的并集，  
  且与剩下的 $n-k$ 个关于列的 Gershgorin 圆盘不相交，  
  则 $S$ 恰好包含 $A$ 的 $k$ 个特征值 (按照它们的代数重数计算)

- ③ 特殊地，若某个关于列的 Gershgorin 圆盘与其余 $n-1$ 个关于列的 Gershgorin 圆盘不相交，  
  则它恰好包含 $A$ 的 $1$ 个特征值，且该特征值一定是单重的.

- ④ 考虑第 $j$ 列的 Gershgorin 圆盘 $G_i(A^{\mathrm T})$ 的元素到原点的最远距离  
  即为 $\text{Col}'_j(A) + |a_{jj}| = \sum_{i\neq j}^n |a_{i,j}| + |a_{jj}| = \sum_{i=1}^n |a_{i,j}|$，即 $A$ 第 $j$ 列的绝对列和.  
  注意到 $A$ 的模最大特征值落在 $G(A^{\mathrm T}):= \bigcup_{j=1}^n G_j(A^{\mathrm T})$ 中，因此我们有:   
  $$
  \rho(A) = \max_{1\leq i\leq n} |\lambda_i(A)| \leq \max_{1\leq j\leq n} \sum_{i=1}^n |a_{i,j}| = \|A\|_1
  $$
  这与谱半径定理的结论 $\rho(A)\leq \|A\|_1$ 是吻合的.

****

注意到只要 $S\in \mathbb C^{n\times n}$ 非奇异，$S^{-1}AS$ 与 $A$ 就有相同的特征值，   
故我们能将 Gershgorin 圆盘定理应用于 $S^{-1}AS$，从而得到 $A$ 的更精细的特征值包容集.  
最简单的方式是选取 $D=\text{diag}\{d_1,\dots,d_n\}\ (d_i>0\text{ for all }i=1,\dots,n)$   
对 $D^{-1}AD$ 及其转置 $D A^{-1}D^{-1}$ 应用 Gershgorin 圆盘定理就得到如下结果:  
**(Matrix Analysis 推论 $6.1.6$)**  
任意给定 $A=[a_{i,j}]\in \mathbb C^{n\times n}$ 和 $D=\text{diag}\{d_1,\dots,d_n\}\ (d_i>0\text{ for all }i=1,\dots,n)$   
我们都有以下命题成立: 

- ① $A$ 的 $n$ 个特征值一定落在 $G(D^{-1}AD)=\bigcup_{i=1}^n G_i(D^{-1}AD) = \bigcup_{i=1}^n \{z\in \mathbb C: |z-a_{i,i}|\leq \frac{1}{d_i}\sum_{j\neq i}^n d_j |a_{i,j}|\}$​ 中    
  此外，若这些圆盘中有 $k$ 个的并集与剩下的 $n-k$ 个都不相交，  
  则这个并集恰好包含 $A$ 的 $k$ 个特征值 (按照它们的代数重数计算)
- ② $A$ 的 $n$ 个特征值一定落在 $G(DA^{\mathrm{T}}D^{-1})=\bigcup_{j=1}^n G_j(DA^{\mathrm{T}}D^{-1}) = \bigcup_{j=1}^n \{z\in \mathbb C: |z-a_{jj}|\leq d_j\sum_{i\neq j}^n \frac{1}{d_i} |a_{i,j}|\}$ 中    
  此外，若这些圆盘中有 $k$ 个的并集与剩下的 $n-k$ 个都不相交，  
  则这个并集恰好包含 $A$ 的 $k$ 个特征值 (按照它们的代数重数计算)
- ③ $\rho(A) \leq \min\{\|D^{-1}AD\|_\infty, \|DA^{\mathrm T}D^{-1}\|_\infty\} = \min\{\|D^{-1}AD\|_\infty, \|D^{-1}AD\|_1\}$

记忆这些结论的诀窍是 "行变换是左乘的，列变换是右乘的".  
因此 $D^{-1}AD = [\frac{1}{d_{i}}a_{i,j} d_j]$ 而 $DA^{\mathrm T}D^{-1} = [d_i a_{j,i}\frac{1}{d_j}]$ (方括号中都是对应矩阵 $(i,j)$ 位置上的元素)  
额外的参数 $d_1,\dots,d_n>0$ 让我们有足够的灵活性对特征值进行任意好的估计.

一个简单的例子: **(这个例子并不好)**
$$
A = 
\begin{bmatrix}
1 & 1\\
0 & 2
\end{bmatrix}\\
\hline
G_1(A) = \{z\in \mathbb C: |z-1| \leq 1\}\\
G_2(A) = \{z\in \mathbb C: |z-2| \leq 0\}
$$
这两个关于行的 Gershgorin 圆盘是相交的，太过粗略了.  
现考虑 $D=\text{diag}\{d_1,d_2\}$，满足 $d_1,d_2>0$，则我们有:  
$$
D^{-1}AD = \begin{bmatrix}
1 & \frac{d_2}{d_1}\\
0 & 2
\end{bmatrix}\\
\hline
G_1(D^{-1}AD) = \{z\in \mathbb C:|z-1|\leq \frac{d_2}{d_1}\}\\
G_2(D^{-1}AD) = \{z\in \mathbb C:|z-2|\leq 0\}
$$
此时两个关于行的 Gershgorin 圆盘是不相交的，而且我们可以控制 $G_1(D^{-1}Ad)$ 的半径 $\frac{d_2}{d_1}$ 任意小.  
这样很容易得到 $A$ 的特征值分别是 $1$ 和 $2$ 

<img src="Matrix Analysis 6.1.7.png" style="zoom:30%;" />

*****

引入自由参数和最优化的思想我们有:  
**(Matrix Analysis 推论 $6.1.8$)**  
任意给定 $A=[a_{i,j}]\in \mathbb C^{n\times n}$，则我们有以下命题成立:

- ① $A$ 的 $n$ 个特征值一定落在 $\bigcap_D G(D^{-1}AD)$ 中  
  其中 $D=\text{diag}\{d_1,\dots,d_n\}$ 取遍所有对角元为正实数的对角阵.

- ② $A$ 的 $n$ 个特征值一定落在 $\bigcap_D G(DA^{\mathrm T}D^{-1})$ 中  
  其中 $D=\text{diag}\{d_1,\dots,d_n\}$ 取遍所有对角元为正实数的对角阵.

- ③ 关于谱半径 $\rho(A)$ 我们有:  
  $$
  \begin{align}
  \rho(A) &\leq \min
  \left\{\min_D \|D^{-1}AD\|_\infty, \min_D \|D^{-1}AD\|_1 \right\}\\
  &=
  \min\left\{\min_{d_1,\dots,d_n>0} 
  \left(\max_{1\leq i\leq n} \frac{1}{p_i}\sum_{j=1}^n p_j |a_{i,j}|\right), 
  \min_{d_1,\dots,d_n>0} 
  \left(\max_{1\leq j\leq n} p_j\sum_{i=1}^n \frac{1}{p_i} |a_{i,j}|\right)\right\}
  
  
  \end{align}
  $$
  其中 $D=\text{diag}\{d_1,\dots,d_n\}$ 取遍所有对角元为正实数的对角阵.

如果能对矩阵的特征值位于 (或者不位于) 某种集合中有一些进一步的信息，  
那么这种信息有可能与 Gersgorin 圆盘定理一道用来对特征值的位置给出更精确的结果.  
例如，若 $A$ 是 Hermite 的，那么它的特征值全都是实数，  
所以它们都在集合 $\mathbb R \cap G(A)$ 中，即实数域上的闭区间的有限并集.

*****

给定复方阵 $A\in \mathbb{C}^{n\times n}$   

- 若对任意 $i = 1,\dots, n$ 有 $|a_{i,i}| \geq \text{Row}'_i(A) = \sum_{j\neq i}^n |a_{i,j}|$ 成立，则称 $A$ 是**对角占优的** (diagonally dominant)  
- 若对任意 $i = 1,\dots, n$ 有 $|a_{i,i}| > \text{Row}'_i(A) = \sum_{j\neq i}^n |a_{i,j}|$ 成立，则称 $A$ 是**严格对角占优的** (strictly diagonally dominant)

根据关于行的 Gershgorin 圆盘定理可知:  
如果 $A$ 是严格对角占优的，则 $0$ 不属于 $A$ 的任何一个关于行的 Gershgorin 圆盘.  
因此严格对角占优阵的特征值一定是非零的.    
此外，如果严格对角占优阵 $A$ 的对角元都是正实数，则所有圆盘都位于复平面的右半平面.  
进一步，如果 $A$ 同时还是 Hermite 阵，则 $A$ 的特征值一定是正实数，位于右半实数轴上.  

我们将上述结果总结在下面的定理中:   
**(Levy–Desplanques 定理, Matrix Analysis 定理 $6.1.10$)**  
若复方阵 $A\in \mathbb{C}^{n\times n}$ 是严格对角占优的，则有:

* $A$ 是非奇异阵 (即 $0$ 不是 $A$ 的特征值)
* 此外，如果 $A$ 的主对角元都是正实数，则 $A$ 的特征值都具有正实部    
  进一步，如果 $A$ 同时还是 Hermite 阵，则 $A$ 正定 (即 $A$ 的特征值都是正实数)

上述定理的第一个论断被称为 **Levy–Desplanques 定理**  
**实际上，我们可以将严格对角占优这个条件放宽一点: (Matrix Analysis 6\.1\.11 P392\)**  
设复方阵 $A\in \mathbb{C}^{n\times n}$的主对角元都是非零的，  
如果 $A$ 至少有 $n-1$ 行是严格对角占优的，则 $A$ 是非奇异阵.

实际上，我们可以将严格对角占优这个条件放宽一点:  
**(Matrix Analysis 定理 $6.1.11$)**  
设复方阵 $A\in \mathbb{C}^{n\times n}$的主对角元都是非零的.  
若 $A$ 对角占优，且至少有 $n-1$ 行是严格对角占优的，则 $A$ 是非奇异阵.  
(不过仅凭对角占优是不足以保证非奇异的)



### 3.1.8 特征值摄动定理

设 $\Lambda = \text{diag}\{\lambda_1,\dots,\lambda_n\}\in \mathbb C^{n\times n}$，$E=[e_{ij}]\in \mathbb C^{n\times n}$  
关于行的 Gershgorin 圆盘定理确保摄动矩阵 $\Lambda+E$ 的特征值 $\hat \lambda_1,\dots,\hat\lambda_n$ 落在以下集合中:  
$$
\bigcup_{i=1}^n \left\{z\in \mathbb C:|z-\lambda_i-e_{ii}| \leq \sum_{j\neq i}|e_{ij}|\right\}
$$
进而落在以下集合中:  
$$
\bigcup_{i=1}^n \left\{z\in \mathbb C:|z-\lambda_i| \leq \sum_{j\neq i}|e_{ij}| + |e_{ii}| = \sum_{j= 1}^n |e_{ij}|\leq \|E\|_\infty\right\}
$$
因此对于摄动矩阵 $\Lambda+E$ 的任意特征值 $\hat \lambda$，都存在 $\Lambda$ 的某个特征值 $\lambda_i$ 使得 $|\hat \lambda - \lambda_i|\leq \|E\|_\infty$   
我们可以利用上述结果来给出一个可对角化的矩阵的特征值摄动的界:  
**(Matrix Analysis 定理 $6.3.1$)**  
设 $A\in \mathbb C^{n\times n}$ 可对角化，即存在非奇异阵 $S\in \mathbb C^{n\times n}$ 使得 $A=S\Lambda S^{-1}$  
其中 $\Lambda = \text{diag}\{\lambda_1,\dots,\lambda_n\}\in \mathbb C^{n\times n}$   
设 $E\in \mathbb C^{n\times n}$   
对于摄动矩阵 $A+E$ 的任意特征值 $\hat \lambda$，都存在 $A$ 的一个特征值 $\lambda$ 使得:  
$$
\|\hat \lambda-\lambda\| \leq \|S^{-1}ES\|_\infty \leq \|S\|_\infty \|S^{-1}\|_\infty\|E\|_\infty 
$$

****

利用有限维赋范空间上范数的等价性，我们可以推广上述结论:  
**(Bauer-Fiker 定理, Matrix Analysis 定理 $6.3.2$)**  
设 $A=S\Lambda S^{-1}\in \mathbb C^{n\times n}$ 是可对角化的矩阵.  
对于任意扰动 $E\in \mathbb C^{n\times n}$，我们有 $S^{-1}(A+E)S = \Lambda + S^{-1}ES$   
因此对于 $A+E$ 的任意扰动特征值 $\hat\lambda$ 都有 $A$ 的一个特征值 $\lambda$ 与之对应，满足:  
$$
|\hat\lambda - \lambda| \leq \|S^{-1}ES\| \leq \|S\|\|S^{-1}\| \|E\|
$$
其中 $\|\cdot\|$ 是 $\mathbb C^{n}$ 上的某个绝对范数的诱导范数 (自然是相容函数)  
我们可以将 $\inf_{S} \|S\|\|S^{-1}\|$ 作为问题的条件数 (的上确界)  
可以证明 $\inf_{S} \|S\|\|S^{-1}\|\leq O(\sqrt{n}) \inf_D \|SD\|\|(SD)^{-1}\|$  
因此我们可以将 $S$ 的每一列归一化得到 $S_0$，并以 $\|S_0\|\|S_0^{-1}\|$ 作为条件数的近似.

- **Lemma 1 (一个简单的观察):**  
  任意给定 $\mathbb C^{n\times n}$ 上的相容范数 $\|\cdot\|$   
  若 $B\in \mathbb C^{n\times n}$ 满足 $I_n-B$ 奇异，则根据谱半径定理我们有 $\|B\|\geq \rho(B)\geq 1$  

- **(Matrix Analysis 定义 $5.4.18$)**    
  设 $V=\mathbb F^n$ (其中 $\mathbb F=\mathbb R\text{ or }\mathbb C$)   
  记 $|x|$ 为 $x\in V$ 逐个元素取模得到的向量.  
  我们说 $|x|\preceq|y|$，当且仅当 $|x_i|\leq |y_i|\ (i=1,\dots,n)$   
  我们称 $V$ 上的范数 $\|\cdot\|$ 是:

  - ① **单调的** (monotone)，如果对于任意满足 $|x|\preceq|y|$ 的 $x,y\in V$ 都有 $\|x\|\leq \|y\|$ 成立.
  - ② **绝对的** (absolute)，如果 $\||x|\| = \|x\|\ (\forall\ x\in V)$

  **Lemma 2 (Matrix Analysis 定理 $5.6.36$)**    
  设 $\|\cdot\|$ 是 $V=\mathbb F^n$ (其中 $\mathbb F=\mathbb R\text{ or }\mathbb C$) 上的范数，而 $\|\cdot\|$ 是由它诱导的 $\mathbb F^{n\times n}$ 上的矩阵范数.  
  则下列命题等价:

  - ① $\|\cdot\|$ 是 $\mathbb F^n$ 上的绝对范数
  - ② $\|\cdot\|$ 是 $\mathbb F^n$ 上的单调范数
  - ③ 对于任意对角阵 $\Lambda=\text{diag}\{\lambda_1,\dots,\lambda_n\}$，都有诱导范数 $\|\Lambda\|=\max_{1\leq i\leq n}|\lambda_i|$ 成立.

- **证明:**   
  设 $\hat \lambda$ 是 $S^{-1}(A+E)S = \Lambda + S^{-1}ES$ 的任意特征值.  
  若 $\hat \lambda$ 也是 $A$ 的特征值，则上述定理中的界显然满足.  
  若 $\hat \lambda$ 不是 $A$ 的特征值，则 $\hat \lambda I - \Lambda$ 是非奇异的.  
  注意到以下矩阵奇异:   
  (因为 $\hat \lambda I_n - \Lambda - S^{-1}ES$ 是奇异矩阵)
  $$
  (\hat\lambda I_n - \Lambda)^{-1} (\hat \lambda I_n - \Lambda- S^{-1}ES) = I_n - (\hat \lambda I_n -\Lambda)^{-1} S^{-1}ES
  $$
  根据 **Lemma 1** 可知 $\|(\hat \lambda I_n -\Lambda)^{-1} S^{-1}ES\|\geq 1$   
  根据 **Lemma 2** 我们有:  
  $$
  \begin{align}
  1 
  &\leq
  \|(\hat \lambda I_n -\Lambda)^{-1} S^{-1}ES\|\\
  &\leq
  \|(\hat \lambda I_n -\Lambda)^{-1}\| \|S^{-1}ES\|\quad (\text{use Lemma 2})\\
  &=
  \max_{1\leq i\leq n} |\hat\lambda-\lambda_i|^{-1}\cdot \|S^{-1}ES\|\\
  &=
  \frac{\|S^{-1}ES\|}{\min_{1\leq i\leq n}|\hat\lambda - \lambda_i|}
  \end{align}
  $$
  于是我们有:
  $$
  \begin{align}
  \min_{1\leq i\leq n}|\hat\lambda - \lambda_i| 
  &\leq 
  \|S^{-1}ES\|\\
  &\leq
  \|S\|\|S^{-1}\|\|E\|
  \end{align}
  $$
  命题得证.
  
- 特殊地，若 $A\in \mathbb C^{n\times n}$ 是正规矩阵 (即满足 $A^{\mathrm H}A=AA^{\mathrm H}$)  
  则 $A$ 可以酉对角化，即特征向量构成的矩阵 $X$ 可以取成酉矩阵，故而是良态的.  
  因此正规特征值问题 (更特殊地, Hermite 特征值问题) 是良态的.   
  **(Matrix Analysis 推论 $6.3.4$)**  
  设 $A\in \mathbb C^{n\times n}$ 是正规矩阵 (可酉对角化: $A=U\Lambda U^{\mathrm H}$)  
  对于任意扰动 $E\in \mathbb C^{n\times n}$ (不需要正规)，我们都有 $U^{\mathrm H}(A+E)U = \Lambda + U^{\mathrm H}EU$    
  因此对于 $A+E$ (不需要正规) 的任意扰动特征值 $\hat\lambda$ 都有 $A$ 的一个特征值 $\lambda$ 与之对应，满足:
  $$
  |\hat\lambda - \lambda| \leq \|U^{\mathrm H}EU\|_2 =\|E\|_2
  $$
  最后一步用到了谱范数 $\|\cdot\|_2$ 的酉不变性.

****

**(Matrix Analysis 引理 $6.3.10$)**  
设 $\lambda\in \mathbb C$ 是 $A\in \mathbb C^{n\times n}$ 的一个单重特征值，$x,y\in \mathbb C^n$ 分别为与 $\lambda$ 相伴的右特征向量和左特征向量.  
则我们有:

- ① $y^{\mathrm H}x \neq 0$ 

- ② 存在非奇异的 $S\in \mathbb C^{n\times n}$，其第一列为 $x$，其逆矩阵 $S^{-1}$ 的第一行为 $\frac{y^{\mathrm H}}{y^{\mathrm H}x}$，使得:  
  $$
  A = S 
  \begin{bmatrix}
  \lambda & \\
  & A_1
  \end{bmatrix} S^{-1}
  $$
  其中 $\lambda$ 不是 $A_1\in \mathbb C^{(n-1)\times (n-1)}$ 的特征值.

**(Matrix Analysis 定理 $6.3.12$)**    
设 $A,E\in \mathbb C^{n\times n}$  
设 $\lambda\in \mathbb C$ 是 $A$ 的一个单重特征值，$x,y\in \mathbb C^n$ 分别为与 $\lambda$ 相伴的右特征向量和左特征向量. 
则我们有:

- ① 任意给定 $\varepsilon>0$，存在 $\delta>0$ 使得对于任意满足 $|t|<\delta$ 的 $t\in \mathbb C$ 都存在 $A+tE$ 的唯一的特征值 $\lambda(t)$ 使得:
  $$
  \left|\lambda(t) - \lambda - t \frac{y^{\mathrm H}Ex}{y^{\mathrm H}x}
  \right| \leq |t| \varepsilon
  $$

- ② $\lambda(t)$ 在 $t=0$ 处可微 (自然是连续的)，满足:  
  $$
  \lim_{t\to 0} \lambda(t) = \lambda\\
  \frac{\mathrm{d}}{\mathrm{d}t}\lambda(t){\Large|}_{t=0} = \frac{y^{\mathrm H}Ex}{y^{\mathrm H}x}
  $$
  **邵老师提供的简单证明:**  
  $$
  \begin{align}
  0 
  &=y^{\mathrm H}[(A+tE) (x+\Delta x) - (x+\Delta x) (\lambda + \Delta \lambda)]\\
  &=
  y^{\mathrm H}(Ax + tEx + A\Delta x + tE\Delta x - x\lambda - \Delta x \lambda - x\Delta \lambda - \Delta x \Delta \lambda)\quad (\text{note that }Ax=x\lambda)\\
  &=
  y^{\mathrm H}(tEx + A\Delta x + tE\Delta x - \Delta x \lambda - x\Delta \lambda - \Delta x \Delta \lambda)
  \quad (\text{omit higher-order terms})\\
  &\approx
  y^{\mathrm H}(tEx + A\Delta x - \Delta x \lambda - x\Delta \lambda)\\
  &=
  ty^{\mathrm H}Ex + y^{\mathrm H}A\Delta x - \lambda y^{\mathrm H}\Delta x  - y^{\mathrm H}x\Delta \lambda \quad (\text{note that }y^{\mathrm H}A=\lambda y^{\mathrm H})\\
  &=
  t y^{\mathrm H}Ex - y^{\mathrm H}x \Delta \lambda
  \end{align}
  $$
  于是我们有:
  $$
  \Delta \lambda \approx 
  \frac{ty^{\mathrm H}Ex}{y^{\mathrm H}x}\\
  \Rightarrow\\
  \frac{\mathrm{d}}{\mathrm{d}t}\lambda(t){\Large|}_{t=0} = \frac{y^{\mathrm H}Ex}{y^{\mathrm H}x}
  $$



## 3.2 Jordan 标准型定理

对于给定的复方阵 $A,B\in \mathbb{C}^{n\times n}$，我们如何判断它们是否相似?    
一种解决方法是选定一类特殊形式的矩阵，  
并检查给定的两个矩阵是否可以通过相似变换化简为同一特殊矩阵.  

- 一方面，根据 Schur 定理 (Matrix Analysis 定理 $2.3.1$) 我们知道，任意复方阵都可酉相似变换为一个上三角阵，  
  但是上三角阵这个类别对我们来说太大了，即我们面临唯一性的问题:  
  两个具有相同主对角线而某些严格上三角元不同的上三角阵仍可能是相似的 (Matrix Analysis $2.4.5$ 节)  

- 另一方面，选用对角阵也不合适，这个类别太小了，即我们面临存在性的问题:  
  并非所有的复方阵都能相似对角化.  

我们可以在上三角阵和对角阵之间做一个巧妙的折中:  
**Jordan 矩阵**是一种特殊的分块对角阵，任意复方阵都可通过相似变换得到本质上唯一的 Jordan 矩阵.   
两个 Jordan 矩阵相似的充分必要条件是它们有相同的对角分块 (不考虑排列顺序)   
此外，在 Jordan 矩阵的相似等价类中，没有其他矩阵在严格意义上具有比 Jordan 矩阵更少的非零非对角元.

> 相似关系仅仅是矩阵理论中众多有意义的等价关系中的一种.  
> 对于任意一个矩阵集合上的等价关系，我们总希望能够确定给定的矩阵 $A,B$ 是否在同一个等价类中.  
> 对于这样的决定性问题，一个经典的方法是对给定的等价关系辨识出一组代表矩阵，称为**标准型** (canonical form)  
> 为检验给定的矩阵 $A,B$ 是否等价，只需检验它们的标准型是否等价.

Jordan 标准型是**相似标准型** (canonical form of similarity) 中最重要的一种.



### 3.2.1 存在性证明

设 $k$ 为正整数，给定复数 $\lambda\in \mathbb C$，我们将特征值 $\lambda$ 的 $k$ 阶 Jordan 块 $J_k(\lambda)\in \mathbb C^{k\times k}$ 定义为:
$$
J_1 (\lambda):= [\lambda]\\
J_2 (\lambda):= \begin{bmatrix}
\lambda & 1\\
& \lambda\end{bmatrix}\\
J_k :=
\begin{bmatrix}
\lambda & 1 & & \\
& \lambda & 1 & \\
&&\ddots & \ddots &\\
&& &\lambda & 1 \\
&&&&\lambda
\end{bmatrix}_{k\times k}
$$
一个 Jordan 矩阵 $J\in \mathbb C^{n\times n}$ 是若干个 Jordan 块的直和:  
$$
J := J_{n_1}(\lambda_1)\oplus \dotsm \oplus J_{n_d}(\lambda_d)\quad (n_1+\dotsm + n_d = n)
$$
本节的主要结果是:   
每个复方阵都与一个本质上唯一的 Jordan 矩阵相似 (即在不考虑对角分块排列的时候是唯一的)    
我们分三步来证明这个结果 (其中有两步已经完成了):

- 第一步:    
  任意给定 $A\in \mathbb C^{n\times n}$，设其不同特征值 $\lambda_1,\dots,\lambda_d$ 的代数重数分别为 $n_1,\dots,n_d$   
  Schur 分解定理保证了 $A$ 酉相似于一个 $d\times d$ 分块上三角阵 $T=[T_{ij}]_{i,j=1}^d$  
  其中 $T_{ij}\in \mathbb R^{n_i\times n_j}$，且每一个对角分块 $T_{ii}$ 分别是对角元全为 $\lambda_i$ 的上三角阵.

  > **(Schur 分解定理, Matrix Analysis 定理 $2.3.1$)**  
  > 设 $A=[a_{i,j}]\in \mathbb C^{n\times n}$ 的特征值为 $\lambda_1,\dots,\lambda_n$ (按任意指定的次序排列).  
  > 则存在一个酉矩阵 $U\in \mathbb C^{n\times n}$ 使得 $T:=U^{\mathrm H} A U = [t_{ij}]$ 是以 $\lambda_1,\dots,\lambda_n$ 为对角元的上三角阵.
  >
  > - 特殊地，若 $A\in \mathbb R^{n\times n}$ 且 $\lambda_1,\dots,\lambda_n$ 均为实数，  
  >   则存在一个实正交阵 $Q\in \mathbb R^{n\times n}$ 使得 $T:=Q^{\mathrm T} A Q = [t_{ij}]$ 是以 $\lambda_1,\dots,\lambda_n$ 为对角元的上三角阵.
  > - 可以验证:   
  >   若 $T = U^{\mathrm H}A^{\mathrm T}U$ 是定理描述的与 $A^{\mathrm T}$ 酉相似的上三角阵，取 $V=\bar U$ (逐元素共轭)，  
  >   则 $V^{\mathrm H}A V = (\bar U)^{\mathrm H} A \bar U = U^{\mathrm T} A \bar U = (U^{\mathrm H}A^{\mathrm T}U)^{\mathrm T} = T^{\mathrm T}$ 是一个下三角阵.  
  >   这表明 $A^{\mathrm T}$ 的 Schur 上三角分解和 $A$ 的 Schur 下三角分解是等价的.
  > - **(Matrix Analysis $2.4.5$ 节)**  
  >   即使固定对角元次序，Schur 分解得到的上三角阵 $T$ 也不一定是唯一的   
  >   也就是说，具有相同主对角线的不同的上三角阵可能是酉相似的.

- 第二步:   
  第一步中的 $d\times d$ 分块上三角阵 $T=[T_{ij}]_{i,j=1}^d$ 相似于 $T_{11}\oplus \dotsm \oplus T_{dd}$ 

  **(Matrix Analysis 定理 $2.4.6.1$)**  
  设 $A\in \mathbb C^{n\times n}$ 的不同特征值 $\lambda_1,\dots,\lambda_d$ 的代数重数分别为 $n_1,\dots,n_d$   
  Schur 分解定理保证了 $A$ 酉相似于一个 $d\times d$ 分块上三角阵 $T=[T_{ij}]_{i,j=1}^d$  
  其中 $T_{ij}\in \mathbb R^{n_i\times n_j}$，且每一个对角分块 $T_{ii}$ 分别是对角元全为 $\lambda_i$ 的上三角阵.    
  因此 $A$ 就相似于 $T_{11}\oplus \dotsm \oplus T_{dd}$ (因为二者具有完全一致的特征值)    
  总之，存在酉矩阵 $U\in \mathbb C^{n\times n}$ 和非奇异矩阵 $S\in \mathbb C^{n\times n}$ 使得:  
  $$
  S^{-1}(U^{\mathrm H}AU)S = S^{-1}TS = \begin{bmatrix}
  T_{11} & &\\
  &\ddots &\\
  && T_{dd}
  \end{bmatrix}
  $$
  **邵老师提供的证明:**   
  考虑 $2\times 2$ 分块的情况，我们假设存在 $X$ 使得:
  $$
  \begin{align}
  \begin{bmatrix}
  T_{11} & T_{12}\\
  & T_{22}
  \end{bmatrix}
  &=
  \begin{bmatrix}
  I & X\\
  & I
  \end{bmatrix}
  \begin{bmatrix}
  T_{11} & \\
  & T_{22}
  \end{bmatrix}
  \begin{bmatrix}
  I & -X\\
  & I
  \end{bmatrix}\quad (\text{note that }\begin{bmatrix}
  I & X\\
  & I
  \end{bmatrix}^{-1} = \begin{bmatrix}
  I & -X\\
  & I
  \end{bmatrix})\\
  
  &=
  \begin{bmatrix}
  T_{11} & XT_{22}\\
  & T_{22}
  \end{bmatrix}
  \begin{bmatrix}
  I & -X\\
  & I
  \end{bmatrix}\\
  &=
  \begin{bmatrix}
  T_{11} & XT_{22} - T_{11}X\\
  & T_{22}
  \end{bmatrix}
  
  \end{align}
  $$
  要使上式成立，就等价于 $X$ 是 Sylvester 方程 $XT_{22}-T_{11}X=0$ 的解.  
  由于 $T_{11},T_{22}$ 没有公共特征值，故根据 Sylvester 定理可知上述 Sylvester 方程具有唯一解.
  
  > **(Sylvester 定理, Matrix Analysis 定理 $2.4.4.1$)**  
  > 设 $A\in \mathbb C^{m\times m},B\in \mathbb C^{n\times n},C\in \mathbb C^{m\times n}$   
  > 当且仅当 $A,B$ 没有公共特征值 (即 $\text{eig}(A)\cap \text{eig}(B) = \emptyset$) 时，Sylvester 方程 $AX-XB=C$ 有唯一解 $X\in \mathbb C^{m\times n}$ 
  
  根据数学归纳法容易将上述结论推广到 $d\times d$ 分块的情况，我们只需递归地对其进行 $2\times 2$ 划分即可.  
  命题得证.
  
- 第三步:  
  主对角元均为 $\lambda \in \mathbb C$ 的上三角阵都相似于一个 Jordan 矩阵.  
  这便是本节我们要证明的事实.

*****

**(幂零 Jordan 块的性质, Matrix Analysis 引理 $3.1.4$)**  
给定正整数 $n\geq 2$ 和 $x\in \mathbb{C}^n$，记 $\mathbb C^n$ 的第 $i$ 个标准单位基向量为 $e_i$，则我们有:

* $J_n(0)e_{i+1} = e_i\ (\forall\ i=1,\dots,n-1)$
* $J_n(0)^{\mathrm T}J_n(0) = \begin{bmatrix} 0 &\\ & I_{n-1} \end{bmatrix}$  
* $[I_n - J_n(0)^{\mathrm T}J_n(0)]x = (x^{\mathrm T}e_1)e_1$
* $J_n(0)^k = \begin{cases} 
  I_n & \text{if }k=0\\
  \begin{bmatrix}  & I_{n-k}\\ 0_{k\times k} &  \end{bmatrix} 
  &\text{if }1 \leq k < n\\ 
  0_{n\times n} & \text{if }k\geq n\end{cases}$ 

**(严格上三角阵的 Jordan 标准型, Matrix Analysis 定理 $3.1.5$)**  
若 $T\in \mathbb C^{n\times n}$ 是严格上三角阵，则存在一个非奇异矩阵 $S\in \mathbb C^{n\times n}$ 和正整数 $n_1\geq \dots \geq n_p\geq 1$ 使得:   
$$
S^{-1}TS = J_{n_1}(0)\oplus \dotsm \oplus J_{n_p}(0)
$$
若 $T$ 是实方阵，则相似矩阵 $S$ 也可取成实的.

- **推论:**   
  若 $T\in \mathbb C^{n\times n}$ 是主对角元均为 $\lambda \in \mathbb C$ 的上三角阵，则 $T-\lambda I_n$ 为严格上三角阵  
  根据上述定理可知存在一个非奇异矩阵 $S\in \mathbb C^{n\times n}$ 和正整数 $n_1\geq \dots \geq n_p\geq 1$ 使得:  
  $$
  S^{-1}(T-\lambda I_n)S = J_{n_1}(0)\oplus \dotsm \oplus J_{n_p}(0)
  $$
  进而有:  
  $$
  \begin{align}
  S^{-1}TS 
  &= J_{n_1}(0)\oplus \dotsm \oplus J_{n_p}(0) + \lambda I_n\\
  &= J_{n_1}(\lambda) \oplus \dotsm \oplus J_{n_p}(\lambda)
  \end{align}
  $$

**证明:**   
我们使用数学归纳法.  
当 $n=1$ 时，命题显然成立.  
当 $n>1$ 时，假设命题对所有小于 $n$ 阶的严格上三角阵都成立.  
我们可将 $T$ 分块成 $T=\begin{bmatrix} 0  & a^{\mathrm T} \\  & T_1 \end{bmatrix}$   
其中 $a\in \mathbb{C}^{n-1}$， $T_1$ 为 $n-1$ 阶的严格上三角阵.  

根据归纳假设，存在 $n-1$ 阶非奇异阵 $S_1$ 使得:
$$
S_1^{-1} T_1S_1 = \begin{bmatrix} J_{n_1}(0) & & &\\ &J_{n_2}(0)&& \\ &&\ddots&\\ && &J_{n_k}(0) \end{bmatrix} = \begin{bmatrix} J_{n_1}(0) &\\  & J \end{bmatrix}
$$
其中 $n_1\geq n_2\geq \dots \geq n_k \geq 1$ 满足 $n_1+n_2 + \dots + n_k = n-1$   
而 $J = J_{n_2}(0) \oplus \dots \oplus J_{n_k}(0)$ 为 $n-1-n_1$ 阶 Jordan 矩阵，显然它满足 $J^{n_1} = 0_{n_1\times n_1}$  

于是我们有:
$$
\begin{bmatrix} 1 & \\ & S_1^{-1} \end{bmatrix}\begin{bmatrix} 0  & a^{\mathrm T} \\ & T_1 \end{bmatrix} \begin{bmatrix} 1 & \\ & S_1 \end{bmatrix} = \begin{bmatrix} 0 & a^{\mathrm T} S_1\\ & S_1^{-1} T_1 S_1 \end{bmatrix}
$$
将 $S_1^{-1}T_1 S_1$ 分块为 $S_1^{-1}T_1 S_1=J_{n_1}(0) \oplus J$  
并对应地将 $a^{\mathrm T}S_1$ 分块为 $[a_1^{\mathrm T},a_2^{\mathrm T}]$，其中 $a_1,a_2$ 维数分别为 $n_1$ 和 $n-1-n_1$   
则我们有: 
$$
\begin{bmatrix} 0 & a^{\mathrm T} S_1\\ & S_1^{-1} T_1 P_1 \end{bmatrix} 
= 
\left[\begin{array}{c|cc} 0 & a_1^{\mathrm T} & a_2^{\mathrm T}\\ \hline &  J_{n_1}(0) & \\ & & J \end{array}\right]
$$
考虑如下的相似变换:
$$
\begin{align}
&\begin{bmatrix} 1 & -a_1^{\mathrm T} J_{n_1}(0) &\\ &I_{n_1}&\\ && I \end{bmatrix}
\begin{bmatrix} 0 & a_1^{\mathrm T} & a_2^{\mathrm T}\\ &  J_{n_1}(0) & \\ & & J \end{bmatrix} 
\begin{bmatrix} 1 & a_1^{\mathrm T} J_{n_1}(0) &\\ &I_{n_1}&\\ && I \end{bmatrix}\\
&= 
\begin{bmatrix} 0 & a_1^{\mathrm T}(I_{n_1}-J_{n_1}^{\mathrm T}(0)J_{n_1}(0)) & a_2^{\mathrm T}\\ &  J_{n_1}(0) & \\  & & J \end{bmatrix} \quad(\text{note that }J_{n_1}^{\mathrm T}(0)J_{n_1}(0) = I_{n_1}-e_1e_1^{\mathrm T})\\
&=
\begin{bmatrix} 0 & (a_1^{\mathrm T}e_1)e_1^{\mathrm T} & a_2^{\mathrm T}\\ &  J_{n_1}(0) & \\ & & J \end{bmatrix}
\end{align}
$$
其中 $e_1$ 为对应维数的 Euclid 空间的第 $1$ 个单位标准基向量.

- ① 若 $a_1^{\mathrm T}e_1 = 0$，则上式化为 $\begin{bmatrix} 0 &  & a_2^{\mathrm T}\\ &  J_{n_1}(0) & \\ & & J \end{bmatrix}$ 并置换相似于 $\begin{bmatrix} J_{n_1}(0) &  & \\ &  0 & a_2^{\mathrm T}\\ & & J \end{bmatrix}$   
  根据归纳假设，$n-n_1$ 阶严格上三角阵 $\begin{bmatrix} 0 & a_2^{\mathrm T}\\ 0 & J \end{bmatrix}$ 相似于一个 $n-n_1$ 阶 Jordan 矩阵  
  因此 $\begin{bmatrix} J_{n_1}(0) &  & \\ &  0 & a_2^{\mathrm T}\\ & & J \end{bmatrix}$ 相似于一个 $n$ 阶 Jordan 矩阵，于是命题对 $n$ 阶的情况也成立.

- ② 若 $a_1^{\mathrm T}e_1 \neq 0$，则可对 $\begin{bmatrix} 0 & (a_1^{\mathrm T}e_1)e_1^{\mathrm T} & a_2^{\mathrm T}\\ &  J_{n_1}(0) & \\ & & J \end{bmatrix}$ 做如下相似变换:
  $$
  \begin{align}
  &\begin{bmatrix} \frac{1}{a_1^{\mathrm T}e_1} & &\\ &I_{n_1}& \\ && \frac1{a_1^{\mathrm T}e_1}I \end{bmatrix} 
  \begin{bmatrix} 0 & (a_1^{\mathrm T}e_1)e_1 & a_2^{\mathrm T}\\ &  J_{n_1}(0) & \\ & & J \end{bmatrix} 
  \begin{bmatrix} (a_1^{\mathrm T}e_1) & &\\ &I_{n_1}& \\ && (a_1^{\mathrm T}e_1)I \end{bmatrix} \\
  &=
  \begin{bmatrix} 0 & e_1^{\mathrm T} & a_2^{\mathrm T}\\ &  J_{n_1}(0) & \\ & & J \end{bmatrix}
  \quad (\text{note that }J_{n_1+1}(0) = \begin{bmatrix} 0 & e_1^{\mathrm T}\\  & J_{n_1}(0) \end{bmatrix}\text{ and }
  e_1a_2^{\mathrm T} = 
  \begin{bmatrix}
  a_2^{\mathrm T}\\
  0_{n_1\times (n-1-n_1)}
  \end{bmatrix})\\
  &=
  \begin{bmatrix}  J_{n_1+1}(0) & e_1a_2^{\mathrm T}\\  & J \end{bmatrix}
  
  \end{align}
  $$
  其中 $e_1$ 为对应维数的 Euclid 空间的第 $1$ 个单位标准基向量.  

  对 $\begin{bmatrix}  J_{n_1+1}(0) & e_1a_2^{\mathrm T}\\  & J \end{bmatrix}$ 做如下的相似变换:
  $$
  \begin{align}
  &\begin{bmatrix} I_{n_1+1} & e_2a_2^{\mathrm T}\\ & I \end{bmatrix} 
  \begin{bmatrix} J_{n_1+1}(0) & e_1a_2^{\mathrm T}\\ & J \end{bmatrix} 
  \begin{bmatrix} I_{n_1+1} & -e_2a_2^{\mathrm T}\\ & I \end{bmatrix} \\
  &=
  \begin{bmatrix} J_{n_1+1}(0) & - J_{n_1+1}(0)e_2a_2^{\mathrm T}+e_1a_2^{\mathrm T} +e_2a_2^{\mathrm T} J\\ & J \end{bmatrix} 
  \quad (\text{note that }J_{n_1+1}(0) e_2 = e_1)\\
  &=
  \begin{bmatrix} J_{n_1+1}(0) & e_2a_2^{\mathrm T} J\\ & J \end{bmatrix}
  \end{align}
  $$
  一般来说，可对 $\begin{bmatrix}  J_{n_1+1}(0) & e_ka_2^{\mathrm T} J^{k-1}\\  & J \end{bmatrix}\ (k=1,\dots,n_1)$ 做如下相似变换:  
  $$
  \begin{align}
  &\begin{bmatrix} I_{n_1+1} & e_{k+1}a_2^{\mathrm T}J^{k-1}\\ & I \end{bmatrix} 
  \begin{bmatrix} J_{n_1+1}(0) & e_ka_2^{\mathrm T} J^{k-1}\\ & J \end{bmatrix} 
  \begin{bmatrix} I_{n_1+1} & -e_{k+1}a_2^{\mathrm T}J^{k-1}\\ & I \end{bmatrix} \\
  &=
  \begin{bmatrix} J_{n_1+1}(0) & - J_{n_1+1}(0)e_{k+1}a_2^{\mathrm T} J^{k-1}+e_ka_2^{\mathrm T} J^{k-1} +e_ka_2^{\mathrm T} J^k\\ & J \end{bmatrix} 
  \quad (\text{note that }J_{n_1+1}(0) e_{k+1} = e_k)\\
  &=
  \begin{bmatrix} J_{n_1+1}(0) & e_ka_2^{\mathrm T} J^k\\ & J \end{bmatrix}
  \end{align}
  $$
  注意到 $J^{n_1} = 0_{n_1\times n_1}$ (因为 $J$ 中的幂零 Jordan 块的阶数均小于等于 $n_1$)  
  因此我们重复上述相似变换 $n_1$ 次便可将 $\begin{bmatrix}  J_{n_1+1}(0) & e_1a_2^{\mathrm T}\\  & J \end{bmatrix}$ 化为 $\begin{bmatrix}  J_{n_1+1}(0) & \\  & J \end{bmatrix}$   
  于是命题对 $n$ 阶的情况也成立.

综上所述，定理得证.

****

**(Jordan 标准型的存在性, Matrix Analysis 定理 $3.1.11$)**  
给定复方阵 $A\in \mathbb{C}^{n\times n}$  
设其互不相同的特征值为 $\lambda_1,\dots,\lambda_d$，代数重数分别为 $n_1,\dots,n_d$ (满足 $n = n_1 + \dots + n_d\  (n_1\geq \dotsm \geq n_d \geq 1)$)  
则存在一个非奇异阵 $S\in \mathbb{C}^{n\times n}$ 和一系列正整数 $\{n_1^{(i)}\}_{i=1}^{p_1},\dots,\{n_d^{(i)}\}_{i=1}^{p_d}$ 满足:
$$
\begin{cases}
n_1 = \sum_{i=1}^{p_1} n_1^{(i)} & (n_1^{(1)}\geq \dotsm \geq n_1^{(p_1)}\geq 1)\\
\quad\dots\\
n_d =  \sum_{i=1}^{p_d} n_d^{(i)} & (n_d^{(1)}\geq \dotsm \geq n_d^{(p_d)}\geq 1)\\
\end{cases}
$$
使得:
$$
{\begin{cases}
J^{(1)}(\lambda_1) = J_{n_1^{(1)}} (\lambda_1) \oplus \dotsm \oplus J_{n^{(p_1)}_1} (\lambda_1)\\
\qquad\quad\dotsm\\
J^{(d)}(\lambda_d) = J_{n_d^{(1)}} (\lambda_d) \oplus \dotsm \oplus J_{n^{(p_d)}_d} (\lambda_d)\\
\end{cases}}\\
S^{-1}AS = J = J^{(1)}(\lambda_1) \oplus \dotsm \oplus J^{(d)}(\lambda_d)
$$
如果 $A$ 退化为仅有实特征值的实方阵，则相似矩阵 $S$ 也可取为实方阵.

**证明:**     
Schur 分解定理和 Matrix Analysis 定理 $2.4.6.1$ 保证了存在一个酉矩阵 $U\in \mathbb C^{n\times n}$ 和非奇异矩阵 $S_0\in \mathbb C^{n\times n}$ 使得:
$$
U^{\mathrm H}AU = T\ (\text{where }\text{diag}(T) = T_{11}\oplus \dotsm \oplus T_{dd} = \lambda_1 I_{n_1}\oplus \dotsm \oplus \lambda_d I_{n_d})\\
S^{-1}TS = \text{diag}(T) = T_{11}\oplus \dotsm \oplus T_{dd} = \lambda_1 I_{n_1}\oplus \dotsm \oplus \lambda_d I_{n_d}
$$
根据 Matrix Analysis 定理 $3.1.5$ 及其推论我们又可知道，对于任意 $k=1,\dots,d$，  
都存在一个非奇异阵 $S_k\in \mathbb C^{n_k\times n_k}$ 和一系列满足 $\sum_{i=1}^{p_k} n_k^{(i)}=n_k$ 的正整数 $n_k^{(1)}\geq \dotsm \geq n_k^{(p_k)}\geq 1$ 使得:
$$
S_k^{-1}T_{kk}S_k = J^{(k)}(\lambda_k) = J_{n_k^{(1)}}(\lambda_k) \oplus \dotsm \oplus J_{n_k^{(p_k)}}(\lambda_k)
$$
若取 $S = US_0(S_1\oplus \dotsm \oplus S_d)$，则我们有:
$$
\begin{align}
S^{-1}AS
&=
(S_1\oplus \dotsm \oplus S_d)^{-1} S_0^{-1} U^{\mathrm H}\cdot A\cdot US_0(S_1\oplus \dotsm \oplus S_d)\\
&=
(S_1^{-1}\oplus \dotsm \oplus S_d^{-1}) S_0^{-1}T S_0 (S_1\oplus \dotsm \oplus S_d)\\
&=
(S_1^{-1}\oplus \dotsm \oplus S_d^{-1}) (T_{11}\oplus \dotsm \oplus T_{dd}) (S_1\oplus \dotsm \oplus S_d)\\
&=
(S_1^{-1}T_{11}S_1)\oplus \dotsm \oplus (S_d^{-1}T_{dd}S_d)\\
&=
J^{(1)}(\lambda_1)\oplus \dotsm \oplus J^{(d)}(\lambda_d)\\
&=
J
\end{align}
$$
实际上，与 $A$ 相似的 Jordan 矩阵 $J = J_1(\lambda_1) \oplus \dotsm \oplus J_d(\lambda_d)$ 是唯一的 (不考虑直和项的排列)  
这个 Jordan 矩阵 $J\in \mathbb C^{n\times n}$ 即为 $A\in \mathbb C^{n\times n}$ 的 **Jordan 标准型**.   
我们称 $J_k(\lambda_k)\ (k=1,\dots,d)$ 为**相似的标准分块** (canonical blocks for similarity)  
它们由 $A$ 唯一确定 (不考虑直和项的排列)

但上述定理只说明了 Jordan 标准型的存在性，我们该如何说明其唯一性 (不考虑直和项的排列) 呢?



### 3.2.2 唯一性证明

Jordan 标准型的唯一性论断基于以下两个事实:

- ① 秩是一个相似不变量
- ② 若两个相似矩阵通过一个纯量矩阵 (形如 $\lambda I$) 进行平移，则平移后它们仍是相似的.

***

注意到非幂零 Jordan 块和幂零 Jordan 块可通过幂运算来区分:
$$
\rank(J_m(\lambda)^k) = \begin{cases}
m\ (\forall\ k\in \mathbb Z_+) & \text{if }\lambda \neq 0\\
\begin{cases}
m - k - 1 & \text{if }0\leq k\leq m-1\\
0 & \text{if }k\geq m
\end{cases} & \text{if }\lambda = 0
\end{cases}
$$
因此非幂零 Jordan 块求幂不会出现降秩，而幂零 Jordan 块求幂会逐步降秩，直到秩为零.  
这启发我们用 $\rank ((A-\lambda I_n)^k) = \rank ((J-\lambda I_n)^k)$ 来刻画特征值 $\lambda$ 对应的 Jordan 块的个数.

设 $A\in \mathbb C^{n\times n},\lambda \in \mathbb C,k\in \mathbb N$，我们定义:   
$$
r_k(A,\lambda) := \begin{cases}
n & \text{if }k=0\\
\rank((A-\lambda I_n)^k) & \text{if }k\geq 1
\end{cases}\\

w_k (A,\lambda) := \begin{cases}
n - r_1(A,\lambda) & \text{if }k=1\\
r_{k-1}(A,\lambda) - r_k(A,\lambda) & \text{if }k\geq 2
\end{cases}
$$
我们定义 $A$ 关于特征值 $\lambda$ (设代数重数为 $n_\lambda$) 的 **Weyr 特征**为 $w(A,\lambda):= (w_1(A,\lambda),\dots,w_{n_\lambda}(A,\lambda))$   

设 $J$ 是一个与 $A$ 相似的 Jordan 矩阵，显而易见 $w(J,\lambda)=w(A,\lambda)$   
因此我们可以看出 $w_k(A,\lambda)$ 等于与特征值 $\lambda$ 对应的 Jordan 块中阶数 $\geq k$ 的个数.  
于是 $w_k(A,\lambda) - w_{k+1}(A,\lambda)$ 即为与特征值 $\lambda$ 对应的 Jordan 块中阶数 $= k$ 的个数.  
这表明一个与 $A$ 相似的 Jordan 矩阵 $J$ 的构造完全由 $A$ 关于不同特征值的 Weyr 特征所决定.  
我们将上述讨论总结为以下引理:

**(Matrix Analysis 引理 $3.1.18$)**  

- 设 $\lambda\in \mathbb C$ 是 $A\in \mathbb C^{n\times n}$ 的一个给定的特征值，代数重数是 $n_\lambda$   
  若 $w_1(A,\lambda),\dots,w_{n_\lambda}(A,\lambda)$ 的与 $\lambda\in \mathbb C$ 相关的 Weyr 特征，  
  则一个与 $A$ 相似的 Jordan 矩阵 $J$ 中与 $\lambda$ 相关的 $k$ 阶 Jordan 块 $J_k(\lambda)$ 的个数为 $w_k(A,\lambda)-w_{k+1}(A,\lambda)\ (k=1,\dots,n_\lambda)$ 
- 两个复方阵 $A,B\in \mathbb C^{n\times n}$ 相似，当且仅当:
  - ① 它们具有相同的特征值 $\lambda_1,\dots,\lambda_d$ 及其代数重数 (这 $d$ 个特征值之间互不相同)
  - ② 每一个与特征值相关的 Weyr 特征都是相同的.  
    即 $w_k(A,\lambda_i) = w_k(B,\lambda_i)\ (\forall\ i=1,\dots,d,k = 1,\dots,n_i)$ 

如果 $A$ 可相似变换为两个 Jordan 矩阵 $J,\widetilde J$ (其中 $\widetilde J \neq J$)，  
那么它们关于任意特征值的 Weyr 特征都是相同的，即关于任意特征值的 Jordan 块都是相同的 (不考虑排列)
**Jordan 标准型的唯一性得证.**

*****

现在我们给出完整的 Jordan 标准型定理:    
**(Jordan 标准型定理, Matrix Analysis 定理 $3.1.11$)**    
给定复方阵 $A\in \mathbb{C}^{n\times n}$  
设其互不相同的特征值为 $\lambda_1,\dots,\lambda_d$，代数重数分别为 $n_1,\dots,n_d$ (满足 $n = n_1 + \dots + n_d\  (n_1\geq \dotsm \geq n_d \geq 1)$)  
则存在一个非奇异阵 $S\in \mathbb{C}^{n\times n}$ 和**由 $A$ 唯一确定的**一系列正整数 $\{n_1^{(i)}\}_{i=1}^{p_1},\dots,\{n_d^{(i)}\}_{i=1}^{p_d}$ 满足:
$$
\begin{cases}
n_1 = \sum_{i=1}^{p_1} n_1^{(i)} & (n_1^{(1)}\geq \dotsm \geq n_1^{(p_1)}\geq 1)\\
\quad\dots\\
n_d =  \sum_{i=1}^{p_d} n_d^{(i)} & (n_d^{(1)}\geq \dotsm \geq n_d^{(p_d)}\geq 1)\\
\end{cases}
$$
使得:
$$
{\begin{cases}
J^{(1)}(\lambda_1) = J_{n_1^{(1)}} (\lambda_1) \oplus \dotsm \oplus J_{n^{(p_1)}_1} (\lambda_1)\\
\qquad\quad\dotsm\\
J^{(d)}(\lambda_d) = J_{n_d^{(1)}} (\lambda_d) \oplus \dotsm \oplus J_{n^{(p_d)}_d} (\lambda_d)\\
\end{cases}}\\
S^{-1}AS = J = J^{(1)}(\lambda_1) \oplus \dotsm \oplus J^{(d)}(\lambda_d)
$$
其中 Jordan 矩阵由 $A$ 唯一确定 (不考虑直和项的排列)，称为 $A$ 的 Jordan 标准型.  
特殊地，如果 $A$ 退化为仅有实特征值的实方阵，则相似矩阵 $S$ 也可取为实方阵.



### 3.2.3 数值稳定性

值得注意的是，在前文定理的证明中，推导 Jordan 标准型的过程是一个明确的算法.  
尽管这个算法在理论上可以用来计算已知矩阵的 Jordan 标准型，但在实际应用中的数值稳定性是很差的.  
事实上，到目前为止仍没有一种数值稳定的算法能够计算方阵的 Jordan 标准型.  
其根源在于，$\rank(A)$ 不是方阵 $A$ 元素的连续函数，Jordan 标准型也不是.  
方阵元素的微小扰动就可能导致其 Jordan 标准型的剧烈变化.

考虑以下 $k$ 阶幂零 Jordan 矩阵和单元素扰动:
$$
J_k(0) = 
\begin{bmatrix}
0 & 1 & & \\
& 0 & 1 & \\
&&\ddots & \ddots &\\
&& &0 & 1 \\
&&&&0
\end{bmatrix}
\qquad
\Delta J = 
\begin{bmatrix}
0 &  & & \\
& 0 &  & \\
&&\ddots &  &\\
&& &0 &  \\
\varepsilon &&&&0
\end{bmatrix}\\

J_k(0) +\Delta J= 
\begin{bmatrix}
0 & 1 & & \\
& 0 & 1 & \\
&&\ddots & \ddots &\\
&& &0 & 1 \\
\varepsilon&&&&0
\end{bmatrix}
$$
注意到 $J_k(0)+\Delta J$ 的特征多项式为 $\lambda^k - \varepsilon = 0$ (其中 $\varepsilon = r_0 e^{i\theta_0} \in \mathbb C$ 满足 $|\varepsilon|=r_0\ll 1$)  
(这个特征多项式很容易根据 Frobenius 友阵的性质一眼看出来, 参见后文)  
记 $\omega = \exp(\frac{2\pi i}k)$ 为 $k$ 次单位根，则 $J_k(0)+\Delta J$ 的特征值为 $\sqrt[k]{r_0} e^{i\theta_0}, \sqrt[k]{r_0} e^{i\theta_0}\cdot\omega,\sqrt[k]{r_0} e^{i\theta_0}\cdot\omega^2,\dots,\sqrt[k]{r_0} e^{i\theta_0}\cdot \omega^{k-1}$   
扰动后的特征值会分布在以 $0$ 为圆心，以 $|\sqrt[k]{\varepsilon}| = |\sqrt[k]{r_0} e^{i\theta_0}| = \sqrt[k]{r_0}$ 为半径的圆内.  
随着 $k$ 增大，$\sqrt[k]{r_0}$ 会接近于 $1$，因此这些特征值会离 $0$ 越来越远，很小的扰动会被放得很大.

对应地，考虑以下关于 $\lambda\in \mathbb C$ 的 $k$ 阶 Jordan 矩阵和单元素扰动:  
$$
J_k(\lambda) +\Delta J = 
\begin{bmatrix}
\lambda & 1 & & \\
& \lambda & 1 & \\
&&\ddots & \ddots &\\
&& &\lambda & 1 \\
\varepsilon&&&&\lambda
\end{bmatrix} = (J_k(0) + \Delta J) + \lambda I_k
$$
根据前面的结论可知 $J_k(\lambda)+\Delta J$ 的特征值为 $\lambda + \sqrt[k]{r_0} e^{i\theta_0}, \lambda + \sqrt[k]{r_0} e^{i\theta_0}\cdot\omega,\lambda + \sqrt[k]{r_0} e^{i\theta_0}\cdot\omega^2,\dots,\lambda + \sqrt[k]{r_0} e^{i\theta_0}\cdot \omega^{k-1}$   
扰动后的特征值会分布在以 $\lambda$ 为圆心，以 $|\sqrt[k]{\varepsilon}| = |\sqrt[k]{r_0} e^{i\theta_0}| = \sqrt[k]{r_0}$ 为半径的圆内.  
随着 $k$ 增大，$\sqrt[k]{r_0}$ 会接近于 $1$，因此这些特征值会离 $\lambda$ 越来越远，很小的扰动会被放得很大.  
(难道特征值 $\lambda$ 很大时，特征值的扰动就不明显了吗? 不能说不明显，扰动已经是一个宏观量了)

> 实际上我们有更一般的结论:  
> **(Ostrowski 特征值扰动界)**  
> 给定 $A\in \mathbb C^{n\times n}$  
> 对于特征值扰动问题 $\begin{cases}
> Ax = x \lambda\\
> (A+\Delta A) \widetilde x =\widetilde x \widetilde \lambda\end{cases}$ 我们有 $|\widetilde \lambda - \lambda| \leq O(\sqrt[n]{\|\Delta A\|})$

有趣的是，加入这样的单元素扰动后，$J_k(\lambda) +\Delta J$ 便可对角化了 (重特征值分裂为单特征值)   
根据 **Bauer-Fiker 定理**可知:   
当 $\varepsilon\to 0$ 时，特征向量构成的矩阵 $X$ 会越来越病态，   
(即越来越接近于奇异，具体来说，是所有特征向量都趋近于同一个特征向量)  
因此特征值的扰动问题会越来越病态.   

> **(Bauer-Fiker 定理, Matrix Analysis 定理 $6.3.2$)**  
> 设 $A=S\Lambda S^{-1}\in \mathbb C^{n\times n}$ 是可对角化的矩阵.  
> 对于任意扰动 $E\in \mathbb C^{n\times n}$，我们有 $S^{-1}(A+E)S = \Lambda + S^{-1}ES$   
> 因此对于 $A+E$ 的任意扰动特征值 $\hat\lambda$ 都有 $A$ 的一个特征值 $\lambda$ 与之对应，满足:  
> $$
> |\hat\lambda - \lambda| \leq \|S^{-1}ES\| \leq \|S\|\|S^{-1}\| \|E\|
> $$
> 其中 $\|\cdot\|$ 是 $\mathbb C^{n}$ 上的某个绝对范数的诱导范数 (自然是相容函数)

尽管有这样的局限性， Jordan 标准型还是值得认真了解的，它是理论分析的重要工具.  
当我们要证明方阵有关的结论时，可以先证明它关于对角矩阵成立，  
下一步再证明它关于上三角矩阵或 Jordan 矩阵成立.  
又或者利用任一复方阵可由一个可对角化的方阵任意逼近的事实进行分析论证.  
(这个事实是基于一个有用的观察: 方阵的重特征值可以通过微小的扰动分解为不同的特征值)

***

有时下面的结论会很有帮助:  
Jordan 标准型可以将 Jordan 块中的 $+1$ 元素都取代为一个任意的 $\varepsilon\neq 0$ 

**(Matrix Analysis 推论 $3.1.21$)**   
设 $\varepsilon \neq 0\in \mathbb C$    
给定复方阵 $A\in \mathbb{C}^{n\times n}$  
设其互不相同的特征值为 $\lambda_1,\dots,\lambda_d$，代数重数分别为 $n_1,\dots,n_d$ (满足 $n = n_1 + \dots + n_d\  (n_1\geq \dotsm \geq n_d \geq 1)$)   
则存在一个非奇异阵 $S(\varepsilon)\in \mathbb{C}^{n\times n}$ 和**由 $A$ 唯一确定的**一系列正整数 $\{n_1^{(i)}\}_{i=1}^{p_1},\dots,\{n_d^{(i)}\}_{i=1}^{p_d}$ 满足:
$$
\begin{cases}
n_1 = \sum_{i=1}^{p_1} n_1^{(i)} & (n_1^{(1)}\geq \dotsm \geq n_1^{(p_1)}\geq 1)\\
\quad\dots\\
n_d =  \sum_{i=1}^{p_d} n_d^{(i)} & (n_d^{(1)}\geq \dotsm \geq n_d^{(p_d)}\geq 1)\\
\end{cases}
$$
使得:
$$
{\begin{cases}
J^{(1)}(\lambda_1,\varepsilon) = J_{n_1^{(1)}} (\lambda_1,\varepsilon) \oplus \dotsm \oplus J_{n^{(p_1)}_1} (\lambda_1,\varepsilon)\\
\qquad\quad\dotsm\\
J^{(d)}(\lambda_d,\varepsilon) = J_{n_d^{(1)}} (\lambda_d,\varepsilon) \oplus \dotsm \oplus J_{n^{(p_d)}_d} (\lambda_d,\varepsilon)
\end{cases}\quad (\text{where }
J_m(\lambda,\varepsilon) :=
\begin{bmatrix}
\lambda & \varepsilon & & \\
& \lambda & \varepsilon & \\
&&\ddots & \ddots &\\
&& &\lambda & \varepsilon \\
&&&&\lambda
\end{bmatrix}_{m\times m})}\\
S(\varepsilon)^{-1}AS(\varepsilon) = J(\varepsilon) = J^{(1)}(\lambda_1,\varepsilon) \oplus \dotsm \oplus J^{(d)}(\lambda_d,\varepsilon)
$$
其中 Jordan 矩阵 $J(\varepsilon)$ 由 $A$ 唯一确定 (不考虑直和项的排列)，称为 $A$ 的 Jordan 标准型.  
特殊地，如果 $A$ 退化为仅有实特征值的实方阵，则相似矩阵 $S(\varepsilon)$ 也可取为实方阵.

**证明: **  
根据 Jordan 标准型定理可知:    
存在一个非奇异阵 $S_0\in \mathbb{C}^{n\times n}$ 和**由 $A$ 唯一确定的**一系列正整数 $\{n_1^{(i)}\}_{i=1}^{p_1},\dots,\{n_d^{(i)}\}_{i=1}^{p_d}$ 满足:
$$
\begin{cases}
n_1 = \sum_{i=1}^{p_1} n_1^{(i)} & (n_1^{(1)}\geq \dotsm \geq n_1^{(p_1)}\geq 1)\\
\quad\dots\\
n_d =  \sum_{i=1}^{p_d} n_d^{(i)} & (n_d^{(1)}\geq \dotsm \geq n_d^{(p_d)}\geq 1)\\
\end{cases}
$$
使得:
$$
{\begin{cases}
J^{(1)}(\lambda_1,1) = J_{n_1^{(1)}} (\lambda_1,1) \oplus \dotsm \oplus J_{n^{(p_1)}_1} (\lambda_1,1)\\
\qquad\quad\dotsm\\
J^{(d)}(\lambda_d,1) = J_{n_d^{(1)}} (\lambda_d,1) \oplus \dotsm \oplus J_{n^{(p_d)}_d} (\lambda_d,1)
\end{cases}
\quad (\text{where }
J_m(\lambda,1) :=
\begin{bmatrix}
\lambda & 1 & & \\
& \lambda & 1 & \\
&&\ddots & \ddots &\\
&& &\lambda & 1 \\
&&&&\lambda
\end{bmatrix}_{m\times m})}\\
S_0^{-1}AS_0 = J = J^{(1)}(\lambda_1,1) \oplus \dotsm \oplus J^{(d)}(\lambda_d,1)
$$
注意到:
$$
D_m(\varepsilon) = \text{diag}\{1,\varepsilon,\varepsilon^2,\dots,\varepsilon^{m-1}\}\\
D_m(\varepsilon)^{-1} J_m(\lambda,1) D_m(\varepsilon) = J_m(\lambda,\varepsilon)
$$
我们定义:
$$
\begin{cases}
D^{(1)}(\varepsilon) = D_{n_1^{(1)}}(\varepsilon)\oplus \dotsm \oplus D_{n_1^{(p_1)}}(\varepsilon)\\
\qquad\ \ \ \ \dotsm\\
D^{(d)}(\varepsilon) = D_{n_d^{(1)}}(\varepsilon)\oplus \dotsm \oplus D_{n_d^{(p_d)}}(\varepsilon)\\
D(\varepsilon) = D^{(1)}(\varepsilon) \oplus \dotsm \oplus D^{(d)}(\varepsilon)\\
S(\varepsilon)=S_0 D(\varepsilon)
\end{cases}
$$
则我们有:
$$
\begin{align}
S(\varepsilon)^{-1} A S(\varepsilon)
&=
D(\varepsilon)^{-1} (S_0^{-1}AS_0)D(\varepsilon)\\
&=
(D^{(1)}(\varepsilon) \oplus \dotsm \oplus D^{(d)}(\varepsilon))^{-1} J (D^{(1)}(\varepsilon) \oplus \dotsm \oplus D^{(d)}(\varepsilon))\\
&=
(D^{(1)}(\varepsilon)^{-1} \oplus \dotsm \oplus D^{(d)}(\varepsilon)^{-1}) (J^{(1)}(\lambda_1,1) \oplus \dotsm \oplus J^{(d)}(\lambda_d,1)) (D^{(1)}(\varepsilon) \oplus \dotsm \oplus D^{(d)}(\varepsilon))\\
&=
\{D^{(1)}(\varepsilon)^{-1} J^{(1)}(\lambda_1,1)D^{(1)}(\varepsilon)\} \oplus \dotsm \oplus
\{D^{(d)}(\varepsilon)^{-1} J^{(d)}(\lambda_d,1)D^{(d)}(\varepsilon)\}\\
&=
J^{(1)}(\lambda_1,\varepsilon) \oplus \dotsm \oplus J^{(d)}(\lambda_d,\varepsilon)\\
&=
J(\varepsilon)
\end{align}
$$
命题得证.



### 3.2.4 实 Jordan 标准型

与复方阵不同的是，实方阵的复特征值一定是成对共轭出现的.   
任意给定 $A\in \mathbb R^{n\times n}$ 的一对复共轭特征值 $\lambda,\bar \lambda\in \mathbb C$，我们都有:  
$$
\begin{align}
w_k(A,\lambda)
&=
\rank((A-\lambda I_n)^k)\\
&=
\rank(\overline{(A-\lambda I_n)^k})\\
&=
\rank(\{{\overline{A-\lambda I_n}}\}^k)\\
&=
\rank((A-\bar \lambda I_n)^k)\\
&=
w_k(A,\bar \lambda)
\end{align}\quad (k=1,2,\dots)
$$

> **(Matrix Analysis 引理 $3.1.18$)**  
>
> - 设 $\lambda\in \mathbb C$ 是 $A\in \mathbb C^{n\times n}$ 的一个给定的特征值，代数重数是 $n_\lambda$   
>   若 $w_1(A,\lambda),\dots,w_{n_\lambda}(A,\lambda)$ 的与 $\lambda\in \mathbb C$ 相关的 Weyr 特征，  
>   则一个与 $A$ 相似的 Jordan 矩阵 $J$ 中与 $\lambda$ 相关的 $k$ 阶 Jordan 块 $J_k(\lambda)$ 的个数为 $w_k(A,\lambda)-w_{k+1}(A,\lambda)\ (k=1,\dots,n_\lambda)$ 
> - 两个复方阵 $A,B\in \mathbb C^{n\times n}$ 相似，当且仅当:
>   - ① 它们具有相同的特征值 $\lambda_1,\dots,\lambda_d$ 及其代数重数 (这 $d$ 个特征值之间互不相同)
>   - ② 每一个与特征值相关的 Weyr 特征都是相同的.  
>     即 $w_k(A,\lambda_i) = w_k(B,\lambda_i)\ (\forall\ i=1,\dots,d,k = 1,\dots,n_i)$ 

**Matrix Analysis 引理 $3.1.18$** 确保了 $A$ 的任何关于特征值 $\lambda,\bar\lambda$ 的 Jordan 构造都是相同的.  
因此 $A$ 的复特征值的各阶 Jordan 分块中，同阶的分块总是成对共轭出现的.

考虑如下的 Jordan 矩阵:
$$
\begin{bmatrix}
J_k(\lambda) & \\
& J_k(\bar \lambda)\end{bmatrix} 
=
\left[\begin{array}{ccccc|ccccc}
\lambda & 1 & & &&&&& \\
& \lambda & 1 & &&&&& \\
&&\ddots & \ddots &&&&&\\
&& &\lambda & 1 &&&& \\
&&&&\lambda&&&&\\
\hline
&&&&&\bar\lambda & 1 & & & \\
&&&&&& \bar\lambda & 1 & & \\
&&&&&&&\ddots & \ddots &\\
&&&&&&& &\bar\lambda & 1 \\
&&&&&&&&&\bar\lambda
\end{array}\right]\in \mathbb C^{2k\times 2k}
$$
它置换相似于如下的分块上三角矩阵:   
只需将前 $k$ 行和后 $k$ 行交错排列，再将前 $k$ 列和后 $k$ 列交错排列即可.  
置换矩阵可通过将 $\mathbb C^{2k}$ 的前 $k$ 个和后 $k$ 个标准单位基向量交错排列.
$$
D_k(\lambda)
=
\begin{bmatrix}
D_1(\lambda) & I_2 & & & \\
& D_1(\lambda) & I_2 & & \\
&&\ddots & \ddots &\\
&& &D_1(\lambda) & I_2  \\
&&&&D_1(\lambda)\\
\end{bmatrix}\in \mathbb C^{2k\times 2k}\quad (\text{where }D_1(\lambda) = \begin{bmatrix}
\lambda & \\
& \bar \lambda\end{bmatrix}\in \mathbb C^{2\times 2})
$$
进一步，设 $\lambda = \alpha + i\beta$ (其中 $\alpha,\beta\in \mathbb C$)，则我们有:  
$$
C_1(\alpha,\beta) := 
\begin{bmatrix} 
\alpha & \beta\\
-\beta & \alpha
\end{bmatrix}
= U_1D_1(\lambda) U_1^{H} 
=
(\frac12\begin{bmatrix}
-i & -i\\
1 & -1
\end{bmatrix}) \cdot
\begin{bmatrix}
\lambda & \\
& \bar \lambda
\end{bmatrix}\cdot 
(\frac12\begin{bmatrix}
-i & -i\\
1 & -1
\end{bmatrix})^{\mathrm H}
$$
记 $U_k = \underset{k}{\underbrace{U_1\oplus \dotsm \oplus U_1}}\in \mathbb C^{n\times n}$ (它显然是酉矩阵)，则我们有:  
$$
\begin{align}
U_k D_k(\lambda) U_k^{\mathrm H}
&=
(\underset{k}{\underbrace{U_1\oplus \dotsm \oplus U_1}}) 
\begin{bmatrix}
D_1(\lambda) & I_2 & & & \\
& D_1(\lambda) & I_2 & & \\
&&\ddots & \ddots &\\
&& &D_1(\lambda) & I_2  \\
&&&&D_1(\lambda)\\
\end{bmatrix}
(\underset{k}{\underbrace{U_1\oplus \dotsm \oplus U_1}})^{\mathrm H}\\
&=
\begin{bmatrix}
U_1D_1(\lambda)U_1^{\mathrm H} & U_1I_2U_1^{\mathrm H} & & & \\
& U_1D_1(\lambda)U_1^{\mathrm H} & U_1I_2U_1^{\mathrm H} & & \\
&&\ddots & \ddots &\\
&& &U_1D_1(\lambda)U_1^{\mathrm H} & U_1I_2U_1^{\mathrm H}  \\
&&&&U_1D_1(\lambda)U_1^{\mathrm H}\\
\end{bmatrix}\\
&=
\begin{bmatrix}
C_1(\alpha,\beta) & I_2 & & & \\
& C_1(\alpha,\beta) & I_2 & & \\
&&\ddots & \ddots &\\
&& &C_1(\alpha,\beta) & I_2  \\
&&&&C_1(\alpha,\beta)\\
\end{bmatrix}\\
&=
C_k(\alpha,\beta)

\end{align}
$$
上述结论将我们引导到如下的实 Jordan 标准型定理:    
**(实 Jordan 标准型定理, Matrix Analysis 定理 $3.4.1.5$​)**    
给定实方阵 $A\in \mathbb{R}^{n\times n}$，设其互不相同的特征值为:

- $d_1$ 个实特征值 $\mu_1,\dots,\mu_{d_1}$，代数重数分别为 $n_1,\dots,n_{d_1}$
- $d_2$ 对复共轭特征值 $\begin{cases}
  \lambda_j = \alpha_j + i\beta_j\\
  \bar\lambda_j = \alpha_j - i\beta_j\end{cases} (j=1,\dots,d_2)$，代数重数分别为 $m_1,\dots,m_{d_2}$  
- 上述代数重数满足 $\begin{cases}
  n_1\geq \dotsm \geq n_{d_1} \geq 1\\
  m_1\geq \dotsm \geq m_{d_m} \geq 1\\
  n= n_1+\dotsm + n_{d_1} + 2(m_1 + \dotsm + m_{d_2})
  \end{cases}$

则存在一个实非奇异阵 $S\in \mathbb{R}^{n\times n}$ 和**由 $A$ 唯一确定的**一系列正整数 $\{n_1^{(j)}\}_{j=1}^{p_1},\dots,\{n_{d_1}^{(j)}\}_{j=1}^{p_{d_1}},\{m_1^{(j)}\}_{j=1}^{q_{1}},\dots,\{m_{d_2}^{(j)}\}_{j=1}^{q_{d_2}}$ 满足:  
$$
\begin{cases}
n_1 = \sum_{j=1}^{p_1} n_1^{(j)} & (n_1^{(1)}\geq \dotsm \geq n_1^{(p_1)}\geq 1)\\
\quad\dots\\
n_{d_1} =  \sum_{j=1}^{p_{d_1}} n_{d_1}^{(j)} & (n_{d_1}^{(1)}\geq \dotsm \geq n_{d_1}^{(p_{d_1})}\geq 1)\\
m_{1} = \sum_{j=1}^{q_1} m_1^{(j)} & (m_1^{(1)}\geq \dotsm \geq m_1^{(q_1)}\geq 1)\\
\quad\dots\\
m_{d_2} = \sum_{j=1}^{q_{d_2}} m_{d_2}^{(j)} & (m_{d_2}^{(1)}\geq \dotsm \geq m_{d_2}^{(q_{d_2})}\geq 1)\\
\end{cases}
$$
使得:
$$
{\begin{cases}
J^{(1)}(\mu_1) = J_{n_1^{(1)}} (\mu_1) \oplus \dotsm \oplus J_{n^{(p_1)}_1} (\mu_1)\\
\qquad\quad\dotsm\\
J^{(d_1)}(\mu_{d_1}) = J_{n_{d_1}^{(1)}} (\mu_{d_1}) \oplus \dotsm \oplus J_{n^{(p_{d_1})}_{d_1}} (\mu_{d_1})\\
C^{(1)} (\alpha_1,\beta_1) = C_{m_{1}^{(1)}}(\alpha_1,\beta_1)\oplus \dotsm \oplus C_{m_{1}^{(q_1)}}(\alpha_1,\beta_1)\\
\qquad\quad\dotsm\\
C^{(d_2)} (\alpha_{d_2},\beta_{d_2}) = C_{m_{d_2}^{(1)}}(\alpha_{d_2},\beta_{d_2})\oplus \dotsm \oplus C_{m_{d_2}^{(q_{d_2})}}(\alpha_{d_2},\beta_{d_2})\\
\end{cases}}\\
S^{-1}AS = J^{(1)}(\mu_1) \oplus \dotsm \oplus J^{(d_1)}(\mu_{d_1}) \oplus C^{(1)}(\alpha_1,\beta_1) \oplus
\dotsm \oplus C^{(d_2)}(\alpha_{d_2},\beta_{d_2})
$$
我们称 $S^{-1}AS = J^{(1)}(\mu_1) \oplus \dotsm \oplus J^{(d_1)}(\mu_{d_1}) \oplus C^{(1)}(\alpha_1,\beta_1) \oplus
\dotsm \oplus C^{(d_2)}(\alpha_{d_2},\beta_{d_2})$ 为 $A\in \mathbb R^{n\times n}$ 的**实 Jordan 标准型**  
其中相似矩阵 $S\in \mathbb{R}^{n\times n}$ 之所以可以是实方阵，  
是因为我们已经证明了 $A$ 与它的实 Jordan 标准型在复数域上是相似的，  
进而根据 Matrix Analysis 定理 $1.3.29$ 可知 $A$ 与它的实 Jordan 标准型在实数域上也是相似的.

> **(Matrix Analysis 定理 $1.3.29$)**   
> 两个在复数域上相似的实矩阵在实数域上也是相似的.  

*****

下面的推论总结了几个复方阵与实方阵相似的有用的判别法:  
**(Matrix Analysis 推论 $3.4.1.7$)**  
对于给定复方阵 $A\in \mathbb C^{n\times n}$，下列命题等价:

- ① $A$ 相似于一个实方阵
- ② 对于 $A$ 的任意非零特征值 $\lambda\in \mathbb C$ 以及正整数 $k\in \mathbb Z_+$，$k$ 阶 Jordan 块 $J_k(\lambda),J_k(\bar \lambda)$ 的个数相等.
- ③ 对于 $A$ 的任意非实特征值 $\lambda\in \mathbb C\backslash\mathbb R$ 以及正整数 $k\in \mathbb Z_+$，$k$ 阶 Jordan 块 $J_k(\lambda),J_k(\bar \lambda)$ 的个数相等.
- ④ 对于 $A$ 的任意非实特征值 $\lambda\in \mathbb C\backslash\mathbb R$，$\lambda,\bar \lambda$ 具有相同的 Weyr 特征，  
  即 $\rank((A-\lambda I_n)^k) = \rank((A-\bar \lambda I_n)^k)\ (\forall\ k\in \mathbb Z_+)$ 
- ⑤ 对于 $A$ 的任意非实特征值 $\lambda\in \mathbb C\backslash\mathbb R$，$A$ 关于 $\lambda$ 的 Weyr 特征和 $\bar A$ 关于 $\lambda$ 的 Weyr 特征相同，  
  即 $\rank((A-\lambda I_n)^k) = \rank((\bar A-\lambda I_n)^k)\ (\forall\ k\in \mathbb Z_+)$ 成立.
- ⑥ $A$ 与 $\bar A$ 相似

**(Matrix Analysis 推论 $3.4.1.9$)**    
对于任意 $A\in \mathbb C^{n\times n}$，$A\bar A$ 都与 $\bar A A$ 相似，进而也与一个实矩阵相似.  
这是因为**矩阵乘积的谱不变性**保证了 $A\bar A$ 和 $\bar A A$ 的非零特征值的 Jordan 构造是相同的，故它们是相似的.  
**(存疑: 这里的解释有误)**

> **(矩阵乘积的谱不变性, Matrix Analysis 定理 $1.3.22$)**   
> 任意给定矩阵 $A \in \mathbb{C}^{m\times n},B \in \mathbb{C}^{n\times m}$ (其中 $m\geq n$)  
> 则我们有 $AB\in \mathbb{C}^{m\times m},BA \in \mathbb{C}^{n\times n}$，且 $\text{eig}(AB) = \text{eig}(BA) \cup \{\underbrace{0, \ldots, 0}_{m-n \text{ times}}\}$
> 即 $AB$ 的 $m$ 个特征值即 $BA$ 的 $n$ 个特征值附加上 $m-n$ 个零特征值.  
> 换句话说，二者的特征多项式满足: $p_{AB}(\lambda) = \lambda^{m-n} p_{BA}(\lambda)$   
> **这意味着 $AB,BA$ 的非零特征值是完全相同的 (包括重数)，而零特征值的个数相差 $m-n$ 个.**    

**(Matrix Analysis 推论 $3.4.1.8$)**    
设 $A=\begin{bmatrix} B & C\\ 0 & 0 \end{bmatrix} \in \mathbb{C}^{n\times n}$ (其中 $B \in \mathbb{C}^{m\times m}$)   
若 $B$ 相似于一个实方阵，则 $A$ 也相似于一个实方阵.  

- **证明:**  
  设非奇异阵 $S_0\in \mathbb C^{m\times n}$ 使得 $R_0:= S_0 BS_0^{-1}$ 为一个实矩阵.  
  记 $S= S_0 \oplus I_{n-m}$，考虑以下与 $A$ 相似的矩阵:
  $$
  A_1 := SAS^{-1} 
  = \begin{bmatrix}
  S_0 & \\
  & I_{n-m}
  \end{bmatrix} 
  \begin{bmatrix}
  B & C\\
  0 & 0
  \end{bmatrix}
  \begin{bmatrix}
  S_0^{-1} & \\
  & I_{n-m}
  \end{bmatrix}
  =
  \begin{bmatrix}
  S_0 BS_0^{-1} & S_0C\\
  0 & 0
  \end{bmatrix}
  =
  \begin{bmatrix}
  R_0 & S_0 C\\
  0 & 0
  \end{bmatrix}
  $$
  任意给定 $A$ 的非零特征值 $\lambda \neq 0$，对于任意正整数 $k\in \mathbb Z_+$ 我们都有:  
  $$
  (A_1 -\lambda I_n)^k = 
  \begin{bmatrix}
  (R_0 - \lambda I_m)^k & *\\
  & (-\lambda)^k I_{n-m}
  \end{bmatrix}\\
  
  (\bar A_1 - \lambda I_n)^k = 
  \begin{bmatrix}
  (\bar R_0 - \lambda I_m)^k & *\\
  & (-\lambda)^k I_{n-m}
  \end{bmatrix} = 
  \begin{bmatrix}
  (R_0 - \lambda I_m)^k & *\\
  & (-\lambda)^k I_{n-m}
  \end{bmatrix}\\
  
  \rank((A_1 -\lambda I_n)^k) = \rank((\bar A_1 -\lambda I_n)^k) = \rank((R_0 -\lambda I_{m})^k) + (n-m)
  $$
  因此 $A_1$ 与 $\bar A_1$ 相似，故 $A_1$ 相似于一个实矩阵，从而 $A$ 也相似于一个实矩阵.



### 3.2.5 Jordan 标准型的应用

#### (1) Jordan 矩阵的构造

> **(Jordan 标准型定理, Matrix Analysis 定理 $3.1.11$)**    
> 给定复方阵 $A\in \mathbb{C}^{n\times n}$  
> 设其互不相同的特征值为 $\lambda_1,\dots,\lambda_d$，代数重数分别为 $n_1,\dots,n_d$ (满足 $n = n_1 + \dots + n_d\  (n_1\geq \dotsm \geq n_d \geq 1)$)  
> 则存在一个非奇异阵 $S\in \mathbb{C}^{n\times n}$ 和**由 $A$ 唯一确定的**一系列正整数 $\{n_1^{(i)}\}_{i=1}^{p_1},\dots,\{n_d^{(i)}\}_{i=1}^{p_d}$ 满足:
> $$
> \begin{cases}
> n_1 = \sum_{i=1}^{p_1} n_1^{(i)} & (n_1^{(1)}\geq \dotsm \geq n_1^{(p_1)}\geq 1)\\
> \quad\dots\\
> n_d =  \sum_{i=1}^{p_d} n_d^{(i)} & (n_d^{(1)}\geq \dotsm \geq n_d^{(p_d)}\geq 1)\\
> \end{cases}
> $$
> 使得:
> $$
> {\begin{cases}
> J^{(1)}(\lambda_1) = J_{n_1^{(1)}} (\lambda_1) \oplus \dotsm \oplus J_{n^{(p_1)}_1} (\lambda_1)\\
> \qquad\quad\dotsm\\
> J^{(d)}(\lambda_d) = J_{n_d^{(1)}} (\lambda_d) \oplus \dotsm \oplus J_{n^{(p_d)}_d} (\lambda_d)\\
> \end{cases}}\\
> S^{-1}AS = J = J^{(1)}(\lambda_1) \oplus \dotsm \oplus J^{(d)}(\lambda_d)
> $$
> 其中 Jordan 矩阵由 $A$ 唯一确定 (不考虑直和项的排列)，称为 $A$ 的 Jordan 标准型.  
> 特殊地，如果 $A$ 退化为仅有实特征值的实方阵，则相似矩阵 $S$ 也可取为实方阵.

Jordan 矩阵的结构:
$$
J = 
\begin{bmatrix}
J^{(1)}(\lambda_1)\\
& J^{(2)}(\lambda_2) \\
& & \ddots \\
& & & J^{(d)}(\lambda_d)
\end{bmatrix}
$$
其中 $\lambda_1,\dots,\lambda_d$ 互不相同，$J_1(\lambda_1),\dots ,J_d(\lambda_d)$ 为对应的 Jordan 矩阵:  
$$
{\begin{cases}
J^{(1)}(\lambda_1) = J_{n_1^{(1)}} (\lambda_1) \oplus \dotsm \oplus J_{n^{(p_1)}_1} (\lambda_1)
\text{ where }n_1 = \sum_{i=1}^{p_1} n_1^{(i)} & (n_1^{(1)}\geq \dotsm \geq n_1^{(p_1)}\geq 1)\\
\qquad\quad\ \ \dotsm\\
J^{(d)}(\lambda_d) = J_{n_d^{(1)}} (\lambda_d) \oplus \dotsm \oplus J_{n^{(p_d)}_d} (\lambda_d)
\text{ where } n_d = \sum_{i=1}^{p_d} n_d^{(i)} & (n_d^{(1)}\geq \dotsm \geq n_d^{(p_d)}\geq 1) \\
\end{cases}}
$$
记 Jordan 标准型 $J$ 中 Jordan 块的总数为 $p = \sum_{i=1}^d p_i$，则我们有以下结论:

* 关于 $\lambda_i$ 的 Jordan 矩阵 $J_i(\lambda_i)$ 的阶数 $n_i$ 就等于 $\lambda_i$ 的代数重数.  
  关于 $\lambda_i$ 的 Jordan 块的总数 $p_i$ 就等于 $\lambda_i$ 的几何重数，表示 $J$ 关于 $\lambda_i$ 能且仅能找到 $p_i$ 个线性无关的特征向量.    
  这也给出了几何重数 $\leq$ 代数重数这一事实的直观解释.  

  $A$ 的特征值 $\lambda$ 是**半简单的** (semi-simple) 当且仅当 $\lambda$ 的几何重数 $=$ 代数重数，  
  即等价于 Jordan 标准型中关于 $\lambda$ 的所有 Jordan 块都是 $1\times 1$ 的.
  
  Jordon 标准型 $J$ 为对角阵 (即 $A$ 可对角化) 当且仅当所有特征值都是半简单的，即几何重数等于代数重数.
  
* 只知道 $A$ 所有的特征值及其代数重数和几何重数，并不足以确定 Jordan 标准型 $J$ 的结构.  
  根据之前的结论，Jordan 矩阵 $J$ 的结构 (不考虑 Jordan 块的排序)  
  由 $A$ 关于 $\lambda_1,\lambda_2,\dots,\lambda_d$ 的 **Weyr 特征** $w(J,\lambda_1),w(J,\lambda_2),...,w(J,\lambda_d)$ 唯一确定.  

  设 $A\in \mathbb C^{n\times n},\lambda \in \mathbb C,k\in \mathbb N$，我们定义:
  $$
  r_k(A,\lambda) := \begin{cases}
  n & \text{if }k=0\\
  \rank((A-\lambda I_n)^k) & \text{if }k\geq 1
  \end{cases}\\
  
  w_k (A,\lambda) := \begin{cases}
  n - r_1(A,\lambda) & \text{if }k=1\\
  r_{k-1}(A,\lambda) - r_k(A,\lambda) & \text{if }k\geq 2
  \end{cases}
  $$
  我们定义 $A$ 关于特征值 $\lambda$ (设代数重数为 $n_\lambda$) 的 **Weyr 特征**为 $w(A,\lambda):= (w_1(A,\lambda),\dots,w_{n_\lambda}(A,\lambda))$       
  我们可以看出 $w_k(A,\lambda)$ 等于与特征值 $\lambda$ 对应的 Jordan 块中阶数 $\geq k$ 的个数.  
  于是 $w_k(A,\lambda) - w_{k+1}(A,\lambda)$ 即为与特征值 $\lambda$ 对应的 Jordan 块中阶数 $= k$ 的个数.   

  记 $w(A,\lambda_i):= (w_1(A,\lambda_i),\dots,w_{n_i}(A,\lambda_i))$ 中非负项的最大指标为 $q_i$   
  显然它代表与 $\lambda_i$ 对应的 Jordan 块的最大阶数，它实际上是 $\lambda_i$ 作为极小多项式 $m(\lambda)$ 的根的重数.

据此我们可以手算一些简单形式的小方阵的 Jordan 标准型:

* ① 求出 $A$ 的所有互不相同的值 $\lambda_1,\dots,\lambda_d$ 及其代数重数 $n_1,\dots,n_d$ 

* ② 对 $\lambda_1,\dots,\lambda_d$ 中的每个特征值 $\lambda_i$，计算 $w(A,\lambda_i):= (w_1(A,\lambda_i),\dots,w_{n_i}(A,\lambda_i))$:
  $$
  r_k(A,\lambda) := \begin{cases}
  n & \text{if }k=0\\
  \rank((A-\lambda I_n)^k) & \text{if }k\geq 1
  \end{cases}\\
  
  w_k (A,\lambda) := \begin{cases}
  n - r_1(A,\lambda) & \text{if }k=1\\
  r_{k-1}(A,\lambda) - r_k(A,\lambda) & \text{if }k\geq 2
  \end{cases}
  $$
  记 $w(A,\lambda_i):= (w_1(A,\lambda_i),\dots,w_{n_i}(A,\lambda_i))$ 中非负项的最大指标为 $q_i$   
  计算 Weyr 特征相邻两项的差值 $w_1 - w_2,\  w_2-w_3,\dots, w_{q_i-1}-w_{q_i},\  w_{q_i}$   
  其中 $w_k -w _{k+1}$ 就代表与 $\lambda_i$ 对应的阶数 $= k$ 的 Jordan 块的个数.

上述算法应用于一般的数值计算时会失效，毕竟计算矩阵的秩本身就是一个数值不稳定的过程.



#### (2) 矩阵函数

**(复述 FDU 高等线性代数 2. 范数中的内容)**  
一般地，设 $A\in \mathbb C^{n\times n}$ 可对角化，$A=S\Lambda S^{-1}$，$\Lambda = \text{diag}(\lambda_1,\dots,\lambda_n)$   
给定一个定义域包含 $\lambda_1,\dots,\lambda_n$ 的复值函数 $f$   
我们定义**初等矩阵函数**为: $f(A) := Sf(\Lambda)S^{-1} = S\text{diag}\{f(\lambda_1),\dots,f(\lambda_n)\}S^{-1}$   
可以证明这个定义与相似矩阵 $S$ 的选择是无关的 (即不同的 $S$ 定义出的 $f(A)$ 是一致的)  
这表明可对角化的矩阵的初等矩阵函数具有良好的定义.

特殊地，如果 $f$ 在包含 $\lambda_1,\dots,\lambda_n$ 的某个开域解析，则我们有:  
$$
\begin{align}
f(A)
&:= Sf(\Lambda)S^{-1}\\
&= S\text{diag}\{f(\lambda_1),\dots,f(\lambda_n)\}S^{-1}\\
&= S\left\{\frac{1}{2\pi i}\oint_{\Gamma} f(\lambda)(\lambda I-\Lambda)^{-1}{\mathrm d}\lambda\right\} S^{-1}\\
&= \frac{1}{2\pi i}\oint_{\Gamma} f(\lambda)S(\lambda I-\Lambda)^{-1} S^{-1}{\mathrm d}\lambda\\
&= \frac{1}{2\pi i}\oint_{\Gamma} f(\lambda)(\lambda I-A)^{-1}{\mathrm d}\lambda
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

**下面我们使用 Jordan 标准型给出矩阵函数的计算原理:   ** 
设复方阵 $A\in \mathbb C^{n\times n}$ 的 Jordan 标准型为:
$$
S^{-1}AS
= J = 
\begin{bmatrix}
J^{(1)}(\lambda_1)\\
& J^{(2)}(\lambda_2) \\
& & \ddots \\
& & & J^{(d)}(\lambda_d)
\end{bmatrix}
$$
其中 $\lambda_1,\dots,\lambda_d$ 互不相同，$J_1(\lambda_1),\dots ,J_d(\lambda_d)$ 为对应的 Jordan 矩阵:  
$$
{\begin{cases}
J^{(1)}(\lambda_1) = J_{n_1^{(1)}} (\lambda_1) \oplus \dotsm \oplus J_{n^{(p_1)}_1} (\lambda_1)
\text{ where }n_1 = \sum_{i=1}^{p_1} n_1^{(i)} & (n_1^{(1)}\geq \dotsm \geq n_1^{(p_1)}\geq 1)\\
\qquad\quad\ \ \dotsm\\
J^{(d)}(\lambda_d) = J_{n_d^{(1)}} (\lambda_d) \oplus \dotsm \oplus J_{n^{(p_d)}_d} (\lambda_d)
\text{ where } n_d = \sum_{i=1}^{p_d} n_d^{(i)} & (n_d^{(1)}\geq \dotsm \geq n_d^{(p_d)}\geq 1) \\
\end{cases}}
$$
给定解析的标量函数 $f$，我们定义矩阵函数 $f(A)$ 为:  
$$
f(A):= Sf(J)S^{-1}\\
f(J):= f(J^{(1)}(\lambda_1)) \oplus \dotsm \oplus f(J^{(d)}(\lambda_d))\\
{\begin{cases}
f(J^{(1)}(\lambda_1)) = f(J_{n_1^{(1)}}(\lambda_1)) \oplus \dotsm \oplus f(J_{n^{(p_1)}_1} (\lambda_1))\\
\qquad\qquad\ \ \ \dotsm\\
f(J^{(d)}(\lambda_d)) = f(J_{n_d^{(1)}}(\lambda_d)) \oplus \dotsm \oplus f(J_{n^{(p_d)}_d} (\lambda_d))\\
\end{cases}}
$$
因此问题归结为如何计算关于特征值 $\lambda$ 的 $m$ 阶 Jordan 块 $J_m(\lambda)$ 的矩阵函数 $f(J_m(\lambda))$  
考虑函数 $f$ 在 $x=\lambda$ 处的 Taylor 展开式:  
$$
f(x) = f(\lambda) + \sum_{k=1}^\infty \frac{1}{k!}f^{(k)}(\lambda)(x-\lambda)^k
$$
我们可以类似地写出矩阵函数 $f(J_m(\lambda))$ 的展开式:  
$$
\begin{align}
f(J_m(\lambda))
&=
f(\lambda) I_m + \sum_{k=1}^\infty \frac{1}{k!}f^{(k)}(\lambda) (J_m(\lambda)-\lambda I_m)^k\\
&=
f(\lambda) I_m + \sum_{k=1}^\infty \frac{1}{k!}f^{(k)}(J_m(0))^k

\end{align}
$$
回忆起幂零 Jordan 块的性质:    
**(幂零 Jordan 块的性质, Matrix Analysis 引理 $3.1.4$)**  
给定正整数 $n\geq 2$ 和 $x\in \mathbb{C}^n$，记 $\mathbb C^n$ 的第 $i$ 个标准单位基向量为 $e_i$，则我们有:

* $J_n(0)e_{i+1} = e_i\ (\forall\ i=1,\dots,n-1)$
* $J_n(0)^{\mathrm T}J_n(0) = \begin{bmatrix} 0 &\\ & I_{n-1} \end{bmatrix}$  
* $[I_n - J_n(0)^{\mathrm T}J_n(0)]x = (x^{\mathrm T}e_1)e_1$
* $J_n(0)^k = \begin{cases} 
  I_n & \text{if }k=0\\
  \begin{bmatrix}  & I_{n-k}\\ 0_{k\times k} &  \end{bmatrix} 
  &\text{if }1 \leq k < n\\ 
  0_{n\times n} & \text{if }k\geq n\end{cases}$ 

我们有:  
$$
\begin{align}
f(J_m(\lambda))
&=
f(\lambda) I_m + \sum_{k=1}^\infty \frac{1}{k!}f^{(k)}(\lambda) (J_m(\lambda)-\lambda I_m)^k\\
&=
f(\lambda) I_m + \sum_{k=1}^\infty \frac{1}{k!}f^{(k)}(\lambda)(J_m(0))^k\\
&=
f(\lambda) I_m + \sum_{k=1}^{m-1} \frac{1}{k!}f^{(k)}(\lambda)(J_m(0))^k\\
&=
f(\lambda) I_m + \sum_{k=1}^{m-1} \frac{1}{k!}f^{(k)}(\lambda)
\begin{bmatrix}  
& I_{m-k}\\ 
0_{k\times k} &  \end{bmatrix}\\
&=
\begin{bmatrix}
f(\lambda) & f'(\lambda) & \frac{1}{2!}f''(\lambda) & \dotsm & \frac{1}{(m-1)!}f^{(m-1)}(\lambda)\\
& f(\lambda) & f'(\lambda) & \ddots & \vdots\\
& & \ddots & \ddots & \frac{1}{2!}f''(\lambda)\\
& & & f(\lambda) & f'(\lambda) \\
& & & & f(\lambda)
\end{bmatrix}
\end{align}
$$


#### (3) 线性常微分方程组

考虑一阶初值问题:  
$$
x'(t) = Ax(t)\\
x(0) = x_0
$$
其中 $x(t)=[x_1(t),\dots,x_n(t)]^{\mathrm T}$ 为 $\mathbb C\mapsto \mathbb C^n$ 的函数，$x^{(0)}\in \mathbb C^n$ 为给定向量.

若 $A$ 不是对角阵，则我们称上述方程组为**耦合的** (coupled)，  
即 $x'_i(t)$ 不仅与 $x_i(t)$ 有关，还可能和 $x(t)$ 的其他分量函数有关.  
这种耦合性使得问题难以求解.  
我们可通过将 $A$ 化为几乎对角 (甚至完全对角) 的矩阵来减少 (甚至消除) 耦合的数量，从而简化问题的求解.

设 $A=SJS^{-1}$，其中 $J$ 为 $A$ 的 Jordan 标准型.  
记 $y(t):=S^{-1}x(t)$ 和 $y^{(0)} := S^{-1}x^{(0)}$ 则我们有:
$$
y'(t) = J y(t)\\
y(0) = y^{(0)}
$$

- 若 $J$ 为对角阵 $\Lambda := \text{diag}\{\lambda_1,\dots,\lambda_n\}$ (即 $A$ 可对角化: $A=S^{-1}\Lambda S$)，则上述方程组化为一系列独立的方程:  
  $$
  \begin{cases}
  y'_k (t) = \lambda_k y_k(t)\\
  y_k(0) = y^{(0)}_k
  \end{cases}\ (k=1,\dots,n)
  $$
  解得 $y_k(t) = y_k(0) e^{\lambda_k t} = y_k^{(0)} e^{\lambda_k t}\ (k=1,\dots,n)$  
  其紧凑形式为 $y(t) =  \exp(t\Lambda)y^{(0)}$   
  于是我们有 $x(t) = Sy(t) = S \exp(t\Lambda)y^{(0)} = S\exp (t\Lambda) S^{-1} x^{(0)} = \exp(tA)x^{(0)}$ 

- 若 $J$ 不是对角的，则求解会更加复杂，但最终形式是类似的.  
  注意到 $J$ 中不同 Jordan 块对应的元素不是耦合的，因此我们只需考虑 $J:= J_m(\lambda)$ 为单个 Jordan 块的情形就够了:  
  $$
  {\begin{cases}
  y'_1(t) = \lambda y_1(t) + y_2(t)\\
  \qquad\dotsm\\
  y'_{m-1}(t) =  \lambda y_{m-1}(t) + y_m(t)\\
  y'_{m}(t) = \lambda y_m(t)
  \end{cases}}\\
  y(0) = y^{(0)}
  $$
  解得:
  $$
  y_k(t) = e^{\lambda t} \sum_{i=k}^m y_i(0) \frac{t^{i-k}}{(i-k)!}\ (k=1,\dots,m)\\
  \Leftrightarrow\\
  y(t) 
  =
  \begin{bmatrix}
  y_1(t)\\
  y_2(t)\\
  \vdots\\
  y_{m-1}(t)\\
  y_m(t)
  \end{bmatrix}
  =
  \begin{bmatrix} 
  e^{\lambda t}& e^{\lambda t} t& \frac12e^{\lambda t} t^2& \dots& \frac{1}{(m-1)!}e^{\lambda t} t^{m-1}\\ &e^{\lambda t}&e^{\lambda t} t&\ddots&\vdots\\ 
  &&\ddots&\ddots&\frac12e^{\lambda t} t^2\\ 
  &&&e^{\lambda t} & e^{\lambda t} t\\ &&&& e^{\lambda t}
  \end{bmatrix}
  \begin{bmatrix}
  y_1(0)\\
  y_2(0)\\
  \vdots\\
  y_{m-1}(0)\\
  y_m(0)
  \end{bmatrix}
  =
  \exp(tJ_m(\lambda)) y^{(0)}
  $$
  推广至一般情况 (即 $J$ 为有限个 Jordan 块的直和时) 即有:   
  $$
  y(t) = \exp(tJ) y^{(0)}
  $$
  于是我们有 $x(t) = Sy(t) = S \exp(tJ)y^{(0)} = S\exp (t J) S^{-1} x^{(0)} = \exp(tA)x^{(0)}$   
  其中 $\exp(tA)$ 理论上可使用 Jordan 标准型计算，也可用级数 $\exp(tA)=\sum_{k=1}^\infty \frac{(tA)^k}{k!}$ 计算  
  数值计算和符号计算是不相同的，实际应用中会使用 **Schur-Parlett 算法**进行计算.

*****

**(高等线性代数 Homework 5 Problem 3)**  
设 $V$ 是 $\mathbb R$ 上不超过 $5$ 次的多项式全体构成的向量空间，$D$ 是 $V$ 上的求导算子.  
试选取 $V$ 的一组基，在这组基下写出 $D$ 的表示矩阵，并求出其所有特征值.  
利用上述结论求解常微分方程 $y'' - 2y' + y = x^5$ 的通解.

**Solution:**  
$B_{\text{polynomial}} = [1,x,x^2,x^3,x^4,x^5]^{\mathrm T}$ 是 $V$ 的一组基.  

根据 $\begin{cases}
D(1) = 0\\
D(x) = 1\\
D(x^2) = 2x\\
D(x^3) = 3x^2\\
D(x^4) = 4x^3\\
D(x^5) = 5x^4\end{cases}$ 可知求导算子 $D$ 在基 $[1,x,x^2,x^3,x^4,x^5]$ 下的表示矩阵为 $[D]_{B_{\text{polynomial}}}=
\begin{bmatrix}
0 & 1\\
& 0 & 2\\
&  & 0 & 3\\
& & & 0 & 4\\
& & & & 0 & 5\\
& & & &  & 0\end{bmatrix}$     
显然其所有特征值均为 $0$

设 $y=a^{\mathrm T} B_{\text{polynomial}}$ (其中 $a\in \mathbb R^6$)  
则求解常微分方程 $y'' - 2y' + y = x^5$ 的问题就等价于求解以下线性方程组:  
$$
([D]_{B_{\text{polynomial}}})^2 a - 2 [D]_{B_{\text{polynomial}}} a + a 
=
\begin{bmatrix}
1 & -2 & 2\\
& 1 & -4 & 6\\
& & 1 & -6 & 12\\
& &  & 1 & -8 & 20\\
& & & & 1 & -10\\
& & & & & 1\end{bmatrix}
\begin{bmatrix}
a_0\\
a_1\\
a_2\\
a_3\\
a_4\\
a_5
\end{bmatrix}
= 
\begin{bmatrix}
0\\
0\\
0\\
0\\
0\\
1
\end{bmatrix}\\
$$
解得:
$$
\begin{cases}
a_5 = 1\\
a_4 = 10a_5 = 10\\
a_3 = 8a_4 - 20 a_5 = 8\times 10-20\times 1 = 60\\
a_2 = 6a_3 - 12 a_4 = 6\times 60-12\times 10 = 240\\
a_1 = 4a_2 - 6a_3 = 4\times 240 - 6\times 60 = 600\\
a_0 = 2a_1 - 2a_2 = 2\times 600 - 2\times 240 = 720
\end{cases}
$$
因此我们有:  
$$
\begin{align}
y(x)
&=a^{\mathrm T} B_{\text{polynomial}}\\
&= a_5 x^5 + a_4 x^4 + a_3 x^3 + a_2 x^2 + a_1 x + a_0\\
&= x^5 + 10 x^4 + 60 x^3 + 240 x^2 + 600 x + 720
\end{align}
$$

于是通解形式为:  
$$
y(x) = x^5 + 10 x^4 + 60 x^3 + 240 x^2 + 600 x + 720 + C\ (\text{where }C\text{ is a constant})
$$

****

基 $[1,e^x,e^{2x},e^{3x}]$ 张成的向量空间的求导算子为 $D=\begin{bmatrix}
0\\
&1\\
& & 2\\
& & & 3\end{bmatrix}$   
我们可以通过对基 $[1,e^x,e^{2x},e^{3x}]$ 进行数乘，让求导算子 $D$ 的对角元取任意值.

****

**Heaviside 方法**  
考虑求解 $y - \frac12 y' = -\frac12 x^3$ (其中 $y(x)$ 为多项式函数)  
取基 $[1,x,x^2,x^3]$，则求导算子为 $D=\begin{bmatrix}
0 & 1\\
& 0 & 2\\
&  & 0 & 3\\
& & & 0\end{bmatrix}$   
显然其特征值均为 $0$，特征多项式为 $p_D(t)=t^4$  
因此根据 Cayley-Hamilton 定理可知 $D^4 = 0_{{4\times 4}}$   
于是我们有:
$$
\begin{align}
y - \frac12 y' 
&=
(I-\frac12 D)y = -\frac12 x^3\\
\hline
[y]_{[1,x,x^2,x^3]}
&=
-\frac12 (I - \frac12 D)^{-1}
[x^3]_{[1,x,x^2,x^3]}
\quad (\text{Neumann})\\
&=
-\frac12 (I + \frac12 D + \frac14 D^2 + \frac18 D^3 + \frac1{16}D^4 + \dotsm)
\begin{bmatrix}
0\\
0\\
0\\
1
\end{bmatrix}\\
&=
-\frac12 (I+\frac12 D + \frac14 D^2 + \frac18 D^3) 
\begin{bmatrix}
0\\
0\\
0\\
1
\end{bmatrix}\\
&=
-\frac12 
\begin{bmatrix}
1 & \frac12 & \frac12 & \frac34\\
& 1 & 1 & \frac32\\
& & 1 & \frac32\\
& & & 1
\end{bmatrix}
\begin{bmatrix}
0\\
0\\
0\\
1
\end{bmatrix}\\
&=
-\frac12 
\begin{bmatrix}
\frac34\\
\frac32\\
\frac32\\
1
\end{bmatrix}
\end{align}
$$
因此我们有 $y=-\frac12 x^3 + \frac34 x^2 - \frac34 x - \frac38$ 成立，可以验证它满足 $y-\frac12 y' = -\frac12 x^3$ 

****

用求解线性方程组的思想求解积分:  
$$
I_1 = \int \frac{\sin(x)}{\sin(x)+2\cos(x)}{\mathrm d}x\\
I_2 = \int \frac{\cos(x)}{\sin(x) + 2\cos(x)}{\mathrm d}x
$$
我们有:  
$$
I_1 + 2 I_2 = \int 1 dx = x + C_1\\
-2I_1 + I_2 = \int \frac{\cos(x)-2\sin(x)}{\sin(x)+2\cos(x)}{\mathrm d}x = \int \frac{{\mathrm d}(\sin(x)+2\cos(x))}{\sin(x)+2\cos(x)}  = \log(\sin(x)+2\cos(x))+C_2
$$
其中 $\log(\sin(x)+2\cos(x))$ 不必写成 $\log(|\sin(x)+2\cos(x)|)$  
因为上述积分只能在局部做，本质上是分段函数.

求解上述线性方程组即可得:  
$$
I_1 = \frac{1}{5}x - \frac25 \log(\sin(x)+2\cos(x)) + C\\
I_2 = \frac25 x + \frac{1}{5}\log(\sin(x)+2\cos(x)) + C\\
$$


#### (4) $A^{\mathrm T} \sim A$ 

**定理 (Matrix Analysis $3.2.3$ 节, 方阵与其转置的相似性):**   
对于任意复方阵 $A\in \mathbb{C}^{n\times n}$ 都有 $A^{\mathrm T} \sim A$ 成立.  

- 上述结论是符合直观的:
  $$
  \det(\lambda I_n -A) = \det((\lambda I_n-A)^{\mathrm T}) = \det(\lambda I_n - A^{\mathrm T})\\
  \text{And for Jordan block, we have }\begin{bmatrix}
  0 & 1 & \\
  & 0 & \ddots\\
  & & \ddots & 1\\
  & & & 0
  \end{bmatrix}
  \sim
  \begin{bmatrix}
  0 &  & \\
  1& 0 & \\
  & \ddots & \ddots &\\
  & & 1& 0
  \end{bmatrix}
  $$

- 值得注意的是，$A$ 的转置共轭 $A^{\mathrm H}$ 一般不与 $A$ 相似.  
  $A^{\mathrm H}\sim A$ 只在一些特殊情况成立，例如 $A$ 为正规矩阵 (即 $A^{\mathrm H}A=AA^{\mathrm H}$) 时

**证明: **  
我们定义 $K_m$ 为 $m$ 阶**逆序矩阵** (Reversal Matrix):
$$
K_m = \begin{bmatrix} 0 &&&1\\ &&\dotsm&\\ &1&&\\ 1&&&0 \end{bmatrix}
$$
显然 $K_m$ 是对称且**自反** (involutory) 的: $K_m = K_m^{\mathrm T} = K_m^{-1}$    
若将 $m$ 阶逆序矩阵作为关于 $\lambda$ 的 $m$ 阶 Jordan 块 $J_m(\lambda)$ 的相似变换矩阵，则我们有:     
$$
K_m^{-1}J_m(\lambda) K_m 
= K_mJ_m(\lambda) K_m 
= \begin{bmatrix} \lambda &&&\\ 1 & \lambda &&\\ &\ddots&\ddots&\\ &&1&\lambda \end{bmatrix}
=
(J_m(\lambda))^{\mathrm T}
$$
设复方阵 $A\in \mathbb C^{n\times n}$ 的 Jordan 标准型为:
$$
S^{-1}AS
= J = 
\begin{bmatrix}
J^{(1)}(\lambda_1)\\
& J^{(2)}(\lambda_2) \\
& & \ddots \\
& & & J^{(d)}(\lambda_d)
\end{bmatrix}
$$
其中 $\lambda_1,\dots,\lambda_d$ 互不相同，$J_1(\lambda_1),\dots ,J_d(\lambda_d)$ 为对应的 Jordan 矩阵:  
$$
{\begin{cases}
J^{(1)}(\lambda_1) = J_{n_1^{(1)}} (\lambda_1) \oplus \dotsm \oplus J_{n^{(p_1)}_1} (\lambda_1)
\text{ where }n_1 = \sum_{i=1}^{p_1} n_1^{(i)} & (n_1^{(1)}\geq \dotsm \geq n_1^{(p_1)}\geq 1)\\
\qquad\quad\ \ \dotsm\\
J^{(d)}(\lambda_d) = J_{n_d^{(1)}} (\lambda_d) \oplus \dotsm \oplus J_{n^{(p_d)}_d} (\lambda_d)
\text{ where } n_d = \sum_{i=1}^{p_d} n_d^{(i)} & (n_d^{(1)}\geq \dotsm \geq n_d^{(p_d)}\geq 1) \\
\end{cases}}
$$
我们定义方阵 $K\in \mathbb C^{n\times n}$ 为:  
$$
K = K^{(1)} \oplus \dotsm \oplus K^{(d)}\\
{\begin{cases}
K^{(1)} = K_{n_1^{(1)}} \oplus \dotsm \oplus K_{n^{(p_1)}_1}\\
\qquad\quad\ \ \dotsm\\
K^{(d)} = K_{n_d^{(1)}} \oplus \dotsm \oplus K_{n^{(p_d)}_d}\\
\end{cases}}
$$
它也是对称且自反的: $K^{\mathrm T} = K^{-1}=K$  
则我们有:   
$$
\begin{align}
K^{-1}JK 
&=
\left[(K^{(1)})^{-1}J^{(1)}(\lambda_1) K^{(1)}\right]\oplus \dotsm \oplus \left[(K^{(d)})^{-1}J^{(d)}(\lambda_d) K^{(d)}\right]\\
&=
\left[K_{n_1^{(1)}}^{-1} J_{n_1^{(1)}}(\lambda_1) K_{n_1^{(1)}} \oplus \dotsm \oplus K^{-1}_{n_1^{(p_1)}}J_{n_1^{(p_1)}}(\lambda_1) K_{n_1^{(p_1)}}\right]
\oplus
\dotsm
\oplus
\left[K_{n_d^{(1)}}^{-1} J_{n_d^{(1)}}(\lambda_d) K_{n_d^{(1)}} \oplus \dotsm \oplus K^{-1}_{n_d^{(p_d)}}J_{n_d^{(p_d)}}(\lambda_d) K_{n_d^{(p_d)}}\right]\\
&=
\left[(J_{n_1^{(1)}}(\lambda_1))^{\mathrm T} \oplus (J_{n_1}^{(p_1)}(\lambda_1))^{\mathrm T}\right] \oplus \dotsm \oplus 
\left[(J_{n_d^{(1)}}(\lambda_d))^{\mathrm T} \oplus (J_{n_d}^{(p_d)}(\lambda_d))^{\mathrm T}\right]\\
&=
(J^{(1)}(\lambda_1))^{\mathrm T} \oplus \dotsm \oplus (J^{(d)}(\lambda_d))^{\mathrm T}\\
&=
J^{\mathrm T}
\end{align}
$$
从而有:  
$$
\begin{align}
A^{\mathrm T}
&=
(SJS^{-1})^{\mathrm T}\\
&=
S^{-T}J^{\mathrm T}S^{\mathrm T}\\
&=
S^{-T}(K^{-1}JK)S^{T}\\
&=
S^{-T}K^{-1}(S^{-1}AS) K S^{T}\\
&=
(SKS^{\mathrm T})^{-1} A(SKS^{\mathrm T})
\end{align}
\Rightarrow\\
A^{\mathrm T}\sim A
$$
定理得证.

****

**实际上，我们可以从上述推导中发现更多有趣的事实:**  
首先，$A$ 与 $A^{\mathrm T}$ 之间的相似变换矩阵 $SKS^{\mathrm T}$ 是一个非奇异的对称阵.  
于是有以下推论:   

- **(Matrix Analysis 定理 $3.2.3.1$)**  
  对于任意复方阵 $A\in \mathbb{C}^{n\times n}$，存在一个非奇异的复对称阵 $S\in \mathbb{C}^{n\times n}$ 使得 $S^{-1}AS = A^{\mathrm T}$   
  **(Matrix Analysis 定理 $3.2.4.4$)**  
  特殊地，若 $A$ 是非退化的 (即所有特征值的几何重数都等于 $1$)，  
  则 $A$ 与 $A^{\mathrm T}$ 之间的任何一个相似变换矩阵一定是对称阵.   

基于 $A^{\mathrm T}\sim A$ 的推导中，我们发现:

- 一方面，我们有 $A=SJS^{-1} = SJ(KS^{\mathrm T}S^{-T}K)S^{-1} = (SJKS^{\mathrm T})(S^{-T}KS^{-1})$   
  其中 $SJKS^{\mathrm T}$ 和 $S^{-T}KS^{-1}$ 都是对称阵  
  且 $S^{-T}KS^{-1}$ 一定非奇异，而 $PJKP^{\mathrm T}$当且仅当 $J$ 没有零特征值时是非奇异的.  
- 另一方面，我们有 $A=SJS^{-1} = S(KS^{\mathrm T}S^{-T}K)JS^{-1} = (SKS^{\mathrm T})(S^{-T}KJS^{-1})$   
  其中 $SKS^{\mathrm T}$ 和 $S^{-T}KJS^{-1}$ 都是对称阵，  
  且 $SKS^{\mathrm T}$ 一定非奇异，而 $S^{-T}KJS^{-1}$ 当且仅当 $J$ 没有零特征值时是非奇异的.  

综合以上两种分解方式，我们有以下推论:  
**(Matrix Analysis 定理 $3.2.3.2$)**  
任意复方阵都可以表示为两个复对称阵的乘积，且我们至少可要求其中之一为非奇异阵.  
(若上述定理中的复方阵退化为实方阵，则复对称阵退化为实对称阵)

**邵老师的表述:**  
任意给定复方阵 $A\in \mathbb C^{n\times n}$，都存在复对称阵 $S_1,S_2\in \mathbb C^{n\times n}$ 使得 $A=S_1S_2$，且 $S_1,S_2$ 至少有一个非奇异.   
若退化为实方阵 $A\in \mathbb R^{n\times n}$，则存在实对称阵 $S_1,S_2\in \mathbb R^{n\times n}$ 使得 $A=S_1S_2$，且 $S_1,S_2$ 至少有一个非奇异.

- 这一结论说明对称阵的乘积没有任何用处.  
  即我们没有理由期待两个对称阵的乘积具有任何特殊性质，因为它理论上可以是任意方阵.  
- 我们发现实数域上的情况推广到复数域上的情况没有共轭，单纯从实对称推广到复对称 (而不是共轭对称)  

**邵老师提供的证明:**    
首先考虑复数域上的情况:  
设复方阵 $A\in \mathbb C^{n\times n}$ 的 Jordan 标准型为:
$$
P^{-1}AP
= J = 
\begin{bmatrix}
J^{(1)}(\lambda_1)\\
& J^{(2)}(\lambda_2) \\
& & \ddots \\
& & & J^{(d)}(\lambda_d)
\end{bmatrix}
$$
其中 $P\in \mathbb C^{n\times n}$ 为非奇异阵，$\lambda_1,\dots,\lambda_d$ 互不相同，$J_1(\lambda_1),\dots ,J_d(\lambda_d)$ 为对应的 Jordan 矩阵:  
$$
{\begin{cases}
J^{(1)}(\lambda_1) = J_{n_1^{(1)}} (\lambda_1) \oplus \dotsm \oplus J_{n^{(p_1)}_1} (\lambda_1)
\text{ where }n_1 = \sum_{i=1}^{p_1} n_1^{(i)} & (n_1^{(1)}\geq \dotsm \geq n_1^{(p_1)}\geq 1)\\
\qquad\quad\ \ \dotsm\\
J^{(d)}(\lambda_d) = J_{n_d^{(1)}} (\lambda_d) \oplus \dotsm \oplus J_{n^{(p_d)}_d} (\lambda_d)
\text{ where } n_d = \sum_{i=1}^{p_d} n_d^{(i)} & (n_d^{(1)}\geq \dotsm \geq n_d^{(p_d)}\geq 1) \\
\end{cases}}
$$
我们定义 $K_m$ 为 $m$ 阶**逆序矩阵** (Reversal Matrix):
$$
K_m := \begin{bmatrix} 0 &&&1\\ &&\dotsm&\\ &1&&\\ 1&&&0 \end{bmatrix}\\
$$
我们定义 $\widehat J_m(\lambda)$ 为关于 $\lambda$ 的 $m$ 阶**逆序 Jordan 块**:  
$$
\widehat J_m(\lambda) := J_m(\lambda) K_m =
\begin{bmatrix}  &&1&\lambda\\ &\dotsm&\dotsm&\\ 1&\lambda&&\\ \lambda&&& \end{bmatrix}
$$
则我们有:  
$$
J_m(\lambda) =
\begin{bmatrix}
\lambda & 1 & \\
& \lambda & \ddots\\
& & \ddots & 1\\
& & & \lambda
\end{bmatrix} 
=
\begin{bmatrix} 0 &&&1\\ &&\dotsm&\\ &1&&\\ 1&&&0 \end{bmatrix}
\begin{bmatrix}  &&1&\lambda\\ &\dotsm&\dotsm&\\ 1&\lambda&&\\ \lambda&&& \end{bmatrix}
=
K_m \widehat J_m (\lambda)
$$
我们按 Jordan 矩阵 $J$ 的划分和特征值定义:  
$$
K := K^{(1)} \oplus \dotsm \oplus K^{(d)}\\
{\begin{cases}
K^{(1)} = K_{n_1^{(1)}} \oplus \dotsm \oplus K_{n^{(p_1)}_1}
\text{ where }n_1 = \sum_{i=1}^{p_1} n_1^{(i)} & (n_1^{(1)}\geq \dotsm \geq n_1^{(p_1)}\geq 1)\\
\qquad\quad\ \ \dotsm\\
K^{(d)} = K_{n_d^{(1)}} \oplus \dotsm \oplus K_{n^{(p_d)}_d}
\text{ where } n_d = \sum_{i=1}^{p_d} n_d^{(i)} & (n_d^{(1)}\geq \dotsm \geq n_d^{(p_d)}\geq 1) \\
\end{cases}}\\
\hline
\widehat J := \widehat J^{(1)}(\lambda_1) \oplus \dotsm \oplus \widehat J^{(d)}(\lambda_d)\\
{\begin{cases}
\widehat J^{(1)}(\lambda_1) = \widehat J_{n_1^{(1)}} (\lambda_1) \oplus \dotsm \oplus \widehat J_{n^{(p_1)}_1} (\lambda_1)
\text{ where }n_1 = \sum_{i=1}^{p_1} n_1^{(i)} & (n_1^{(1)}\geq \dotsm \geq n_1^{(p_1)}\geq 1)\\
\qquad\quad\ \ \dotsm\\
\widehat J^{(d)}(\lambda_d) = \widehat J_{n_d^{(1)}} (\lambda_d) \oplus \dotsm \oplus \widehat J_{n^{(p_d)}_d} (\lambda_d)
\text{ where } n_d = \sum_{i=1}^{p_d} n_d^{(i)} & (n_d^{(1)}\geq \dotsm \geq n_d^{(p_d)}\geq 1) \\
\end{cases}}
$$
则我们有:  
$$
\begin{align}
J 
&= J^{(1)}(\lambda_1) \oplus \dotsm \oplus J^{(d)}(\lambda_d)\\
&= (K^{(1)} \widehat J^{(1)}(\lambda_1)) \oplus \dotsm \oplus (K^{(d)} \widehat J^{(d)}(\lambda_d))\\
&= (K^{(1)} \oplus \dotsm \oplus K^{(d)}) (\widehat J^{(1)}(\lambda_1) \oplus \dotsm \oplus \widehat J^{(d)}(\lambda_d))\\
&=K\widehat J

\end{align}
$$
于是我们有:  
$$
\begin{align}
A &= P JP^{-1}\quad (\text{note that }J=K\widehat J)\\
&= P(K\widehat J) P^{-1}\\
&= (PKP^{\mathrm T}) (P^{-T}\widehat J P^{-1})\quad (\text{denote }
\begin{cases}
S_1 := PKP^{\mathrm T}\\
S_2 := P^{-T}\widehat J P^{\mathrm T}
\end{cases})\\
&= S_1 S_2
\end{align}
$$
其中 $\begin{cases}
S_1 := PKP^{\mathrm T}\\
S_2 := P^{-T}\widehat J P^{\mathrm T}
\end{cases}$ 均为对称阵，且 $S_1$ 一定是非奇异阵 (因为 $K$ 一定是非奇异阵)  
实际上，我们还发现 $K,\widehat J$ 是可交换的，即 $J = K\widehat J = \widehat J K$  
因此我们还可以这样定义 $S_1, S_2$:  
$$
\begin{align}
A &= P J P^{-1}\quad (\text{note that }J=K\widehat J)\\
&= P(\widehat JK) P^{-1}\\
&= (P\widehat JP^{\mathrm T}) (P^{-\mathrm T}K P^{-1})\quad (\text{denote }
\begin{cases}
S_1 := P\widehat JP^{\mathrm T}\\
S_2 := P^{-\mathrm T} K P^{\mathrm T}
\end{cases})\\
&= S_1 S_2
\end{align}
$$
那么此时 $S_2$ 一定是非奇异阵 (因为 $K$ 一定是非奇异阵)

综合上述两种定义方法，我们可知:  
任意给定复方阵 $A\in \mathbb C^{n\times n}$，都存在复对称阵 $S_1,S_2\in \mathbb C^{n\times n}$ 使得 $A=S_1S_2$，且 $S_1,S_2$ 至少有一个非奇异. 

*****

下面考虑实数域上的情况:  
设实方阵 $A\in \mathbb R^{n\times n}$ 的**实 Jordan 标准型**为:
$$
P^{-1}AP = J=J^{(1)}(\mu_1) \oplus \dotsm \oplus J^{(d_1)}(\mu_{d_1}) \oplus C^{(1)}(\alpha_1,\beta_1) \oplus
\dotsm \oplus C^{(d_2)}(\alpha_{d_2},\beta_{d_2})
$$
其中 $P\in \mathbb R^{n\times n}$ 为非奇异阵  
实数 $\mu_1,\dots,\mu_{d_1}$ 为 $A$ 的互不相同的实特征值，代数重数分别为 $n_1,\dots,n_{d_1}$  
共轭复数 $\begin{cases}
\lambda_j = \alpha_j + i\beta_j\\
\bar\lambda_j = \alpha_j - i\beta_j\end{cases} (j=1,\dots,d_2)$ 为 $A$ 的 $d_2$ 对复共轭特征值，代数重数分别为 $m_1,\dots,m_{d_2}$   
上述代数重数满足 $\begin{cases}
n_1\geq \dotsm \geq n_{d_1} \geq 1\\
m_1\geq \dotsm \geq m_{d_m} \geq 1\\
n= n_1+\dotsm + n_{d_1} + 2(m_1 + \dotsm + m_{d_2})
\end{cases}$  
对应的非共轭 Jordan 块和共轭 Jordan 块为:  
$$
\begin{cases}
J^{(1)}(\mu_1) = J_{n_1^{(1)}} (\mu_1) \oplus \dotsm \oplus J_{n^{(p_1)}_1} (\mu_1)\text{ where }n_1 = \sum_{j=1}^{p_1} n_1^{(j)}\ \ (n_1^{(1)}\geq \dotsm \geq n_1^{(p_1)}\geq 1)\\
\qquad\quad\dotsm\\
J^{(d_1)}(\mu_{d_1}) = J_{n_{d_1}^{(1)}} (\mu_{d_1}) \oplus \dotsm \oplus J_{n^{(p_{d_1})}_{d_1}} (\mu_{d_1})\text{ where }n_{d_1} =  \sum_{j=1}^{p_{d_1}} n_{d_1}^{(j)}\ \ (n_{d_1}^{(1)}\geq \dotsm \geq n_{d_1}^{(p_{d_1})}\geq 1)\\
C^{(1)} (\alpha_1,\beta_1) = C_{m_{1}^{(1)}}(\alpha_1,\beta_1)\oplus \dotsm \oplus C_{m_{1}^{(q_1)}}(\alpha_1,\beta_1)\text{ where }m_{1} = \sum_{j=1}^{q_1} m_1^{(j)}\ \ (m_1^{(1)}\geq \dotsm \geq m_1^{(q_1)}\geq 1)\\
\qquad\quad\dotsm\\
C^{(d_2)} (\alpha_{d_2},\beta_{d_2}) = C_{m_{d_2}^{(1)}}(\alpha_{d_2},\beta_{d_2})\oplus \dotsm \oplus C_{m_{d_2}^{(q_{d_2})}}(\alpha_{d_2},\beta_{d_2})\text{ where }m_{d_2} = \sum_{j=1}^{q_{d_2}} m_{d_2}^{(j)} \ \ (m_{d_2}^{(1)}\geq \dotsm \geq m_{d_2}^{(q_{d_2})}\geq 1)\\
\end{cases}
$$
其中:
$$
C_1(\alpha,\beta) := 
\begin{bmatrix} 
\alpha & \beta\\
-\beta & \alpha
\end{bmatrix}\\

C_m(\alpha,\beta) 
=
\begin{bmatrix}
C_1(\alpha,\beta) & I_2 & & & \\
& C_1(\alpha,\beta) & I_2 & & \\
&&\ddots & \ddots &\\
&& &C_1(\alpha,\beta) & I_2  \\
&&&&C_1(\alpha,\beta)\\
\end{bmatrix}
$$
和复 Jordan 标准型一样，非共轭 Jordan 块分解为同阶逆序矩阵和同阶逆序非共轭 Jordan 块的乘积:   
$$
J_m(\mu) =
\begin{bmatrix}
\mu & 1 & \\
& \mu & \ddots\\
& & \mu & 1\\
& & & \mu
\end{bmatrix} 
=
\begin{bmatrix} 0 &&&1\\ &&\dotsm&\\ &1&&\\ 1&&&0 \end{bmatrix}
\begin{bmatrix}  &&1&\mu\\ &\dotsm&\dotsm&\\ 1&\mu&&\\ \mu&&& \end{bmatrix}
=
K_m \widehat J_m (\mu)
$$
类似地，共轭 Jordan 块可以分解为:  
$$
\begin{align}
C_m(\alpha,\beta) 
&=
\begin{bmatrix}
C_1(\alpha,\beta) & I_2 & & \\
& C_1(\alpha,\beta) &\ddots & \\
&&\ddots & I_2 \\
&& &C_1(\alpha,\beta)\\
\end{bmatrix}\\
&=
\begin{bmatrix}
 & & & I_2\\
& & \dotsm & \\
& I_2 & & \\
I_2&& &\\
\end{bmatrix}
\begin{bmatrix}
 & & I_2 & C(\alpha,\beta)\\
& I_2 & \dotsm & \\
I_2& C(\alpha,\beta) & & \\
C(\alpha,\beta)&& &\\
\end{bmatrix}\\
&=
\widetilde K_m 
\widehat C_m(\alpha,\beta)
\end{align}
$$
余下构造 $\widetilde K$ 和 $\widehat J$ 使得 $J = \widetilde K \widehat J =  \widehat J \widetilde K$ 的逻辑与复数域上的情况是类似的.  
以及构造实对称阵 $\begin{cases}
S_1 := P\widetilde KP^{\mathrm T}\\
S_2 := P^{-T}\widehat J P^{\mathrm T}
\end{cases}$ 或 $\begin{cases}
S_1 := P\widehat JP^{\mathrm T}\\
S_2 := P^{-T}\widetilde K P^{\mathrm T}
\end{cases}$ 的逻辑与复数域上的情况也是类似的.  
因此任意给定实方阵 $A\in \mathbb R^{n\times n}$，都存在实对称阵 $S_1,S_2\in \mathbb R^{n\times n}$ 使得 $A=S_1S_2$，且 $S_1,S_2$ 至少有一个非奇异.

*****

**广义特征值问题:** $Ax=Bx\lambda$ **(存疑)**  
其中 $A,B\in \mathbb C^{n\times n}$，$x\in \mathbb C^n$ 和 $\lambda\in \mathbb C$   
广义特征值 $\lambda$ 就是 $\det(A-\lambda B)=0$ 的根.  
处理一般的广义特征值问题，单靠 Jordan 标准型是不够的，需要使用 Kronecker 标准型.    

不过我们可以求解一类特殊的广义特征值问题，即 $\begin{cases}
A^{\mathrm H}=A\\
B^{\mathrm H}=B\succ 0\end{cases}$ 的情况.  
此时 $A,B$ 可以同时合同对角化，即存在非奇异阵 $C\in \mathbb C^{n\times n}$ 使得 $\begin{cases}
C^{\mathrm H}AC = \Lambda\\
C^{\mathrm H}BC = I_n\end{cases}$  
则我们有:
$$
\begin{align}
\det(A-\lambda B)
&=
\det(C\Lambda C^{\mathrm H} - \lambda CC^{\mathrm H})\\
&=
\det(C) \det(\Lambda -\lambda I_n) \det(C^{\mathrm H})\\
&=
\det(C) \det(\Lambda -\lambda I_n) \overline{\det(C)}\\
&=
|\det(C)|^2 \det(\Lambda -\lambda I_n)
\end{align}
$$
因此求解 $\det(A-\lambda B)=0$ 就等价于求解 $\det(\Lambda - \lambda I_n)=0$   
也就是说，此时广义特征值问题 $Ax=Bx\lambda$ 就等价于一般的特征值问题 $B^{-1}A x = x\lambda$

值得注意的是，对于 $\begin{cases}
A^{\mathrm T}=A\\
B^{\mathrm T}=B\end{cases}$ 的情况我们目前来说是处理不了的.  
正如本节证明的命题所示，我们没有理由期待两个对称阵的乘积具有任何特殊性质.



#### (5) 收敛矩阵 & 幂有界矩阵

我们称复方阵 $A\in \mathbb C^{n\times n}$ 是**收敛的** (convergent)，如果当 $k\to\infty$ 时 $A^k$ 的每个元素都趋于 $0$  
我们称复方阵 $A\in \mathbb C^{n\times n}$ 是**幂有界的** (power bounded)，如果 $\{A^k:k\in \mathbb Z_+\}$ 的所有元素构成 $\mathbb C$ **(存疑 $\mathbb C^{n\times n}$?)** 的有界子集.  
收敛矩阵一定是幂有界的，但幂有界矩阵不一定是收敛的 (单位阵 $I_n$ 就是这样一个例子)

设复方阵 $A\in \mathbb C^{n\times n}$ 的 Jordan 标准型为:
$$
S^{-1}AS
= J = 
\begin{bmatrix}
J^{(1)}(\lambda_1)\\
& J^{(2)}(\lambda_2) \\
& & \ddots \\
& & & J^{(d)}(\lambda_d)
\end{bmatrix}
$$
其中 $\lambda_1,\dots,\lambda_d$ 互不相同，$J_1(\lambda_1),\dots ,J_d(\lambda_d)$ 为对应的 Jordan 矩阵:  
$$
{\begin{cases}
J^{(1)}(\lambda_1) = J_{n_1^{(1)}} (\lambda_1) \oplus \dotsm \oplus J_{n^{(p_1)}_1} (\lambda_1)
\text{ where }n_1 = \sum_{i=1}^{p_1} n_1^{(i)} & (n_1^{(1)}\geq \dotsm \geq n_1^{(p_1)}\geq 1)\\
\qquad\quad\ \ \dotsm\\
J^{(d)}(\lambda_d) = J_{n_d^{(1)}} (\lambda_d) \oplus \dotsm \oplus J_{n^{(p_d)}_d} (\lambda_d)
\text{ where } n_d = \sum_{i=1}^{p_d} n_d^{(i)} & (n_d^{(1)}\geq \dotsm \geq n_d^{(p_d)}\geq 1) \\
\end{cases}}
$$
根据 $A^k = (SJS^{-1})^k = SJ^k S^{-1}$ 可知 $A^k\to 0_{n\times n}\ (n\to\infty)$ 的充要条件是 $J^k\to 0_{n\times n}\ (n\to\infty)$  
由于 Jordan 标准型 $J$ 是有限个 Jordan 块的直和，故我们只需研究单个 Jordan 块的性质即可.  
考虑关于特征值 $\lambda$ 的 $m$ 阶 Jordan 块 $J_m(\lambda)$，我们有 $(J_m(\lambda))^k
=
(J_m(0) + \lambda I_m)^k$​ 成立.  

回忆起幂零 Jordan 块的性质:    
**(幂零 Jordan 块的性质, Matrix Analysis 引理 $3.1.4$)**  
给定正整数 $n\geq 2$ 和 $x\in \mathbb{C}^n$，记 $\mathbb C^n$ 的第 $i$ 个标准单位基向量为 $e_i$，则我们有:

* $J_n(0)e_{i+1} = e_i\ (\forall\ i=1,\dots,n-1)$
* $J_n(0)^{\mathrm T}J_n(0) = \begin{bmatrix} 0 &\\ & I_{n-1} \end{bmatrix}$  
* $[I_n - J_n(0)^{\mathrm T}J_n(0)]x = (x^{\mathrm T}e_1)e_1$
* $J_n(0)^k = \begin{cases} \begin{bmatrix}  & I_{n-k}\\ 0_{k\times k} &  \end{bmatrix} 
  &\text{if }1 \leq k < n\\ 
  0_{n\times n} & \text{if }k\geq n\end{cases}$ 

则对于任意 $k\geq m$ 我们都有:   
$$
\begin{align}
(J_m(\lambda))^k
&=
(J_m(0) + \lambda I_m)^k\\
&=
\sum_{i=0}^k \binom{k}{i}(J_m(0))^i \cdot \lambda^{k-i} \\
&=
\lambda^k I_m + \sum_{i=1}^k \binom{k}{i}(J_m(0))^i \cdot \lambda^{k-i}\\
&=
\lambda^k I_m + \sum_{i=1}^{m-1}
\lambda^{k-i}\binom{k}{i} 
\begin{bmatrix}
 & I_{m-i}\\
 0_{i\times i} 
\end{bmatrix}\\
&=
\begin{bmatrix}
\lambda^k & \lambda^{k-1}\binom{k}{1} & \dotsm & \lambda^{k-m+2}\binom{k}{m-2} & \lambda^{k-m+1}\binom{k}{m-1}\\
& \lambda^k & \lambda^{k-1}\binom{k}{1} & \ddots & \lambda^{k-m+2}\binom{k}{m-2}\\
& & \ddots & \ddots & \vdots\\
& & & \lambda^k & \lambda^{k-1}\binom{k}{1}\\
& & & & \lambda^k
\end{bmatrix}
\end{align}
$$

- 一方面，根据 $(J_m(\lambda))^k \to 0_{m\times m}\ (k\to \infty)$ 蕴涵着 $\lambda^k \to 0\ (k\to\infty)$，即 $|\lambda|<1$ 

- 另一方面，假设 $|\lambda|<1$ (不妨设 $\lambda \neq 0$, 因为 $\lambda=0$ 时的收敛性是显然的)  
  对于任意 $i=1,\dots,m-1$ 我们都有:
  $$
  \begin{align}
  \left|\lambda^{k-i}\binom{k}{i}\right| 
  &= 
  \left|\lambda^{k-i}\frac{k!}{i!(k-i)!}\right|\\
  &\leq 
  \left|\lambda^{k-i}\frac{k^i}{i!}\right|\\
  &=
  \left|\frac{\lambda^k k^i}{\lambda^i i!}\right|\\
  &=\frac{1}{|\lambda|^i i!}\frac{k^i}{(\frac{1}{|\lambda|})^k}\to 0\ (k\to \infty)
  \end{align}\Rightarrow
  \lim_{k\to\infty}\lambda^{k-i}\binom{k}{i} = 0\ (i=1,\dots,m-1)
  $$
  因此当 $|\lambda|<1$ 时，$(J_m(\lambda))^k$ 的严格上三角元 $\lambda^{k-i}\binom{k}{i}\ (i=1,\dots,m-1)$ 在 $k\to\infty$ 下均趋近于 $0$   
  考虑到 $(J_m(\lambda))^k$ 的对角元 $\lambda^k$ 在 $k\to\infty$ 下也趋近于 $0$，故可知 $(J_m(\lambda))^k \to 0_{m\times m}\ (k\to \infty)$

综上所述，$(J_m(\lambda))^k \to 0_{m\times m}\ (k\to \infty)$ 的充要条件是 $|\lambda|<1$  
因此根据之前的讨论可知 $A$ **收敛** (即 $A^k\to 0_{m\times m}\ (k\to\infty)$) 的**充要条件**是 $A$ 的所有特征值的模长均 $<1$ 

为了分析幂有界的充要条件，我们只需额外考虑 $|\lambda|=1$ 的情况.  
显然当前仅当 Jordan 块 $J_m(\lambda)$ 的阶数 $m=1$ 时，其元素在 $k\to\infty$ 时才是有界的.  
因此 $A$ **幂有界** (即 $\{A^k:k\in \mathbb Z_+\}$ 的所有元素构成 $\mathbb C$ 的有界子集) 的**充要条件**是:  
$A$ 的所有特征值的模长均 $\leq 1$，且模长为 $1$ 的特征值对应的 Jordan 块都是 $1\times 1$ 的   
(即其模长为 $1$ 的特征值都是半简单的: 几何重数等于代数重数)

我们将上述讨论总结成以下定理:  
**(方阵收敛/幂有界的充要条件, Matrix Analysis 定理 $3.2.5.2$)**  
任意给定复方阵 $A\in \mathbb C^{n\times n}$，我们都有:

- $A$ 是收敛的 (即 $A^k\to 0_{m\times m}\ (k\to\infty)$)，当且仅当 $A$ 的所有特征值的模长均 $<1$ 
- $A$ 是幂有界的 (即 $\{A^k:k\in \mathbb Z_+\}$ 的所有元素构成 $\mathbb C$ 的有界子集)，   
  当且仅当 $A$ 的所有特征值的模长均 $\leq 1$，且模长为 $1$ 的特征值对应的 Jordan 块都是 $1\times 1$ 的   
  (即其模长为 $1$ 的特征值都是半简单的: 几何重数等于代数重数)



#### (6) Jordan-Chevalley 分解

在前文我们反复使用过这一事实:  
**任意 Jordan 块都能分解为一个幂零 Jordan 块和一个对角阵的和.**  
考虑关于 $\lambda$ 的 Jordan 块 $J_m(\lambda)$，它可分解为:  
$$
J_m(\lambda) = J_m(0) + \lambda I_m
$$
更一般地，设复方阵 $A\in \mathbb C^{n\times n}$ 的 Jordan 标准型为:
$$
S^{-1}AS
= J = 
\begin{bmatrix}
J^{(1)}(\lambda_1)\\
& J^{(2)}(\lambda_2) \\
& & \ddots \\
& & & J^{(d)}(\lambda_d)
\end{bmatrix}
$$
其中 $\lambda_1,\dots,\lambda_d$ 互不相同，$J_1(\lambda_1),\dots ,J_d(\lambda_d)$ 为对应的 Jordan 矩阵:  
$$
{\begin{cases}
J^{(1)}(\lambda_1) = J_{n_1^{(1)}} (\lambda_1) \oplus \dotsm \oplus J_{n^{(p_1)}_1} (\lambda_1)
\text{ where }n_1 = \sum_{i=1}^{p_1} n_1^{(i)} & (n_1^{(1)}\geq \dotsm \geq n_1^{(p_1)}\geq 1)\\
\qquad\quad\ \ \dotsm\\
J^{(d)}(\lambda_d) = J_{n_d^{(1)}} (\lambda_d) \oplus \dotsm \oplus J_{n^{(p_d)}_d} (\lambda_d)
\text{ where } n_d = \sum_{i=1}^{p_d} n_d^{(i)} & (n_d^{(1)}\geq \dotsm \geq n_d^{(p_d)}\geq 1) \\
\end{cases}}
$$
我们可以将 Jordan 标准型写为一个幂零矩阵和一个对角阵的和:  
$$
J_D:= D_1 \oplus \dotsm \oplus D_d\\
\begin{cases}
D_1 = \lambda_1 I_{n_1^{(1)}} \oplus \dotsm \oplus \lambda_1 I_{n_1^{(p_1)}} = \lambda_1 I_{n_1}\\
\quad\ \ \dotsm\\
D_d = \lambda_d I_{n_d^{(1)}} \oplus \dotsm \oplus \lambda_d I_{n_d^{(p_d)}} = \lambda_d I_{n_d}
\end{cases}\\
J_N:= N_1 \oplus \dotsm \oplus N_d \\
\begin{cases}
N_1 = J_{n_1^{(1)}}(0) \oplus \dotsm \oplus J_{n_1^{(p_1)}}(0)\\
\quad \ \dotsm\\
N_d = J_{n_d^{(1)}}(0) \oplus \dotsm \oplus J_{n_d^{(p_d)}}(0)
\end{cases}\\
$$
我们有 $J=J_N+J_D$ 成立.  
可定义 $\begin{cases}
A_N := S J_N S^{-1}\\
A_D := S J_D S^{-1}\end{cases}$ (它们满足 $A_N+A_D= SJ_NS^{-1}+SJ_DS^{-1}=S(J_N+J_D)S^{-1}=SJS^{-1}=A$)  
显然 $A_N$ 是一个幂零矩阵 (注意到 $A_N^k=SJ_N^k S^{-1}$, 因此 $A_N$ 继承了 $J_D$ 的幂零性)  
而 $A_D$ 是一个可相似对角化的矩阵.

由于 $J_N,J_D$ 分块方式一致且 $J_D$ 为对角阵，故我们有 $J_N J_D = J_D J_N$   
进而有 $A_NA_D = A_D A_N$ 

我们将上述讨论总结成以下定理:  
**(Jordan-Chevalley 分解, Matrix Analysis $3.2.7$ 节)**  
任意复方阵 $A\in \mathbb C^{n\times n}$ 都存在唯一的分解 $A = A_N + A_D$  
其中 $A_N$ 是幂零矩阵，$A_D$ 是可相似对角化的矩阵，且它们乘法可交换 (即 $A_NA_D = A_D A_N$)



#### (7) 与非退化矩阵的交换性

**(Toeplitz 矩阵)**    
若 $A\in \mathbb C^{n\times n}$ 的任意一条与主对角线平行的对角线的元素都相同，则我们称 $A$ 为 **Toeplitz 矩阵**:  
$$
\begin{align}
A 
&:= \begin{bmatrix}
a_0 & a_1 & a_2 & \dotsm & a_{n-1}\\
a_{-1} & a_0 & a_1 & \ddots &\vdots\\
a_{-2} & a_{-1} & \ddots & \ddots & a_2\\
\vdots& \ddots & \ddots & a_0 & a_1\\
a_{-(n-1)}& \dotsm & a_{-2} & a_{-1}& a_0
\end{bmatrix}\quad (\text{where }a_{-(n-1)},\dots,a_{-1},a_0,a_1,\dots,a_{n-1}\in \mathbb C)\\
&=
a_0 I_n + \sum_{k=1}^{n-1} a_k (J_n(0))^k + \sum_{k=1}^{n-1} a_{-k} (J_n^{\mathrm T}(0))^k
\end{align}
$$
其中:  
$$
J_n(0) := \begin{bmatrix}
0 & 1 & & \\
& 0 & 1 & \\
&&\ddots & \ddots &\\
&& &0 & 1 \\
&&&&0
\end{bmatrix}_{n\times n}\text{ and }J_n(0)^k = \begin{cases} \begin{bmatrix}  & I_{n-k}\\ 0_{k\times k} &  \end{bmatrix} 
&\text{if }1 \leq k < n\\ 
0_{n\times n} & \text{if }k\geq n\end{cases}\\

J_n^{\mathrm T}(0) := \begin{bmatrix}
0 &  & & \\
1& 0 &  & \\
&1&\ddots &  &\\
&&\ddots &0 &  \\
&&&1&0
\end{bmatrix}_{n\times n}\text{ and }(J_n^{\mathrm T}(0))^k = \begin{cases} \begin{bmatrix}  & 0_{k\times k}\\ I_{n-k} &  \end{bmatrix} 
&\text{if }1 \leq k < n\\ 
0_{n\times n} & \text{if }k\geq n\end{cases}
$$
根据它们对标准基 $\{e_1,\dots,e_n\}$ 的作用，  
我们称 $J_n(0)$ 为**后向位移** (backward shift)，称 $J_n^{\mathrm T}(0)$ 为**前向位移** (forward shift)  
记 $K_n$ 为 $n$ 阶反序矩阵，则我们有 $J_n^{\mathrm T}(0) = K_n^{-1}J_n(0)K_n$，进而有 $A^{\mathrm T} = K_n^{-1}AK_n$ 

**(上三角 Toeplitz 矩阵)**    
上三角 Toeplitz 矩阵 $A\in \mathbb C^{n\times n}$ 可以表示为 $J_n(0)$ 的多项式:  
$$
A:= \begin{bmatrix}
a_0 & a_1 & a_2 & \dotsm & a_{n-1}\\
& a_0 & a_1 & \ddots &\vdots\\
 &  & \ddots & \ddots & a_2\\
&  & & a_0 & a_1\\
&  &  & & a_0
\end{bmatrix} 
=
a_0 I_n + a_1 J_n(0) + a_2 (J_n(0))^2 + \dotsm + a_{n-1}(J_n(0))^{n-1}
$$
容易验证 $n$ 阶上三角 Toeplitz 矩阵是一个**交换代数**:

- ① $n$ 阶上三角 Toeplitz 矩阵的线性组合以及乘积仍是 $n$ 阶上三角 Toeplitz 矩阵
- ② 任意两个 $n$ 阶上三角 Toeplitz 矩阵一定可交换
- ③ $n$ 阶上三角 Toeplitz 矩阵 $A$ 非奇异，当且仅当 $a_0\neq 0$  
  此时 $A^{-1} = b_0 I_n + b_1 J_n(0) + b_2 (J_n(0))^2 + \dotsm + b_{n-1}(J_n(0))^{n-1}$   
  其系数满足 $\begin{cases}
  b_0 = a_0^{-1}\\
  b_k = a_0^{-1}(\sum_{m=0}^{k-1}a_{k-m}b_m)\ (k=1,\dots,n-1)\end{cases}$

*****

**(Matrix Analysis 引理 $3.2.4.1$)**  
若复方阵 $A\in \mathbb C^{n\times n}$ 可与 $n$ 阶 Jordan 块 $J_n(\lambda)$ 交换，即满足 $AJ_n(\lambda)=J_n(\lambda)A$  
则 $A$ 是上三角 Toeplitz 矩阵，因而存在不超过 $n-1$ 次的复系数多项式 $g(t)$ 使得 $A=g(J_n(\lambda))$ 

- **证明:**  
  将 $A$ 按列记为 $A=[a_1,\dots,a_n]$   
  则等式 $AJ_n(\lambda)=J_n(\lambda)A$ 可等价表示为:    
  (其中 $\otimes$ 代表 Kronecker 乘积，$\text{vec}(\cdot)$ 代表向量化操作符)
  $$
  J_n(\lambda)A - AJ_n(\lambda) = 0_{n\times n}\\
  \Leftrightarrow\\
  (I_n \otimes J_n(\lambda) - J_n^{\mathrm T}(\lambda)\otimes I_n)\text{vec}(A) = \text{vec}(0_{n\times n})\\
  \Leftrightarrow\\
  \begin{bmatrix}
  J_{n}(0) \\
  -I_n & J_{n}(0)\\
  & \ddots & \ddots & \\
  & & -I_n & J_{n}(0)
  \end{bmatrix}
  \begin{bmatrix}
  a_1\\
  a_2\\
  \vdots\\
  a_n
  \end{bmatrix}
  =
  \begin{bmatrix}
  0_n\\
  0_n\\
  \vdots\\
  0_n
  \end{bmatrix}\\
  \Leftrightarrow\\
  \begin{cases}
  J_n(0) a_1 = 0_n\\
  J_n(0) a_k = a_{k-1}\ (k=2,\dots,n)
  \end{cases}
  $$
  我们可知 $a_1,\dots,a_n$ 具有如下形式:  
  $$
  \begin{align}
  a_1 &= c_0 e_1\\
  a_2 &= c_1 e_1 + c_0 e_2\\
  a_3 &= c_2 e_1 + c_1 e_2 + c_0 e_3\\
  &\dotsm\\
  a_n &= c_{n-1} e_1 + \dotsm + c_0 e_n
  \end{align}\quad (\text{where }c_0,\dots,c_{n-1}\in \mathbb C)
  $$
  因此 $A$ 一定是**上三角 Toeplitz 矩阵**:
  $$
  A =
  [a_1,a_2,\dots,a_n]
  =
  \begin{bmatrix}
  c_0 & c_1 & c_2 & \dotsm & c_{n-1}\\
  & c_0 & c_1 & \ddots &\vdots\\
  & & \ddots & \ddots & c_2\\
  & & & c_0 & c_1\\
  & & & & c_0
  \end{bmatrix}\quad (\text{where }c_0,\dots,c_{n-1}\in \mathbb C)
  $$
  注意到幂零 Jordan 块 $J_n(0)$ 满足 $J_n(0)^k = \begin{cases} \begin{bmatrix}  & I_{n-k}\\ 0_{k\times k} &  \end{bmatrix} 
  &\text{if }1 \leq k < n\\ 
  0_{n\times n} & \text{if }k\geq n\end{cases}$   
  因此 $A$ 一定可以写成幂零 Jordan 块 $J_n(0)$ 的一个不超过 $n-1$ 次的复系数多项式，  
  进而可以写成任意 Jordan 块 $J_n(\lambda)\ (\lambda\in \mathbb C)$ 的一个不超过 $n-1$ 次的复系数多项式:  
  $$
  \begin{align}
  A &= 
  \begin{bmatrix}
  c_0 & c_1 & c_2 & \dotsm & c_{n-1}\\
  & c_0 & c_1 & \ddots &\vdots\\
  & & \ddots & \ddots & c_2\\
  & & & c_0 & c_1\\
  & & & & c_0
  \end{bmatrix}\\
  &=
  c_0 I_n 
  + 
  c_1 \begin{bmatrix}  & I_{n-1}\\ 0_{1\times 1} &  \end{bmatrix} 
  +
  c_2 \begin{bmatrix}  & I_{n-2}\\ 0_{2\times 2} &  \end{bmatrix}
  +
  \dotsm
  +
  c_{n-1} \begin{bmatrix}  & I_{1}\\ 0_{(n-1)\times (n-1)} &  \end{bmatrix}\\
  &=
  c_0 I_n + c_1 J_n(0) + c_2 J_n^2(0) + \dotsm + c_{n-1} J_n^{n-1}(0)\\
  &=
  c_0 I_n + c_1 [J_n(\lambda) - \lambda I_n] + c_2 [J_n(\lambda) - \lambda I_n]^2 + \dotsm + c_{n-1} [J_n(\lambda)-\lambda I_n]^{n-1}\\
  &\overset{\Delta}=
  g(J_n(\lambda))
  \end{align} \quad (\text{where }c_0,\dots,c_{n-1}\in \mathbb C)
  $$
  引理得证.

*****

**(Matrix Analysis 定理 $3.2.4.2$)**  
设复方阵 $B\in \mathbb C^{n\times n}$ 是非退化的 (即每个不同的特征值几何重数都是 $1$)  
若复方阵 $A\in \mathbb C^{n\times n}$ 与 $B$ 可交换 (即 $AB=BA$)，则存在一个至多 $n-1$ 次的多项式 $g(t)$ 使得 $A=g(B)$  

- **逆命题:**   
  若对于任意一个能与 $B\in \mathbb C^{n\times n}$ 交换的复方阵 $A\in \mathbb C^{n\times n}$，  
  都存在一个至多 $n-1$ 次的多项式 $g(t)$ 使得 $A=g(B)$，  
  则 $B$ 一定是非退化的方阵 (即每个不同的特征值几何重数都是 $1$)

- **Lemma: (Matrix Analysis 推论 $2.4.4.2$)**  
  设 $B,C\in \mathbb C^{n\times n}$ 是分块对角的，共形地划分为 $\begin{cases}
  B = B_1 \oplus \dotsm \oplus B_k\\
  C = C_1 \oplus \dotsm \oplus C_k\end{cases}$  且满足 $\text{eig}(B_i)\cap \text{eig}(C_j)= \emptyset\ (\forall\ i\neq j)$   
  若 $A\in \mathbb C^{n\times n}$ 满足 $AB=CA$，  
  则 $A$ 也可与 $B,C$ 共形地划分为 $A = A_1\oplus \dotsm \oplus A_k$ 且有 $A_i B_i = C_i A_i\ (i=1,\dots,k)$   
  (这是 Sylvester 定理的直接推论，证明参见 FDU 高等线性代数 1. 复数与复矩阵)

- **证明:**  
  设 $B$ 的 Jordan 标准型为 $J_B = S^{-1}BS$ (其中 $S\in \mathbb C^{n\times n}$ 为某个非奇异阵)  
  $$
  AB=BA\\
  \Leftrightarrow\\
  A(SJ_BS^{-1}) = (SJ_BS^{-1})A\\
  \Leftrightarrow\\
  (S^{-1}AS)J_B = J_B(S^{-1}AS)
  $$
  要证明存在一个至多 $n-1$ 次的多项式 $g(t)$ 使得 $A=g(B)$  
  就等价于证明存在一个至多 $n-1$ 次的多项式 $g(t)$ 使得 $S^{-1}AS = S^{-1}g(B)S = g(S^{-1}BS) = g(J_B)$   
  因此我们只需假设 $B$ 本身是一个非退化的 Jordan 矩阵 (即 $B$ 的每个不同特征值都只对应一个 Jordan 块) 就够了.

  设 $B = J_{n_1}(\lambda_1)\oplus \dotsm \oplus J_{n_d}(\lambda_d)$ (其中 $\lambda_1,\dots,\lambda_d\in \mathbb C$ 互不相同)  
  **Lemma (Matrix Analysis 推论 $2.4.4.2$)** 保证 $A$ 也是分块对角的，记为 $A=A_{11}\oplus \dotsm \oplus A_{dd}$   
  且这些分块满足 $A_{ii}J_{n_i}(\lambda_i) = J_{n_i}(\lambda_i)A_{ii}\ (i=1,\dots,d)$   
  根据 **Matrix Analysis 引理 $3.2.4.1$** 可知 $A_{ii}\ (i=1,\dots,d)$ 是**上三角 Toeplitz 矩阵**

  要构造一个至多 $n-1$ 次的多项式 $g(t)$ 满足 $A=g(B) = g(J_{n_1}(\lambda_1)\oplus \dotsm \oplus J_{n_d}(\lambda_d))$    
  只需构造 $d$ 个至多 $n-1$ 次的多项式 $g_i(t)\ (i=1,\dots,d)$ 满足 $\begin{cases}
  g_i(J_{n_i}(\lambda_i))=A_{ii}\\
  g_i(J_{n_j}(\lambda_j))=0_{n_j\times n_j}\ (j\neq i)\end{cases}\ (i=1,\dots,d)$  
  最终 $g(t)=g_1(t)+\dotsm + g_d(t)$ 就是所要求的多项式.  

  于是我们对 $i=1,\dots,d$ 定义:  
  $$
  q_i(t) := \prod_{k\in \{1,\dots,d\}\backslash \{i\}} (t-\lambda_k)^{n_k}
  $$
  注意到 $q_i(t)$ 是 $\sum_{j\neq i} n_j = n-n_i$ 次多项式，且对于任意 $j\neq i$ 都满足:  
  $$
  \begin{align}
  q_i(J_{n_j}(\lambda_j))
  &=
  \prod_{k\in \{1,\dots,d\}\backslash\{i\}} (J_{n_j}(\lambda_j) - \lambda_k I_{n_j})^{n_k}\\
  &=
  \left\{\prod_{k\in \{1,\dots,d\}\backslash\{i,j\}}(J_{n_j}(\lambda_j) - \lambda_k I_{n_j})^{n_k}\right\}
  (J_{n_j}(\lambda_j) - \lambda_j I_{n_j})^{n_j}\\
  &=
  \left\{\prod_{k\in \{1,\dots,d\}\backslash\{i,j\}}(J_{n_j}(\lambda_j) - \lambda_k I_{n_j})^{n_k}\right\}
  (J_{n_j}(0))^{n_j}\\
  &=
  \left\{\prod_{k\in \{1,\dots,d\}\backslash\{i,j\}}(J_{n_j}(\lambda_j) - \lambda_k I_{n_j})^{n_k}\right\}
  0_{n_j\times n_j}\\
  &=
  0_{n_j\times n_j}
  \end{align}
  $$
  而 $q_i(J_{n_i}(\lambda_i)) = \prod_{k\neq i} (J_{n_i}(\lambda_i)-\lambda_k I_{n_i})^{n_k}$ 作为 $d-1$ 个非奇异的上三角 Toeplitz 矩阵的乘积，  
  也一定是非奇异的上三角 Toeplitz 矩阵的乘积，  
  于是其逆 $[q_i(J_{n_i}(\lambda_i))]^{-1}$ 存在且是上三角 Toeplitz 矩阵，  
  进而可知 $[q_i(J_{n_i}(\lambda_i))]^{-1} A_{ii}$ 也是上三角 Toeplitz 矩阵.  
  因此我们可以找到一个不超过 $n_i-1$ 次的多项式 $r(t)$ 使得 $r(J_{n_i}(\lambda_i)) = [q_i(J_{n_i}(\lambda_i))]^{-1} A_{ii}\ (i=1,\dots,d)$   

  现在我们终于可以定义 $g_i(t)$ 了，它是不超过 $(n-n_i) + (n_i-1)=n-1$ 次的多项式:  
  $$
  g_i(t) := q_i(t) r_i(t)\ (i=1,\dots,d)
  $$
  容易验证它满足 $\begin{cases}
  g_i(J_{n_i}(\lambda_i))=A_{ii}\\
  g_i(J_{n_j}(\lambda_j))=0_{n_j\times n_j}\ (j\neq i)\end{cases}\ (i=1,\dots,d)$  
  最后我们成功构造出不超过 $n-1$ 次的多项式 $g(t):=g_1(t)+\dotsm + g_d(t)$ 满足 $A=g(B) = g(J_{n_1}(\lambda_1)\oplus \dotsm \oplus J_{n_d}(\lambda_d))$  
  定理得证.

*****

**(Matrix Analysis 推论 $3.2.4.4$)**  
给定 $A,S\in \mathbb C^{n\times n}$，设 $A$ 是非退化的 (即每个不同的特征值几何重数都是 $1$)  
则我们有以下结论:

- ① 若 $AS=SA^{\mathrm T}$，则 $S$ 是对称的

  **证明:**  
  **Matrix Analysis 定理 $3.2.3.1$** 保证了存在一个非奇异的复对称阵 $P\in \mathbb{C}^{n\times n}$ 使得 $PAP^{-1} = A^{\mathrm T}$  
  因此我们有 $AS=SA^{\mathrm T} = S(PAP^{-1})$，  
  从而有 $A(SP) = (SP)A$ 

  > **(Matrix Analysis 定理 $3.2.3.1$)**  
  > 对于任意复方阵 $A\in \mathbb{C}^{n\times n}$，存在一个非奇异的复对称阵 $S\in \mathbb{C}^{n\times n}$ 使得 $S^{-1}AS = A^{\mathrm T}$   

  这样一来，**Matrix Analysis 定理 $3.2.4.2$** 便保证了存在一个多项式 $g(t)$ 满足 $SP=g(A)$   
  于是我们有:  
  $$
  \begin{align}
  PS^{\mathrm T}
  &=
  (SP)^{\mathrm T}\\
  &=
  (g(A))^{\mathrm T}\\
  &=
  g(A^{\mathrm T})\\
  &=
  g(PAP^{-1})\\
  &=
  Pg(A)P^{-1}\\
  &=
  P(SP)P^{-1}\\
  &=
  PS
  \end{align}
  $$
  由于 $P$ 是非奇异阵，故我们有 $S^{\mathrm T}=S$ 成立，表明 $S$ 是对称阵.

- ② 若 $S$ 非奇异，且 $A^{\mathrm T}=S^{-1}AS$，则 $S$ 是对称的  
  (这是 ① 的直接推论，表明当 $A$ 非退化时，$A$ 与 $A^{\mathrm T}$ 之间的任何一个相似变换矩阵一定是对称阵)



### 3.2.6 极小多项式

#### (1) 与 Jordan 标准型的联系

给定复方阵 $A\in \mathbb C^{n\times n}$  
若多项式 $p(t)$ 使得 $p(A)=0_{n\times n}$，则我们称 $p(t)$ 可使 $A$ **零化** (annihilate)  

**Cayley-Hamilton 定理** (Matrix Analysis 定理 $2.4.3.2$) 保证了特征多项式 $p_A(t)$ 可使 $A$ 零化 (即 $p_A(t)=0_{n\times n}$)  
因此能使 $A$ 零化的多项式是存在的，而且次数至少可以是 $n$ 次 (甚至更小)  
我们特别感兴趣的是使 $A$ 零化的最低次数的首 $1$ 多项式，称为**极小多项式** (minimal polynomial)，记作 $m_A(t)$  
可以证明: 任意复方阵 $A\in \mathbb C^{n\times n}$ 极小多项式 $m_A(t)$ 存在且唯一.

**(Matrix Analysis 定理 $3.3.1$)**  
任意给定 $A\in \mathbb C^{n\times n}$，存在唯一的极小多项式 $m_A(t)$，其次数至多为 $n$  
且任意一个可使 $A$ 零化的首 $1$ 多项式 $h(t)$ 都能被极小多项式 $m_A(t)$ 整除，即存在 $q(t)$ 使得 $h(t)=q(t)m_A(t)$ 

- ① **极小多项式的存在性**可由 Cayley-Hamilton 定理说明，也可通过以下论述说明:    
  $n$ 阶复方阵的全体构成线性空间 $\mathbb{C}^{n\times n}$，维数为 $n^2$  
  故对于任意复方阵 $A\in \mathbb{C}^{n\times n}$，$I_n,A,A^2,...,A^{n^2}$ 一定是线性相关的.  
  因此至少存在一个 $k\leq n^2$ 次的首一多项式 $h(t)$ 使得 $h(A)=0_{n\times n}$   
  这表明极小多项式是存在的，且次数至多为 $n^2$.  
  Cayley-Hamilton 定理进一步表明极小多项式的次数至多为 $n$.
- ② 任意一个可使 $A$ 零化的首 $1$ 多项式 $h(t)$ 都能被极小多项式 $m_A(t)$ 整除  
  **(反证法)** 假设 $h(t)$ 不能被 $m_A(t)$ 整除，  
  则存在 $q(t),r(t)$ 使得 $h(t)=q(t)m_A(t)+r(t)$ (其中 $r(t)$ 次数低于 $m_A(t)$)  
  于是我们有 $r(A)=h(A)-q(A)m_A(A)=0_{n\times n}-q(A)\cdot 0_{n\times n}=0_{n\times n}$   
  记 $\widetilde r(t)$ 为 $r(t)$ 首 $1$ 化得到的多项式.  
  这样我们就找到了一个次数低于 $m_A(t)$ 的可使 $A$ 零化的首 $1$ 多项式 $\widetilde r(A)$  
  这与 "$m_A(t)$ 是极小多项式" 的假设矛盾.   
  因此任意一个可使 $A$ 零化的首 $1$ 多项式 $h(t)$ 都能被极小多项式 $m_A(t)$ 整除.
- ③ **极小多项式的唯一性**:  
  若 $m_1(t),m_2(t)$ 均为 $A$ 的极小多项式，则根据 ② 可知它们可以相互整除，因此只相差常数倍.  
  又由于它们均是首 $1$ 多项式，故它们相差的常数倍为 $1$，即它们相等.

根据上述定理我们可得到如下推论:

- **(Matrix Analysis 推论 $3.3.3$)**  
  相似矩阵拥有相同的极小多项式.  
  **证明:**  
  设 $A,B\in \mathbb C^{n\times n}$ 相似，即存在非奇异阵 $S\in \mathbb C^{n\times n}$ 使得 $A=SBS^{-1}$，则我们有:
  $$
  m_B(A) = m_B(SBS^{-1}) = Sm_B(B)S^{-1} = S\cdot 0_{n\times n}\cdot S^{-1} = 0_{n\times n}\ \ \Rightarrow \ \ m_A(t)\ |\ m_B(t)\\
  m_A(B) = m_A(S^{-1}AS) = S^{-1}m_A(A)S = S^{-1}\cdot 0_{n\times n}\cdot S = 0_{n\times n}\ \ \Rightarrow
  \ \ m_B(t)\ |\ m_A(t)
  $$
  因此 $m_A(t)$ 和 $m_B(t)$ 可以相互整除，表明它们只相差常数倍，即存在 $c\in \mathbb C$ 使得 $m_A(t)=cm_B(t)$.  
  注意到 $m_A(t)$ 和 $m_B(t)$ 都是首 $1$ 多项式，故 $c=1$，即有 $m_A(t)=m_B(t)$  
  命题得证.

  但拥有相同的极小多项式的两个不同方阵不一定是相似的  
  例如 $A=J_2(0)\oplus J_2(0)$ 和 $B=J_2(0)\oplus 0_{2\times 2}$ 

- **(Matrix Analysis 推论 $3.3.4$)**  
  任意给定 $A\in \mathbb C^{n\times n}$，其极小多项式 $m_A(t)$ 都能整除特征多项式 $p_A(t)$   
  此外，特征多项式 $p_A(t)$ 的每一个根 (即 $A$ 的每一个特征值) 都是极小多项式 $m_A(t)$ 的根.

  第一个命题是显然的.  
  第二个命题可以这样说明:  
  对于 $A$ 的任意特征对 $(\lambda,x)$ 都有 $0_n=0_{n\times n}x = q_A(A)x = q_A(\lambda) x$，因此有 $q_A(\lambda)=0$ 成立.  
  (注意特征向量 $x$ 天然满足 $x\neq 0_n$)

  这个推论提供了计算小型矩阵 $A$ 的极小多项式的一种方法:  
  逐一筛选出特征多项式 $p_A(\lambda)$ 所有能零化 $A$ 的因式，并从中选取次数最小的因式，即得极小多项式 $m_A(t)$.  
  尽管从数值计算的角度来说，这不是一个好的算法，但对于形式简单的小矩阵还是有用的.

- **(高等代数学, 命题 $6.3.3$)**  
  分块对角阵 $A=A_1\oplus \dotsm \oplus A_d$ 的极小多项式 $m_A(t)$ 是 $A_1,\dots,A_d$ 极小多项式 $m_{A_1}(t),\dots,m_{A_d}(t)$ 的最小公倍式.  
  这是显然的，因为 $m_{A_1}(t),\dots,m_{A_d}(t)$ 中的每一个都能整除 $m_A(t)$，因此 $m_A(t)$ 是它们的公倍式.  
  又由于 $m_A(t)$ 是极小多项式，故 $m_A(t)$ 是 $m_{A_1}(t),\dots,m_{A_d}(t)$ 的最小公倍式.

*****

复方阵 $A\in \mathbb C^{n\times n}$ 的极小多项式 $m_A(t)$ 与 Jordan 标准型 $J$ 之间存在密切的联系.  

**(Matrix Analysis 定理 $3.3.6$)**  
设复方阵 $A\in \mathbb C^{n\times n}$ 的 Jordan 标准型为:
$$
S^{-1}AS
= J = 
\begin{bmatrix}
J^{(1)}(\lambda_1)\\
& J^{(2)}(\lambda_2) \\
& & \ddots \\
& & & J^{(d)}(\lambda_d)
\end{bmatrix}
$$
其中 $\lambda_1,\dots,\lambda_d$ 互不相同，$J_1(\lambda_1),\dots ,J_d(\lambda_d)$ 为对应的 Jordan 矩阵:  
$$
{\begin{cases}
J^{(1)}(\lambda_1) = J_{n_1^{(1)}} (\lambda_1) \oplus \dotsm \oplus J_{n^{(p_1)}_1} (\lambda_1)
\text{ where }n_1 = \sum_{i=1}^{p_1} n_1^{(i)} & (n_1^{(1)}\geq \dotsm \geq n_1^{(p_1)}\geq 1)\\
\qquad\quad\ \ \dotsm\\
J^{(d)}(\lambda_d) = J_{n_d^{(1)}} (\lambda_d) \oplus \dotsm \oplus J_{n^{(p_d)}_d} (\lambda_d)
\text{ where } n_d = \sum_{i=1}^{p_d} n_d^{(i)} & (n_d^{(1)}\geq \dotsm \geq n_d^{(p_d)}\geq 1) \\
\end{cases}}
$$
定义:
$$
\begin{align}
r_i 
&:= \min\{k\in \mathbb Z_+: (J^{(i)}(\lambda_i) - \lambda_i I_{n_i})^k = 0_{n_i\times n_i}\}\\
&= \min\{n_i^{(1)},\dots,n_i^{(p_i)}\}
\end{align}\ (i=1,\dots,d)
$$
即 $r_i$ 为 $A$ 关于特征值 $\lambda_i$ 的所有 Jordan 块 $J_{n_i^{(1)}}(\lambda_i),\dots,J_{n_i^{(p_i)}}(\lambda_i)$ 的最大的阶.  
则 $J^{(i)}(\lambda_i)$ 的极小多项式 (即最小次数的首一零化多项式) 为 $m_{J^{(i)}(\lambda_i)}(t)=(t-\lambda_i)^{r_i}$
因此 $A$ 的极小多项式为:  
$$
m_A(t)= m_J(t)= \prod_{i=1}^d m_{J^{(i)}(\lambda_i)}(t) = \prod_{i=1}^d (t-\lambda_i)^{r_i}
$$

- 上述结果在计算极小多项式时没有太大的帮助，  
  因为确定一个矩阵的 Jordan 标准型要比确定它的极小多项式更困难.  
  (在已知特征值的情况下，极小多项式可通过试错法简单确定)

***

**(Matrix Analysis 推论 $3.3.8$)**  
复方阵 $A\in \mathbb C^{n\times n}$ 可相似对角化，当且仅当其极小多项式没有重根，即其所有特征值对应的所有 Jordan 块的阶数都是 $1$  
换言之，当且仅当 $\prod_{i=1}^d(t-\lambda_i)$ 可以零化 $A$，即 $(A-\lambda_1 I_n)\dotsm (A-\lambda_d I_n)=0_{n\times 0}$

这个判别法对于判断一个给定的方阵是否可以对角化是有实际用途的.  
只要我们知道它不同的特征值 $\lambda_1,\dots,\lambda_d$，就可以构造多项式 $\prod_{i=1}^d(t-\lambda_i)$ 并观察它是否使 $A$ 零化.  

- 如果它能使 $A$ 零化，它必定是 $A$ 的极小多项式.  
  这是因为没有更低次数的多项式能以 $A$ 的所有不同的特征值作为其零点了
- 如果它不能使 $A$ 零化，那么 $A$ 不可对角化.  

将上述结果总结成以下等价形式是有助益的.  
**(Matrix Analysis 推论 $3.3.10$)**  
给定 $A\in \mathbb C^{n\times n}$，设 $m_A(t)$ 是其极小多项式，则以下命题等价:

- ① $m_A(t)$ 是不同线性因子的乘积
- ② $A$ 的每一个特征值作为 $m_A(t)=0$ 的根的重数都是 $1$ 
- ③ 对 $A$ 的每个特征值 $\lambda$ 都有 $m_A'(\lambda) =0 $ 
- ④ $A$ 可对角化



#### (2) Frobenius 友阵

迄今为止，我们考虑的都是寻找能使给定方阵 $A\in \mathbb C^{n\times n}$ 零化的最小次数的首 $1$ 多项式，即极小多项式 $m_A(t)$   
现在我们研究其逆过程:   
给定一个 $n$ 次首 $1$多项式 $m(t):= t^n + a_{n-1}t^{n-1} + \dots + a_1t + a_0$，   
是否存在一个 $n$ 阶矩阵 $A$ 使它以 $m(t)$ 为极小多项式?

答案是肯定的:  
$$
A:= 
\begin{bmatrix}
0 & & &  & -a_0\\
1 & 0 &  & & -a_1\\
& 1 & \ddots & & \vdots\\
& & \ddots & 0 & -a_{n-2}\\
& & & 1 & -a_{n-1}
\end{bmatrix}
$$

记 $e_1,\dots,e_n$ 为 $\mathbb C^n$ 的单位标准正交基向量.  
观察 $A$ 的前 $n-1$ 列，我们有:   
$$
\begin{cases} e_2 = Ae_1\\ e_3  = Ae_2= A^2 e_1\\ e_4 = Ae_3 = A^3 e_1\\ \quad\dotsm\\ e_{n} = Ae_{n-1} = A^{n-1}e_1\\ \end{cases}
$$
至于 $A$ 的第 $n$ 列，我们有:
$$
\begin{align}
Ae_n 
&= -a_{n-1} e_n - a_{n-2} e_{n-1} - \dotsm - a_1 e_2 -a_0 e_1\\
&= -a_{n-1} A^{n-1} e_1 - a_{n-2} A^{n-2}e_1 -\dots - a_1 Ae_1 -a_0 e_1 \\
&= (A^n -m(A)) e_1 
\end{align}
$$
根据 $e_n = A^{n-1}e_1$ 还有 $Ae_n = A^n e_1$    
因此我们有 $(A^n-m(A)) e_1 = A^n e_1$   
于是有 $m(A)e_1 = 0_n$    
进而有:
$$
\begin{align}
m(A)e_k
&=
m(A)(A^{k-1}e_1)\\
&=
A^{k-1}m(A)e_1\\
&=
A^{k-1}0_{n}\\
&=
0_n
\end{align}\ (k=1,\dots,n)
$$
这表明 $m(A)$ 的第 $1,\dots,n$ 列均为零向量 $0_n$，因此 $m(A)=0_{n\times n}$   
即 $m(t)$ 是 $A$ 的零化多项式.

*****

为证明 $m(t)$ 是 $A$ 的极小多项式，  
可设存在一个次数为 $s<n$ 的首一多项式 $m_*(t)$ 满足 $m_*(A) = 0_{n\times n}$   
设 $m_*(t)= t^s+b_{s-1}t^{s-1} + b_{s-2}t^{s-2} + \dots + b_1t + b_0$，则我们有:
$$
\begin{align}
m_*(A) e_1 
&= A^se_1 + b_{s-1}A^{s-1}e_1 + \dots + b_1A e_1+ b_0 e_1 \\
&= e_{s+1} + b_{s-1}e_{s} + \dots + b_1 e_2 + b_0 e_1\\
&= 0_n
\end{align}
$$
但 $\mathbb C^n$ 的单位标准正交基向量 $e_{s+1}, e_s, \dots , e_2, e_1$ 是线性无关的，上式显然与之矛盾.  
因此任意一个次数为 $s<n$ 的首一多项式都不能零化 $A$.  
故 $m(t)$ 就是 $A$ 的极小多项式.  

****

这样我们就基于给定的 $n$ 次首一多项式 $m(t)$ 构造了以 $m(t)$ 为极小多项式的 $n$ 阶方阵 $A$:  
$$
A:= 
\begin{bmatrix}
0 & & &  & -a_0\\
1 & 0 &  & & -a_1\\
& 1 & \ddots & & \vdots\\
& & \ddots & 0 & -a_{n-2}\\
& & & 1 & -a_{n-1}
\end{bmatrix}\\
\text{where }m(t):= t^n + a_{n-1}t^{n-1} + \dots + a_1t + a_0
$$
我们称这样的 $A$ 为 $m(t)$ 的**友矩阵** (companion matrix).   
(需要注意区分的是, 方阵的伴随矩阵的英文是 Adjugate Matrix)  
由于 $m(t)$ 的次数与 $A$ 的阶数相同，因此 $m(t)$ 也正是 $A$ 的特征多项式.

值得注意的是，任意 Frobenius 友阵都是**非退化的** (nonderogatory)   
(即所有特征值的几何重数都是 $1$，又即每个不同的特征值只对应一个 Jordan 块)  
尽管非退化的方阵 $A\in \mathbb C^{n\times n}$ 不一定都是 Frobenius 友阵，  
但非退化的方阵 $A\in \mathbb C^{n\times n}$ 一定相似于特征多项式 $p_A(t)$ 对应的 Frobenius 友阵，因为它们具有相同的 Jordan 标准型.

上面的论述可以总结为以下结论:  
**(Matrix Analysis 定理 $3.3.15$)**  
设 $n$ 阶方阵 $A\in \mathbb{C}^{n\times n}$ 的特征多项式和极小多项式分别为 $p_A(t) $ 和 $m_A(t)$   
则下列命题等价: 

* $m_A(t)$ 的次数为 $n$
* $m_A(t) \equiv p_A(t)$
* $A$ 是非退化的 (nonderogatory)，即 $A$ 所有特征值的几何重数都是 $1$ (即每个不同的特征值只对应一个 Jordan 块)
* $A$ 相似于 $p_A(t)$ 对应的 Frobenius 友阵

**The End**
