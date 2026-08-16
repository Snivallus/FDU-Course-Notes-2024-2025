# FDU 高等线性代数 4. Hermite 特征值

本文根据邵美悦老师授课内容整理而成，并参考了以下教材：  

* Matrix Analysis (R. Horn & C. Johnson) Chapter $4$
* 矩阵分析 (R. Horn & C. Johnson) 第 $4$ 章

欢迎批评指正!

## 4.1 Hermite 阵

考虑复方阵 $A\in \mathbb C^{n\times n}$  
若 $A^{\mathrm T}=A$，则我们称 $A$ 是**对称的** (symmetric)  
若 $A^{\mathrm T}=-A$，则我们称 $A$ 是**反对称的** (skew symmetric)   
若 $A^{\mathrm H} = A$，则我们称 $A$ 是 **Hermite 的** (Hermitian)  
若 $A^{\mathrm H} = -A$，则我们称 $A$ 是**反 Hermite 的** (skew Hermitian)    
若存在某个 $\theta\in \mathbb R$ 使得 $e^{i\theta}A$ 是 Hermite 的，则我们称 $A$ 是**本性 Hermite 的** (essentially Hermitian)

对于任意复方阵 $A\in \mathbb C^{n\times n}$ 我们有如下结论:

- $A+ A^{\mathrm H},A^{\mathrm H}A,A^{\mathrm H}A$ 都是 Hermite 阵，而 $A-A^{\mathrm H}$ 是反 Hermite 阵
- 若 $A$ 是 Hermite 阵，则其正整数次幂 $A^{k}\ (k\geq 1)$ 都是 Hermite 阵.  
  进一步，若 $A$ 还是非奇异的，则其负整数次幂 $A^{k}\ (k\leq -1)$ 也都是 Hermite 阵.
- 若 $A$ 是 Hermite 阵，则 $iA$ 是反 Hermite 阵.  
  若 $A$ 是反 Hermite 阵，则 $iA$ 是 Hermite 阵.
- 若 $A$ 是 Hermite 阵，则 $A$ 的对角元均为实数. 

***

- 任意复方阵 $A\in \mathbb C^{n\times n}$ 都可以唯一地分解为 $A = \text{Re}(A) + \text{Im}(A)$  
  其中 $\text{Re}(A) = \frac12(A+\bar A)$ 称为 $A$ 的**实部**，而 $\text{Im}(A) = \frac12(A-\bar A)$ 称为 $A$ 的**虚部**.

- 任意复方阵 $A\in \mathbb C^{n\times n}$ 都可以唯一地分解为 $A = S(A) + C(A)$  
  其中 $S(A) = \frac12(A+A^{\mathrm T})$ 称为 $A$ 的**对称部分**，而 $C(A) = \frac12(A-A^{\mathrm T})$ 称为 $A$ 的**反对称部分**.

- 任意复方阵 $A\in \mathbb C^{n\times n}$ 都可以唯一地分解为 $A = H(A) + K(A)$  
  其中 $H(A) = \frac12(A+A^{\mathrm H})$ 称为 $A$ 的 **Hermite 部分**，而 $K(A) = \frac12(A-A^{\mathrm H})$ 称为 $A$ 的**反 Hermite 部分**.

  通常我们会从 $K(A)$ 中提取一个 $i$ 出来，就得到 $A$ 的 **Toeplitz 分解**:   
  $$
  A = H(A) + iK(A)\text{ where }\begin{cases}
  H(A) = \frac12 (A+A^{\mathrm H})\\
  K(A) = \frac1{2i} (A-A^{\mathrm H})
  \end{cases}
  $$

***

**(Matrix Analysis 定理 $4.1.4$)**  
复方阵 $A\in\mathbb C^{n\times n}$ 是 Hermite 阵，当且仅当下列条件至少有一条满足:  

- ① 对于任意 $x\in \mathbb C^{n}$，$x^{\mathrm H}Ax$ 都是实数
- ② $A$ 是正规矩阵，且特征值均为实数
- ③ 对于任意 $S\in \mathbb C^{n\times n}$，$S^{\mathrm H}AS$ 都是 Hermite 阵.

**必要性证明:**    
设复方阵 $A\in\mathbb C^{n\times n}$ 是 Hermite 阵，则我们有:

- ① 对于任意 $x\in \mathbb C^{n}$，我们有 $\overline{x^{\mathrm H}Ax} = (x^{\mathrm H}Ax)^{\mathrm H} = x^{\mathrm H}A^{\mathrm H}x = x^{\mathrm H}Ax$，表明 $x^{\mathrm H}Ax$ 是实数
- ② 根据 $A^{\mathrm H}=A$ 我们立即有 $AA^{\mathrm H}=A^{\mathrm H}A$，表明 $A$ 是正规矩阵.  
  对于 $A$ 的任意特征值 $\lambda$，我们总能找到一个单位特征向量 $x\in \mathbb C^n$ (满足 $Ax=\lambda x$ 且 $\|x\|_2 = 1$)   
  于是 $\lambda = \lambda x^{\mathrm H}x = x^{\mathrm H}\lambda x = x^{\mathrm H}Ax$，根据结论 ① 可知 $\lambda$ 是实数.
- ③ 对于任意 $S\in \mathbb C^{n\times n}$，$(S^{\mathrm H}AS)^{\mathrm H} = S^{\mathrm H}A^{\mathrm H}S = S^{\mathrm H}AS$，表明 $S^{\mathrm H}AS$ 是 Hermite 阵  
  **(合同变换保持共轭对称性)**

**充分性证明:**

- ① 若对于任意 $x\in \mathbb C^{n}$，$x^{\mathrm H}Ax$ 都是实数，  
  则对于任意 $x,y\in \mathbb C^n$，$(x+y)^{\mathrm H}A(x+y) = (x^{\mathrm H}Ax + y^{\mathrm H}Ay)  + (x^{\mathrm H}Ay + y^{\mathrm H}Ax)$ 是实数.  
  注意到 $x^{\mathrm H}Ax$ 和 $y^{\mathrm H}Ay$ 是实数  
  于是我们知道对于任意 $x,y\in \mathbb C^n$，$x^{\mathrm H}Ay + y^{\mathrm H}Ax$ 都是实数.

  任取 $k,j\in \{1,\dots,n\}$  

  - 令 $\begin{cases}
    x=e_k\\
    y=e_j\end{cases}$ 根据 $x^{\mathrm H}Ay + y^{\mathrm H}Ax = a_{kj} + a_{jk}$ 是实数可知 $\text{Im}(a_{kj}) = -\text{Im}(a_{jk})$ 
  - 令 $\begin{cases}
    x=ie_k\\
    y=e_j\end{cases}$ 根据 $x^{\mathrm H}Ay + y^{\mathrm H}Ax = ia_{kj} + a_{jk}$ 是实数可知 $\text{Re}(a_{kj}) = \text{Re}(a_{jk})$ 

  因此 $\bar {a_{kj}} = a_{jk}\ (\forall\ k,j\in \{1,\dots,n\})$，表明 $A^{\mathrm H}=A$ (即 $A$ 为 Hermite 阵)

- ② 若 $A$ 是正规矩阵，根据 **Matrix Analysis 定理 $2.5.3$** 可知 $A$ 可酉对角化  
  即存在一个酉矩阵 $U\in \mathbb C^{n\times n}$，使得 $U^{\mathrm H}AU=\Lambda = \text{diag}(\lambda_1,\dots,\lambda_n)$   
  其中 $\lambda_1,\dots,\lambda_n\in \mathbb C$ 是 $A$ 的特征值.  
  一般来说 $A^{\mathrm H} = (U\Lambda U^{\mathrm H})^{\mathrm H} = U\Lambda^{\mathrm H} U^{\mathrm H} = U\bar \Lambda U^{\mathrm H}$ 

  若额外假设 $A$ 的特征值 $\lambda_1,\dots,\lambda_n$ 均为实数，  
  则我们知道 $\bar\Lambda =\Lambda$，进而有 $A^{\mathrm H} = U\bar \Lambda U^{\mathrm H} = U\Lambda U^{\mathrm H} = A$，表明 $A$ 是 Hermite 阵

- ③ 若对于任意 $S\in \mathbb C^{n\times n}$，$S^{\mathrm H}AS$ 都是 Hermite 阵  
  令 $S=I_n$，即可知 $I_n^{\mathrm H}AI_n = A$ 为 Hermite 阵.

***

Hermite 阵是正规矩阵的特例，因此有关正规矩阵的结果均适用于 Hermite 阵，例如:

- Hermite 阵可酉对角化.
- Hermite 阵的不同特征值对应的特征向量是正交的.
- $\mathbb C^n$ 存在一组由给定 Hermite 阵的特征向量组成的标准正交基.

设 $A,B\in \mathbb C^{n\times n}$ 为 Hermite 阵，  
则 $A,B$ 的实数线性组合一定是 Hermite 阵，但 $A,B$ 的复数线性组合不一定 Hermite 阵.  
(虚数单位 $i$ 作用在 Hermite 阵上会将其变为反 Hermite 阵)

另外， 由于 $(AB)^{\mathrm H} = B^{\mathrm H}A^{\mathrm H} =BA$，  
故 $AB$ 是 Hermite 阵当且仅当 $A,B$ 乘法可交换 (commutative)，即 $BA = AB$.  
关于可交换的 Hermite 阵的一个著名结果如下:  
**(Matrix Analysis 定理 $4.1.6$)**  
设 $\mathcal F$ 是一个给定的非空的 Hermite 矩阵族.  
当且仅当 $\mathcal F$ 中的 Hermite 阵两两可交换时，它们可以同时酉对角化.  
即当且仅当 $AB=BA\ (\forall\ A,B\in \mathcal F)$ 时，  
存在一个酉矩阵 $U\in \mathbb C^{n\times n}$ 使得对于任意 $A\in \mathcal F$，$UAU^{\mathrm H}$ 都是对角阵.



## 4.2 变分性质

考虑到 Hermite 阵 $A \in \mathbb C^{n\times n}$的特征值均为实数，  
我们不妨约定 $A$ 的特征值总是按非减的次序排列: $\lambda_{\min} = \lambda_1(A) \leq \dots \leq \lambda_n(A) = \lambda_{\max}$ 

由于 Hermite 阵自然是正规矩阵，故其特征向量具有良好的性质.

>对于正规矩阵 $A\in \mathbb C^{n\times n}$ 的任意特征对 $(\lambda,x)$ 我们都有 $Ax=x\lambda$ 和 $x^{\mathrm H}A=\lambda x^{\mathrm H}$ 成立.    
>现考虑 $A$ 的不同特征值的特征对 $(\lambda,x),(\mu,y)$ (其中 $\lambda\neq \mu$)  
>利用共轭算子的性质我们有:   
>$$
>\begin{align}
>(\lambda-\mu)\langle x,y\rangle
>&=
>\langle \lambda x, y\rangle - \langle x, \bar \mu y\rangle\\
>&=
>\langle Ax,y\rangle - \langle x, A^{\mathrm H}y\rangle\\
>&=
>0
>\end{align}
>$$
>根据 $\lambda\neq \mu$ 可知 $\langle x,y\rangle=0$，这表明 $A$ 的不同特征值对应的特征向量相互正交.  
>而 $A$ 关于任意特征值 $\lambda$ 的特征向量构成的任意非空集合所生成的子空间都包含一组特征向量构成的标准正交基.  
>(这个性质对于任意方阵 $A\in \mathbb C^{n\times n}$ 都成立，因为同一特征值的特征向量是可以线性组合的)  
>因此 $\mathbb C^n$ 存在一组由 $A$ 的特征向量构成的标准正交基.



### 4.2.1 Rayleigh-Ritz 定理

**(Rayleigh-Ritz 定理, Matrix Analysis 定理 $4.2.2$)**  
设 $A\in \mathbb C^{n\times n}$ 是 Hermite 阵，特征值按非减的次序排列: $\lambda_{\min} = \lambda_1(A) \leq \dots \leq \lambda_n(A) = \lambda_{\max}$    
给定整数 $1\leq i_1<\dotsm < i_m \leq n$，设 $x_{i_1},\dots,x_{i_m}$ 是标准正交的，且 $Ax_{i_k} = x_{i_k}\lambda_{i_k}\ (k=1,\dots,m)$  
记 $S:=\text{span}\{x_{i_1},\dots,x_{i_m}\}$，则我们有:  
$$
\begin{align}
\lambda_{i_1} 
&=
\min_{0_n\neq x\in S} \frac{x^{\mathrm H}Ax}{x^{\mathrm H}x}\\
&=
\min_{\{x|x\in S\text{ such that }\|x\|_2=1\}} x^{\mathrm H}Ax\\
&\leq
\max_{\{x|x\in S\text{ such that }\|x\|_2=1\}} x^{\mathrm H}Ax\\
&=
\max_{0_n\neq x\in S} \frac{x^{\mathrm H}Ax}{x^{\mathrm H}x}\\
&=
\lambda_{i_m}
\end{align}
$$

- 对于任意满足 $\|x\|_2=1$ 的 $x\in S$ 都有 $\lambda_{i_1}\leq x^{\mathrm H}Ax\leq \lambda_{i_m}$ 成立.  
  左侧和右侧的不等号的取等条件分别为 $Ax=x\lambda_{i_1}$ 和 $Ax=x\lambda_{i_m}$  

- 特殊地，对于任意满足 $\|x\|_2=1$ 的 $x\in \mathbb C^n$ 都有 $\lambda_\min \leq x^{\mathrm H}Ax\leq \lambda_{\max}$ 成立.  
  左侧和右侧的不等号的取等条件分别为 $Ax=x\lambda_\min$ 和 $Ax=x\lambda_{\max}$   
  (它可以根据 $A$ 的谱分解直接证明，原理与一般情况的证明是一致的)  
  换言之，我们有:
  $$
  \begin{align}
  \lambda_1 &= \min_{x\in\mathbb{C}^n\backslash \{0_n\}} \frac{x^{\mathrm{H}}Ax}{x^{\mathrm{H}}x} = \min_{\|x\|_2=1} x^{\mathrm{H}}Ax\\
  
  \lambda_n &= \max_{x\in\mathbb{C}^n\backslash \{0_n\}} \frac{x^{\mathrm{H}}Ax}{x^{\mathrm{H}}x} = \max_{\|x\|_2=1} x^{\mathrm{H}}Ax\\
  
  \end{align}
  $$
  **几何解释:**  
  连续实值函数 $f(x)=x^{\mathrm H}Ax$ 在单位球面 $\{x\in \mathbb C^n:\|x\|_2=1\}$ (这是个紧集) 上的最大值为 $\lambda_\max$，最小值为 $\lambda_{\min}$

**证明:**  
由于 $S$ 中的非零向量总可以标准化，故我们不妨考虑单位向量 $x\in S$ (满足 $\|x\|_2=1$)  
注意到 $S:=\text{span}\{x_{i_1},\dots,x_{i_m}\}$，故存在 $\alpha_1,\dots,\alpha_m\in \mathbb C$ 使得 $x = \sum_{k=1}^m \alpha_k x_{i_k}$   
而 $x_{i_1},\dots,x_{i_m}$ 的标准正交性确保了:  
$$
\begin{align}
x^{\mathrm H}x
&=
\left(\sum_{k=1}^m \alpha_k x_{i_k}\right)^{\mathrm H} \left(\sum_{k=1}^m \alpha_k x_{i_k}\right)\\
&=
\sum_{p,q=1}^m \bar \alpha_p \alpha_q x_{i_p}^{\mathrm H}x_{x_{i_q}}\\
&=
\sum_{k=1}^m |\alpha_k|^2\quad (\text{note that }x^{\mathrm H}x=\|x\|_2^2 = 1)\\
&=
1\\
\hline
x^{\mathrm H}Ax
&=
\left(\sum_{k=1}^m \alpha_k x_{i_k}\right)^{\mathrm H} A\left(\sum_{k=1}^m \alpha_k x_{i_k}\right)\\
&=
\left(\sum_{k=1}^m \alpha_k x_{i_k}\right)^{\mathrm H} \left(\sum_{k=1}^m \alpha_k A x_{i_k}\right)\\
&=
\left(\sum_{k=1}^m \alpha_k x_{i_k}\right)^{\mathrm H} \left(\sum_{k=1}^m \alpha_k x_{i_k} \lambda_{i_k}\right)\\
&=
\sum_{p,q=1}^m \bar \alpha_p \alpha_q x_{i_p}^{\mathrm H}x_{x_{i_q}} \lambda_{i_q}\\
&=
\sum_{k=1}^m |\alpha_k|^2 \lambda_{i_k}
\end{align}
$$
因此 $x^{\mathrm H}Ax = \sum_{k=1}^m |\alpha_k|^2 \lambda_{i_k}$ (其中 $\sum_{k=1}^m |\alpha_k|^2=1$) 是 $\lambda_{i_1},\dots,\lambda_{i_m}$ 的凸组合.  
于是我们有:
$$
\begin{align}
\lambda_{i_1}
&=
\lambda_{i_1} \sum_{k=1}^m |\alpha_k|^2\\
&\leq
\sum_{k=1}^m |\alpha_k|^2 \lambda_{i_k}\\
&=
x^{\mathrm T}Ax\\
\hline
\lambda_{i_m}
&=
\lambda_{i_m} \sum_{k=1}^m |\alpha_k|^2\\
&\geq
\sum_{k=1}^m |\alpha_k|^2 \lambda_{i_k}\\
&=
x^{\mathrm T}Ax\\
\end{align}\quad(\forall\ x\in S\text{ such that }\|x\|_2=1)
$$
前者当且仅当 $x$ 为 $A$ 关于 $\lambda_{i_1}$ 的特征向量时取等，后者当且仅当 $x$ 为 $A$ 关于 $\lambda_{i_m}$ 的特征向量时取等.  
因此我们有:
$$
\lambda_{i_1} = \min_{\{x|x\in S\text{ such that }\|x\|_2=1\}} x^{\mathrm H}Ax\\
\lambda_{i_m} = \max_{\{x|x\in S\text{ such that }\|x\|_2=1\}} x^{\mathrm H}Ax
$$
命题得证.

*****

考虑如下问题:  
$$
\begin{align}
\max \quad &  2x^2 +xy + y^2 + xz\\
\text{s.t. }\quad &x^2+y^2+z^2 =1\\
&\qquad\Leftrightarrow\\
\max\quad & 
\begin{bmatrix}
x\\
y\\
z
\end{bmatrix}^{\mathrm T}
\begin{bmatrix}
2 & \frac12 & \frac12\\
\frac12 & 1 & 0\\
\frac12 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
x\\
y\\
z
\end{bmatrix}\\

\text{s.t.}\quad &\left\| \begin{bmatrix}
x\\
y\\
z
\end{bmatrix}\right\|_2 = 1
\end{align}
$$

根据 Rayleigh-Ritz 定理可知上述问题的最优值就是系数矩阵 (一个 Hermite 阵) 的最大特征值，  
而最优解就是最大特征值对应的一个单位特征向量.



### 4.2.2 Courant-Fischer 定理

**(子空间的交, Matrix Analysis 引理 $4.2.3$)**  

- ① 设 $S_1,S_2$ 是有限维向量空间 $V$ 的子空间.  
  我们有 $\dim(S_1\cap S_2)+\dim(S_1+S_2) = \dim(S_1)+\dim(S_2)$   
  因此我们有:   
  $$
  \begin{align}
  \dim(S_1\cap S_2) &= \dim(S_1)+\dim(S_2)-\dim(S_1+S_2)\\
  &\geq \dim(S_1)+\dim(S_2)-\dim(V)
  \end{align}
  $$

- ② 归纳法指出:  
  设 $S_1,\dots,S_k$ 是有限维向量空间 $V$ 的子空间.  
  我们有 $\dim(S_1\cap \dotsm \cap S_k)\geq \dim(S_1)+\dotsm + \dim(S_2) - (k-1)\dim(V)$ 成立.

****

**(Courant–Fischer min-max 定理, Matrix Analysis 定理 $4.2.6$)**  
给定 Hermite 阵 $A \in \mathbb C^{n\times n}$，特征值按非减的次序排列: $\lambda_{\min} = \lambda_1 \leq \dots \leq \lambda_n = \lambda_{\max}$    
记 $S$ 为 $\mathbb C^n$ 的子空间，则我们有:
$$
\begin{align}
\lambda_i
&= \min_{S\subseteq  \mathbb C^n:\dim(S)=i}\left\{ \max_{x\neq 0_n\in S} \frac{x^{\mathrm H}Ax}{x^{\mathrm H}x}\right\}\\
&= \max_{S\subseteq  \mathbb C^n:\dim(S)=n-i+1}\left\{ \min_{x\neq 0_n\in S} \frac{x^{\mathrm H}Ax}{x^{\mathrm H}x}\right\}\\
\end{align}\quad (i=1,\dots,n)
$$
**证明:**  
$A \in \mathbb C^{n\times n}$ 是 Hermite 阵 (自然是正规矩阵)，一定可以酉对角化. 
即存在酉矩阵 $U\in \mathbb C^{n\times n}$ 和对角阵 $\Lambda = \text{diag}\{\lambda_1,\dots,\lambda_n\}$ 使得 $A=U\Lambda U^{\mathrm H}$  
其中 $U$ 的列向量组 $\{u_1,u_2,\dots,u_n\}$ 构成 $\mathbb C^n$ 的一组标准正交基.  

对于任意给定的 $i=1,\dots,n$，定义 $U_{(i)}= \text{span}\{u_i,u_{i+1},\dots,u_n\}$   
则 $U_{(i)}$ 是 $\mathbb C^n$ 的子空间，维数 $\dim(U_{(i)}) = n-i+1$   
对于 $\mathbb C^n$ 任意给定的 $i$ 维子空间 $S$，我们都有:  
$$
\begin{align}
\dim(S\cap U_{(i)})
&=
\dim(S) + \dim(U_{(i)}) -\dim (S+U_{(i)})\\
&\geq
\dim(S) + \dim(U_{(i)}) - \dim(\mathbb C^n)\\
&\geq
i + (n-i+1) - n\\
&= 1
\end{align}
$$
这表明 $S\cap U_{(i)}$ 一定有非零向量，即 $(S\cap U_{(i)})\backslash\{0_n\} \neq \emptyset$ 

任意给定 $\mathbb C^n$ 的 $i$ 维子空间 $S$  
不失一般性，可取单位向量 $x\in (S\cap U_{(i)})$，我们都有:
$$
\begin{align}
x^{\mathrm H}Ax 
&= x^{\mathrm H} (U\Lambda U^{\mathrm H})x\\
&= (U^{\mathrm H}x)^{\mathrm H} \Lambda (U^{\mathrm H}x)\quad (\text{Denote }y:=U^{\mathrm H}x,\text{ note that }\|y\|_2 = \|U^{\mathrm H}x\|_2=1)\\
&=
y^{\mathrm H}\Lambda y\quad (\text{Let }y =  \sum_{k=i}^n u_k \alpha_k,\text{ where }\sum_{k=i}^n |\alpha_k|^2=1)\\
&=
\sum_{k=i}^n |\alpha_k|^2 \lambda_k\quad (\text{note that }\lambda_i\leq \lambda_{i+1}\leq \dotsm \leq \lambda_n)\\
&\geq
\lambda_i \sum_{k=i}^n |\alpha_k|^2\quad (\text{note that }\sum_{k=i}^n |\alpha_k|^2=1)\\
&=
\lambda_i
\end{align}
$$
上式的不等号至少在 $S=\text{span}\{u_1,u_2,\dots,u_i\}$ 且 $x$ 与 $u_i$ 线性相关时取等.  
因此我们有:  
$$
\begin{align}
\lambda_i 
&= \min_{S\subseteq  \mathbb C^n:\dim(S)=i}\left\{ \max_{\{x|x\in S\text{ such that }\|x\|_2=1\}} x^{\mathrm H}Ax\right\}\\
&= \min_{S\subseteq  \mathbb C^n:\dim(S)=i}\left\{ \max_{x\neq 0_n\in S} \frac{x^{\mathrm H}Ax}{x^{\mathrm H}x}\right\}\\
\end{align}\quad (i=1,\dots,n)
$$
对 $-A$ 应用上述结论即得:   
(注意对 $-A$ 来说，特征值非减次序为 $-\lambda_n\leq \dots\leq -\lambda_1$，因此 $-\lambda_{i}$ 是 $-A$ 的第 $n-i+1$ 小的特征值)
$$
\begin{align}
-\lambda_i 
&= \min_{S\subseteq  \mathbb C^n:\dim(S)=n-i+1}\left\{ \max_{\{x|x\in S\text{ such that }\|x\|_2=1\}} -x^{\mathrm H}Ax\right\}\\
&= \min_{S\subseteq  \mathbb C^n:\dim(S)=n-i+1}\left\{ \max_{x\neq 0_n\in S} -\frac{x^{\mathrm H}Ax}{x^{\mathrm H}x}\right\}\\
\end{align}\quad (i=1,\dots,n)
$$
于是有:  
$$
\begin{align}
\lambda_i 
&= \min_{S\subseteq  \mathbb C^n:\dim(S)=n-i+1}\left\{ \max_{\{x|x\in S\text{ such that }\|x\|_2=1\}} x^{\mathrm H}Ax\right\}\\
&= \min_{S\subseteq  \mathbb C^n:\dim(S)=n-i+1}\left\{ \max_{x\neq 0_n\in S} \frac{x^{\mathrm H}Ax}{x^{\mathrm H}x}\right\}\\
\end{align}\quad (i=1,\dots,n)
$$
命题得证.

****

**(Matrix Analysis 推论 $4.2.10$)**   
给定 Hermite 阵 $A \in \mathbb C^{n\times n}$，特征值按非减的次序排列: $\lambda_{\min} = \lambda_1 \leq \dots \leq \lambda_n = \lambda_{\max}$   
设 $S$ 是 $\mathbb C^n$ 的一个给定的 $k$ 维子空间，给定 $c\in \mathbb R$

- ① 若对于任意单位向量 $x\in S$ 都有 $x^{\mathrm H}Ax \leq (<)\ c$，则 $\lambda_k\leq (<)\ c$  
- ② 若对于任意单位向量 $x\in S$ 都有 $x^{\mathrm H}Ax \geq (>)\ c$，则 $\lambda_{n-k+1}\geq (>)\ c$  

****

**(Matrix Analysis 推论 $4.2.12$)**   
给定 Hermite 阵 $A \in \mathbb C^{n\times n}$，特征值按非减的次序排列: $\lambda_{\min} = \lambda_1 \leq \dots \leq \lambda_n = \lambda_{\max}$   
设 $S$ 是 $\mathbb C^n$ 的一个给定的 $k$ 维子空间

- ① 若对于任意单位向量 $x\in S$ 都有 $x^{\mathrm H}Ax \leq 0$，则 $A$ 至少有 $k$ 个非正的特征值.  
  若对于任意单位向量 $x\in S$ 都有 $x^{\mathrm H}Ax < 0$，则 $A$ 至少有 $k$ 个负的特征值.
- ② 若对于任意单位向量 $x\in S$ 都有 $x^{\mathrm H}Ax \geq 0$，则 $A$ 至少有 $k$ 个非负的特征值.  
  若对于任意单位向量 $x\in S$ 都有 $x^{\mathrm H}Ax > 0$，则 $A$ 至少有 $k$ 个正的特征值. 



## 4.3 特征值不等式

**(子空间的交, Matrix Analysis 引理 $4.2.3$)**  

- ① 设 $S_1,S_2$ 是有限维向量空间 $V$ 的子空间.  
  我们有 $\dim(S_1\cap S_2)+\dim(S_1+S_2) = \dim(S_1)+\dim(S_2)$   
  因此我们有:   
  $$
  \begin{align}
  \dim(S_1\cap S_2) &= \dim(S_1)+\dim(S_2)-\dim(S_1+S_2)\\
  &\geq \dim(S_1)+\dim(S_2)-\dim(V)
  \end{align}
  $$

- ② 归纳法指出:  
  设 $S_1,\dots,S_k$ 是有限维向量空间 $V$ 的子空间.  
  我们有 $\dim(S_1\cap \dotsm \cap S_k)\geq \dim(S_1)+\dotsm + \dim(S_2) - (k-1)\dim(V)$ 成立.

### 4.3.1 Weyl 不等式

Weyl 不等式是很多不等式的基础.   
它描述了当受到 Hermite 加性扰动 $B$ 后，Hermite 阵 $A$ 的特征值会发生什么变化.

**(Weyl 不等式, Matrix Analysis 定理 $4.3.1$)**  
给定 Hermite 阵 $A,B\in \mathbb C^{n\times n}$  
设 $\{\lambda_i(A)\}_{i=1}^n,\{\lambda_i(B)\}_{i=1}^n,\{\lambda_i(A+B)\}_{i=1}^n$ 为 $A,B,A+B$ 的非减次序的特征值.  
任意给定 $i=1,2,\dots,n$

- ① 对于任意 $j=1,\dots,i$ 都有 $\lambda_j(A) + \lambda_{1+i-j}(B) \leq \lambda_{i}(A+B)$ 成立  
  上式对某一对 $(i,j)$ 取等，当且仅当存在非零向量 $x$ 使得 $\begin{cases}
  Ax = x\lambda_j(A)\\
  Bx = x\lambda_{1+i-j}(B)\\
  (A+B)x = x\lambda_{i}(A+B)\end{cases}$   
  若 $A,B,A+B$ 不存在公共特征向量，则上述不等式都是严格不等式.
- ② 对于任意 $j=i,\dots,n$ 都有 $\lambda_i(A+B) \leq \lambda_j(A) + \lambda_{n+i-j}(B)$ 成立    
  上式对某一对 $(i,j)$ 取等，当且仅当存在非零向量 $x$ 使得 $\begin{cases}
  Ax = x\lambda_j(A)\\
  Bx = x\lambda_{n+i-j}(B)\\
  (A+B)x = x\lambda_{i}(A+B)\end{cases}$   
  若 $A,B,A+B$ 不存在公共特征向量，则上述不等式都是严格不等式.

**证明:**    
任意给定 $i = 1,2,\dots, n$  

**首先证明** $\forall\ j = 1,\dots,i,\ \ \lambda_j(A) + \lambda_{1+i-j}(B) \leq \lambda_{i}(A+B)$:    
对于任意给定的 $j = 1,2,\dots,i$   
设 $S_1,S_2,S_3$ 分别为 $\mathbb C^n$ 的 $(n-j+1),(n-(1+i-j)+1),i$ 维子空间，于是有:   
$$
\begin{align}
\dim(S_1\cap S_2\cap S_3)
&\geq 
\dim(S_1) + \dim (S_2) + \dim (S_3) - (3-1)\dim(\mathbb C^n)\\
&=
(n-j+1) + (n - (1+i-j)+1) + i - 2n\\
&=
1
\end{align}
$$
故 $(S_1 \cap S_2 \cap S_3)/\{0_n\} \neq \emptyset$     
因此可取 $x_0 \neq 0_n \in (S_1 \cap S_2 \cap S_3)$  
则根据 **Courant-Fischer min-max 定理**可知:
$$
{\begin{cases} 
\lambda_i(A) =\underset{S\subseteq \mathbb C^n:\dim(S)=i}{\min}
\left\{\underset{x\neq 0_n\in S}{\max}  \frac{x^{\mathrm H}Ax}{x^{\mathrm H}x}\right\}\\ 
\lambda_i(A) =\underset{S\subseteq \mathbb C^n:\dim(S)=n-i+1}{\max}
\left\{\underset{x\neq 0_n\in S}{\min}  \frac{x^{\mathrm H}Ax}{x^{\mathrm H}x}\right\}\end{cases}}\\
\Rightarrow\\
\begin{align}
\lambda_j(A) + \lambda_{1+i-j}(B) 
&\leq \frac{x_0^{\mathrm H}Ax_0}{x_0^{\mathrm H}x_0} + \frac{x_0^{\mathrm H}Bx_0}{x_0^{\mathrm H}x_0}\quad (\text{note that }x_0\in S_1\text{ and }x_0\in S_2)\\
&= \frac{x_0^{\mathrm H}(A+B)x_0}{x_0^{\mathrm H}x_0}\qquad\quad (\text{note that }x_0\in S_3) \\
&\leq \lambda_{i}(A+B)
\end{align}
$$
**其次证明** $\forall\ j = i,\dots ,n,\ \ \lambda_i(A+B) \leq \lambda_j(A) + \lambda_{n+i-j}(B)$:   
对于任意给定的 $j = i,\dots ,n$   
设 $S_1,S_2,S_3$ 分别为 $\mathbb C^n$ 的 $j,(n+i-j),(n-i+1)$ 维子空间，于是有:  
$$
\begin{align}
\dim(S_1\cap S_2\cap S_3)
&\geq 
\dim(S_1) + \dim (S_2) + \dim (S_3) - (3-1)\dim(\mathbb C^n)\\
&=
j + (n+i-j) + (n-i+1)\\
&=
1
\end{align}
$$
故 $(S_1 \cap S_2 \cap S_3)/\{0_n\} \neq \emptyset$   
因此可取 $x_0 \neq 0_n \in (S_1 \cap S_2 \cap S_3)$  
则根据 **Courant-Fischer min-max 定理**可知:
$$
{\begin{cases} 
\lambda_i(A) =\underset{S\subseteq \mathbb C^n:\dim(S)=i}{\min}
\left\{\underset{x\neq 0_n\in S}{\max}  \frac{x^{\mathrm H}Ax}{x^{\mathrm H}x}\right\}\\ 
\lambda_i(A) =\underset{S\subseteq \mathbb C^n:\dim(S)=n-i+1}{\max}
\left\{\underset{x\neq 0_n\in S}{\min}  \frac{x^{\mathrm H}Ax}{x^{\mathrm H}x}\right\}\end{cases}}\\
\Rightarrow\\
\begin{align}
\lambda_i(A+B) 
&\leq \frac{x_0^{\mathrm H}(A+B)x_0}{x_0^{\mathrm H}x_0}\qquad\quad(\text{note that }x_0\in S_3)\\
&= \frac{x_0^{\mathrm H} A x_0}{x_0^{\mathrm H}x_0} + \frac{x_0^{\mathrm H} B x_0}{x_0^{\mathrm H}x_0}\quad (\text{note that }x_0\in S_1\text{ and }x_0\in S_2)\\
&\leq \lambda_j(A) + \lambda_{n+i-j}(B)
\end{align}
$$
命题得证.

****

> **(Weyl 不等式, Matrix Analysis 定理 $4.3.1$)**  
> 给定 Hermite 阵 $A,B\in \mathbb C^{n\times n}$  
> 设 $\{\lambda_i(A)\}_{i=1}^n,\{\lambda_i(B)\}_{i=1}^n,\{\lambda_i(A+B)\}_{i=1}^n$ 为 $A,B,A+B$ 的非减次序的特征值.  
> 任意给定 $i=1,2,\dots,n$
>
> - ① 对于任意 $j=1,\dots,i$ 都有 $\lambda_j(A) + \lambda_{1+i-j}(B) \leq \lambda_{i}(A+B)$ 成立  
>   上式对某一对 $(i,j)$ 取等，当且仅当存在非零向量 $x$ 使得 $\begin{cases}
>   Ax = x\lambda_j(A)\\
>   Bx = x\lambda_{1+i-j}(B)\\
>   (A+B)x = x\lambda_{i}(A+B)\end{cases}$   
>   若 $A,B,A+B$ 不存在公共特征向量，则上述不等式都是严格不等式.
> - ② 对于任意 $j=i,\dots,n$ 都有 $\lambda_i(A+B) \leq \lambda_j(A) + \lambda_{n+i-j}(B)$ 成立    
>   上式对某一对 $(i,j)$ 取等，当且仅当存在非零向量 $x$ 使得 $\begin{cases}
>   Ax = x\lambda_j(A)\\
>   Bx = x\lambda_{n+i-j}(B)\\
>   (A+B)x = x\lambda_{i}(A+B)\end{cases}$   
>   若 $A,B,A+B$ 不存在公共特征向量，则上述不等式都是严格不等式.

给定 Hermite 阵 $A,B\in \mathbb C^{n\times n}$  
设 $\{\lambda_i(A)\}_{i=1}^n,\{\lambda_i(B)\}_{i=1}^n,\{\lambda_i(A+B)\}_{i=1}^n$ 为 $A,B,A+B$ 的非减次序的特征值. 

- **(Matrix Analysis 推论 $4.3.5$)**   
  若 $\rank(B)=r<n$ (即 $B$ 恰好有 $n-r$ 个零特征值，显然有 $\lambda_{n-r}(B)\leq 0,\lambda_{r+1}(B)\geq 0$)  
  则下列命题成立:  

  - ① 对于任意 $i=1,\dots,n-r$ 都有 $\lambda_i(A+B) \leq \lambda_{i+r}(A)+\lambda_{n-r}(B) \leq \lambda_{i+r}(A)$ 成立.  
    上式对某个 $i$ 取等，当且仅当 $\lambda_{n-r}(B)=0$​ 且存在非零向量 $x$ 使得 $\begin{cases}
    Ax = x\lambda_{i+r}(A)\\
    Bx = 0_n\\
    (A+B)x = x\lambda_{i}(A+B)\end{cases}$ 
  - ② 对于任意 $i=r+1,\dots,n$​ 都有 $\lambda_{i-r}(A)\leq \lambda_{i-r}(A)+\lambda_{r+1}(B) \leq \lambda_i(A+B)$​ 成立.  
    上式对某个 $i$​ 取等，当且仅当 $\lambda_{r+1}(B)=0$​ 且存在非零向量 $x$​ 使得 $\begin{cases}
    Ax = x\lambda_{i-r}(A)\\
    Bx = 0_n\\
    (A+B)x = x\lambda_{i}(A+B)\end{cases}$

  若对于 $A$ 的每个特征向量 $x$ 都有 $Bx\neq 0_n$ 成立，则上述不等式都是严格成立的.

- **(Matrix Analysis 推论 $4.3.3$)**  
  若 $B$ 恰好有 $a$ 个负特征值和 $b$ 个正特征值   
  (显然 $\lambda_{a+1}(B)\geq 0$ 且 $\lambda_{n-b}\leq 0$，当且仅当 $n>a+b$ (即 $B$ 奇异) 时取等)  
  则下列命题成立:  

  - ① 对于任意 $i=1,\dots,n-b$ 都有 $\lambda_i(A+B) \leq \lambda_{i+b}(A)+\lambda_{n-b}(B) \leq \lambda_{i+b}(A)$​ 成立.     
    上式对某个 $i$ 取等，当且仅当 $B$ 奇异且存在非零向量 $x$ 使得 $\begin{cases}
    Ax = x\lambda_{i+b}(A)\\
    Bx = 0_n\\
    (A+B)x = x\lambda_{i}(A+B)\end{cases}$ 
  - ② 对于任意 $i=a+1,\dots,n$ 都有 $\lambda_{i-a}(A)\leq \lambda_{i-a}(A)+\lambda_{a+1}(B) \leq \lambda_i(A+B)$​ 成立.    
    上式对某个 $i$ 取等，当且仅当 $B$ 奇异且存在非零向量 $x$ 使得 $\begin{cases}
    Ax = x\lambda_{i-a}(A)\\
    Bx = 0_n\\
    (A+B)x = x\lambda_{i}(A+B)\end{cases}$

  若 $B$ 非奇异 (即 $n=a+b$) 或对于 $A$ 的每个特征向量 $x$ 都有 $Bx\neq 0_n$ 成立，则上述不等式都是严格成立的.

- **(Matrix Analysis 推论 $4.3.7$)**   
  若 $B$ 恰好有 $1$ 个负特征值和 $1$ 个正特征值  
  (显然 $\lambda_{2}(B)\geq 0$ 且 $\lambda_{n-1}\leq 0$，当且仅当 $n>1+1=2$ (即 $B$ 奇异) 时取等)   则下列命题成立:  

  - ① 对于任意 $i=1,\dots,n-1$ 都有 $\lambda_i(A+B) \leq \lambda_{i+1}(A)+\lambda_{n-1}(B) \leq \lambda_{i+1}(A)$​ 成立.     
    上式对某个 $i$ 取等，当且仅当 $B$ 奇异且存在非零向量 $x$ 使得 $\begin{cases}
    Ax = x\lambda_{i+1}(A)\\
    Bx = 0_n\\
    (A+B)x = x\lambda_{i}(A+B)\end{cases}$ 
  - ② 对于任意 $i=2,\dots,n$ 都有 $\lambda_{i-1}(A)\leq \lambda_{i-1}(A)+\lambda_{2}(B) \leq \lambda_i(A+B)$​ 成立.    
    上式对某个 $i$ 取等，当且仅当 $B$ 奇异且存在非零向量 $x$ 使得 $\begin{cases}
    Ax = x\lambda_{i-1}(A)\\
    Bx = 0_n\\
    (A+B)x = x\lambda_{i}(A+B)\end{cases}$

  若 $B$ 非奇异 (即 $n=1+1=2$) 或对于 $A$ 的每个特征向量 $x$ 都有 $Bx\neq 0_n$ 成立，则上述不等式都是严格成立的.  
  上述结论可以写成更紧凑的形式:  
  $$
  \begin{align}
  & \lambda_1 (A+B)\leq \lambda_2(A)\\
  \lambda_{i-1}(A)\leq & \lambda_i(A+B) \leq \lambda_{i+1}(A)\quad (i=2,\dots,n-1)\\
  \lambda_{n-1}(A) \leq &\lambda_n (A+B)
  \end{align}
  $$

- 注意到当 $B=zz^{\mathrm H}$ (其中 $z\neq 0_n\in \mathbb C^n$) 时，$B$ 恰好有一个正特征值 $z^{\mathrm H}z$     
  因此当 $n\geq 2$ 时我们一定有 $\lambda_{1}(B)=0=\lambda_{n-1}(B)$ 成立.  
  对应地，当 $B=-zz^{\mathrm H}$ (其中 $z\neq 0_n\in \mathbb C^n$) 时，$B$ 恰好有一个负特征值 $-z^{\mathrm H}z$   
  因此当 $n\geq 2$ 时我们一定有 $\lambda_{2}(B)=0=\lambda_{n}(B)$ 成立.  

  **(秩 $1$ Hermite 摄动的交错定理, Matrix Analysis 推论 $4.3.9$)**  
  若 $n\geq 2$ 且 $B=zz^{\mathrm H}$ (其中 $z\neq 0_n\in \mathbb C^n$)，则下列命题成立:

  - ① 对于任意 $i=1,\dots,n-1$ 都有 $\lambda_i(A+B) \leq \lambda_{i+1}(A)+\lambda_{n-1}(B) = \lambda_{i+1}(A)$​ 成立.     
    上式对某个 $i$ 取等，当且仅当 $B$ 奇异且存在非零向量 $x$ 使得 $\begin{cases}
    Ax = x\lambda_{i+1}(A)\\
    Bx = 0_n\\
    (A+B)x = x\lambda_{i}(A+B)\end{cases}$ 
  - ② 对于任意 $i=1,\dots,n$ 都有 $\lambda_{i}(A) = \lambda_{i}(A)+\lambda_{1}(B) \leq \lambda_i(A+B)$​ 成立.    
    上式对某个 $i$ 取等，当且仅当 $B$ 奇异且存在非零向量 $x$ 使得 $\begin{cases}
    Ax = x\lambda_{i}(A)\\
    Bx = 0_n\\
    (A+B)x = x\lambda_{i}(A+B)\end{cases}$

  上述结论可以写成更紧凑的形式:   
  $$
  \begin{align}
  \lambda_{i}(A)\leq & \lambda_i(A+B) \leq \lambda_{i+1}(A)\quad (i=1,\dots,n-1)\\
  \lambda_{n}(A) \leq &\lambda_n (A+B)
  \end{align}
  $$

  *****

  对应地，若 $n\geq 2$ 且 $B=-zz^{\mathrm H}$ (其中 $z\neq 0_n\in \mathbb C^n$)，则下列命题成立:

  - ① 对于任意 $i=1,\dots,n$ 都有 $\lambda_i(A+B) \leq \lambda_{i}(A)+\lambda_{n}(B) = \lambda_{i}(A)$​ 成立.     
    上式对某个 $i$ 取等，当且仅当 $B$ 奇异且存在非零向量 $x$ 使得 $\begin{cases}
    Ax = x\lambda_{i}(A)\\
    Bx = 0_n\\
    (A+B)x = x\lambda_{i}(A+B)\end{cases}$ 
  - ② 对于任意 $i=2,\dots,n$ 都有 $\lambda_{i-1}(A)= \lambda_{i-1}(A)+\lambda_{2}(B) \leq \lambda_i(A+B)$ 成立.    
    上式对某个 $i$ 取等，当且仅当 $B$ 奇异且存在非零向量 $x$ 使得 $\begin{cases}
    Ax = x\lambda_{i-1}(A)\\
    Bx = 0_n\\
    (A+B)x = x\lambda_{i}(A+B)\end{cases}$

  上述结论可以写成更紧凑的形式:    
  $$
  \begin{align}
  & \lambda_1 (A+B)\leq \lambda_1(A)\\
  \lambda_{i-1}(A)\leq & \lambda_i(A+B) \leq \lambda_{i}(A)\quad (i=2,\dots,n)\\
  \end{align}
  $$

  ***

  换言之，若 $A\in \mathbb C^{n\times n}$ 是 Hermite 阵 (其中 $n\geq 2$)，则对于任意非零向量 $z\in \mathbb C^n$ 都有:  
  $$
  \begin{align}
  \lambda_{i}(A)\leq & \lambda_i(A+zz^{\mathrm H}) \leq \lambda_{i+1}(A)\quad (i=1,\dots,n-1)\\
  \lambda_{n}(A) \leq &\lambda_n (A+zz^{\mathrm H})\\
  \hline
  & \lambda_1 (A-zz^{\mathrm H})\leq \lambda_1(A)\\
  \lambda_{i-1}(A)\leq & \lambda_i(A-zz^{\mathrm H}) \leq \lambda_{i}(A)\quad (i=2,\dots,n)\\
  \end{align}
  $$

- **(Weyl 不等式的最常用形式, Matrix Analysis 推论 $4.3.15$)**   
  根据 Weyl 不等式可知下列命题成立:

  - ① 对于任意 $i=1,\dots,n$ 都有 $\lambda_i(A+B) \leq \lambda_{i}(A)+\lambda_{n}(B)$​ 成立.     
    上式对某个 $i$ 取等，当且仅当存在非零向量 $x$ 使得 $\begin{cases}
    Ax = x\lambda_{i}(A)\\
    Bx = x\lambda_n(B)\\
    (A+B)x = x\lambda_{i}(A+B)\end{cases}$ 
  - ② 对于任意 $i=1,\dots,n$ 都有 $\lambda_{i}(A)+\lambda_{1}(B) \leq \lambda_i(A+B)$ 成立.    
    上式对某个 $i$ 取等，当且仅当 $B$ 奇异且存在非零向量 $x$ 使得 $\begin{cases}
    Ax = x\lambda_{i}(A)\\
    Bx = x\lambda_1(B)\\
    (A+B)x = x\lambda_{i}(A+B)\end{cases}$

  上述结论可以写成更紧凑的形式:   
  (若 $A,B$ 没有公共特征向量，则下列不等式严格成立)  
  $$
  \lambda_1(B)\leq \lambda_i(A+B) - \lambda_i(A) \leq \lambda_n (B)\quad (i=1,\dots,n)
  $$
  注意到对于 Hermite 阵 $B\in \mathbb C^{n\times n}$ 来说有 $\|B\|_2 = \rho(B)=\max\{|\lambda_1(B)|,|\lambda_n(B)|\}$ 成立  
  故我们还可以得到:
  $$
  \begin{align}
  \max_{1\leq i\leq n}\{\lambda_i(A+B)-\lambda_i(A)\} 
  &\leq \lambda_n(B)\\
  \min_{1\leq i\leq n}\{\lambda_i(A+B)-\lambda_i(A)\}
  &\geq \lambda_1(B)\\
  \hline
  -\|B\|_2 \leq \lambda_1(B) \leq \lambda_i(A+B) - \lambda_i(A) &\leq \lambda_n(B) \leq \|B\|_2 \quad (i=1,\dots,n)
  \\
  \max_{1\leq i\leq n}|\lambda_i(A+B)-\lambda_i(A)| 
  &\leq \|B\|_2 = \max\{|\lambda_1(B)|,|\lambda_n(B)|\}
  \end{align}
  $$
  
- 注意到当 $B$ 是 Hermite 半正定阵时我们有 $\lambda_1(B)\geq 0$ 成立，当且仅当 $B$ 奇异时取等.  
  此外，$B$ 是 Hermite 正定阵当且仅当 $B$ 非奇异.  
  **(单调定理, Matrix Analysis 推论 $4.3.12$)**  
  若 $B$ 是 Hermite 半正定阵，则对于任意 $i=1,\dots,n$ 都有 $\lambda_i(A)\leq \lambda_i(A) + \lambda_1(B)\leq \lambda_i(A+B)$ 成立.   
  上式对某个 $i$ 取等，当且仅当 $B$ 奇异且存在非零向量 $x$ 使得 $\begin{cases}
  Ax = x\lambda_{i}(A)\\
  Bx = 0_n\\
  (A+B)x = x\lambda_{i}(A+B)\end{cases}$ 

  若 $B$ 非奇异 (这意味着 $B$ 是 Hermite 正定阵)，则上述不等式严格成立.  
  即对于任意 $i=1,\dots,n$ 都有 $\lambda_i(A) < \lambda_i(A+B)$ 成立.  
  此外，根据 **Matrix Analysis 推论 $4.3.15$** 我们还有:  
  $$
  \begin{align}
  \max_{1\leq i\leq n}|\lambda_i(A+B)-\lambda_i(A)| 
  &= \|\text{eig}(A+B) - \text{eig}(A)\|_\infty \leq \lambda_n(B) = \|B\|_2\\
  \min_{1\leq i\leq n}|\lambda_i(A+B)-\lambda_i(A)|
  &\geq \lambda_1(B) = \frac{1}{\|B^{-1}\|_2}
  \end{align}
  $$



### 4.3.2 Hoffman-Wielandt 不等式

回忆起一般的特征值摄动定理:     
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
  (它要比 Weyl 不等式的最常用形式更强)
  
- **(Matrix Analysis 定义 $5.4.18$)**    
  设 $V=\mathbb F^n$ (其中 $\mathbb F=\mathbb R\text{ or }\mathbb C$)   
  记 $|x|$ 为 $x\in V$ 逐个元素取模得到的向量.  
  我们说 $|x|\preceq|y|$，当且仅当 $|x_i|\leq |y_i|\ (i=1,\dots,n)$   
  我们称 $V$ 上的范数 $\|\cdot\|$ 是:

  - ① **单调的** (monotone)，如果对于任意满足 $|x|\preceq|y|$ 的 $x,y\in V$ 都有 $\|x\|\leq \|y\|$ 成立.
  - ② **绝对的** (absolute)，如果 $\||x|\| = \|x\|\ (\forall\ x\in V)$

  可以证明在有限维赋范空间上单调范数和绝对范数的概念是等价的.

***

在数值应用中，原矩阵 $A$ 与摄动矩阵 $E$ 通常是实对称的 (自然是正规的)
可以证明: 当 $A$ 和 $A+E$ 均为正规矩阵时，其所有特征值的扰动都有 Frobenius 范数上界.  
**(Hoffman-Wielandt 不等式, Matrix Analysis 定理 $6.3.5$)**  
设 $A,E\in \mathbb C^{n\times n}$ 均为正规矩阵.  
设 $\lambda_1,\dots,\lambda_n\in \mathbb C$ 是 $A$ 的以某种次序排列的特征值，  
而 $\hat\lambda_1,\dots,\hat\lambda_n\in \mathbb C$ 是 $A+E$ 的以某种次序排列的特征值.   
则存在 $\{1,\dots,n\}$ 的某个排列 $\pi$ 使得:  
$$
\sum_{i=1}^n |\hat\lambda_{\pi(i)} - \lambda_i|^2 \leq \|E\|_{\mathrm F}^2 = \tr(E^{\mathrm H}E)
$$

- **(Matrix Analysis 推论 $6.3.8$)**  
  设 $A\in \mathbb C^{n\times n}$ 为 Hermite 阵，而 $A+E$ 是正规矩阵.  
  设 $\lambda_1,\dots,\lambda_n\in \mathbb R$ 是 $A$ 的以非减次序排列的特征值，  
  而 $\hat\lambda_1,\dots,\hat\lambda_n\in \mathbb C$ 是 $A+E$ 的以实部非减次序排列的特征值 (即满足 $\text{Re}(\hat \lambda_1)\leq \dots \leq \text{Re}(\hat \lambda_n)$)  
  记 $\Delta \Lambda := \text{diag}\{\hat\lambda_1-\lambda_1,\dots,\hat \lambda_n - \lambda_n\}$    
  则我们有:  
  $$
  \sum_{i=1}^n |\hat \lambda_{i}-\lambda_i|^2 = \|\Delta \Lambda\|_{\mathrm F}^2  \leq \|E\|_{\mathrm F}^2 = \tr(E^{\mathrm H}E)
  $$

我们目前能够证明的版本要比 **Matrix Analysis 推论 $6.3.8$** 还要特殊:  
**(Hoffman-Wielandt 不等式的最常用形式)**  
设 $A,E\in \mathbb C^{n\times n}$ 均为 Hermite 阵.   
设 $\lambda_1,\dots,\lambda_n\in \mathbb R$ 是 $A$ 的以非减次序排列的特征值，  
而 $\hat\lambda_1,\dots,\hat\lambda_n\in \mathbb R$ 是 $A+E$ 的以非减次序排列的特征值.    
记 $\Delta \Lambda := \text{diag}\{\hat\lambda_1-\lambda_1,\dots,\hat \lambda_n - \lambda_n\}$    
则我们有:  
$$
\sum_{i=1}^n |\hat \lambda_{i}-\lambda_i|^2 = \|\Delta \Lambda\|_{\mathrm F}^2  \leq \|E\|_{\mathrm F}^2 = \tr(E^{\mathrm H}E)
$$

- **Lemma: (Rellich 定理) ([On the Rellich eigendecomposition](https://arxiv.org/pdf/2211.15539))** 
  给定 $\mathbb R$ 中的开区间 $(a,b)$   
  解析形式的 Hermite 阵 $A(t)\in \mathbb C^{n\times n}$ 具有解析形式的谱分解:  
  $$
  \begin{cases}
  A(t) = Q(t)\Lambda(t)Q(t)^{\mathrm H}\\
  Q(t)^{\mathrm H} Q(t) = I_n
  \end{cases}\quad (t\in (a,b))
  $$
  其中 $A(t),Q(t),\Lambda(t)$ 的元素都是关于 $t\in (a,b)$ 的解析函数，  
  且对于任意 $t\in (a,b)$，$A(t),Q(t),\Lambda(t)$ 分别为 Hermite 阵、酉矩阵和对角元均为实数的对角阵.
  
- **证明:**   
  考虑 $A(t)= A+tE\ (0< t < 1)$  
  根据 Kato-Reillich 定理可知 $A(t)\in \mathbb C^{n\times n}$ 具有解析形式的谱分解:
  $$
  \begin{cases}
  A(t) = Q(t)\Lambda(t)Q(t)^{\mathrm H}\\
  Q(t)^{\mathrm H} Q(t) = I_n
  \end{cases}\quad (0< t< 1)
  $$
  对 $Q(t)^{\mathrm H}Q(t)=I_n$ 求导得到:  
  $$
  Q'(t)^{\mathrm H}Q(t) + Q(t)^{\mathrm H}Q'(t) = 0_{n\times n}
  $$
  这说明 $K(t):= Q'(t)^{\mathrm H}Q(t)$ 为反 Hermite 阵，其对角元为纯虚数.    
  记其对角部分为 $\text{diag}(K(t))=iD(t)$ (其中 $D(t)$ 的对角元均为实数)

  对 $\Lambda(t) = Q(t)^{\mathrm H} A(t)Q(t)$ 求导得到:  
  $$
  \begin{align}
  \Lambda'(t)
  &=
  Q'(t)^{\mathrm H} A(t) Q(t)
  +
  Q(t)^{\mathrm H} A'(t) Q(t)
  +
  Q(t)^{\mathrm H} A(t) Q'(t)\quad (\text{note that }\begin{cases}
  A(t) = Q(t)\Lambda(t)Q(t)^{\mathrm H}\\
  A'(t) = E\end{cases})\\
  &=
  Q'(t)Q(t)^{\mathrm H}\Lambda (t) + Q(t)^{\mathrm H} E Q(t) + \Lambda(t) Q(t)^{\mathrm H} Q'(t)\\
  &=
  K(t) \Lambda(t) + Q(t)^{\mathrm H}EQ(t) + \Lambda (t) K(t)^{\mathrm H}
  
  \end{align}
  $$
  
  由于 $\Lambda(t)$ 是对角阵，故 $\Lambda'(t)$ 的非对角部分一定为零，于是我们有: 
  $$
  \begin{align}
  \Lambda'(t)
  &=\text{diag}(\Lambda'(t))\\
  &=
  \text{diag}(K(t) \Lambda(t) + Q(t)^{\mathrm H}EQ(t) + \Lambda (t) K(t)^{\mathrm H})\\
  &=
  \text{diag}(K(t)) \Lambda(t) + \text{diag}(Q(t)^{\mathrm H}EQ(t)) + \Lambda(t)\text{diag}(K(t)^{\mathrm H})\\
  &=
  (iD(t))\cdot\Lambda(t) + \text{diag}(Q(t)^{\mathrm H}EQ(t)) + \Lambda(t)\cdot (iD(t))^{\mathrm H}\\
  &=
  iD(t)\Lambda(t) + \text{diag}(Q(t)^{\mathrm H}EQ(t)) - i D(t)\Lambda(t)\\
  &=
  \text{diag}(Q(t)^{\mathrm H}EQ(t))
  \end{align}
  $$
  于是我们有:  
  $$
  \begin{align}
  \sqrt{\sum_{i=1}^n |\hat \lambda_{i}-\lambda_i|^2}
  &=
  \|\Delta \Lambda\|_{\mathrm F}\\
  &=
  \|\Lambda(1)-\Lambda(0)\|_{\mathrm F}\\
  &=
  \left\| \int_0^1 \Lambda'(t){\mathrm d}t\right\|_{\mathrm F}\\
  &\leq
  \int_0^1 \|\Lambda'(t)\|_{\mathrm F}\ {\mathrm d}t \\
  &=
  \int_0^1 \|\text{diag}(Q(t)^{\mathrm H}EQ(t))\|_{\mathrm F}\ {\mathrm d}t\\
  &\leq
  \int_0^1 \|Q(t)^{\mathrm H}EQ(t)\|_{\mathrm F}\ {\mathrm d}t\quad (\text{note that }\|\cdot\|_{\mathrm F}\text{ is unitary-invariant})\\
  &=
  \int_0^1 \|E\|_{\mathrm F} \ {\mathrm d}t\\
  &=
  \|E\|_{\mathrm F}
  \end{align}
  $$
  
  因此我们有:  
  $$
  \sum_{i=1}^n |\hat \lambda_{i}-\lambda_i|^2 = \|\Delta \Lambda\|_{\mathrm F}^2  \leq \|E\|_{\mathrm F}^2 = \tr(E^{\mathrm H}E)
  $$
  命题得证.

****

**回忆起 Weyl 不等式的最常用形式:**   
若 $A,E\in \mathbb C^{n\times n}$ 为 Hermite 阵，且特征值按非减次序排列，则我们有:
$$
-\|E\|_2 = \lambda_1(E)\leq \lambda_i(A+E) - \lambda_i(A) \leq \lambda_n(E)\leq \|E\|_2\quad (i=1,\dots,n)\\

\Leftrightarrow\\

\max_{1\leq i\leq n}|\lambda_i(A+E)-\lambda_i(A)| = \|\Delta \Lambda\|_2 \leq \|E\|_2\\
\text{where }\Delta \Lambda = \text{diag}\{\lambda_1(A+E)-\lambda_1(A),\dots, \lambda_n(A+E)-\lambda_n(A)\}
$$
**对比 Hoffman-Wielandt 不等式的最常用形式:**   
若 $A,E\in \mathbb C^{n\times n}$ 为 Hermite 阵，且特征值按非减次序排列，则我们有:
$$
\sum_{i=1}^n |\lambda_{i}(A+E)-\lambda_i(A)|^2 = \|\Delta \Lambda\|_{\mathrm F}^2  \leq \|E\|_{\mathrm F}^2 = \tr(E^{\mathrm H}E)\\

\text{where }\Delta \Lambda = \text{diag}\{\lambda_1(A+E)-\lambda_1(A),\dots, \lambda_n(A+E)-\lambda_n(A)\}
$$
**总结起来就是:**    
若 $A,E\in \mathbb C^{n\times n}$ 为 Hermite 阵，且特征值按非减次序排列，则我们有:
$$
\|\Delta \Lambda\|_{\mathrm F} \leq \|E\|_{\mathrm F}\\
\|\Delta \Lambda\|_2 \leq \|E\|_2\\
\text{where }\Delta \Lambda = \text{diag}\{\lambda_1(A+E)-\lambda_1(A),\dots, \lambda_n(A+E)-\lambda_n(A)\}
$$
事实上，对于一般的酉不变范数 $\|\cdot\|$ 都有 $\|\Delta \Lambda\| \leq \|E\|$   
(这是 Matrix Analysis 定理 $7.4.9.3$ 的直接推论，我们后面会提到) 


****

**(邵老师的补充)**  
设正规矩阵 $A\in \mathbb C^{n\times n}$ 非奇异，数值计算得到的近似谱分解为:  
$$
\begin{cases}
A \approx X\hat \Lambda X^{\mathrm H}\\
X^{\mathrm H}X\approx I_n
\end{cases}
$$
设特征值以非减次序排列，则对于任意 $k=1,\dots,n$ 我们都有:  
$$
\begin{align}
\lambda_k(X^{\mathrm H}AX) - \lambda_k(A)
&=
\lambda_k \left( 
X^{\mathrm H}(A-\lambda_k(A) I) X + \lambda_k(A)(X^{\mathrm H}X-I) + \lambda_k(A)I
\right) - \lambda_k(A)\\
&=
\lambda_k(X^{\mathrm H}(A-\lambda_k(A)I) X + \lambda_k(A)(X^{\mathrm H}X-I))\\
&=
\lambda_k (B+E)
\end{align}
$$
其中 $\begin{cases}
B:=X^{\mathrm H}(A-\lambda_k(A)I) X\\
E:= \lambda_k(A)(X^{\mathrm H}X-I)\end{cases}$      
根据 Weyl 不等式可知:  
$$
|\lambda_k(B+E)-\lambda_k(B)| \leq \|E\|_2
$$
注意到 $A-\lambda_k(A)I$ 的第 $k$ 个特征值是 $0$，前 $k-1$ 个特征值为负值，后 $n-k$ 个特征值为正值.  
根据惯性定理 (即合同变换不改变特征值符号) 可知 $B=X^{\mathrm H}(A-\lambda_k(A)I) X$ 的特征值具有相同分布:  
即第 $k$ 个特征值是 $0$，前 $k-1$ 个特征值为负值，后 $n-k$ 个特征值为正值  
因此 $\lambda_k(B) = 0$   
于是上式变为:
$$
|\lambda_k(B+E)| \leq \|E\|_2
$$
因此我们有:  
$$
\begin{align}
|\lambda_k(X^{\mathrm H}AX) - \lambda_k(A)| 
&=
|\lambda_k (B+E)|\\
&\leq
\|E\|_2\\
&=
\|\lambda_k(A)(X^{\mathrm H}X-I)\|_2\\
&=
|\lambda_k(A)| \|X^{\mathrm H}X-I\|_2
\end{align}
$$
于是第 $k=1,\dots,n$ 个特征值的相对误差为:  
$$
\left|\frac{\lambda_k(X^{\mathrm H}AX)-\lambda_k(A)}{\lambda_k(A)}\right| \leq \|X^{\mathrm H}X-I\|_2
$$


### 4.3.3 Cauchy 交错定理

Cauchy 交错定理 (Cauchy Interlacing Theorem) 描述了 $n$ 阶 Hermite 阵与其 $n-1$ 阶主子阵的特征值关系.  
**(Cauchy 交错定理, Matrix Analysis 定理 $4.3.17$)**  
给定 Hermite 阵 $A \in \mathbb C^{n\times n}$，特征值按非减的次序排列: $\lambda_1(A) \leq \dots \leq \lambda_n(A)$   
考虑 $A$ 的 $n-1$ 主子阵 $B = A_{(1:n-1,1:n-1)}\in \mathbb C^{(n-1)\times (n-1)}$，并记 $A=\begin{bmatrix}
B & y\\
y^{\mathrm H} & a\end{bmatrix}$  
特征值按非减的次序排列: $\lambda_1(B) \leq \dots \leq \lambda_{n-1}(B)$    
则我们有如下的交错性质:
$$
\lambda_1(A) \leq \lambda_1(B) \leq \lambda_2(A) \leq \dotsm \leq \lambda_{n-1}(A) \leq \lambda_{n-1}(B)
\leq \lambda_{n}(A)\\

\Leftrightarrow\\

\lambda_i(A) \leq \lambda_i(B) \leq \lambda_{i+1}(A)\quad (\forall\ i=1,\dots,n-1)
$$

其中 $\lambda_i(A) = \lambda_i(B)$ 成立当且仅当存在非零向量 $z\in \mathbb C^{n-1}$ 使得 $\begin{cases}
Bz = z\lambda_i(B)\\
Bz = z\lambda_i(A)\\
y^{\mathrm H}z = 0\end{cases}$   
而 $\lambda_i(B)= \lambda_{i+1}(A)$ 成立当且仅当存在非零向量 $z\in \mathbb C^{n-1}$ 使得 $\begin{cases}
Bz = z\lambda_i(B)\\
Bz = z\lambda_{i+1}(A)\\
y^{\mathrm H}z = 0\end{cases}$    
若 $B$ 没有与 $y$ 正交的特征向量，则上述不等式均为严格不等式.  

- **上述定理表明:**   
  Hermite 阵无论是加边扩充还是删边约简，其新旧特征值必定是交错的.   
  当然，加边扩充和删边约简不一定要在最后一行和最后一列进行，它可以在一行和对应的列进行.

- **证明:**    
  **首先证明对于任意 $i=1,\dots,n-1$ 都有 $\lambda_i(A)\leq \lambda_i(B)$:**  
  任意给定 $\mathbb C^{n-1}$ 的一个 $i$ 维子空间 $S$，都可定义 $\tilde S = \left\{y:=\begin{bmatrix}x\\0\end{bmatrix}: x\in S\subseteq \mathbb C^{n-1}\right\}$   
  显然 $\tilde S$ 为 $\mathbb C^n$ 的 $i$ 维子空间，但 $\mathbb C^n$ 的 $i$ 维子空间全体是比 $\tilde S$ 多的.  
  根据 **Courant\-Fischer min\-max 定理 (Matrix Analysis 定理 $4.2.6$)** 可知:  
  $$
  \begin{align}
  \lambda_i(B)
  &=
  \min_{\{S\subseteq \mathbb C^{n-1}:\dim(S)=i\}} 
  \left\{
  \max_{x\neq 0_{n-1} \in S}\frac{x^{\mathrm H}Bx}{x^{\mathrm H}x}
  \right\}\\
  
  &= 
  \min_{\left\{
  \tilde S = \{y:=\begin{bmatrix}x\\0\end{bmatrix}: x\in S\subseteq \mathbb C^{n-1}\}{\Large |} \dim(S)=i
  \right\}}
  \left\{
  \max_{y\neq 0_n \in S}\frac{y^{\mathrm H}Ay}{y^{\mathrm H}y}
  \right\}\\
  
  &\geq
  \min_{\{V\subseteq \mathbb C^n : \dim(V) = i\}} 
  \left\{
  \max_{y\neq 0_n\in V} \frac{y^{\mathrm H}Ay}{y^{\mathrm H}y}
  \right\}\\
  
  &=
  \lambda_i(A)
  \end{align}
  $$
  **其次证明对于任意 $i=1,\dots,n-1$ 都有 $\lambda_i(B)\leq \lambda_{i+1}(A)$:**  
  我们对 $-A,-B$ 应用刚才的结论，则有 $-\lambda_{n-i+1}(A) \leq -\lambda_{n-1-i+1}(B)\ (\forall\ i=1,\dots,n-1)$   
  将 $n-1-i+1$ 替换为 $i$ 可知 $\lambda_i(B)\leq \lambda_{i+1}(A)\ (\forall\ i=1,\dots,n-1)$   

  综上所述，我们有:
  $$
  \lambda_i(A) \leq \lambda_i(B) \leq \lambda_{i+1}(A)\quad (\forall\ i=1,\dots,n-1)
  $$
  命题得证.

- **(包容定理, Matrix Analysis 定理 $4.3.28$)**  
  归纳法指出:   
  考虑 Hermite 阵 $A$ 的 $n-m$ 主子阵 $B = A_{(1:n-m,1:n-m)}\in \mathbb C^{(n-m)\times (n-m)}$  
  特征值按非减的次序排列: $\lambda_1(B) \leq \dots \leq \lambda_{n-m}(B)$  
  并记 $A=\begin{bmatrix}
  B & C\\
  C^{\mathrm H} & D\end{bmatrix}$ (其中 $C\in \mathbb C^{(n-m)\times m}, D\in \mathbb C^{m\times m}$)  
  则我们有如下的交错性质:
  $$
  \lambda_i(A) \leq \lambda_i(B) \leq \lambda_{i+m}(A)\quad (\forall\ i=1,\dots,n-m)
  $$
  其中 $\lambda_i(A) = \lambda_i(B)$ 成立当且仅当存在非零向量 $z\in \mathbb C^{n-m}$ 使得 $\begin{cases}
  Bz = z\lambda_i(B)\\
  Bz = z\lambda_i(A)\\
  C^{\mathrm H}z = 0_m\end{cases}$   
  而 $\lambda_i(B)= \lambda_{i+m}(A)$ 成立当且仅当存在非零向量 $z\in \mathbb C^{n-m}$ 使得 $\begin{cases}
  Bz = z\lambda_i(B)\\
  Bz = z\lambda_{i+m}(A)\\
  C^{\mathrm H}z = 0\end{cases}$    
  若 $B$ 没有与 $C$ 的列向量组正交的特征向量，则上述不等式均为严格不等式.

- **(Poincaré 分离定理, Matrix Analysis 推论 $4.3.37$)**  
  设 $A\in \mathbb C^{n\times n}$ 是 Hermite 阵，$1\leq m\leq n$  
  设 $u_1,\dots,u_n\in \mathbb C^n$ 标准正交  
  记 $U=[u_1,\dots,u_n]\in \mathbb C^{n\times n}$ 和 $V=[u_1,\dots,u_m]\in \mathbb C^{n\times m}$  
  记 $B:= V^{\mathrm H}AV = [u_i^{\mathrm H}Au_j]_{i,j=1}^{m}\in \mathbb C^{m\times m}$ (它就是 $U^{\mathrm H}AU$ 的 $m$ 阶顺序主子阵)  
  注意到 $U^{\mathrm H}AU$ 的特征值与 $A$ 的特征值完全相同.  
  设 $A,B$ 的特征值以非减次序排列，则我们有:  
  $$
  \lambda_i(A) \leq \lambda_i(B) \leq \lambda_{i+(n-m)}(A)\quad (\forall\ i=1,\dots,m)
  $$
  这也说明对于任意列标准正交的 $V\in \mathbb C^{n\times m}$ 我们都有:  
  $$
  \sum_{i=1}^m \lambda_i(A) \leq \sum_{i=1}^m \lambda_i(V^{\mathrm H}AV) \leq \sum_{i=1}^m \lambda_{n-m+i}(A)
  $$
  注意到左右不等号均可取等，于是我们顺理成章地得到了**樊氏迹极小化原理** (Ky Fan's Trace Minimization Principle)

- **(樊氏迹极小化原理, Rayleigh-Ritz 定理的推广, Matrix Analysis 推论 $4.3.39$)**  
  设 $A\in \mathbb C^{n\times n}$ 是 Hermite 阵，$1\leq m\leq n$，则我们有:  
  $$
  \begin{align}
  \min_{\begin{subarray}{}
  V\in \mathbb C^{n\times m}\\
  V^{\mathrm H}V = I_m
  \end{subarray}}
  \tr(V^{\mathrm H}AV) &= \lambda_1(A) + \dotsm + \lambda_m(A)\\
  
  \max_{\begin{subarray}{}
  V\in \mathbb C^{n\times m}\\
  V^{\mathrm H}V = I_m
  \end{subarray}}
  \tr(V^{\mathrm H}AV) &= \lambda_{n-m+1}(A) + \dotsm + \lambda_n(A)\\
  \end{align}
  $$
  对于 $m=1,\dots,n-1$，上式中的最大 (小) 值可以对这样一个矩阵 $V\in \mathbb C^{n\times m}$ 取到，  
  这个矩阵的列是与 $A$ 的前 $m$ 大 (小) 的特征值相伴的标准正交的特征向量.  
  对于 $m=n$，对任意酉矩阵 $V\in \mathbb C^{n\times n}$ 都有 $\tr(V^{\mathrm H}AV)=\tr(AVV^{\mathrm H}) = \tr(A)=\lambda_1(A)+\dotsm+\lambda_n(A)$   
  樊氏迹极小化原理在理论上是简单的，但在优化领域很实用.

**Cauchy 交错定理的 Matlab 代码验证:**

```matlab
% Generate a random Hermitian matrix of size 10x10
rng(51);
n = 10;
H = randn(n) + 1i * randn(n);  % Create a random complex matrix
H = 0.5 * (H + H');  % Make it Hermitian

% Compute eigenvalues of the original matrix
eigvals_H = sort(real(eig(H)));

% Initialize figure
figure;
hold on;
grid on;
xlabel('Index');
ylabel('Eigenvalue');
title('Eigenvalues of Hermitian Matrix and Its Submatrices');

% Plot eigenvalues of the original matrix
plot(eigvals_H, n * ones(n, 1), 'o', 'DisplayName', 'Eigenvalues of H');

% Loop over all possible submatrices and compute their eigenvalues
for k = n-1:-1:1
    % Extract the top-left kxk submatrix
    H_k = H(1:k, 1:k);
    
    % Compute eigenvalues of the submatrix
    eigvals_H_k = sort(real(eig(H_k)));
    
    % Plot eigenvalues of the submatrix on the appropriate y-line
    plot(eigvals_H_k, k * ones(k, 1), 'o', 'DisplayName', sprintf('Eigenvalues of H_{%dx%d}', k, k));
end

% Adjust plot properties
legend show;
ylim([-1, n+1]);
hold off;
```

运行结果:

<img src="interlacing of eigenvalues of submatrices.png" style="zoom:40%;" />

****

我们已经讨论了特征值交错定理的两个例子:

- **① (秩 $1$ Hermite 摄动的交错定理, Matrix Analysis 推论 $4.3.9$)**   
  若 $A\in \mathbb C^{n\times n}$ 是 Hermite 阵 (其中 $n\geq 2$)，则对于任意非零向量 $z\in \mathbb C^n$ 都有:  
  $$
  \begin{align}
  \lambda_{i}(A)\leq & \lambda_i(A+zz^{\mathrm H}) \leq \lambda_{i+1}(A)\quad (i=1,\dots,n-1)\\
  \lambda_{n}(A) \leq &\lambda_n (A+zz^{\mathrm H})\\
  \hline
  & \lambda_1 (A-zz^{\mathrm H})\leq \lambda_1(A)\\
  \lambda_{i-1}(A)\leq & \lambda_i(A-zz^{\mathrm H}) \leq \lambda_{i}(A)\quad (i=2,\dots,n)\\
  \end{align}
  $$

- **② (Cauchy 交错定理, Matrix Analysis 定理 $4.3.17$​)**    
  给定 Hermite 阵 $A \in \mathbb C^{n\times n}$，特征值按非减的次序排列: $\lambda_1(A) \leq \dots \leq \lambda_n(A)$   
  考虑 $A$ 的 $n-1$ 主子阵 $B = A_{(1:n-1,1:n-1)}\in \mathbb C^{(n-1)\times (n-1)}$，并设 $A=\begin{bmatrix}
  B & y\\
  y^{\mathrm H} & a\end{bmatrix}$  
  特征值按非减的次序排列: $\lambda_1(B) \leq \dots \leq \lambda_{n-1}(B)$    
  则我们有如下的交错性质:
  $$
  \lambda_1(A) \leq \lambda_1(B) \leq \lambda_2(A) \leq \dotsm \leq \lambda_{n-1}(A) \leq \lambda_{n-1}(B)
  \leq \lambda_{n}(A)\\
  
  \Leftrightarrow\\
  
  \lambda_i(A) \leq \lambda_i(B) \leq \lambda_{i+1}(A)\quad (\forall\ i=1,\dots,n-1)
  $$


这表明若对一个给定的 Hermite 矩阵进行秩 $1$ Hermite 摄动、删边或加边，则新旧特征值必定是交错的.  
事实上，上述两个特征值交错定理相互蕴含.  

**那反过来呢?**  
任意给定两组交错的实数，  
它们是否一定是一个 Hermite 阵加边扩充前后的新旧特征值，  
或者一定是一个 Hermite 阵加上一个秩 $1$ Hermite 阵前后的新旧特征值?  
下面两个定理给出了肯定的回答:

- **(Matrix Analysis 定理 $4.3.21$)**   
  任意给定两组交错的实数:  
  $$
  \lambda_1 \leq \mu_1 \leq \lambda_2 \leq \dots \leq \lambda_{n-1} \leq \mu_{n-1} \leq \lambda_{n}
  $$
  记 $\Mu := \text{diag}\{\mu_1,\dots,\mu_{n-1}\}$  
  则存在实数 $a\in \mathbb R$ 和实向量 $y\in \mathbb R^{n-1}$ 使得 $\begin{bmatrix}
  \Mu & y\\
  y^{\mathrm T} & a\end{bmatrix}\in \mathbb R^{n\times n}$ 的特征值是 $\lambda_1,\dots,\lambda_n$ 

- **(Matrix Analysis 定理 $4.3.26$​)**    
  任意给定两组交错的实数:  
  $$
  \lambda_1 \leq \mu_1 \leq \lambda_2 \leq \dots \leq \lambda_{n-1} \leq \mu_{n-1} \leq \lambda_{n} \leq \mu_n
  $$
  记 $\Lambda = \text{diag}\{\lambda_1,\dots,\lambda_n\}$  
  则存在实向量 $z\in \mathbb R^{n}$ 使得 $\Lambda+zz^{\mathrm T}$ 的特征值是 $\mu_1,\dots,\mu_n$   
  
  **(Lowner 定理, Applied Numerical Linear Algebra, J. W. Demmel, 定理 $5.10$)**   
  特殊地，如果 $\lambda_1 < \mu_1 < \lambda_2 < \dots < \lambda_{n-1} < \mu_{n-1} < \lambda_{n} < \mu_n$，则 $z\in \mathbb R^n$ 的构造如下:
  $$
  z_i = 
  \left(\frac{\prod_{1\leq j\leq n} (\mu_j-\lambda_{n-i+1})}{\prod_{1\leq j\leq n,j\neq n-i+1}(\lambda_j-\lambda_{n-i+1})}\right)^{\frac12}\ (i=1,\dots,n)
  $$
  (证明参见 FDU 数值算法 Homework 09 Problem 06)



## 4.4 推广到奇异值

### 4.4.1 与 Hermite 特征值的联系

**(Matrix Analysis 定理 $7.3.3$)**   
给定复矩阵 $A\in \mathbb C^{m\times n}$，记 $q := \min\{m,n\}$  
设 $\sigma_1\geq \sigma_2 \geq \dots \geq \sigma_q$ 为 $A$ 的奇异值  
定义 Hermite 阵 $\tilde A = \begin{bmatrix} 0_{m\times m}& A\\ A^{\mathrm H} & 0_{n\times n} \end{bmatrix} \in \mathbb C^{(m+n)\times (m+n)}$  
则 $\tilde A$ 的特征值为 $-\sigma_1 \leq \dots\leq -\sigma_q \leq \underbrace{0=\dots = 0}_{|m-n|} \leq \sigma_q \leq \dots \leq \sigma_1$   

**证明:**  
首先假设 $m\geq n$，设 $A$ 的奇异值分解为:  
$$
\begin{align}
A 
&= U\Sigma V^{\mathrm H} \\
&= [U_1,U_2] 
\begin{bmatrix}
\Sigma_n\\
0_{(m-n)\times n}
\end{bmatrix} 
V^{\mathrm H}\\
&=
U_1\Sigma_n V^{\mathrm H}
\end{align}
$$
其中 $U\in \mathbb C^{m\times m}$ 和 $V\in \mathbb C^{n\times n}$ 为酉矩阵，$U_1\in \mathbb C^{m\times n}$ 由 $U$ 的前 $n$ 列构成，$\Sigma_n = \text{diag}\{\sigma_1,\dots,\sigma_n\}$   
于是我们有:
$$
\begin{align}
A &= U_1\Sigma_1V^{\mathrm H}\\
0_{m\times n} & = U_2 0_{(m-n)\times n}V^{\mathrm H}\\
A^{\mathrm H} &= V\Sigma_1 U_1^{\mathrm H}\\
0_{n\times m} &= V0_{n\times (m-n)}U_2^{\mathrm H}
\end{align}
$$
定义 $m+n$ 阶矩阵:  
$$
Q := 
\begin{bmatrix}
\frac{\sqrt{2}}2U_1 & -\frac{\sqrt{2}}{2}U_1 & U_2\\
\frac{\sqrt{2}}{2}V & \frac{\sqrt{2}}{2}V & 0_{m\times (n-m)}
\end{bmatrix}\in \mathbb C^{n\times n}
$$
容易验证 $Q$ 是酉矩阵:  
$$
\begin{align}
Q^{\mathrm H}Q
&=
\begin{bmatrix}
\frac{\sqrt{2}}2U_1 & -\frac{\sqrt{2}}{2}U_1 & U_2\\
\frac{\sqrt{2}}{2}V & \frac{\sqrt{2}}{2}V & 0_{m\times (n-m)}
\end{bmatrix}^{\mathrm H}
\begin{bmatrix}
\frac{\sqrt{2}}2U_1 & -\frac{\sqrt{2}}{2}U_1 & U_2\\
\frac{\sqrt{2}}{2}V & \frac{\sqrt{2}}{2}V & 0_{m\times (n-m)}
\end{bmatrix}\\
&=
\begin{bmatrix}
\frac{\sqrt2}{2}U_1^{\mathrm H} & \frac{\sqrt{2}}{2}V^{\mathrm H}\\
-\frac{\sqrt2}{2}U_1^{\mathrm H} & \frac{\sqrt{2}}{2}V^{\mathrm H}\\
U_2^{\mathrm H} & 0_{m\times (n-m)}
\end{bmatrix} 
\begin{bmatrix}
\frac{\sqrt{2}}2U_1 & -\frac{\sqrt{2}}{2}U_1 & U_2\\
\frac{\sqrt{2}}{2}V & \frac{\sqrt{2}}{2}V & 0_{m\times (n-m)}
\end{bmatrix}\\
&=
\begin{bmatrix}
\frac12 U_1^{\mathrm H}U_1 + \frac12 V^{\mathrm H}V & -\frac12 U_1^{\mathrm H}U_1 + \frac12 V^{\mathrm H}V & \frac{\sqrt{2}}{2}U_1^{\mathrm H}U_2\\
-\frac12 U_1^{\mathrm H}U_1 + \frac12 V^{\mathrm H}V & \frac12 U_1^{\mathrm H}U_1 + \frac12 V^{\mathrm H}V & -\frac{\sqrt{2}}{2}U_1^{\mathrm H}U_2\\
\frac{\sqrt2}{2}U_2^{\mathrm H}U_1 & -\frac{\sqrt2}{2}U_2^{\mathrm H}U_1 & U_2^{\mathrm H}U
\end{bmatrix}\\
&=
\begin{bmatrix}
I_n & 0_{n\times n} & 0_{n\times (m-n)}\\
0_{n\times n} & I_n & 0_{n\times (m-n)}\\
0_{(m-n)\times n} & 0_{(m-n)\times n} & I_{m-n}
\end{bmatrix}\\
&=
I_{m+n}
\end{align}
$$
同时我们有:  
$$
\begin{align}
Q^{\mathrm H}\tilde A Q
&=
\begin{bmatrix}
\frac{\sqrt{2}}2U_1 & -\frac{\sqrt{2}}{2}U_1 & U_2\\
\frac{\sqrt{2}}{2}V & \frac{\sqrt{2}}{2}V & 0_{m\times (n-m)}
\end{bmatrix}^{\mathrm H}
\begin{bmatrix} 0_{m\times m}& A\\ A^{\mathrm H} & 0_{n\times n} \end{bmatrix}
\begin{bmatrix}
\frac{\sqrt{2}}2U_1 & -\frac{\sqrt{2}}{2}U_1 & U_2\\
\frac{\sqrt{2}}{2}V & \frac{\sqrt{2}}{2}V & 0_{m\times (n-m)}
\end{bmatrix}\\
&=
\begin{bmatrix}
\frac{\sqrt2}{2}U_1^{\mathrm H} & \frac{\sqrt{2}}{2}V^{\mathrm H}\\
-\frac{\sqrt2}{2}U_1^{\mathrm H} & \frac{\sqrt{2}}{2}V^{\mathrm H}\\
U_2^{\mathrm H} & 0_{m\times (n-m)}
\end{bmatrix}
\begin{bmatrix}
\frac{\sqrt2}{2}AV & \frac{\sqrt{2}}{2}AV & 0_{m\times (n-m)}\\
\frac{\sqrt2}{2}A^{\mathrm H}U_1 & -\frac{\sqrt{2}}{2}A^{\mathrm H}U_1 & A^{\mathrm H}U_2
\end{bmatrix}\\

&=
\begin{bmatrix}
\frac{1}{2}U_1^{\mathrm H}AV + \frac12 V^{\mathrm H}A^{\mathrm H}U_1 
& \frac{1}{2}U_1^{\mathrm H}AV - \frac12 V^{\mathrm H}A^{\mathrm H}U_1
& \frac{\sqrt2}{2}V^{\mathrm H}A^{\mathrm H}U_2\\
-\frac{1}{2}U_1^{\mathrm H}AV + \frac12 V^{\mathrm H}A^{\mathrm H}U_1 
& -\frac{1}{2}U_1^{\mathrm H}AV - \frac12 V^{\mathrm H}A^{\mathrm H}U_1 
& \frac{\sqrt2}{2}V^{\mathrm H}A^{\mathrm H}U_2\\
\frac{\sqrt 2}{2}U_2^{\mathrm H}A V &  \frac{\sqrt 2}{2}U_2^{\mathrm H}AV & 0_{(m-n)\times (m-n)}
\end{bmatrix}\\

&=
\begin{bmatrix}
\Sigma_n & 0_{n\times n} & 0_{n\times (m-n)}\\
0_{n\times n} & -\Sigma_n & 0_{n\times (m-n)}\\
0_{(m-n)\times n} & 0_{(m-n)\times n} & 0_{(m-n)\times (m-n)}
\end{bmatrix}
\end{align}
$$
因此 $\tilde A$ 的特征值为 $-\sigma_1 \leq \dots\leq -\sigma_n \leq \underbrace{0=\dots = 0}_{m-n} \leq \sigma_n \leq \dots \leq \sigma_1$   

当 $m<n$ 时我们可对 $A^{\mathrm H}$ 应用上述结论便得到:  
$\begin{bmatrix} 0_{n\times n}& A^{\mathrm H}\\ A & 0_{m\times m} \end{bmatrix} \in \mathbb C^{(m+n)\times (m+n)}$ 的特征值为 $-\sigma_1 \leq \dots\leq -\sigma_m \leq \underbrace{0=\dots = 0}_{n-m} \leq \sigma_m \leq \dots \leq \sigma_1$  
注意到: 
$$
\begin{align}
&\begin{bmatrix}
& I_n\\
I_m & 
\end{bmatrix}^{-1}
\begin{bmatrix} 0_{n\times n}& A^{\mathrm H}\\ A & 0_{m\times m} \end{bmatrix}
\begin{bmatrix}
& I_n\\
I_m & 
\end{bmatrix}\\
&=
\begin{bmatrix}
& I_m\\
I_n & 
\end{bmatrix}
\begin{bmatrix} A^{\mathrm H}& 0_{n\times n}\\ 0_{m\times m} & A\end{bmatrix}\\
&=
\begin{bmatrix} 0_{m\times m}& A\\ A^{\mathrm H} & 0_{n\times n}\end{bmatrix}\\
&=\tilde A
\end{align}
$$
因此 $\tilde A$ 的特征值为 $-\sigma_1 \leq \dots\leq -\sigma_m \leq \underbrace{0=\dots = 0}_{n-m} \leq \sigma_m \leq \dots \leq \sigma_1$ 

综上所述，$\tilde A$ 的特征值为 $-\sigma_1 \leq \dots\leq -\sigma_q \leq \underbrace{0=\dots = 0}_{|m-n|} \leq \sigma_q \leq \dots \leq \sigma_1$ (其中 $q=\min\{m,n\}$)

****

**另一种观点: 重排**      
不失一般性，假设 $m\geq n$，设 $A$ 的奇异值分解为:  
$$
\begin{align}
A 
&= U\Sigma V^{\mathrm H} \\
&= [U_1,U_2] 
\begin{bmatrix}
\Sigma_n\\
0_{(m-n)\times n}
\end{bmatrix} 
V^{\mathrm H}\\
&=
U_1\Sigma_n V^{\mathrm H}
\end{align}
$$
其中 $U\in \mathbb C^{m\times m}$ 和 $V\in \mathbb C^{n\times n}$ 为酉矩阵，$U_1\in \mathbb C^{m\times n}$ 由 $U$ 的前 $n$ 列构成，$\Sigma_n = \text{diag}\{\sigma_1,\dots,\sigma_n\}$   
则我们有:  
$$
\begin{align}
\tilde A 
&= 
\begin{bmatrix} 0_{m\times m}& A\\ A^{\mathrm H} & 0_{n\times n} \end{bmatrix}\\
&= 
\begin{bmatrix} 0_{m\times m}& U\Sigma V^{\mathrm H}\\ V\Sigma^{\mathrm H} U^{\mathrm H} & 0_{n\times n} \end{bmatrix}\\
&=
\begin{bmatrix}
U&\\
& V
\end{bmatrix}
\begin{bmatrix} 0_{m\times m}& \Sigma \\ \Sigma^{\mathrm H} & 0_{n\times n} \end{bmatrix}
\begin{bmatrix}
U^{\mathrm H}&\\
& V^{\mathrm H}
\end{bmatrix}\\
\end{align}
$$
于是 $\tilde A$ 酉相似于:
$$
\begin{align}
\begin{bmatrix} 0_{m\times m}& \Sigma \\ \Sigma^{\mathrm H} & 0_{n\times n} \end{bmatrix} 
&=
\begin{bmatrix} 0_{n\times n}& 0_{n\times (m-n)}&\Sigma_n \\ 
0_{(m-n)\times n} &0_{(m-n)\times (m-n)} & 0_{(m-n)\times n}\\
\Sigma_n^{\mathrm H} & 0_{n\times (m-n)} & 0_{n\times n} \end{bmatrix}
\end{align}
$$
它可以进行对称的行列重排，得到:  
(值得注意的是，对称的行列重排也是酉相似变换)  
$$
\left[\begin{array}{cc|cc|cc|cc|cccc}
0 & \sigma_1 \\
\sigma_1 & 0\\
\hline
& & 0 & \sigma_1 \\
& & \sigma_1 & 0\\
\hline
& & & &  &\ddots \\
& & & & \ddots &\\
\hline
& & & & & & 0 & \sigma_n\\
& & & & & & \sigma_n & 0\\
\hline
& & & & & & & & 0 & 0 & \dotsm & 0\\
& & & & & & & & 0 & 0 & \dotsm & 0\\
& & & & & & & & \vdots & \vdots & & \vdots\\
& & & & & & & & 0 & 0 & \dotsm & 0
\end{array}\right]
$$
其中右下角的 $0$ 分块的阶数为 $(m+n)-2n = m-n$  
因此 $\tilde A$ 的特征值为 $\pm \sigma_1,\pm \sigma_2,\dotsm ,\pm \sigma_n$ 以及 $m-n$ 个 $0$ 



### 4.4.2 Rayleigh-Ritz 定理

设 $A\in \mathbb C^{m\times n}$ 的奇异值按非增次序排列: $\sigma_1(A) \geq \dotsm \geq \sigma_q(A)$ (其中 $q=\min\{m,n\}$)  
设 $A^{\mathrm H}A \in \mathbb C^{n\times n}$ 的特征值按非减次序排列: $\lambda_1(A^{\mathrm H}A) \leq \dotsm \leq \lambda_n(A^{\mathrm H}A)$    
设 $AA^{\mathrm H} \in \mathbb C^{m\times m}$ 的特征值按非减次序排列: $\lambda_1(AA^{\mathrm H}) \leq \dotsm \leq \lambda_m(AA^{\mathrm H})$   

值得注意的是，无论矩阵维数 $m,n$ 的大小关系是怎样的，  
最大特征值 $\lambda_n(A^{\mathrm H}A)$ 和 $\lambda_m(AA^{\mathrm H})$ 都对应 $A$ 的最大奇异值 $\sigma_1(A)$:  
$$
\max_{x\neq 0_n\in \mathbb C^n}\frac{x^{\mathrm H}(A^{\mathrm H}A)x}{x^{\mathrm H}x} = \lambda_n(A^{\mathrm H}A) = (\sigma_1(A))^2 = \lambda_m(AA^{\mathrm H}) = \max_{x\neq 0_m\in \mathbb C^m}\frac{x^{\mathrm H}(AA^{\mathrm H})x}{x^{\mathrm H}x}\\

\Leftrightarrow\\

\|A\|_2=\max_{\|x\|_2=1} \|Ax\|_2 = \sigma_1(A) = \max_{\|x\|_2=1}\|A^{\mathrm H}x\|_2 = \|A^{\mathrm H}\|_2
$$
而最小特征值和最小奇异值就不同了，它们的对应关系与矩阵维数 $m,n$ 有关.  

- ① 当 $m=n$ 时 $(q=\min\{m,n\}=n)$，我们有:  
  $$
  \min_{x\neq 0_n\in \mathbb C^n}\frac{x^{\mathrm H}(A^{\mathrm H}A)x}{x^{\mathrm H}x} = \lambda_1(A^{\mathrm H}A) = (\sigma_n(A))^2 = \lambda_1(AA^{\mathrm H}) = \min_{x\neq 0_n\in \mathbb C^n}\frac{x^{\mathrm H}(AA^{\mathrm H})x}{x^{\mathrm H}x}\\
  
  \Leftrightarrow\\
  
  \min_{\|x\|_2=1} \|Ax\|_2 = \sigma_n(A) = \min_{\|x\|_2=1}\|A^{\mathrm H}x\|_2
  $$

- ② 当 $m>n$ 时 $(q=\min\{m,n\}=n)$，我们有:  
  $$
  \min_{x\neq 0_m\in \mathbb C^m}\frac{x^{\mathrm H}(AA^{\mathrm H})x}{x^{\mathrm H}x} = \lambda_1(AA^{\mathrm H}) = 0\\
  \min_{x\neq 0_n\in \mathbb C^n}\frac{x^{\mathrm H}(A^{\mathrm H}A)x}{x^{\mathrm H}x} = \lambda_1(A^{\mathrm H}A) = (\sigma_n(A))^2\\
  \Leftrightarrow\\
  
  \min_{\|x\|_2=1} \|A^{\mathrm H}x\|_2 = 0\\
  \min_{\|x\|_2=1}\|Ax\|_2 = \sigma_n(A)
  $$

- ③ 当 $m<n$ 时 $(q=\min\{m,n\}=m)$，我们有:  
  $$
  \min_{x\neq 0_n\in \mathbb C^n}\frac{x^{\mathrm H}(A^{\mathrm H}A)x}{x^{\mathrm H}x} = \lambda_1(A^{\mathrm H}A) = 0\\
  \min_{x\neq 0_m\in \mathbb C^m}\frac{x^{\mathrm H}(AA^{\mathrm H})x}{x^{\mathrm H}x} = \lambda_1(AA^{\mathrm H}) = (\sigma_m(A))^2\\
  \Leftrightarrow\\
  
  \min_{\|x\|_2=1} \|Ax\|_2 = 0\\
  \min_{\|x\|_2=1}\|A^{\mathrm H}x\|_2 = \sigma_m(A)
  $$

****

设 $\mathbb C^{m\times n}$ 的矩阵范数 $\|\cdot\|$ 可由 $\mathbb C^n$ 上的范数 $\|\cdot\|_\alpha$ 和 $\mathbb C^m$ 上的**绝对范数** $\|\cdot\|_\beta$ 诱导:  
$$
\|A\|:= \sup_{\|x\|_\alpha=1} \|Ax\|_\beta \quad (\forall\ A\in \mathbb C^{m\times n})
$$
任意给定矩阵 $A$，其任意子矩阵 $B$ 的诱导范数一定小于等于 $A$ 的诱导范数.  
(邵老师: 这可能只是一个充分条件，因为从证明来看只需保证与补零相关的性质即可)  

- **(Matrix Analysis 定义 $5.4.18$)**    
  设 $V=\mathbb F^n$ (其中 $\mathbb F=\mathbb R\text{ or }\mathbb C$)   
  记 $|x|$ 为 $x\in V$ 逐个元素取模得到的向量.  
  我们说 $|x|\preceq|y|$，当且仅当 $|x_i|\leq |y_i|\ (i=1,\dots,n)$   
  我们称 $V$ 上的范数 $\|\cdot\|$ 是:

  - ① **单调的** (monotone)，如果对于任意满足 $|x|\preceq|y|$ 的 $x,y\in V$ 都有 $\|x\|\leq \|y\|$ 成立.
  - ② **绝对的** (absolute)，如果 $\||x|\| = \|x\|\ (\forall\ x\in V)$

  可以证明在有限维赋范空间上单调范数和绝对范数的概念是等价的.

**证明:**

- ① 首先考虑 $A$ 删去最后一列得到的子矩阵 $B$，我们有:  
  $$
  \begin{align}
  \|B\| 
  &=
  \sup_{\|x\|_\alpha=1} \|Bx\|_\beta\\
  &=
  \sup_{\|x\|_\alpha = 1} \left\| 
  \begin{bmatrix}
  B & 0_m
  \end{bmatrix}
  \begin{bmatrix}
  x\\0
  \end{bmatrix}\right\|_\beta\\
  &=
  \sup_{\|x\|_\alpha=1} \left\|A \begin{bmatrix}
  x\\0
  \end{bmatrix}\right\|_\beta
  
  \quad (\text{denote }y:= \begin{bmatrix}
  x\\0
  \end{bmatrix})\\
  &\leq
  
  \sup_{\|y\|_\alpha=1} \|Ay\|_\beta\\
  &=
  \|A\|
  \end{align}
  $$
  归纳法表明，对于 $A$ 删去任意列得到的子矩阵 $B$，我们都有 $\|B\|\leq \|A\|$

- ② 其次考虑 $A$ 删去最后一行得到的子矩阵 $B$，我们有:  
  $$
  \begin{align}
  \|B\| 
  &=
  \sup_{\|x\|_\alpha=1} \|Bx\|_\beta\\
  &=
  \sup_{\|x\|_\alpha = 1} \left\| 
  \begin{bmatrix}
  B\\
  0_n^{\mathrm T}
  \end{bmatrix}
  x\right\|_\beta\\
  &=
  \sup_{\|x\|_\alpha = 1} \left\| 
  \begin{bmatrix}
  Bx\\
  0
  \end{bmatrix}\right\|_\beta\quad (\text{note that }\|\cdot\|_\beta \text{ is an absolute norm, so that }\left\| 
  \begin{bmatrix}
  Bx\\
  0
  \end{bmatrix}\right\|_\beta\leq \|Ax\|_\beta)\\
  &\leq
  \sup_{\|x\|_\alpha=1} \left\|Ax\right\|_\beta\\
  &=
  \|A\|
  \end{align}
  $$
  归纳法表明，对于 $A$ 删去任意行得到的子矩阵 $B$，我们都有 $\|B\|\leq \|A\|$

综合①②，我们可知对于 $A$ 的任意子矩阵 $B$，我们都有 $\|B\|\leq \|A\|$ 成立.

****

**(最大奇异值的双线性形式)**  
设 $A\in \mathbb C^{m\times n}\ (m,n\geq 2)$ 的奇异值按非增次序排列: $\sigma_1(A) \geq \dotsm \geq \sigma_q(A)$ (其中 $q=\min\{m,n\}$)  
则我们有:  
$$
\begin{align}
\max_{\begin{subarray}{}
x\in \mathbb C^n \backslash \{0_n\}\\
y\in \mathbb C^m \backslash \{0_m\}
\end{subarray}} \frac{|y^{\mathrm H}Ax|}{\|x\|_2\|y\|_2} 
&= \sigma_\max = \sigma_1(A)\\

\min_{\begin{subarray}{}
x\in \mathbb C^n \backslash \{0_n\}\\
y\in \mathbb C^m \backslash \{0_m\}
\end{subarray}} \frac{|y^{\mathrm H}Ax|}{\|x\|_2\|y\|_2} 
&= 0
\end{align}
$$
**证明:**   
设 $A\in \mathbb C^{m\times n}$ 的奇异值分解为 $A =
U\Sigma V^{\mathrm H}$  
其中 $U\in \mathbb C^{m\times m}$ 和 $V\in \mathbb C^{n\times n}$ 为酉矩阵，而 $\Sigma\in \mathbb C^{m\times n}$ 的对角元均为非负实数.

任意给定 $x\in \mathbb C^n \backslash \{0_n\}$ 和 $y\in \mathbb C^m \backslash \{0_m\}$   
根据 $\text{span}\{U\} = \mathbb C^m$ 和 $\text{span}\{V\}=\mathbb C^n$ 可知:  
存在 $\alpha\in \mathbb C^n\backslash \{0_n\}$ 和 $\beta \in \mathbb C^m\backslash \{0_m\}$ 使得 $\begin{cases}
x=V\alpha\\
y=U\beta\end{cases}$  
根据 $l_2$ 范数的酉不变性可知:  
$$
\begin{align}
\frac{|y^{\mathrm H}Ax|}{\|x\|_2\|y\|_2}
&=
\frac{|(U\beta)^{\mathrm H}A(V\alpha)|}{\|U\beta\|_2 \|V\alpha\|_2}\\
&=
\frac{|\beta^{\mathrm H} U^{\mathrm H}AV\alpha|}{\|\alpha\|_2\|\beta\|_2}\\
&=
\frac{|\beta^{\mathrm H}\Sigma \alpha|}{\|\alpha\|_2\|\beta\|_2}
\end{align}
$$

- ① 首先假设 $m = n$，则可设 $\Sigma = \text{diag}\{\sigma_1,\dots,\sigma_n\}$  
  其中 $\sigma_\max = \sigma_1 \geq \dotsm \geq \sigma_n = \sigma_\min \geq 0$   
  于是我们有:  
  $$
  \begin{align}
  \frac{|y^{\mathrm H}Ax|}{\|x\|_2\|y\|_2} 
  &= 
  \frac{|\beta^{\mathrm H}\Sigma \alpha|}{\|\alpha\|_2\|\beta\|_2}\quad (\text{use Cauchy-Schwarz inequality})\\
  &\leq
  \frac{\|\Sigma^{\frac12}\alpha\|_2 \|\Sigma^{\frac12}\beta\|_2}{\|\alpha\|_2\|\beta\|_2}\\
  &\leq
  \frac{\sqrt{\sigma_\max} \|\alpha\|_2 \cdot\sqrt{\sigma_\max}\|\beta\|_2 }{\|\alpha\|_2\|\beta\|_2}\\
  &=
  \sigma_\max
  \end{align}
  $$
  当且仅当 $x=v_1,y=u_1$ 或 $x=-v_1,y=-u_1$ 时取等.   
  因此我们有:
  $$
  \max_{\begin{subarray}{}
  x\in \mathbb C^n \backslash \{0_n\}\\
  y\in \mathbb C^n \backslash \{0_n\}
  \end{subarray}} \frac{|y^{\mathrm H}Ax|}{\|x\|_2\|y\|_2} = 
  \max_{\begin{subarray}{}
  \alpha\in \mathbb C^n \backslash \{0_n\}\\
  \beta\in \mathbb C^n \backslash \{0_n\}
  \end{subarray}} \frac{|\beta^{\mathrm H}\Sigma \alpha|}{\|\alpha\|_2\|\beta\|_2} = \sigma_\max
  $$
  另一方面我们有:  
  $$
  \begin{align}
  \frac{|y^{\mathrm H}Ax|}{\|x\|_2\|y\|_2}
  &=
  \frac{|\beta^{\mathrm H}\Sigma \alpha|}{\|\alpha\|_2\|\beta\|_2}\geq 0
  \end{align}
  $$
  当且仅当 $\beta\ \bot\ \Sigma \alpha$ (即 $y\ \bot\ Ax$) 时取等.  
  因此我们有:  
  $$
  \min_{\begin{subarray}{}
  x\in \mathbb C^n \backslash \{0_n\}\\
  y\in \mathbb C^n \backslash \{0_n\}
  \end{subarray}} \frac{|y^{\mathrm H}Ax|}{\|x\|_2\|y\|_2} 
  =
  \min_{\begin{subarray}{}
  \alpha\in \mathbb C^n \backslash \{0_n\}\\
  \beta\in \mathbb C^n \backslash \{0_n\}
  \end{subarray}} \frac{|\beta^{\mathrm H}\Sigma \alpha|}{\|\alpha\|_2\|\beta\|_2}
  = 0
  $$

- ② 其次假设 $m>n$，则我们可记:
  $$
  \begin{align}
  A 
  &=
  U\Sigma V^{\mathrm H}\\
  &=
  [U_1,U_2]
  \begin{bmatrix}
  \Sigma_1\\
  0_{(m-n)\times n}
  \end{bmatrix}
  V^{\mathrm H}\\
  &=
  U_1\Sigma_1 V^{\mathrm H}
  \end{align}
  $$
  其中 $U_1\in \mathbb C^{m\times n}$ 由 $U$ 的前 $n$ 列构成，而 $\Sigma_1\in \mathbb C^{n\times n}$ 是对角元均为非负实数的对角阵.   
  将 $\beta\in \mathbb C^m\backslash \{0_m\}$ 对应地划分为 $\beta = \begin{bmatrix} \beta_1\\ \beta_2\end{bmatrix}$ (其中 $\beta_1\in \mathbb C^n$)  
  则我们有:  
  $$
  \begin{align}
  \frac{|y^{\mathrm H}Ax|}{\|x\|_2\|y\|_2}
  &=
  \frac{|\beta^{\mathrm H}\Sigma\alpha|}{\|\alpha\|_2\|\beta\|_2}\\
  &=
  \frac{
  \left|\begin{bmatrix}
  \beta_1\\
  \beta_2
  \end{bmatrix}^{\mathrm H}
  \begin{bmatrix}
  \Sigma_1\\
  0_{(m-n)\times n}
  \end{bmatrix}\alpha\right|}
  {\|\alpha\|_2\|\beta\|_2}\\
  &=
  \frac{
  |\beta^{\mathrm H}_1\Sigma\alpha|}
  {\|\alpha\|_2\|\beta\|_2}\quad (\text{note that }\|\beta_1\|_2\leq \|\beta\|_2)\\
  
  &\leq
  \frac{|\beta^{\mathrm H}_1\Sigma\alpha|}{\|\alpha\|_2\|\beta_1\|_2}\quad (\text{use conclusion of case (1)})\\
  
  &=
  \sigma_\max
  \end{align}
  $$
  注意到当 $x=v_1,y=u_1$ 时我们有 $\frac{|y^{\mathrm H}Ax|}{\|x\|_2\|y\|_2} = \frac{|u_1^{\mathrm H}Av_1|}{\|v_1\|_2\|u_1\|_2}
  = |u_1^{\mathrm H}(u_1\sigma_\max)| = \sigma_\max$  
  因此我们有:  
  $$
  \max_{\begin{subarray}{}
  x\in \mathbb C^n \backslash \{0_n\}\\
  y\in \mathbb C^m \backslash \{0_m\}
  \end{subarray}} \frac{|y^{\mathrm H}Ax|}{\|x\|_2\|y\|_2} = \sigma_\max
  $$
  另一方面我们有:  
  $$
  \begin{align}
  \frac{|y^{\mathrm H}Ax|}{\|x\|_2\|y\|_2}
  &=
  \frac{|\beta^{\mathrm H}\Sigma \alpha|}{\|\alpha\|_2\|\beta\|_2}\\
  &= 
  \frac{|\beta_1^{\mathrm H}\Sigma_1\alpha|}{\|\alpha\|_2 \|\beta\|_2}\\
  &\geq
  0
  \end{align}
  $$
  当且仅当 $\beta_1\ \bot\ \Sigma_1 \alpha$ (即 $y\ \bot\ Ax$) 时取等.  
  因此我们有:  
  $$
  \min_{\begin{subarray}{}
  x\in \mathbb C^n \backslash \{0_n\}\\
  y\in \mathbb C^m \backslash \{0_m\}
  \end{subarray}} \frac{|y^{\mathrm H}Ax|}{\|x\|_2\|y\|_2} 
  =
  \min_{\begin{subarray}{}
  \alpha\in \mathbb C^n \backslash \{0_n\}\\
  \beta\in \mathbb C^m \backslash \{0_m\}
  \end{subarray}} \frac{|\beta^{\mathrm H}\Sigma \alpha|}{\|\alpha\|_2\|\beta\|_2}
  = 0
  $$

- ③ 最后假设 $m<n$，根据 ② 中的结论我们有:    
  (显然 $A,A^{\mathrm H}$ 具有相同的奇异值)
  $$
  \begin{align}
  \max_{\begin{subarray}{}
  x\in \mathbb C^n \backslash \{0_n\}\\
  y\in \mathbb C^m \backslash \{0_m\}
  \end{subarray}} \frac{|y^{\mathrm H}Ax|}{\|x\|_2\|y\|_2} 
  &=
  \max_{\begin{subarray}{}
  x\in \mathbb C^n \backslash \{0_n\}\\
  y\in \mathbb C^m \backslash \{0_m\}
  \end{subarray}} \frac{|x^{\mathrm H}A^{\mathrm H}y|}{\|x\|_2\|y\|_2} = \sigma_\max(A^{\mathrm H}) = \sigma_\max(A)=\sigma_\max\\
  
  \min_{\begin{subarray}{}
  x\in \mathbb C^n \backslash \{0_n\}\\
  y\in \mathbb C^m \backslash \{0_m\}
  \end{subarray}} \frac{|y^{\mathrm H}Ax|}{\|x\|_2\|y\|_2} 
  &=
  \min_{\begin{subarray}{}
  x\in \mathbb C^n \backslash \{0_n\}\\
  y\in \mathbb C^m \backslash \{0_m\}
  \end{subarray}} \frac{|x^{\mathrm H}A^{\mathrm H}y|}{\|x\|_2\|y\|_2} = 0
  \end{align}
  $$

综上所述，我们有:  
$$
\begin{align}
\max_{\begin{subarray}{}
x\in \mathbb C^n \backslash \{0_n\}\\
y\in \mathbb C^m \backslash \{0_m\}
\end{subarray}} \frac{|y^{\mathrm H}Ax|}{\|x\|_2\|y\|_2} 
&= \sigma_\max = \sigma_1(A)\\

\min_{\begin{subarray}{}
x\in \mathbb C^n \backslash \{0_n\}\\
y\in \mathbb C^m \backslash \{0_m\}
\end{subarray}} \frac{|y^{\mathrm H}Ax|}{\|x\|_2\|y\|_2} 
&= 0
\end{align}
$$


### 4.4.3 Courant-Fischer 定理

> **(Courant–Fischer min-max 定理, Matrix Analysis 定理 $4.2.6$)**  
> 给定 Hermite 阵 $A \in \mathbb C^{n\times n}$，特征值按非减的次序排列: $\lambda_{\min} = \lambda_1 \leq \dots \leq \lambda_n = \lambda_{\max}$    
> 记 $S$ 为 $\mathbb C^n$ 的子空间，则我们有:
> $$
> \begin{align}
> \lambda_i
> &= \min_{S\subseteq  \mathbb C^n:\dim(S)=i}\left\{ \max_{x\neq 0_n\in S} \frac{x^{\mathrm H}Ax}{x^{\mathrm H}x}\right\}\\
> &= \max_{S\subseteq  \mathbb C^n:\dim(S)=n-i+1}\left\{ \min_{x\neq 0_n\in S} \frac{x^{\mathrm H}Ax}{x^{\mathrm H}x}\right\}\\
> \end{align}\quad (i=1,\dots,n)
> $$

**(奇异值的 Courant-Fischer 定理, Matrix Analysis 定理 $7.3.8$)**  
设 $A\in \mathbb C^{m\times n}$ 的奇异值按非增次序排列: $\sigma_1(A) \geq \dotsm \geq \sigma_q(A)$ (其中 $q=\min\{m,n\}$)   
则对于任意 $i=1,\dots,q$ 我们都有: 
$$
\begin{align}
\sigma_i(A)
&=
\min_{S\subseteq \mathbb C^n:\dim(S)=n-i+1}\left\{ \max_{x\neq 0_n\in S} 
\frac{\|Ax\|_2}{\|x\|_2}
\right\}\\

&=
\max_{S\subseteq \mathbb C^n:\dim(S)=i} \left\{
\min_{x\neq 0_n\in S} \frac{\|Ax\|_2}{\|x\|_2}
\right\}
\end{align}
$$
**证明:**   
设 $A^{\mathrm H}A \in \mathbb C^{n\times n}$ 的特征值按非减次序排列: $\lambda_1(A^{\mathrm H}A) \leq \dotsm \leq \lambda_n(A^{\mathrm H}A)$   
无论矩阵维度 $m,n$ 的大小关系如何，对于任意 $i=1,\dots,q$ 我们都有:
$$
\begin{align}
\sigma_i^2(A)
&=
\lambda_{n-i+1}(A^{\mathrm H}A)\\
&=
\min_{S\subseteq  \mathbb C^n:\dim(S)=n-i+1}\left\{ \max_{x\neq 0_n\in S} \frac{x^{\mathrm H}(A^{\mathrm H}A)x}{x^{\mathrm H}x}\right\} 
=
\left\{
\min_{S\subseteq  \mathbb C^n:\dim(S)=n-i+1}\left\{ \max_{x\neq 0_n\in S} \frac{\|Ax\|_2}{\|x\|_2}\right\}
\right\}^2
\\
&= \max_{S\subseteq  \mathbb C^n:\dim(S)=i}\left\{ \min_{x\neq 0_n\in S} \frac{x^{\mathrm H}Ax}
{x^{\mathrm H}x}\right\}
=
\left\{
\min_{S\subseteq  \mathbb C^n:\dim(S)=i}\left\{ \max_{x\neq 0_n\in S} \frac{\|Ax\|_2}{\|x\|_2}\right\}
\right\}^2\\
\end{align}
$$
从而有:  
$$
\begin{align}
\sigma_i(A)
&=
\min_{S\subseteq \mathbb C^n:\dim(S)=n-i+1}\left\{ \max_{x\neq 0_n\in S} 
\frac{\|Ax\|_2}{\|x\|_2}
\right\}\\

&=
\max_{S\subseteq \mathbb C^n:\dim(S)=i} \left\{
\min_{x\neq 0_n\in S} \frac{\|Ax\|_2}{\|x\|_2}
\right\}
\end{align}\quad (i=1,\dots,q)
$$



### 4.4.4 Cauchy 交错定理

**(奇异值的 Cauchy 交错定理, Matrix Analysis 推论 $7.3.6$)**  
设 $A\in \mathbb C^{m\times n}$ 的奇异值按非增次序排列: $\sigma_1 \geq \dotsm \geq \sigma_q$ (其中 $q=\min\{m,n\}$)  
设 $B$ 为 $A$ 删去任意一行或一列得到的子矩阵，其奇异值按非增次序排列: $\hat \sigma_1\geq \dotsm \geq \hat \sigma_q$   
其中如果 $m\geq n$ 且 $B$ 是 $A$ 删除一列得到的，或 $m\leq n$ 且 $B$ 是 $A$ 删除一行得到的，则定义 $\hat \sigma_q = 0$  
可以证明 $A$ 和 $\tilde A$ 的奇异值有如下交错形式:  
$$
\sigma_1 \geq \hat \sigma_1 \geq \sigma _2 \geq \hat \sigma_2 \geq \dotsm \geq \sigma_q \geq \hat \sigma_q \geq 0
$$
**证明:**  
定义:
$$
\tilde A := \begin{bmatrix}
0_{m\times m} & A\\
A^{\mathrm H} & 0_{n\times n}
\end{bmatrix}
$$
根据 **Matrix Analysis 定理 $7.3.3$** 可知其特征值为:
$$
-\sigma_1 \leq \dots\leq -\sigma_q \leq \underbrace{0=\dots = 0}_{|m-n|} \leq \sigma_q \leq \dots \leq \sigma_1
$$
从 $A$ 中删去第 $i$ 行，对应于删除 $\tilde A$ 的第 $i$ 行和第 $i$ 列.  
从 $A$ 中删去第 $j$ 列，对应于删除 $\tilde A$ 的第 $m+j$ 行和第 $m+j$ 列.  
记删除结果为 $\tilde B$   
根据 Hermite 特征值的 Cauchy 交错定理 (Matrix Analysis 定理 $4.3.17$) 可知 $\tilde A$ 和 $\tilde B$ 的特征值是交错的: 
$$
-\sigma_1 \leq -\hat \sigma_1 \leq \dotsm \leq -\sigma_q \leq -\hat \sigma_q \leq \underbrace{0=\dots = 0}_{|m-n|} \leq \hat \sigma_q \leq  \sigma_q \leq \dots \leq \hat \sigma_1 \leq \sigma_1
$$
命题得证.



### 4.4.5 Weyl 不等式

**(奇异值的 Weyl 不等式, Matrix Analysis 推论 $7.3.5$)**  
设 $A\in \mathbb C^{m\times n}$ 的奇异值按非增次序排列: $\sigma_1(A) \geq \dotsm \geq \sigma_q(A)$ (其中 $q=\min\{m,n\}$)   
设 $B\in \mathbb C^{m\times n}$ 的奇异值按非增次序排列: $\sigma_1(B) \geq \dotsm \geq \sigma_q(B)$   
则对于任意 $i=1,\dots,q$ 我们都有:  
$$
|\sigma_i(A)-\sigma_i(B)| \leq \|A-B\|_2
$$
**证明:**  
记 $E = A-B\in \mathbb C^{m\times n}$，并定义:  
$$
\tilde A := \begin{bmatrix}
0_{m\times m} & A\\
A^{\mathrm H} & 0_{n\times n}
\end{bmatrix}\qquad 
\tilde E := \begin{bmatrix}
0_{m\times m} & E\\
E^{\mathrm H} & 0_{n\times n}
\end{bmatrix}
$$
根据 **Matrix Analysis 定理 $7.3.3$** 可知其特征值为:  
$$
-\sigma_1(A) \leq \dots\leq -\sigma_q(A) \leq \underbrace{0=\dots = 0}_{|m-n|} \leq \sigma_q(A) \leq \dots \leq \sigma_1(A)\\

-\sigma_1(E) \leq \dots\leq -\sigma_q(E) \leq \underbrace{0=\dots = 0}_{|m-n|} \leq \sigma_q(E) \leq \dots \leq \sigma_1(E)\\
$$
根据 Hermite 特征值的 Weyl 不等式 **(Matrix Analysis 定理 $4.3.1$)** 可知:  
对于任意 $i=1,\dots,q$ 我们都有:  
$$
|\sigma_i(A+E)-\sigma_i(A)| \leq \|\tilde E\|_2 = \|E\|_2\\
\Leftrightarrow\\
|\sigma_i(A)-\sigma_i(B)| \leq \|A-B\|_2
$$
命题得证.



### 4.4.6 Hoffman-Wielandt 不等式

**(奇异值的 Hoffman-Wielandt 不等式, Matrix Analysis 推论 $7.3.5$)**  
设 $A\in \mathbb C^{m\times n}$ 的奇异值按非增次序排列: $\sigma_1(A) \geq \dotsm \geq \sigma_q(A)$ (其中 $q=\min\{m,n\}$)   
设 $B\in \mathbb C^{m\times n}$ 的奇异值按非增次序排列: $\sigma_1(B) \geq \dotsm \geq \sigma_q(B)$   
则我们有:
$$
\sum_{i=1}^q (\sigma_i(A)-\sigma_i(B))^2 \leq \|A-B\|_{\mathrm F}^2
$$
**证明:**  
记 $E = A-B\in \mathbb C^{m\times n}$，并定义:  
$$
\tilde A := \begin{bmatrix}
0_{m\times m} & A\\
A^{\mathrm H} & 0_{n\times n}
\end{bmatrix}\qquad 
\tilde E := \begin{bmatrix}
0_{m\times m} & E\\
E^{\mathrm H} & 0_{n\times n}
\end{bmatrix}
$$
根据 **Matrix Analysis 定理 $7.3.3$** 可知其特征值为:  
$$
-\sigma_1(A) \leq \dots\leq -\sigma_q(A) \leq \underbrace{0=\dots = 0}_{|m-n|} \leq \sigma_q(A) \leq \dots \leq \sigma_1(A)\\

-\sigma_1(E) \leq \dots\leq -\sigma_q(E) \leq \underbrace{0=\dots = 0}_{|m-n|} \leq \sigma_q(E) \leq \dots \leq \sigma_1(E)\\
$$
根据 Hermite 特征值的 Hoffman-Wielandt 不等式 **(Matrix Analysis 定理 $6.3.5$)** 可知:  
$$
2\sum_{i=1}^q (\sigma_i(A+E)-\sigma_i(A))^2 \leq \|\tilde E\|_{\mathrm F}^2 = 2\|E\|_{\mathrm F}^2\\
\Leftrightarrow\\
\sum_{i=1}^q (\sigma_i(A)-\sigma_i(B))^2 \leq \|A-B\|_{\mathrm F}^2
$$
命题得证.



### 4.4.7 推广到酉不变范数

任意给定 $A,B\in \mathbb C^{m\times n}$ (假设奇异值按非增次序排列)  
Weyl 不等式说的是 $\|\Sigma(A) - \Sigma(B)\|_2 \leq \|A-B\|_2$   
Hoffman-Wielandt 不等式说的是 $\|\Sigma(A)-\Sigma(B)\|_{\mathrm F} \leq \|A-B\|_{\mathrm F}$  
事实上，类似的等式对 $\mathbb C^{m\times n}$ 上的每个酉不变范数都是成立的.

**(Matrix Analysis 定理 $7.4.9.1$)**  
设 $A,B\in \mathbb C^{m\times n}$ 的奇异值分解为 $A=U_1\Sigma(A)V_1^{\mathrm H}$ 和 $B=U_2\Sigma(B)V_2^{\mathrm H}$  
其中 $U_1,U_2\in \mathbb C^{m\times m}$ 和 $V_1,V_2\in \mathbb C^{n\times n}$ 均为酉矩阵，  
而 $\Sigma(A)$ 和 $\Sigma(B)$ 的对角元按非增次序排列.  
则对于 $\mathbb C^{m\times n}$ 上的任意一个酉不变范数 $\|\cdot\|$ 都有 $\|\Sigma(A)-\Sigma(B)\|\leq \|A-B\|$ 成立.

- **(Mirsky 定理, Matrix Analysis 推论 $7.4.9.3$)**  
  设 $A,B\in \mathbb C^{n\times n}$ 是 Hermite 阵，其谱分解为 $A=U_1\Lambda(A)U_1^{\mathrm H}$ 和 $B=U_1\Lambda(B)U_2^{\mathrm H}$  
  其中 $U_1,U_2\in \mathbb C^{n\times n}$ 均为酉矩阵  
  而 $\Lambda(A)$ 和 $\Lambda(B)$ 的对角元按非增次序排列 (注意: 与特征值的约定习惯相反)  
  则对于 $\mathbb C^{m\times n}$ 上的任意一个酉不变范数 $\|\cdot\|$ 都有 $\|\Lambda(A)-\Lambda(B)\|\leq \|A-B\|$ 成立.

  **证明:**  
  选取 $\mu\in [0,\infty)$ 使得 $A+\mu I$ 和 $B+\mu I$ 均为半正定矩阵.  
  则我们有:  
  $$
  \begin{align}
  \Sigma(A+\mu I) &= \Lambda(A+\mu I)\\
  \Sigma(B+\mu I) &= \Lambda(B+\mu I)
  \end{align}
  $$
  根据 **Matrix Analysis 定理 $7.4.9.1$** 可知:  
  对于 $\mathbb C^{m\times n}$ 上的任意一个酉不变范数 $\|\cdot\|$ 我们都有:
  $$
  \begin{align}
  \|\Lambda(A)-\Lambda(B)\|
  &=
  \|\Lambda(A+\mu I) -  \Lambda(B+\mu I)\|\\
  &=
  \|\Sigma(A+\mu I) - \Sigma(B+\mu I)\|\\
  &\leq
  \|(A+\mu I) - (B+\mu I)\|\\
  &=
  \|A-B\|
  \end{align}
  $$



### 4.4.8 Eckart-Young 定理

**(Eckart-Young 定理)**  
设 $A\in \mathbb C^{m\times n}$ 的奇异值分解为 $A = \sum_{i=1}^q u_i \sigma_i v_i^{\mathrm H}$   
其中 $q=\min\{m,n\}$，奇异值按非增次序排列: $\sigma_1 \geq \dotsm \geq \sigma_q$   
任意给定 $k=1,\dots,q$，考虑 $A$ 的秩不超过 $k$ 的逼近:

- ① 在谱范数意义下，我们有:  
  $$
  \min_{B\in \mathbb C^{m\times n}:\rank(B)\leq k} \|A-B\|_2 = \sigma_{k+1}(A)
  $$
  使 $\|A-B\|_2$ 取到最小值 $\sigma_{k+1}(A)$ 的 $B$ 不唯一，只要可以使 $A-B$ 的最大奇异值为 $\sigma_{k+1}(A)$ 即可.  
  其中一个取法便是 $B=A_k:= \sum_{i=1}^k u_i \sigma_i v_i^{\mathrm H}$ 

- ② 在 Frobenius 范数意义下，我们有: 
  $$
  \min_{B\in \mathbb C^{m\times n}:\rank(B)\leq k} \|A-B\|_{\mathrm F} = \left(\sum_{i=k+1}^q (\sigma_i(A))^2 \right)^{\frac12}
  $$
  若 $\sigma_k(A)>\sigma_{k+1}(A)$，则使 $\|A-B\|_{\mathrm F}$ 取到最小值 $\left(\sum_{i=k+1}^q (\sigma_i(A))^2 \right)^{\frac12}$ 的 $B$ 是唯一的:   
  即 $B=A_k:= \sum_{i=1}^k u_i \sigma_i v_i^{\mathrm H}$ 

- ③ 在一般的酉不变范数意义下，我们有:  
  $$
  \min_{B\in \mathbb C^{m\times n}:\rank(B)\leq k} \|A-B\|  = \left\|\sum_{i=k+1}^q u_i \sigma_i v_i^{\mathrm H}\right\|
  $$
  使 $\|A-B\|$ 取到最小值 $\left\|\sum_{i=k+1}^q u_i \sigma_i v_i^{\mathrm H}\right\|$ 的一个取法便是 $B=A_k:= \sum_{i=1}^k u_i \sigma_i v_i^{\mathrm H}$  

- ④ 值得说明的是，将约束条件 $\rank(B)\leq k$ 修改为 $\rank(B)=k$ 是不合适的.  
  否则最小值是不一定存在的，于是要将 $\min$ 改成 $\inf$.   
  例如对零矩阵 $0_{m\times n}$ 进行秩 $1$ 逼近，下确界存在且为 $0$，但取不到 (即最小值不存在)，  
  而对零矩阵 $0_{m\times n}$ 进行秩不超过 $1$ 的逼近就不会出现上述问题.

**证明:**  

- ① 首先在谱范数意义下考虑秩不超过 $k$ 的最佳逼近.  
  根据奇异值的 Weyl 不等式可知:  
  $$
  \begin{align}
  \|A-B\|_2
  &\geq
  \max_{1\leq i\leq q}|\sigma_i(A)-\sigma_i(B)|\\
  &\geq
  |\sigma_{k+1}(A)-\sigma_{k+1}(B)|\quad (\text{note that }\rank(B)\leq k\text{ so that }\sigma_{k+1}(B) = 0)\\
  &=
  \sigma_{k+1}(A)
  \end{align}
  $$
  上述不等式至少在 $B=A_k:= \sum_{i=1}^k u_i \sigma_i v_i^{\mathrm H}$ 时取等.

- ② 其次在 Frobenius 范数意义下考虑秩不超过 $k$ 的最佳逼近.  
  根据奇异值的 Hoffman-Wielandt 不等式可知:  
  $$
  \begin{align}
  \|A-B\|_{\mathrm F}^2
  &\geq
  \sum_{i=1}^q |\sigma_i(A)-\sigma_i(B)|^2\\
  &\geq
  \sum_{i=k+1}^q |\sigma_i(A)-\sigma_i(B)|^2\quad (\text{note that }\rank(B)\leq k\text{ so that }\sigma_{k+1}(B)=\dotsm = \sigma_{q}(B) = 0)\\
  &=
  \sum_{i=k+1}^q (\sigma_i(A))^2
  \end{align}
  $$
  上述不等式当且仅当 $B=A_k:= \sum_{i=1}^k u_i \sigma_i v_i^{\mathrm H}$ 时取等.  

  取等条件的充分性是容易验证的，下面证明取等条件的必要性:  
  若 $\|A-B\|_{\mathrm F}^2 = \sum_{i=k+1}^q (\sigma_i(A))^2$，则对于任意 $i=1,\dots,k$ 必定有 $\sigma_i(B)=\sigma_i(A)$ 成立.  
  **(存疑: 如何说明必要性?)**  
  (可能要使用 trace，有些麻烦)

****

**代数最小二乘问题:**    
给定设计矩阵 $X\in \mathbb C^{n\times d}$ 和观测向量 $y\in \mathbb C^n$  
我们使用目标是最小化残差: 
$$
\min_{\beta\in \mathbb C^d} \|y-X\beta\|^2 
$$
等价于求解一个线性方程组 $(X^{\mathrm T}X)\beta = X^{\mathrm T}y$ (称为法方程)

**几何最小二乘问题 (主成分分析):**  
给定数据点 $y_1,\dots,y_n\in \mathbb C^d$ 和维度 $k\ll q:= \max\{n,d\}$    
要找最佳的 $k$ 维仿射空间 $\{Ax+b:x\in \mathbb C^d\}$ 使得 $y_1,\dots,y_n$ 的投影方差最大化，即使得垂直部分最小化.  

任意给定 $k$ 维仿射空间 $V:=\{Ax+b:x\in \mathbb C^k\}$   
其中 $A\in \mathbb C^{d\times k},b\in \mathbb C^d$ 且 $\rank(A)=\dim(\text{span}(A))=k$ 
记 $y_i$ 在 $V:=\{Ax+b:x\in \mathbb C^k\}$ 中的投影为 $\tilde y_i$  
则依向量 $b$ 平移后的点 $y_i-b$ 和 $\tilde y_i -b$ 具有如下关系:
$$
\tilde y_i - b = P(y_i-b) = AA^\dagger (y_i-b)\\
\Leftrightarrow\\
\tilde y_i = AA^\dagger(y_i-b) + b
$$
其中 $P:= AA^\dagger$ 是 $\mathbb C^d$ 到 $\text{span}(A)$ 的正交投影算子  
记 $P_\bot:= I_d - AA^\dagger$      
于是垂直分量的 $l_2$ 范数和为:
$$
\begin{align}
\sum_{i=1}^n \|\tilde y_i - y_i\|_2^2
&=
\sum_{i=1}^n \|AA^\dagger (y_i-b) + b - y_i\|^2_2\\
&=
\sum_{i=1}^n \|(I-AA^\dagger)(y_i-b)\|_2^2\\
&=
\sum_{i=1}^n \|P_\bot (y_i-b)\|^2_2
\end{align}
$$
我们的目标是最小化垂直分量的 $l_2$ 范数和:  
$$
\min_{A\in \mathbb C^{d\times k},b\in \mathbb C^d:\rank(A)\leq k} \sum_{i=1}^n \|P_\bot (y_i-b)\|^2_2\quad (\text{where }P_\bot:= I_d - AA^\dagger)
$$
目标函数对 $b$ 求梯度可得:  
$$
\begin{align}
\nabla_b \left\{
\sum_{i=1}^n \|P_\bot (y_i-b)\|^2_2
\right\} 
&= -2\sum_{i=1}^n P_\bot (y_i-b)\\
&= -2P_\bot \left(\sum_{i=1}^n y_i - nb \right)
\end{align}
$$
令 $\nabla_b \left\{
\sum_{i=1}^n \|P_\bot (y_i-b)\|^2_2
\right\}=-2P_\bot \left(\sum_{i=1}^n y_i - nb \right) = 0_d$ 可得 $b_\star = \frac1n \sum_{i=1}^n y_i$   
这表明无论 $A\in \mathbb C^{d\times k}$ 如何选取，数据点 $y_1,\dots,y_n$ 的重心总是截距向量 $b$ 的一个最优解.  
(但并非唯一的最优解，它处于 $\text{span}(A_\star)$ 中的分量是可以松动的，其中 $A_\star$ 是 $A$ 的一个最优解)  
将 $b=b_\star = \frac1n \sum_{i=1}^n y_i$ 代入原问题，就将其等价转换为: 
$$
\min_{A\in \mathbb C^{d\times k}:\rank(A)\leq k} \sum_{i=1}^n \|P_\bot (y_i - b_\star)\|^2\quad (\text{where }\begin{cases}
P_\bot := I_d - AA^\dagger\\
b_\star = \frac1n \sum_{i=1}^n y_i\end{cases})
$$
记 $Z:=[y_1-b_\star,\dots,y_n-b_\star]\in \mathbb C^{d\times n}$   
则目标函数可化简为: 
$$
\begin{align}
\sum_{i=1}^n \|P_\bot (y_i - b_\star)\|^2
&=
\|P_\bot Z\|_{\mathrm F}^2\\
&=
\|(I_d-AA^\dagger) Z\|_{\mathrm F}^2\\
&=
\|Z-AA^\dagger Z\|_{\mathrm F}^2
\end{align}
$$
因此优化问题变为:  
$$
\min_{A\in \mathbb C^{d\times k}:\rank(A)\leq k} \|Z-AA^\dagger Z\|_{\mathrm F}^2
$$
设 $Z\in \mathbb C^{d\times n}$ 的奇异值分解为 $Z:= \sum_{i=1}^q u_i \sigma_i v_i^{\mathrm H} = U\Sigma V^{\mathrm H}$    
其中 $q:= \max\{n,d\}$，奇异值按非增次序排列: $\sigma_1 \geq \dotsm \geq \sigma_q$，对角阵 $\Sigma := \text{diag}\{\sigma_1,\dots,\sigma_q\}$  
而 $U:=[u_1,\dots,u_q]\in \mathbb C^{d\times q}$ 和 $V:=[v_1,\dots,v_q]\in \mathbb C^{d\times q}$ 的列向量组标准正交.  
假设 $\sigma_k > \sigma_{k+1}$，则根据 **Eckart-Young 定理**可知上述优化问题的最优解 $A_\star$ 必然满足:  
$$
A_\star A_{\star}^\dagger Z = \sum_{i=1}^k u_i \sigma_i v_i^{\mathrm H}
$$
因此 $A_\star$ 只需保证 $A_\star A^\dagger_\star = \sum_{i=1}^k u_ku_k^{\mathrm H} =  U_kU_k^{\mathrm H}$ 即可:  
(其中 $U_k := [u_1,\dots,u_k]\in \mathbb C^{d\times k}$) 
$$
\begin{align}
(A_\star A_\star^\dagger)Z
&=
\left(\sum_{i=1}^k u_ku_k^{\mathrm H}\right)
\left(\sum_{i=1}^q u_i \sigma_i v_i^{\mathrm H}\right)\\
&=
\sum_{i=1}^k u_k \left(\sum_{i=1}^q (u_k^{\mathrm H}u_i) \sigma_i v_i^{\mathrm H}\right)\quad (\text{note that }u_k^{\mathrm H}u_i =
\delta_{k,i} = \begin{cases}
1 & \text{if }i=k\\
0 & \text{otherwise}
\end{cases})\\
&=
\sum_{i=1}^k u_k \sigma_k v_k^{\mathrm H}
\end{align}
$$
$A_\star$ 的最简单取法便是 $A_\star = U_k = [u_1,\dots,u_k]\in \mathbb C^{d\times k}$   

**The End**
