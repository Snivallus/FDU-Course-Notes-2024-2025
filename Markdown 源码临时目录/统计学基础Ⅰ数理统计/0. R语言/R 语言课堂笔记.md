# R 语言——侯曦燕

[elements of statistics - Elements of Statistics (quarto.pub)](https://oishihou.quarto.pub/elements-of-statistics/#/title-slide)

- 整除操作符 `%/%` (返回商的下取整)

  ```R
  # 计算 15 除以 2 的整数部分
  result1 <- 15 %/% 2
  print(result1)  # 输出应为 7
  
  # 计算 -15 除以 2 的整数部分
  result2 <- -15 %/% 2
  print(result2)  # 输出应为 -8
  ```

- 阶乘函数 `factorial()`

  ```R
  # 计算 5 的阶乘
  fact5 <- factorial(5)
  print(fact5)  # 输出应为 120
  ```

- 组合数函数 `choose()`

  ```R
  # 计算从 5 个元素中选取 3 个的组合数
  comb53 <- choose(5, 3)
  print(comb53)  # 输出应为 10
  ```

- 字符串拼接函数：`paste()` 和 `paste0()`

  ```R
  paste('welcome','to','R')  # 将输出 "welcome to R"
  paste0('welcome','to','R') # 将输出 "welcometoR" (不带空格)
  
  print(paste("hello", "world", sep = "-")) # 将输出 "hello-world"
  
  paste(1:12, c("st", "nd", "rd", rep("th", 9)))
  # 将输出:
  # [1] "1 st"  "2 nd"  "3 rd"  "4 th"  "5 th"  "6 th"  "7 th"  "8 th"  "9 th" 
  # [10] "10 th" "11 th" "12 th"
  
  paste0(1:12, c("st", "nd", "rd", rep("th", 9)))
  # 将输出: 
  # [1] "1st"  "2nd"  "3rd"  "4th"  "5th"  "6th"  "7th"  "8th"  "9th"  "10th"
  # [11] "11th" "12th"
  ```

- 向量操作函数：`c()`, `vector()`, `seq()`, 和 `rep()`  
  分别定义向量、初始化向量、生成规律序列、复制向量元素.

  - `seq` 函数：  
    - `from`：序列的开始值。
    - `to`：序列的结束值。
    - `by`：序列中每个元素之间的间隔。
    - `length.out`：生成序列的元素总数

  ```R
  # 从 1 到 10 的序列
  sequence1 <- seq(from = 1, to = 10)
  
  # 从 5 到 50，每隔 5 的序列
  sequence2 <- seq(from = 5, to = 50, by = 5)
  
  # 生成长度为 4 的序列，从 1 到 10
  sequence3 <- seq(from = 1, to = 10, length.out = 4)
  ```

  - `rep` 函数：
    - `x`：要复制的向量。
    - `times`：重复次数。
    - `each`：每个元素重复的次数。
    - `length.out`：输出向量的长度。
    - `each` 和 `times` 通常不同时使用，具体取决于重复模式的需求。

  ```R
  # 重复整个向量三次
  repeated_vector <- rep(x = c(1, 2), times = 3)
  
  # 每个元素重复三次
  each_repeated_vector <- rep(x = c(1, 2), each = 3)
  
  # 创建特定长度的重复向量
  fixed_length_repeated_vector <- rep(x = 1:3, length.out = 10)
  ```

- `class()` 函数：

  ```r
  # Character vector
  ltrs <- letters[1:10] 	# "a" "b" "c" "d" "e" "f" "g" "h" "i" "j"
  class(ltrs)				# 输出 "character"
  
  # Factor vector
  fac <- as.factor(ltrs)  
  fac
  # 输出: 
  # [1] a b c d e f g h i j
  # Levels: a b c d e f g h i j
  
  class(fac)				# 输出 "facto"
  ```

- 同一向量中无法混杂不同类型的数据，  
  若出现混杂，则 `c()` 会将低级别的类型强制转换为高级别的类型，从高到低为：  

  1. 列表型 (list)
  2. 字符型 (character)
  3. 复数型 (complex)
  4. 双精度型 (double)
  5. 整型 (integer)
  6. 逻辑型 (logical)

  我们主要记忆，混合数值向量和字符向量时 `numeric` 类型会被强制转换为 `character` 类型.  
  使用 `as.` 函数可以实现强制类型转换.  
  例如 `as.numeric`, `as.logical` 等等.  
  如果 R 不知道强制转换后应该取什么值，则赋值 `NA`：

  ```R
  x <- c('a','b','c')
  as.numeric(x)  # 输出: [1] NA NA NA
  ```

- `cumsum` 是累计求和函数： 

  ```R
  x <- c(1,2,3,4)
  # Cumulative sums
  cumsum(x) # 输出: [1]  1  3  6 10
  ```

- `attributes()` 函数：(获取对象的属性)    
  当我们调用 `attributes(obj)` 时，它返回一个列表，这个列表包含了对象的所有属性.  
  如果对象没有任何属性，函数会返回 `NULL`

- 矩阵：

  - `A+B` 和 `A-B`
  - 矩阵点积 `A*B`
  - 矩阵乘积 `A%*%B`
  - 行列式 `det(A)`
  - 求逆 `solve(A)`
  - 求特征值和特征向量 `eigen(A)`
  - 转置 `t(A)`
  - 取对角元向量 `diag(A)`

- `apply()` 函数：`apply(X, MARGIN, FUN, ...)`

  - `X`: 是要操作的数组或矩阵。
  - `MARGIN`: 指定要应用函数的维度。`MARGIN=1` 表示应用函数到行，`MARGIN=2` 表示应用函数到列。
  - `FUN`: 是要应用的函数。
  - `...`: 其他可能传递给 `FUN` 的参数。

  ```R
  # 创建一个矩阵
  mat <- matrix(1:9, nrow=3, ncol=3)
  
  # 计算每行的总和
  col_sums <- apply(mat, 1, sum)
  
  # j
  apply(mat,2,mean)
  ```

  

  









































