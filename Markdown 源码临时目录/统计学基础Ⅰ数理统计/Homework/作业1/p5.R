# sample data
sublimation_heat_data <- c(
  136.6, 145.2, 151.5, 162.7, 159.1, 
  159.8, 160.8, 173.9, 160.1, 160.4, 
  161.1, 160.6, 160.2, 159.5, 160.3, 
  159.2, 159.3, 159.6, 160.0, 160.2, 
  160.1, 160.0, 159.7, 159.5, 159.5, 
  159.6, 159.5
)

# 第(1)问
if(0){
  # Define the boundaries of groups
  breaks <- seq(135, 175, by=4)
  
  # Create a histogram
  hist(sublimation_heat_data, breaks=breaks, 
       main="金融铱升华热直方图", 
       xlab="升华热区间", ylab="频数",
       col="blue", border="black",
       xaxt='n', yaxt='n' # disable the default axis
  )
  
  # User-defined axis
  # horizontal axis
  axis(1, at=seq(135, 175, by=4), labels=seq(135, 175, by=4), las=1)
  # vertical axis
  axis(2, at=seq(0, 30, by=1), las=1, 
       labels=ifelse(seq(0, 30, by=1) %in% seq(0, 30, by=10), seq(0, 30, by=1), "")
  )
}

# 第(2)问
if(0){
  # calculate Empirical Distribution Function
  edf <- ecdf(sublimation_heat_data)
  
  # create Empirical Distribution Function
  plot(edf, main="金融铱升华热的经验分布图", 
       xlab="升华热", ylab="累积概率", 
       col="blue", lwd=2, 
       xaxt='n', yaxt='n' # disable the default axis
  )
  
  # User-defined axis
  # horizontal axis
  axis(1, at=seq(135, 175, by=10), labels=seq(135, 175, by=10), las=1)
  # vertical axis
  axis(2, at=seq(0, 1, by=0.2), labels=seq(0, 1, by=0.2), las=1)
}

# 第(3)问
if(0){
  # given probability level
  prob_levels <- c(0.90, 0.75, 0.25, 0.05, 0.01)
  
  # calculate quantile
  quantiles <- quantile(sublimation_heat_data, probs = prob_levels)
  print(quantiles)
}

# 第(4)问
if(0){
  library(e1071)
  mean_value <- mean(sublimation_heat_data)
  variance_value <- var(sublimation_heat_data)
  std_deviation <- sd(sublimation_heat_data)
  skewness_value <- skewness(sublimation_heat_data)
  kurtosis_value <- kurtosis(sublimation_heat_data)
  
  cat("mean: ", mean_value, "\n")
  cat("variance: ", variance_value, "\n")
  cat("standard variance: ", std_deviation, "\n")
  cat("skewness: ", skewness_value, "\n")
  cat("kurtosis: ", kurtosis_value, "\n")
}

# 第(5)问
if(1){
  # 使用boxplot()函数创建箱线图
  boxplot(sublimation_heat_data, 
          main="金融铱升华热的箱线图", 
          xlab="升华热", 
          col="lightblue", # 设置箱体颜色
          notch=FALSE, # 如果为TRUE，则在箱体中添加一个缺口以表示中位数的置信区间
          horizontal=TRUE # 设置箱线图的方向，FALSE为垂直，TRUE为水平
  )
  
  # 添加数据点，以更清楚地展示所有观测值
  points(sublimation_heat_data,
         jitter(rep(1, length(sublimation_heat_data))), 
         col="blue", pch=16)
}
