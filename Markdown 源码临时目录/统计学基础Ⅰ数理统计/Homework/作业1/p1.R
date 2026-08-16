students <- data.frame(
  name = c("LAWRENCE", "JEFFERY", "EDWARD", "PHILLIP", "KIRK",
           "ROBERT", "JACLYN", "DANNY", "CLAY", "HENRY", 
           "LESLIE", "JOHN", "WILLIAM", "MARTHA", "LEWIS", 
           "AMY", "ALFRED", "CHRIS", "FREDRICK", "CAROL",   
           "JOE", "MARY", "LINDA", "MARK", "PATTY",  
           "ELIZABET", "JUDY", "LOUISE", "ALICE", "JAMES", 
           "MARIAN", "TIM", "BARBARA", "DAVID", "KATIE",   
           "MICHAEL", "SUSAN", "JANE", "LILLIE", "ROBERT"),
  age = c(17, 14, 16, 16, 17, 15, 12, 15, 15, 14,   
          14, 13, 15, 16, 14, 15, 14, 14, 14, 14,
          13, 15, 17, 15, 14, 14, 14, 12, 13, 12, 
          16, 12, 13, 13, 12, 13, 13, 12, 12, 12),
  height = c(172, 169, 167, 167, 167, 164, 162, 162, 162, 159, 
             159, 159, 159, 159, 157, 157, 157, 157, 154, 154,   
             154, 152, 152, 152, 152, 152, 152, 149, 149, 149,
             147, 147, 147, 145, 145, 142, 137, 135, 127, 125),
  gender = c("M", "M", "M", "M", "M", "M", "F", "M", "M", "M", 
             "F", "M", "M", "F", "M", "F", "M", "M", "M", "F",   
             "M", "F", "F", "M", "F", "F", "F", "F", "F", "M",
             "F", "M", "F", "M", "F", "M", "F", "F", "F", "M"),
  weight = c(78.1, 51.3, 50.8, 58.1, 60.8, 58.1, 65.8, 48.1, 47.7, 54,   
             64.5, 44.5, 50.4, 50.8, 41.8, 50.8, 44.9, 44.9, 42.2, 38.1,  
             47.7, 41.8, 52.7, 47.2, 38.6, 41.3, 36.8, 55.8, 48.6, 58.1,
             52.2, 38.1, 50.8, 35.9, 43.1, 43.1, 30.4, 33.6, 29.1, 35.9)
)

# print data frame
# print(students)

# 第(1)问
if(0){
  # Extract the first letter of each name
  first_letters <- substr(students$name, 1, 1)
  
  # Calculate the frequency of each first letter
  letter_frequency <- table(first_letters)
  
  # Print the frequency table
  print(letter_frequency)
}

# 第(2)问
if(0){
  # mean
  mean_weight <- mean(students$weight)
  
  # variance
  variance_weight <- var(students$weight)
  
  # standard deviation
  sd_weight <- sd(students$weight)
  
  # median
  median_weight <- median(students$weight)
  
  # range
  range_weight <- range(students$weight)
  range_value <- diff(range_weight)
  
  # print statistics
  cat("mean:", mean_weight, "\n")
  cat("variance:", variance_weight, "\n")
  cat("standard deviation:", sd_weight, "\n")
  cat("median:", median_weight, "\n")
  cat("minimum:", range_weight[1], "\n")
  cat("maximum:", range_weight[2], "\n")
  cat("range:", range_value, "\n")
}

# 第(3)问
if(0){
  # Define the boundaries of groups
  breaks <- seq(120, 176, by=8)
  
  # Create a histogram
  hist(students$height, breaks=breaks, main="身高分布直方图",
       xlab="身高区间", ylab="频数", 
       col="blue", border="black", 
       right=TRUE, # which means the interval type is (]
       xaxt='n', yaxt='n' # disable the default axis
  )
  
  # User-defined axis
  # horizontal axis
  axis(1, at=seq(120, 176, by=8), labels=seq(120, 176, by=8), las=1)
  # vertical axis
  axis(2, at=seq(0, 20, by=1), las=1, 
       labels=ifelse(seq(0, 20, by=1) %in% seq(0, 20, by=5), seq(0, 20, by=1), "")
  )
}