sample_data <- c(0.5, 0.7, 0.2, 0.7, 0.4, 2.5, 1.5, -0.2, -0.5, 0.1)
# calculate Empirical Distribution Function
sample_ecdf <- ecdf(sample_data)

# create Empirical Distribution Function
plot(sample_ecdf, main="Empirical Distribution Function", 
     xlab="Values", ylab="ECDF", col="blue", lwd=2,
     xaxt='n', yaxt='n' # disable the default axis
)
     
# User-defined axis
# horizontal axis
axis(1, at=seq(-1, 3, by=0.5), labels=seq(-1, 3, by=0.5), las=1)
# vertical axis
axis(2, at=seq(0, 1, by=0.1), las=1, 
      labels=ifelse(seq(0, 1, by=0.1) %in% seq(0, 1, by=0.2), seq(0, 1, by=0.1), "")
)