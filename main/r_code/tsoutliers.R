# See https://stats.stackexchange.com/questions/1142/simple-algorithm-for-online-outlier-detection-of-a-generic-time-series
tsoutliers <- function(x,plot=FALSE, lower_thresh=0.25, upper_thresh=.75)
{
  x <- as.ts(x)  # convert series to a time series object

  # Fit TS model here
  if(frequency(x)>1)  # the frequency function returns the number of periods of the time series
    
    # seasonal ts decomposition by Loess if
    # data has 2 or more periods
    resid <- stl(x,s.window="periodic",robust=TRUE)$time.series[,3]
  else
  {
    tt <- 1:length(x)
    resid <- residuals(loess(x ~ tt))
  }

  # Determine if outlier given fitted model here
  resid.q <- quantile(resid,prob=c(lower_thresh,upper_thresh))
  iqr <- diff(resid.q)
  limits <- resid.q + 1.5*iqr*c(-1,1)
  score <- abs(pmin((resid-limits[1])/iqr,0) + pmax((resid - limits[2])/iqr,0))
  
  if(plot)
  {
    print("PLOTTING>>>>")
    plot(x)
    x2 <- ts(rep(NA,length(x)))
    x2[score>0] <- x[score>0]
    tsp(x2) <- tsp(x)
    points(x2,pch=19,col="red")
    return(invisible(score))
  }
  else
    return(score)
}

# Example
# eps <- rnorm(100, mean = 0, sd = 3)
# mu <- 4
# X_t <- mu + eps
# outliers <- tsoutliers(X_t, plot=TRUE)
