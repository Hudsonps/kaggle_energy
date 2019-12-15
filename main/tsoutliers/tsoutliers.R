# See https://stats.stackexchange.com/questions/1142/simple-algorithm-for-online-outlier-detection-of-a-generic-time-series
tsoutliers <- function(x,plot=FALSE, lower_prob=0.25, upper_prob=.75)
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
  
  # returns the values at the lower_prob and upper_prob -tiles
  resid.q <- quantile(resid, prob=c(lower_prob, upper_prob), type=7)
  
  # upper_prob-tile minus lower_prob-tile
  iqr <- diff(resid.q)
  
  # define lower and upper whiskers
  limits <- resid.q + 1.5*iqr*c(-1,1)
  
  # pmin, pmax: min and max functions applied to vectors
  score <- abs(
    pmin((resid - limits[1]) / iqr,0) + 
      pmax((resid - limits[2]) / iqr,0)
  )
  
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
    # !!! For testing
    # return(list(resid=resid, resid.q=resid.q, iqr=iqr, limits=limits, score=score))
}

# Example
# eps <- rnorm(10, mean = 0, sd = 3)
# mu <- 4
# X_t <- mu + eps
# o <- tsoutliers(X_t, plot=FALSE,lower_prob=0.25, upper_prob=0.75)
