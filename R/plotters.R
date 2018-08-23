# plotting functions for mdrm
# for now, just one and two dimensional outcomes are implemented


#' plot the empirical baseline distribution function.	
#'
#' \samp{plot{x}} plots the baseline CDF based on the estimated jump sizes \eqn{F}, where \eqn{x} is an objet of class \code{DRM}
#'
#' The number of distinct jumps equals the number of unique \eqn{Y} observations (calculated row-wise if Y is a matrix like \code{unique(Y)}).
#' For univariate Y, 95% confidence bands are included.  Note that, given \eqn{m} unique outcomes, model identifiability requires \eqn{\sum_{j=1}^{m}F_j=1}
#' so that there are \eqn{m-1} parameters. We set \eqn{F_{m}\equiv 1-\sum_{j=1}^{m-1}F_j}.
#' 
#' Note that plotting is only implemented for univariate and bivariate outcomes.
#' @param x an object of class 'DRM', created by calling \code{\link{drm}}
#' @return nothing; called for its side effect of creating a plot
#' @seealso \code{\link{drm}}
#' @export
setMethod(f = "plot", signature = "DRM_",
          definition = function(x,...){
            if(!x@is_fitted) stop("Model not yet fit to the data.")
            if(ncol(x@Y)>2){stop("Sorry, plotting is defined only for univariate or bivariate outcomes.")}
            if(ncol(x@Y) == 1){
              m = NROW(x@U)
              fh = x@Fhat[1:(m-1)]
              th_f = c(fh,c(x@beta[-1])) # exclude intercept since it isn't an independent parameter
              HF0 = drmHess(th_f,x@X,x@Y,TRUE) # this should be calculated @ model fit
              IF = -solve(HF0)
              sU = sort(x@U)
              cumV = unlist(lapply(1:m,function(j){
                inMat = which(x@U <= sU[j])
                sum(IF[inMat,inMat])
              }))
              cumSD = sqrt(cumV)*qnorm(0.975)/sqrt(nrow(x@X))
              ordF = x@Fhat[order(x@U[,1])]
              cF = cumsum(ordF)
              YL = c(min(cF-cumSD),max(cF+cumSD))
              op = par(no.readonly = TRUE); on.exit(par(op))
              par(mar=c(3,3.25,3,1),mgp = c(1.75,0.5,0),tcl = -0.33)
              plot(sU,cF,ylim = YL,type = "l",col = "#91320a",
                   lty = 4,las = 1,lwd = 3,
                   ylab = expression(hat(P)(Y<=y)),
                   xlab = if(is.null(colnames(x@Y))) "Y" else colnames(x@Y)[1],
                   main = "Baseline CDF (pointwise 95% C.I.)")
              polygon(x = c(sU,rev(sU)),y = c(cF-cumSD,rev(cF+cumSD)),
                      col = scales::alpha(3,0.4))
              abline(h = c(0,1),lty = 5)
            } else { # two columns
              Ugrid = cdF(x@U,x@Fhat,0,TRUE)
              par(mar = c(2,2,1,1),mgp = c(3,0.67,0))
              image(x = Ugrid$ux,y = Ugrid$uy,z = Ugrid$cdf,
                    col = viridis::viridis(40),las = 1)
              # levelplot(gridU[,3]~gridU[,1]*gridU[,2],pretty = FALSE,contour = TRUE,
              #           at = 0:10/10,col.regions = genRCB(colorScheme,100),
              #           xlab = colnames(x@U)[1],ylab = colnames(x@U)[2],
              #           main="Estimated Baseline Distribution",
              #           colorkey = list(space = "right",at = contour_percentiles,
              #                           labels = as.character(contour_percentiles)))
            }
            invisible()
          })
