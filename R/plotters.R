# plotting functions for mdrm
# for now, just one and two dimensional outcomes are implemented


#' plot the empirical baseline distribution function.	
#'
#' \code{plot(x)} plots the baseline CDF based on the estimated jump sizes \eqn{F}, where \eqn{x} is 
#' an object of class \code{DRM}.
#'
#' The number of distinct jumps equals the number of unique \eqn{Y} observations (calculated row-wise if 
#' Y is a matrix like \code{unique(Y)}).
#' For univariate Y, 95\% confidence bands are included.  Note that, given \eqn{m} unique outcomes, 
#' model identifiability requires \eqn{\sum_{j=1}^{m}F_j=1} so that there are \eqn{m-1} parameters. 
#' We set \deqn{F_{m}= 1-\sum_{j=1}^{m-1}F_j}.
#' 
#' Plotting is only implemented for univariate and bivariate outcomes.
#' @param x an object of class 'DRM', created by calling \code{\link{drm}}
#' @param y ignored.
#' @param ... ignored; part of function signature for compatibility with S3 method.
#' @return \code{invisible()}; called for its side effect of creating a plot.
#' @seealso \code{\link{drm}}
#' @importFrom scales alpha
#' @importFrom viridis viridis
#' @importFrom graphics par plot abline polygon image title
#' @importFrom stats qnorm
#' @export
setMethod(f = "plot", signature = signature(x = "DRM", y = "missing"),
          definition = function(x,y,...){
            if(ncol(x@Y)>2){stop("Sorry, plotting is defined only for univariate or bivariate outcomes.")}
            op = par(no.readonly = TRUE); on.exit(par(op))
            if(ncol(x@Y) == 1){
              m = NROW(x@U)
              p = m-1 # number of independent parameters
              fh = x@Fhat[1:p]
              fcov = matrix(0,nrow=m,ncol=m)
              fcov[1:p,1:p] = vcov(x)[1:p,1:p]
              fcov[,m] = -rowSums(fcov) # of course, this matrix is not full rank
              fcov[m,] = fcov[,m]
              fcov[m,m] = sum(fcov[1:p,1:p])
              
              sU = sort(x@U[,1])
              sF = x@Fhat[order(x@U[,1])]
              fsd = sapply(1:m,function(j){
                ix = which(x@U <= sU[j])
                sum(fcov[ix,ix])
              })
              
              cumSD = sqrt(pmax(fsd,0))*qnorm(0.975)
              cF = cumsum(sF)
              YL = c(min(cF-cumSD),max(cF+cumSD))
              op = par(no.readonly = TRUE); on.exit(par(op))
              par(mar = c(3,3.25,3,1),mgp = c(1.75,0.5,0),tcl = -0.33)
              plot(sU,cF,ylim = YL,type = "l",col = "#91320a",
                   lty = 4,las = 1,lwd = 3,
                   ylab = expression(hat(P)(Y<=y)),
                   xlab = if(is.null(colnames(x@Y))) "Y" else colnames(x@Y)[1],
                   main = "Baseline CDF (pointwise 95% C.I.)")
              ytick = par('yaxp')
              abline(h = seq(from=ytick[1],to=ytick[2],length.out=ytick[3]+1),lty="a5",col="#7f7e7e")
              polygon(x = c(sU,rev(sU)),y = c(cF-cumSD,rev(cF+cumSD)),
                      col = scales::alpha(3,0.5))
            } else { # two columns
              Ugrid = cdF(x@U,x@Fhat,0,TRUE)
              par(mar = c(3.25,3.25,0,0),mgp = c(2,0.5,0),tcl=-0.33)
              image(x = Ugrid$ux,y = Ugrid$uy,z = Ugrid$cdf,
                    col = viridis::viridis(40),las = 1,
                    xlab = colnames(x@U)[1],ylab = colnames(x@U)[2])
            }
            invisible()
          })
