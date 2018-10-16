#' @useDynLib mdrm
#' @importFrom Rcpp sourceCpp


# prototype the DRM class

# parse optional arguments list and return a list with any defaults substituted for passed-in values
parseDots = function(lst){
  defaults = list("tol" = 1.e-5,"maxit" = 150, "verbosity" = 0,
                  "maxstep" = 10,"justbeta" = FALSE, "method" = "Brent")
  lst = lst[!sapply(lst,is.null)] # avoid 'if arg. of length zero' error
  if(length(lst)==0) return(defaults)
  if(max(lengths(lst))>1) stop("drm: options list should contain only length-1 vectors.")
  selected = defaults
  modes = c("numeric","integer","integer","numeric","logical","character")
  names(lst) = tolower(names(lst))
  matchOpt = match(names(lst),names(defaults),NA)
  if(length(matchOpt)){
    for(i in 1:length(matchOpt)){
      if(!is.na(matchOpt[i])){
        modei = match.fun(paste0("as.",modes[matchOpt[i]]))
        selected[matchOpt[i]] = modei(lst[i])  # setting user-defined options for C++ algorithm  
      }
    }
  }
  if(selected[['tol']] < 1.0e-10 || 
     selected[['maxit']] <= 1 || 
     selected[['maxstep']] < 1.e-10 ||
     any(is.na(selected))){
    stop("Bad options passed in fitOptions. Ensure TOL, maxit, and maxStep are positive.")
  }
  selected
}

#' (Multivariate) Density Ratio Model
#'
#' Fits a semi-parametric density ratio model and returns an object of the corresponding class.
#'
#' \code{drm} fits a semi-parametric \emph{density ratio model} under the assumption that the outcomes follow an
#' exponential family distribution.  The probability density function is
#' \deqn{f(y|X)=\frac{dF_0(y)\exp\left(y * X^{\top}\beta\right)}{\int\exp\left(z * X^{\top}\beta\right)dF_{0}(z)}}{f(y|X) = f_0(y)*exp(y'Xb)/Integral(f0(z)*exp(z\cdot Xb)dz)}
#' for a baseline distribution function \eqn{F_0} with density/p.m.f. \eqn{f_0}
#' The functionality is intended to match that of \code{lm}; however, note that arguments \code{weights} and \code{offset} are not supported.
#' Model fitting is done via Newton's method with the inverse-Hessian approximation scheme of Broyden, Flecher, Goldfarb, and Shanno, equivalent to \code{optim}'s BFGS.
#' @param formula  A formula specifying the response and predictors, in the same manner as \code{lm}, \code{glm}, etc.
#' @param data  A data frame from which to extract the variables indicated in the formula
#' @param subset optional parameter for subsetting the data in argument \code{data}, following \code{lm} or \code{glm}
#' @param na.action What to do with missing values. See correponding documentation for \code{glm} (or \code{model.frame})
#' @param contrasts Optional argument specifying contrasts for factor covariates. See documentation for \code{stats::model.matrix}
#' @param fitOptions An optional named list which may be populated with control parameters. Named members may include:
#' \code{tol}, the convergence criterion. This means slightly different things depending on whether \code{method="Brent"} or not.
#' In the first case, convergence is defined as \deqn{2*\|f_{old}-f_{new}\|\leq\|f_{old}\|+\|f_{new}\|+\epsilon}
#' Otherwise, we directly compare the Euclidean norm of the gradient to \code{tol}.
#' \code{maxit}, maximum number of iterations
#' \code{verbosity}, controlling diagnostic output shown during model fitting (higher gives more output and zero gives none);
#' \code{maxStep}, a parameter passed to the line search algorithm which may be shrunk 
#' \code{justBeta}, indicating whether to only fit a point estimate
#' \code{method}, if "Brent", uses Brent's method to find a suitable objective function decrease in the current search direction.
#' Otherwise, uses a cubic approximation to fulfill the Wolfe conditions. Both methods use the BFGS Quasi-Newton algorithm to pick
#' candidate directions. The 'Brent' method seems to be a little more stable for this problem, but both should yield the same estimates.
#' Defaults are \code{list(1.0e-4,150,0,10,FALSE,"Brent")}, respectively.
#' Other elements in \code{fitOptions} will not raise an error, but they are ignored. Names do not need to match in a case-sensitive manner.
#' @return A DRM object.  Use \code{summary(x)} or \code{plot(x)} to obtain summary information or \code{names(x)} to view all fields, 
#' where \code{x} is a drm object. It is an S4 object, so the slots may be explored as usual.
#'
#' 
#' @seealso \code{\link[mdrm:summary-DRM_-method]{summary}}, \code{\link[mdrm:plot-DRM_-method]{plot}}, \code{\link[mdrm:names-DRM_-method]{names}}
#' @export
drm = function(formula,data,subset,na.action,contrasts=NULL,fitOptions=list()){
  cl = match.call()   # this one is just used to return the call along with the model summary
  mf = match.call(expand.dots = FALSE)
  m = match(c("formula", "data", "subset", "na.action"), names(mf), 0L)  # unlike lm(), no offset or weights!
  mf = mf[c(1L, m)]
  mf$drop.unused.levels = TRUE
  mf$na.action = "na.omit" # no method of dealing with missing values here
  mf[[1L]] = quote(stats::model.frame)
  mf = eval(mf, parent.frame())  # creates the model frame
	mt = attr(mf, "terms")   # passed into model.matrix() instead of a formula
	# for DRM, we exclude the intercept for the input model matrix, but use it when creating the model matrix so that factors
	# have the usual baseline constrast interpretation. Non-standard setting of 'contrasts' argument could cause an issue here.
	attr(mt,"intercept") = 1
	if(is.empty.model(mt)){ stop("Cannot continue with an empty model: please check input formula.") }
	yy = as.matrix(model.response(mf, "numeric"))
	if(is.null(colnames(yy))){ colnames(yy) = paste0("Y",1:ncol(yy)) }
	xx = model.matrix(mt, mf, contrasts)
	if("(Intercept)" %in% colnames(xx)) xx = xx[,-c(which(colnames(xx)=="(Intercept)"))]  # explicitly remove intercept
	if(is.null(colnames(xx))){ colnames(xx) = paste0("X",1:ncol(xx)) }
  defOpt = parseDots(fitOptions)
	Obj = DRM_(X = xx, Y = yy)  # instantiate class member
	res = fitdrm(Obj@Y,Obj@X,TOL=defOpt$tol,MAXIT=defOpt$maxit,verb=defOpt$verbosity,maxStep=defOpt$maxstep,
		justBeta = defOpt$justbeta,method = defOpt$method)  # "internal" fitting function
	if(length(res) == 1L){
		stop("An error occurred in model fitting. Perhaps re-scale the data or try exporing fitOptions.")
	}
	Obj@U = res$U
	Obj@beta = rbind(res$b0,as.matrix(res$beta))
	Obj@Fhat = res$F
	Obj@nUniq = res$nUniq
	Obj@N = nrow(Obj@X)
	Obj@alpha = res$alpha
	Obj@logLik = res$logLik
	Obj@sdbeta = rbind(res$b0sd,as.matrix(res$sdbeta))
	Obj@sdF = res$sdF
	Obj@covHat = res$vcov
	Obj@resid = res$resid
	Obj@iterations = res$iterations
	colnames(Obj@U) = colnames(Obj@Y)
	Obj@p.value=2*(1-pnorm(abs(Obj@beta)/Obj@sdbeta))
	Obj@betaC = rbind(res$b0,res$beta%*%(cov(Obj@resid))) # intercept does not need scaling
	Obj@user.call = cl
	Obj@is_fitted = TRUE
	return(Obj)
}

valDRM = function(object){return(TRUE)}  # this could be expanded

DRM_ = setClass(Class = "DRM_",
	slots = c(X = "matrix",
	          Y = "matrix",
	          U = "matrix", 
	          nUniq = "numeric",
	          beta = "matrix",
	          Fhat = "numeric", 
	          N = "numeric",
	          resid = "matrix",
	          conv = "integer",
	          preProcess = "logical",
	          sdbeta = "matrix",
	          sdF = "numeric",
	          covHat = "matrix",
	          p.value = "matrix",
	          logLik = "numeric",
	          iterations = "integer",
	          alpha = "numeric",
	          betaC = "matrix",
	          user.call = "call",
	          errMsg = "character",
	          is_fitted = "logical"
	),
	prototype = list(preProcess = FALSE,conv = 1L,errMsg = "",is_fitted = FALSE),
	validity = valDRM
)

# notice that for the base R function 'print', the main argument is 'x'. This HAS TO MATCH when 
# overloading print for objects of a new class. Likewise the name of summary()'s argument is 'object'...

#' Print an object of class 'DRM'
#'
#' prints a brief summary of an object retured from calling \code{drm}.  See \code{\link{names}} for a more thorough object description.
#' @param x an object returned from calling \code{drm}
#' @return Information printed to screen.
#' @export
setMethod(f = "print", signature = "DRM_", 
	definition = function(x){
	  if(x@is_fitted){
	    cat("\nAn object of class 'DRM'\n");
	    cat("Call:\n\t"); print(x@user.call)
	    cat(paste0("\n# observations: ",x@N,"\n"))
	    cat(paste0("Unique Outcomes: ",length(x@Fhat),"\n"))
	    cat("\nRegression coefficients: \n")
	    print(round(x@beta,digits=3))
	  } else {
	    cat('\nThis object has not been fit yet.\n')
	  }
	})

#' List the names (attributes) of a DRM object
#'
#' Equivalent to \code{names(d)} when d has class data.frame, lm, etc (i.e. when it is a list).
#' Use x$name to see the contents of the named attribute; for example x$sdbeta 
#' will return the matrix of estimated standard deviations of the model's \deqn{\beta}{beta} coefficients.
#' @param x Object of class 'DRM'
#' @return A list of object names
#' @seealso \code{\link{drm}}, \code{\link[mdrm:summary-DRM_-method]{summary}}
#' @export
setMethod(f = "names", signature = "DRM_",
	definition = function(x){ return(slotNames(x)) })

# show() is just an alias of print()

setMethod(f = "show",signature = "DRM_",
	definition = function(object){
		print(object)
	})
	
#' Allow the familiar `$` accessor with S4 objects
#'
#' While S4 slots are naturally available for DRM_ objects, most users are more familiar with operator$, and it may be more convenient.
#'
#' @param x Object of class 'DRM'
#' @param name a valid name (a member of \code{names(x)})
#' @seealso \code{\link[mdrm:names-DRM_-method]{names}}
setMethod(f = `$`, signature = "DRM_",
	function(x,name){
		return(slot(x,name))
	})
	
#' Get the covariance matrix of model parameters
#'
#' This is just a wrapper which returns object@covHat, to keep a consistent interface.
#'
#' @param object a fitted drm object
#' @param ... more arguments (for consistency with the S3 generic function). They're ignored.
#' @return The variance-covariance matrix of parameter estimates. The top left and bottom right blocks contain the covariance
#' of the jump sizes \eqn{\hat{F}} and the mean parameters \eqn{\hat{\beta}}, respectively.
#' @rdname vcov-method
#' @export
setMethod(f = "vcov",signature = "DRM_",
	definition = function(object,...){
		return(object@covHat)
	})

#' Extract log-likelihood
#' 
#' Provides an S3-generic method for getting log-likelihood from objects of DRM class
#' 
#' @param object a fitted drm object
#' @param ... extra arguments to satisfy method dispatch; are ignored
#' @rdname logLik-method
#' @export
setMethod(f = "logLik",signature = "DRM_",
          definition = function(object,...){
            return(object@logLik)
          })

#' Extract Model Residuals
#'
#' Provides a convenient accessor to model residuals
#' 
#' @param object a fitted drm object
#' @param ... extra arguments to satisfy method dispatch; are ignored
#' @return Model residuals, a matrix of size \eqn{N\times Q} (naturally, of size \code{dim(Y)}). Only raw residuals are available
#' @rdname residuals-method
#' @export
setMethod(f = "residuals",signature = "DRM_",
          definition = function(object,...){
            return(object@resid)
          })

#' Extract Model Residuals
#'
#' Provides a convenient accessor to model residuals
#' 
#' @param object a fitted drm object
#' @param ... extra arguments to satisfy method dispatch; are ignored
#' @return Model residuals, a matrix of size \eqn{N\times Q} (naturally, of size \code{dim(Y)}). Only raw residuals are available
#' @rdname resid-method
#' @export
setMethod(f = "resid",signature = "DRM_",
          definition = function(object,...){
            return(object@resid)
          })


#' Get the sample size used in model fitting
#' 
#' This is mostly for interface with the sandwich package. Just returns \code{object@N}
#'
#' @param object a fitted drm object
#' @param ... more arguments (for consistency with the S3 generic function). It is ignored.
#' @return The sample size N of this model
#' @rdname nobs-method
#' @export
setMethod(f = "nobs",
          signature = "DRM_",
          definition = function(object,...){
            return(object@N)
          })

#' Get model coefficients
#' 
#' It is probably just as easy to use the slot names directly, but for consistency this method is available, too
#' 
#' @param object a fitted drm object
#' @param ... a list; the named argument 'which' is parsed and may take values 'alpha', 'beta', or 'Fhat'.
#' In this case, only that subset of parameters is returned.
#' otherwise (by default), returns a list with both.
#' @return A matrix or vector (if argument \code{which} is used), or a list with components \code{alpha}
#' and \code{beta}
#' @rdname coef-method
#' @export
setMethod(f = "coef",signature = "DRM_",
          definition = function(object,...){
            argL = list(...)
            if("which" %in% names(argL)){
              sn = argL[["which"]][1L]
              if(sn %in% slotNames(object)) return(slot(object,sn))
            }
            list("beta" = object@beta,"alpha" = object@alpha,"Fhat" = object@Fhat)
          })

#' Get model coefficients
#' 
#' It is probably just as easy to use the slot names directly, but for consistency this method is available, too
#' 
#' @param object a fitted drm object
#' @param ... a list; the named argument 'which' is parsed and may take values 'alpha' or 'beta'. In this case, only that subset of parameters is returned.
#' otherwise (by default), returns a list with both.
#' @return A matrix or vector (if argument \code{which} is used), or a list with components \code{alpha}
#' and \code{beta}
#' @rdname coefficients-method
#' @export
setMethod(f = "coefficients",signature = "DRM_",
          definition = function(object,...){
            coef(object,...) # alias for coef()
          })

#' Extract Model Fitted Values
#' 
#' Identical to \code{object@Y - resid(object)}.
#' 
#' @param object a fitted drm object
#' @return The fitted values of the response \code{Y}.
#' @export
setMethod(f = "fitted",signature = "DRM_",
          definition = function(object,...){
            object@Y - object@resid
          })

sig_star = function(pv){
  res = rep("",length(pv))
  res[pv <= 0.10] = "."
  res[pv <= 0.05] = "*"
  res[pv <= 0.01] = "**"
  res[pv <= 0.001] = "***"
  res
}
	
makeLine = function(w,contents){
	if(length(w) == 1){ w = rep(w,length(contents)) }
	contents = substr(contents,1,w)
	chs = nchar(contents)
	extraSp = w - chs + 1 # guarantee at least one space between columns 
	lnn = paste0(contents[1],paste0(rep(" ",extraSp[1]),collapse="")) # first is left-aligned
	for(i in 2:length(contents)){ # rest are right-aligned
		newStr = paste0(paste0(rep(" ",extraSp[i]),collapse=""),contents[i])
		lnn = paste0(lnn,newStr)
	}
	lnn
}

#' Summary table of model coefficients
#'
#' Coefficient estimates, standard deviations, z-scores and p-values for the \emph{parametric} parameters \eqn{\hat{\beta}}
#'
#' Asymptotically, the non-parametric MLE coefficients \deqn{\hat{\beta}}{beta-hat} follow a zero-mean normal distribution.
#' The summary table provides the p-values for a two-sided Wald test.
#' @param object an object of class 'DRM'
#' @return A summary table of Wald tests.
#' @seealso \code{\link[mdrm:lr_test-method]{lr_test}} for testing multiple coefficients via a likelihood-ratio test.
#' @export

setMethod(f = "summary", signature = "DRM_",
	definition = function(object){
		cFmt = "%.3f" # controls rounding
		Q = NCOL(object@beta)
		nameSpacer = max(15,min(max(nchar(colnames(object@X))),25)) # name column is between 15 and 25 characters wide
		pvalu = matrix(ifelse(object@p.value < 1.0e-6,"<1.0e-6",sprintf(cFmt,object@p.value)),ncol=Q)
		sigs = matrix(sig_star(c(object@p.value)),ncol=Q)
		zscores = matrix(sprintf(cFmt,object@beta/object@sdbeta),ncol=Q)
		cat("Call:\n"); print(object@user.call)
		cat("\nDensity ratio model fit:\n\nCoefficients:\n")
		ww = c(nameSpacer+2,10,10,8,8)  # widths of each column in output table
		cat(makeLine(ww,c("","Estimate","Std. Error","z value","Pr(>|z|)")))
		Ynames = colnames(object@Y)
		Xnames = c("Intercept",colnames(object@X))
		for(j in seq_along(Ynames)){
			cat(paste0("\nOutcome: ",Ynames[j],"\n"))
			for(i in seq_along(Xnames)){
				cat(makeLine(ww,c(Xnames[i],sprintf(cFmt,object@beta[i,j]),sprintf(cFmt,object@sdbeta[i,j]),zscores[i,j],pvalu[i,j])))
				cat(paste0("  ",sigs[i,j],"\n")) # significance codes
			}
			cat("\n")
		}
		rcov = cov(object@resid)
		cat("Residual (co)variance:\n")
		prmatrix(format(rcov,digits = 3),rowlab=rep('',NROW(rcov)),collab = rep('',NCOL(rcov)),
		         quote = FALSE)
		cat("\n\nLog-likelihood:",sprintf(cFmt,logLik(object)))
		cat("\n# of parameters in model:",Q*NROW(object@beta) + NROW(object@U) - 1,"\n")
	})

#' Likelihood ratio test
#'
#' A function to perform likelihood ratio test for likelihood ratio tests
#'
#' This is a generic function which is implemented as an S4 method for the DRM class
#'
#' Calculates the result of a \deqn{\chi^2}{chi-squared} test for \deqn{H_0:}{H0:} specified elements of \deqn{\beta}{beta} are zero
#' vs. the alternative in which the likelihood is evaluated at \deqn{\hat{\beta}_{MLE}}{beta_MLE}.
#'
#' @param obj An object of the appropriate class
#' @param tstMat The parameter values corresponding to the null hypothesis (assuming that the MLE is the alternative)
#' @param literal A boolean indicating whether the provided values of tstMat should be used verbatim or as indicators of which elements of
#' the MLE to set to zero.
#' @return A list detailing the results of the likelihood ratio test.
#' @docType methods
#' @rdname lr_test-method
#' @aliases lr_test,character,ANY-method
#' @export
setGeneric(name = "lr_test", def = function(obj,tstMat,literal){standardGeneric("lr_test")})

#' Likelihood ratio test for drm object
#'
#' Method for drm class objects to conduct a likelihood ratio test.  Uses the likelihood function of the density ratio model to conduct inference on \eqn{\beta}
#'
#' @param obj A fitted drm object
#' @param tstMat The parameter values for \eqn{\beta}
#' @param literal A boolean indicating whether the provided values of tstMat should be used verbatim or as indicators of which elements of
#' the MLE to set to zero.
#' @seealso \code{\link[mdrm:summary-DRM_-method]{summary}}
setMethod(f = "lr_test", signature = "DRM_",
	definition = function(obj,tstMat,literal = FALSE){
	  stopifnot(obj@is_fitted)
		if(!isTRUE(all.equal(dim(tstMat),dim(obj@beta))))
		  stop(paste0("Error: dimension of test matrix must have ",nrow(obj@beta),
		              "rows and ",ncol(obj@beta)," columns."))
		tstB = tstMat
		if(!literal){
			tstMat = ifelse(as.numeric(tstMat) == 0,0,1) 
			tstB = obj@beta*tstMat
		}
		tstDF = sum(tstB!=obj@beta)
		LLalt = LLcalc(obj,tstB)
		ts = 2*(obj@logLik-LLalt)
		pval = 1-pchisq(ts,df=tstDF)
		return(list("Beta_MLE" = obj@beta,"Beta_test" = tstB,"test_statistic" = ts, "df" = tstDF, "p_value" = pval))
})

# the log-likelihood is not difficult to calculate here in R:
drm_loglik = function(oo,tstB){
  stopifnot(oo@is_fitted)
	XB = oo@X%*%tstB
	LL = sum(oo@Y*XB)+sum(log(oo@Fhat)*oo@nUniq)
	E = exp(oo@U%*%t(XB))*oo@Fhat
	LL - sum(log(colSums(E)))
}

#' Bootstrap estimation of residuals' (co)variance
#'
#' In the case of scaling the estimated coefficients \eqn{\beta} by the residual covariance, estimate
#' the standard errors of the scaled parameters via non-parametric bootstrap
#'
#' Word of caution: if the data set and/or requested number of bootstrap samples is very large, we may overflow RAM when summarize is
#' FALSE since storage is on the order \code{B*N*(ncol(X)+ncol(Y))}.  When summarize = TRUE, the factor of \code{N} disappears.
#'
#' @param formula Model formula specifying response(s) and covariates
#' @param data See corresponding documentation for \code{\link{drm}}
#' @param subset See corresponding documentation for \code{\link{drm}}
#' @param na.action  See corresponding documentation for \code{\link{drm}}
#' @param B number of bootstrap replications
#' @param summarize Should this function return the entire length-\code{B} sequences of fitted \eqn{\hat{\beta}}{\beta} and \eqn{\hat{\Sigma}}{\Sigma}, or just the empirical standard deviation of \eqn{\beta\Sigma}?
#' @param contrasts See corresponding documentation for \code{\link{drm}}
#' @param fitOptions See corresponding documentation for \code{\link{drm}}
#' @return If \code{summarize=TRUE}, a matrix of standard deviations. Otherwise, two lists of length \code{B}: the fitted \eqn{\beta} and corresponding \eqn{\Sigma}
#' @seealso \code{\link{drm}}
#' @rdname drmBootstrap-method
#' @export
drmBootstrap = function(formula,data,subset,na.action,B,summarize = FALSE,
                        contrasts = NULL,fitOptions = list()){
  mf = match.call(expand.dots = FALSE)
  m = match(c("formula", "data", "subset", "na.action"), names(mf), 0L)  # unlike lm(), no offset or weights!
  mf = mf[c(1L, m)]
  mf$drop.unused.levels = TRUE
  mf[[1L]] = quote(stats::model.frame)
  mf = eval(mf, parent.frame())
	mt = attr(mf, "terms")
	if(is.empty.model(mt)){ stop("Cannot continue with an empty model: please check input formula.") }
	yy = as.matrix(model.response(mf, "numeric"))
	xx = model.matrix(mt, mf, contrasts)
	defOpt = parseDots(fitOptions)
	B = as.integer(B)[1L]
	if(is.na(B) || B <= 1L) stop("Number of bootstrap samples should exceed 1")
	res = drmBoot(yy,xx,B,defOpt$tol,defOpt$maxit,
	              defOpt$verbosity,defOpt$method)
	if(isTRUE(summarize)){
		Q = NCOL(res$betas[[1]])
		bAdj = sapply(1:B,FUN = function(i){res$betas[[i]]%*%res$sigmas[[i]]})
		bsd = apply(bAdj,1,sd)
		return(matrix(bsd,ncol = Q,byrow = FALSE))
	}
	res # return the lists if we don't summarize
}