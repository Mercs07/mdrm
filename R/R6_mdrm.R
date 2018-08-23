# a simple R6 class for mdrm
# the main purpose is to separate calculations which need only be performed once per data set
# such as extracting unique counts and identifying parameter sizes
# such that relatively simple methods can be exposed

library(R6)
library(Rcpp)
sourceCpp("C:/Users/skm/Dropbox/Rpackage/mdrm/src/util.cpp") # for uniq_counts
library(numDeriv)

drm = R6Class(classname ="drm",
              public = list(
                nParam = NULL,
                initialize = function(Y,X){
                  stopifnot(NROW(X)==NROW(Y),is.numeric(X),is.numeric(Y))
                  private$X = as.matrix(X)
                  private$Y = as.matrix(Y)
                  private$n = NROW(X)
                  private$p = NCOL(X)
                  private$q = NCOL(Y)
                  ucs = uniq_counts(Y)
                  private$U = ucs$U
                  private$uc = ucs$uc
                  private$m = NROW(ucs$U)
                  self$nParam = private$m + private$p*private$q - 1L
                },
                LL = function(theta){
                  stopifnot(length(theta)==self$nParam)
                  alpha = c(theta[1:(private$m-1)],0)
                  beta = matrix(tail(theta,private$p*private$q),ncol = private$q)
                  xb = private$X%*%beta
                  expt = exp(sweep(private$U%*%t(xb),MARGIN = 1,STATS = alpha,FUN = `+`))
                  (sum(private$uc*alpha) + sum(private$Y*xb) - sum(log(colSums(expt))))/private$n
                },
                LL2 = function(theta){
                  if(length(theta)!=self$nParam) stop("LL2: Bad inputs")
                  drm_ll(private$X,private$Y,private$U,private$uc,theta)
                },
                grad = function(theta){
                  stopifnot(length(theta)==self$nParam)
                  pq = private$p*private$q
                  na = private$m - 1
                  alpha = c(theta[1:na],0)
                  beta = matrix(tail(theta,pq),ncol = private$q)
                  xb = private$X%*%beta
                  expt = exp(sweep(private$U%*%t(xb),MARGIN = 1,STATS = alpha,FUN = `+`)) # m x n
                  denoms = colSums(expt) # length n
                  dAlpha = (private$uc - rowSums(sweep(expt,MARGIN = 2,STATS = denoms,FUN = `/`)))[1:na]
                  YEU = private$Y - sweep(t(expt)%*%private$U,MARGIN = 1,STATS = denoms,FUN=`/`) # n x q
                  dBeta = double(pq)
                  for(i in 1:pq){
                    ri = (i-1)%%private$p + 1; ci = (i-1)%/%private$p + 1
                    dBeta[i] = sum(private$X[,ri]*YEU[,ci])
                  }
                  c(dAlpha,dBeta)/private$n
                },
                grad2 = function(theta){
                  if(length(theta)!=self$nParam) stop("grad2: Bad inputs")
                  drm_grad(private$X,private$Y,private$U,private$uc,theta)
                },
                hessian = function(theta){
                  if(length(theta)!=self$nParam) stop("grad2: Bad inputs")
                  drm_hess(private$X,private$Y,private$U,private$uc,theta)
                }
              ),
              # private members are mostly just data
              private = list(
                X = NULL,
                Y = NULL,
                U = NULL,
                uc = integer(0),
                n = NULL,m = NULL,p = NULL,q = NULL
              )
)

# test
library(MASS)
library(microbenchmark)
library(optimx)
P = 4; Q = 3; N = 120
X = matrix(runif(P*N,-1,1),ncol=P)

BB = matrix(round(rnorm(P*Q),1),ncol=Q)
Y = t(apply(X%*%BB,1,mvrnorm,n=1,Sigma=matrix(0.4,Q,Q)+0.67))

m0 = drm$new(Y,X)
th0 = runif(m0$nParam,-1,1)
b0 = c(coef(lm(Y~X))[-c(1),])
th0[(m0$nParam-length(b0)+1):m0$nParam] = b0
m0$LL(th0)
m0$LL2(th0)

H0 = m0$hessian(th0)

microbenchmark(
  m0$grad(th0),
  m0$grad2(th0),
  times = 100
)


g1 = m0$grad(th0)
g2 = m0$grad2(th0)
ghat = numDeriv::grad(m0$LL2,th0)
heshat = numDeriv::hessian(m0$LL2,th0)
range(H0-heshat)
plot(m0$grad(th0),ghat,col = ifelse(1:m0$nParam>N-1,"steelblue","salmon"),pch=19)
abline(a=0,b=1,lty=6,col='#3e3a3f')

# warning: supplying a 'hess' function induces a very expensive eval
# even though BFGS should not use it!
drm_mle = optimx(par = th0,fn = m0$LL2,gr = m0$grad2,
                 #hess = m0$hessian,
                 hessian = FALSE,
                 method = "BFGS",itnmax = 2e3,
                 control = list(maximize = TRUE,trace=2L))

th_hat = t(drm_mle[1:length(th0)])
cbind(c(BB),tail(th_hat,P*Q)) # beta-hat estimate is a miserable failure!

