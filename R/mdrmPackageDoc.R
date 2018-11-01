#  Documentation for the mdrm package

#' mdrm: a package for fitting a semi-parametric \emph{density ratio model}
#'
#' This package's principal function (for now) is \code{drm}, which fits a model
#' for data which follows an exponential family distribution.
#'
#' @section Functions:
#' drm 
#' lr_test
#' summary 
#' plot 
#' print
#' vcov
#' drmBoot
#' 
#' @section Basic usage:
#' The interface follows that of the usual R model fitting: given a \code{data.frame} with outcomes and regressors, the main fitting function \code{drm}
#' is called with a formula and data argument and returns a fitted object. The usual S3 generics (plot, summary, coef, etc.) can be used to
#' extract information about the fit.
#'
#' @docType package
#' @name mdrm
NULL