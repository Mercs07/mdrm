//#pragma once

#ifndef rcppeigen_utils

#define rcppeigen_utils
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

/*
 * Various utility functions which are not specific to the particular model
 * implemented in mdrm
 */

using namespace Eigen;
using MatRef = const Eigen::Ref<const Eigen::MatrixXd>&;
using VecRef = const Eigen::Ref<const Eigen::VectorXd>&;

// set the values of x to zero at locations specified by ii. The purpose of this is to
// fit a model with some parameter(s) fixed at a certain value
void setZeros(Eigen::Ref<Eigen::VectorXd>,const Eigen::Ref<const Eigen::VectorXi>&);

// test for infinite or NAN - this is typically a signal that something has gone horribly wrong in the optimization
bool any_nan(VecRef x);

// sample 'size' integers uniformly distributed on 1,...,N (inclusive!) for selecting nonparametric bootstrap samples
// note: Rcpp includes the R distributions, which can be called as in R and return an Rcpp::NumericVector
VectorXi sample(const int N,const int size);

/*take an input matrix (each row is an observation) and a same-length location vector.
 return the matrix of unique elements and set the location vector so uniq.row(loc[i])=Y.row(i)
 in the context of algorithm-running this function returns U and sets nU as a side-effect
 uniqCnt gets 'trimmed' at the end.
 */
MatrixXd uniq(MatRef Y,VectorXi& uniqCnt,Eigen::Ref<Eigen::VectorXi> inxMap);


template<int T> // T is -1 for Matrix (Eigen::Dynamic) or 1 (VectorXd) (or another constexpr)
void printrng(Eigen::Matrix<double,Eigen::Dynamic,T>& M,const char* name){
  Rcpp::Rcout << "Range of " << name << ": " << M.minCoeff() << ", " << M.maxCoeff() << std::endl;
}

// considered using an Eigen::Map to just 'view' the same data as a vector; however, if we'd like to
// put several matrices into a single vector, they do need to be copied to maintain contiguous storage
// so this futzing around is necessary even if everything is const
VectorXd vectorize(MatRef M);

MatrixXd unVectorize(VecRef vec,const int ncols);

// the rankUpdate method maps X -> X + alpha*A*A^T for scalar (double) alpha
inline MatrixXd AtA(MatRef A){
  const int p(A.cols());
  return MatrixXd(p,p).setZero().selfadjointView<Lower>().rankUpdate(A.adjoint());
}

// compute OLS beta using LLT decomposition.
MatrixXd betaFit(MatRef Y,MatRef X);

// sample covariance matrix
MatrixXd cov(MatRef X);

#endif // end rcppeigen_utils