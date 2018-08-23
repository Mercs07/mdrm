#pragma once

#include <Eigen/Dense>
#include <Rcpp.h>

/*
 * Various utility functions which are not specific to the particular model
 * implemented in mdrm
 */

using namespace Eigen;
using MatRef = const Eigen::Ref<const Eigen::MatrixXd>&;
using VecRef = const Eigen::Ref<const Eigen::VectorXd>&;

// set the values of x to zero at locations specified by ii. The purpose of this is to
// fit a model with some parameter(s) fixed at a certain value
void setZeros(Eigen::Ref<Eigen::VectorXd> x,const Eigen::Ref<const Eigen::VectorXi>& ii){
  const int L = x.size() - 1; int ix;
  for(auto i=0;i < ii.size();++i){
    ix = ii(i);
    if(ix > L || ix < 0) continue; // ignore out of range instead of assuming or freaking out
    x(ix) = 0.;
  }
}

// test for infinite or NAN
bool any_nan(VecRef x){
  bool has_nan = (x.array() != x.array()).any();
  bool has_inf = ((x-x).array() != (x-x).array()).any();
  return has_nan || has_inf;
}

// sample 'size' integers uniformly distributed on 1,...,N (inclusive!) for selecting nonparametric bootstrap samples
// note: Rcpp includes the R distributions, which can be called as in R and return an Rcpp::NumericVector
VectorXi sample(const int N,const size_t size){
  VectorXi res(size);
  double u;
  const double n = static_cast<double>(N); // Rf_runif parameter
  for(auto i=0;i<size;++i){
    u = ::Rf_runif(0.,n);
    res(i) = std::ceil(u);
  }
  return res;
}

/*take an input matrix (each row is an observation) and a same-length location vector.
 return the matrix of unique elements and set the location vector so uniq.row(loc[i])=Y.row(i)
 in the context of algorithm-running this function returns U and sets nU as a side-effect
 uniqCnt gets 'trimmed' at the end.
 */
MatrixXd uniq(MatRef Y,VectorXi& uniqCnt,Eigen::Ref<Eigen::VectorXi> inxMap){
  const int N = Y.rows(), P = Y.cols();
  uniqCnt.setZero();
  MatrixXd U = MatrixXd::Random(N,P);
  int curry = 0; bool match;
  for(int i=0;i<N;i++){
    match = false;
    for(int j=0;j<curry;j++){
      if(Y.row(i)==U.row(j)){ // we could put an epsilon-close criterion here to make a fuzzy comparison
        uniqCnt(j)++;
        match = true;
        inxMap(i) = j;
        break;
      }
    }
    if(!match){
      U.row(curry) = Y.row(i);
      inxMap(i) = curry;
      uniqCnt(curry++)++;
    }
  }
  uniqCnt.conservativeResize(curry);
  return U.block(0,0,curry,P);
}

// [[Rcpp::export]]
Rcpp::List uniq_counts(const Rcpp::NumericMatrix X){
  using MapMxd = Eigen::Map<Eigen::MatrixXd>;
  const MapMxd xx(Rcpp::as<MapMxd>(X));
  VectorXi uc(xx.rows()),uinx(xx.rows());
  MatrixXd ux = uniq(xx,uc,uinx);
  return Rcpp::List::create(
    Rcpp::Named("U") = ux,
    Rcpp::Named("uc") = uc;
  )
}

// sample covariance matrix
MatrixXd cov(MatRef X){
  const int N = X.rows();
  MatrixXd sscp( X.transpose()*X );
  VectorXd xmu = (1./N)*X.colwise().sum();
  sscp -= N*xmu*xmu.transpose();
  return (1./(N-1))*sscp;
}

inline VectorXd vectorize(MatRef M){
  const int P = M.rows(), Q = M.cols();
  VectorXd res(P*Q);
  for(int i=0;i<Q;i++){
    res.segment(i*P,P) = M.col(i);
  }
  return res ;
}

inline MatrixXd unVectorize(VecRef vec,const int ncols){
  const int r = vec.size()/ncols;
  if(r <= 1) return vec.transpose();
  MatrixXd mat(r,ncols);
  for(int i=0;i<ncols;i++){
    mat.col(i) = vec.segment(i*r,r);
  }
  return mat;
}

// the rankUpdate method maps X -> X + alpha*A*A^T for scalar (double) alpha
inline MatrixXd AtA(MatRef A){
  const int p(A.cols());
  return MatrixXd(p,p).setZero().selfadjointView<Lower>().rankUpdate(A.adjoint());
}

// compute OLS beta using LLT decomposition.
MatrixXd betaFit(const MatrixXd &Y,const MatrixXd &X){
  const Eigen::LLT<MatrixXd> thellt(AtA(X)); // compute the Cholesky decomposition of X'X
  return thellt.solve(X.adjoint()*Y);
}