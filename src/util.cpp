// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppEigen.h>
#include "util.h" // header-only functions that *other* functions in here depend on

/*
 * Various utility functions which are not specific to the particular model
 * implemented in mdrm
 */

using namespace Eigen;
using MatRef = const Eigen::Ref<const Eigen::MatrixXd>&;
using VecRef = const Eigen::Ref<const Eigen::VectorXd>&;
using MapMxd = Eigen::Map<Eigen::MatrixXd>;
using MapVxd = Eigen::Map<Eigen::VectorXd>;

// set the values of x to zero at locations specified by ii (SORTED)
void setZeros(Eigen::Ref<Eigen::VectorXd> x,const Eigen::Ref<const Eigen::VectorXi>& ii){
  const int L = x.size() - 1; int ix;
  for(int i=0;i<ii.size();i++){
    ix = ii(i);
    if(ix >= 0 && ix < L) x(ix) = 0.; // skip out of range
  }
}

void setValues(Eigen::Ref<Eigen::VectorXd> x,const Eigen::Ref<const Eigen::VectorXi>& ii,VecRef v){
  const int L = x.size() - 1; int ix;
  if(ii.size() != v.size()) Rcpp::stop("setValues: Mismatched input sizes");
  for(int i=0;i < ii.size();i++){
    ix = ii(i);
    if(ix >= 0 && ix < L) x(ix) = v(i); // skip out of range
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
// though it may be cleaner (and its certainly faster here) to use a std::random instance
VectorXi sample(const int N,const int size){
  VectorXi res(size);
  double u;
  const double n{static_cast<double>(N)}; // Rf_runif parameter
  for(int i=0;i<size;i++){
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

// overloaded version which does not bother with tracking indices
MatrixXd uniq(MatRef Y,VectorXi& uniqCnt){
	const int N = Y.rows();
	uniqCnt.setZero();
	MatrixXd U  = MatrixXd::Random(N,Y.cols());  // NOTE: for 'safety', do a random initialization
	int curry = 0; bool match;
	for(int i=0;i<N;i++){
		match = false;
		for(int j=0;j<curry;j++){
			if(Y.row(i)==U.row(j)){ // we could put an epsilon-close comparison here to make a fuzzy comparison
				uniqCnt(j)++;
				match = true;
				break;
			}
		}
		if(!match){
			U.row(curry) = Y.row(i);
			uniqCnt(curry++)++; // get the index for nU first, then increment it (postfix)
		}
	}
	uniqCnt.conservativeResize(curry);
	return U.topRows(curry);
}

// sample covariance matrix
MatrixXd cov(MatRef X){
  const int N = X.rows();
  MatrixXd sscp( X.transpose()*X );
  VectorXd xmu = (1./N)*X.colwise().sum();
  sscp -= N*xmu*xmu.transpose();
  return (1./(N-1))*sscp;
}

VectorXd vectorize(MatRef M){
  const int P = M.rows(), Q = M.cols();
  VectorXd res(P*Q);
  for(int i=0;i<Q;i++){
    res.segment(i*P,P) = M.col(i);
  }
  return res ;
}

MatrixXd unVectorize(VecRef vec,const int ncols){
  const int r = vec.size()/ncols;
  if(r <= 1) return vec.transpose();
  MatrixXd mat(r,ncols);
  for(int i=0;i<ncols;i++){
    mat.col(i) = vec.segment(i*r,r);
  }
  return mat;
}

MatrixXd betaFit(MatRef Y,MatRef X,bool add_intercept = false){
  MatrixXd X1;
  if(add_intercept){
    X1 = MatrixXd(X.rows(),X.cols()+1); // initial estimate needs X and a column of ones
	  X1.col(0) = VectorXd::Constant(X.rows(),1.0);  // intercept
	  X1.rightCols(X.cols()) = X;
  } else {
    X1 = X;
  }
  const Eigen::LLT<MatrixXd> thellt(AtA(X1));
  return thellt.solve(X1.adjoint()*Y).bottomRows(X.cols());
}