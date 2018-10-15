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

/*
Rcpp::List uniq_counts(const Rcpp::NumericMatrix X){
  const MapMxd xx(Rcpp::as<MapMxd>(X));
  VectorXi uc(xx.rows()),uinx(xx.rows());
  MatrixXd ux = uniq(xx,uc,uinx);
  return Rcpp::List::create(
    Rcpp::Named("U") = ux,
    Rcpp::Named("uc") = uc
  );
}
*/

/*
double drm_ll(Rcpp::NumericMatrix xx,Rcpp::NumericMatrix yy,Rcpp::NumericMatrix uu,
	Rcpp::NumericVector ucnt,Rcpp::NumericVector theta){
		
	const MapMxd X(Rcpp::as<MapMxd>(xx)), Y(Rcpp::as<MapMxd>(yy)), U(Rcpp::as<MapMxd>(uu));
	const MapVxd uc(Rcpp::as<MapVxd>(ucnt)), th(Rcpp::as<MapVxd>(theta));
	const int n = X.rows(),p = X.cols(), q = Y.cols(), m = U.rows();
	const Map<const MatrixXd> beta(th.data() + (m-1),p,q); // view last p*q coefficients as matrix
	VectorXd alpha(m);
	alpha.head(m-1) = th.head(m-1); alpha(m-1) = 0.;
	const MatrixXd XB( X*beta );
	const ArrayXXd E( ((U*XB.transpose()).colwise() + alpha).array().exp() );  // m x n array
  double lik = (Y.cwiseProduct(XB)).sum() + uc.dot(alpha) -  (E.colwise().sum()).log().sum();
  return (1.0/n)*lik;
}
*/

/*
Rcpp::NumericVector drm_grad(Rcpp::NumericMatrix xx,Rcpp::NumericMatrix yy,Rcpp::NumericMatrix uu,
                             Rcpp::NumericVector ucnt,Rcpp::NumericVector theta){
  
  const MapMxd X(Rcpp::as<MapMxd>(xx)), Y(Rcpp::as<MapMxd>(yy)), U(Rcpp::as<MapMxd>(uu));
  const MapVxd uc(Rcpp::as<MapVxd>(ucnt)), th(Rcpp::as<MapVxd>(theta));
  const int n = X.rows(),p = X.cols(), q = Y.cols(), m = U.rows(), nth = theta.size();
  const Map<const MatrixXd> beta(th.data() + (m-1),p,q); // view last p*q coefficients as matrix
  VectorXd alpha(m);
  alpha.head(m-1) = th.head(m-1); alpha(m-1) = 0.;
  const MatrixXd XB( X*beta );
  const ArrayXXd E( ((U*XB.transpose()).colwise() + alpha).array().exp() );  // m x n array
  const ArrayXd denoms( E.colwise().sum() );  // n x 1
  const ArrayXd dalpha( ((E.rowwise()/denoms.transpose()).rowwise().sum()) );
  VectorXd gg(nth);
  gg.head(m-1) = (uc.array() - dalpha).head(m-1);  // first the alphas
  MatrixXd YEU = Y.array()-(E.matrix().transpose()*U).array().colwise()/denoms;  // n * q
  for(int z=0;z<p*q;z++){  // next the betas
    gg(m-1+z) = X.col(z%p).dot(YEU.col(z/p));
  }
  gg *= (1./static_cast<double>(n));
  return Rcpp::wrap(gg);
}
*/

/*
Rcpp::NumericMatrix drm_hess(Rcpp::NumericMatrix xx,Rcpp::NumericMatrix yy,Rcpp::NumericMatrix uu,
                             Rcpp::NumericVector ucnt,Rcpp::NumericVector theta){
  
  const MapMxd X(Rcpp::as<MapMxd>(xx)), Y(Rcpp::as<MapMxd>(yy)), U(Rcpp::as<MapMxd>(uu));
  const MapVxd uc(Rcpp::as<MapVxd>(ucnt)), th(Rcpp::as<MapVxd>(theta));
  const int n = X.rows(),p = X.cols(), q = Y.cols(), m = U.rows(), nth = theta.size();
  const int pq = p*q;
  const Map<const MatrixXd> beta(th.data() + (m-1),p,q); // view last p*q coefficients as matrix
  VectorXd alpha(m);
  alpha.head(m-1) = th.head(m-1); alpha(m-1) = 0.;
  const MatrixXd XB( X*beta );
  const ArrayXXd E( ((U*XB.transpose()).colwise() + alpha).array().exp() );  // m x n array
  const ArrayXd denoms( E.colwise().sum() );  // n x 1
  const ArrayXd d2( denoms*denoms );
  const ArrayXXd EU( E.matrix().transpose()*U );  // n * q array
  const ArrayXXd UA(U), XA(X);
  MatrixXd H(nth,nth);
  // upper block: d/a d/a
  for(int z=0;z<m-1;z++){
    for(int w=0;w<=z;w++){
      H(z,w) = (E.row(z)*E.row(w)/d2.transpose()).sum();
      H(w,z) = H(z,w);
    }
  }
  for(int z=0;z<m-1;z++) H(z,z) -= (E.row(z)/denoms.transpose()).sum(); //extra term on main diagonal
  // off-diagonals: d/a d/b
  ArrayXd xe(n), uue(n);
  int a,b,c,d;
  for(int w=0;w<m-1;w++){
    for(int z=0;z<pq;z++){
      a = z%p;
      b = z/p;
      int offix = m-1+z;
      xe = (E.colwise()*(UA.col(b)-U(w,b))).colwise().sum();
      H(offix,w) = ((XA.col(a)*E.row(w).transpose()*xe)/d2).sum();
      H(w,offix) = H(offix,w);
    }
  }
  // lower block: d/b d/b
  for(int y=0;y<pq;y++){
    a = y%p; b = y/p;
    for(int z=0;z<pq;z++){
      c = z%p; d = z/p;
      uue = (E.colwise()*(UA.col(b)*UA.col(d))).colwise().sum();
      // for(int i=0;i<D.n;i++){ uue(i) = (D.U.col(b).array()*D.U.col(d).array()*E.col(i)).sum();}
      H(m-1+y,m-1+z) = ((XA.col(a)*XA.col(c)*(EU.col(b)*EU.col(d)-uue*denoms))/d2).sum();
      H(m-1+z,m-1+y) = H(m-1+y,m-1+z);
    }
  }
  H *= (1.0/static_cast<double>(n));
  return Rcpp::wrap(H);
}
*/


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

MatrixXd betaFit(MatRef Y,MatRef X){
  const Eigen::LLT<MatrixXd> thellt(AtA(X)); // compute the Cholesky decomposition of X'X
  return thellt.solve(X.adjoint()*Y);
}