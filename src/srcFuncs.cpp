#include <RcppEigen.h>  // the documentation never mentions this header and it is included in RcppExports.cpp, but it is also necessary to include here
#include <string>
#include <cmath>  // std::ceil. Also includes rand() which causes R CMD check to complain about the RNG. just don't use the actual rand() function.

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp11)]]

// NR - based routines
#include "util.h"
#include "quasinewtonEigen.h"    // from 3rd edition of NR - fancy cubic interpolation which is probably totally unnecessary, viz. the large number of gradient-only optimizers
#include "brent_linmin_EigenR.h" // Brent's method with Golden Section search - more stable, seemingly, than the cubic interpolation approach above

using Rcpp::Named;
using namespace Eigen;

using MapVxi = Eigen::Map<Eigen::VectorXi>;
using MapMxi = Eigen::Map<Eigen::MatrixXi>;
using MapVxd = Eigen::Map<Eigen::VectorXd>;
using MapMxd = Eigen::Map<Eigen::MatrixXd>;
using MatRef = const Eigen::Ref<const Eigen::MatrixXd>&;
using VecRef = const Eigen::Ref<const Eigen::VectorXd>&;


// Object to hold the data plus all current values
class mdrm{
public:
  int n,p,q,nParam,m;
  bool initialized;
  double stpmax;  // the max. step size in lineSearch
  MatrixXd X,Y,U; // data members
  int verbosity;  // keep order of declaration consistent with order in constructor initializer list
  VectorXd uc;    // counts of unique values
  VectorXi Uinx;
  mdrm(){ initialized = false; } // this constructor used in conjunction with Brent method
  mdrm(MatRef xx,MatRef yy,int verb) : X(xx), Y(yy), verbosity(verb){
    n = X.rows();
    p = X.cols();
    q = Y.cols();
    if(n != Y.rows()){Rcpp::stop("mdrm: Number of outcomes doesn't match number of covariates.");}
    VectorXi uy(n);
    Uinx = VectorXi(n);
    U = uniq(Y,uy,Uinx);
    m = U.rows();
    uc = uy.cast<double>();
    nParam = p*q + m - 1;
    initialized = true; // not really needed here since this is used with void dfpmin() version
  }
  void init(MatrixXd xx,MatrixXd yy,int verb = 0){  // fake constructor pairing with the stub above
    X = xx; Y = yy;
    n = X.rows(); p = X.cols(); q = Y.cols();
    VectorXi uy(n);
    Uinx = VectorXi(n);
    U = uniq(Y,uy,Uinx);
    m = U.rows();
    uc = uy.cast<double>();
    nParam = p*q + m - 1;
    initialized = true;
    verbosity = verb;
  }
  double operator()(VecRef th) const { // negative log-likelihood, scaled by sample size
    const MatrixXd XB( X*unVectorize(th.tail(p*q),q) );
    VectorXd alp(m);
    alp.head(m-1) = th.head(m-1); alp(m-1) = 0.;
    const ArrayXXd E( ((U*XB.transpose()).colwise() + alp).array().exp() );  // m x n array
    double lik = (Y.cwiseProduct(XB)).sum() + uc.dot(alp) -  (E.colwise().sum()).log().sum();
    return (-1.0/n)*lik;
  }
  void df(VecRef th,Eigen::Ref<VectorXd> gg) const; // this signature is for compatibility with the NR Brent method
  VectorXd gradF(VecRef th) const;
  MatrixXd yhat(VecRef theta) const { // calculate predicted values for given parameters Use these to calculate residual covariance, goodness of fit, etc.
    const MatrixXd XB = X*unVectorize(theta.tail(p*q),q);
    VectorXd alp(m);
    alp.head(m-1) = theta.head(m-1); alp(m-1) = 0.;
    const MatrixXd E( ((U*XB.transpose()).colwise() + alp).array().exp() );  // m x n array
    const ArrayXd d( E.colwise().sum() ); // n denominators, summed over j=1,...,m
    return (E.transpose()*U).array().colwise()/d;
  }
  MatrixXd scores(VecRef) const; // n x nParam matrix of scores for use in multiplier bootstrap test.  Their colwise sum should equal the gradient
};

// one global class member for Brent method
mdrm G;

// wrappers for Brent method
double drmLL(VecRef theta){
  if(!G.initialized) Rcpp::stop("Attempt to call drmLL without initializing G");
  return G(theta);
}

VectorXd drmGrad(VecRef theta){
  VectorXd res(theta.size());
  G.df(theta,res);
  return res;
}

void mdrm::df(VecRef th,Eigen::Ref<VectorXd> gg) const { // negative gradient, scaled by sample size
  if(gg.size() != nParam){gg.resize(nParam);}
  const MatrixXd XB( X*unVectorize(th.tail(p*q),q) );
  VectorXd alp(m);
  alp.head(m-1) = th.head(m-1); alp(m-1) = 0.;
  const ArrayXXd E( ((U*XB.transpose()).colwise() + alp).array().exp() );  // m x n array
  const ArrayXd denoms( E.colwise().sum() );  // n x 1
  const ArrayXd dalp( ((E.rowwise()/denoms.transpose()).rowwise().sum()) );
  gg.head(m-1) = (uc.array() - dalp).head(m-1);  //first the alphas
  MatrixXd YEU = Y.array()-(E.matrix().transpose()*U).array().colwise()/denoms;  // n * q
  for(int z=0;z<p*q;z++){  //next the betas
    gg(m-1+z) = X.col(z%p).dot(YEU.col(z/p));
  }
  gg *= (-1./n);
}

VectorXd mdrm::gradF(VecRef th) const { // gradient with respect to F, for testing purposes with d2LLF
  const MatrixXd XB = X*unVectorize(th.tail(p*q),q);
  ArrayXd F(m);
  F.head(m - 1) = th.head(m - 1);
  F(m-1) = 1. - th.head(m-1).sum();
  VectorXd res(p*q + m-1);
  const ArrayXXd E = (XB*U.transpose()).array().exp(); // n x m
  const MatrixXd Ef = E.rowwise()*F.transpose();
  const ArrayXd denoms = E.matrix()*F.matrix(); // n x 1
  double fM = uc(m-1)/F(m-1);
  VectorXd pt1 = uc.head(m-1).array()/F.head(m-1).array() - fM;
  VectorXd pt2 = ((E.leftCols(m-1).colwise() - E.col(m-1)).colwise()/denoms).colwise().sum();
  res.head(m-1) = pt1 - pt2;
  VectorXd tmp(n);
  for(int a=0;a<p;a++){
    for(int b=0;b<q;b++){
      int pos = m - 1 + a%p + b/p;
      // Rcpp::Rcout << "(a,b) = (" << a << "," << b << "), pos = " << pos << std::endl;
      tmp = (Y.col(b).array() - (Ef*U.col(b)).array()/denoms);
      res(pos) = X.col(a).dot(tmp);
    }
  }
  return res/n;
}

MatrixXd mdrm::scores(VecRef th) const { // the only wrinkle here is that we need to match up elements of Y with their respective elements in U
  const MatrixXd XB( X*unVectorize(th.tail(p*q),q) );
  VectorXd alp(m);
  alp.head(m-1) = th.head(m-1); alp(m-1) = 0.;
  const ArrayXXd E( ((U*XB.transpose()).colwise() + alp).array().exp() );  // m x n array
  const ArrayXd denoms( E.colwise().sum() );
  const ArrayXXd Escl = E.rowwise()/denoms.transpose();
  MatrixXd res(n,nParam);
  const MatrixXd yHats( yhat(th) );
  const ArrayXXd resid(Y - yHats);
  res.leftCols(m-1) = -Escl.topRows(m-1).transpose();
  for(int i=0;i<n;i++){
    if(Uinx(i) < m-1) res(i,Uinx(i)) += 1.; // need to avoid the last jump size
  }
  for(int z=0;z<p*q;z++){
    res.col(z+m-1) = X.col(z%p).array()*resid.col(z/p); // X.col(a)*(Y.col(b) - \hat{Y}.col(b)
  }
  /*
   for(int z = p*q; z < nParam; z++){
   res.col(p*q+z) = 1. - Escl.row(z).transpose();
   }
   */
  return res;
}


// calculate log-likelihood at current beta, alpha values.
// check which ones of these to exclude from final product
// [[Rcpp::export]]
double LL(const Eigen::VectorXd th,const Eigen::MatrixXd X,const Eigen::MatrixXd Y){
  const mdrm D(X,Y,0);
  if(th.size() != D.nParam) Rcpp::stop("Wrong number of parameters in LL");
  return -D(th);
}

// calculate log-likelihood with beta, F values
// [[Rcpp::export]]
double LLF(const Eigen::VectorXd th,const Eigen::MatrixXd X,const Eigen::MatrixXd Y){
  const mdrm D(X,Y,0);
  VectorXd F(D.m);
  F.head(D.m-1) = th.head(D.m-1);
  F(D.m-1) = 1. - F.head(D.m-1).sum();
  if(F.minCoeff() <= 0) Rcpp::stop("Input parameters don't form a distribution!");
  MatrixXd beta(unVectorize(th.tail(D.p*D.q),D.q));
  MatrixXd XB(X*beta);
  double ll0 = (Y.array()*XB.array()).sum();
  VectorXd lnF = F.array().log();
  ll0 += lnF.dot(D.uc);
  ArrayXXd E = (D.U*XB.transpose()).array().exp();
  ArrayXXd EF = E.colwise()*F.array();
  ArrayXd d = EF.colwise().sum().log();
  ll0 -= d.sum();
  return ll0/D.n;
}

// expose gradients to R
// [[Rcpp::export]]
Rcpp::NumericVector gradA(const Eigen::VectorXd th,const Eigen::MatrixXd X,const Eigen::MatrixXd Y){
  const mdrm D(X,Y,0);
  if(th.size() != D.nParam){
    std::string errmsg = "gradA: parameter vector has ";
    errmsg += std::to_string(th.size()) + ", but the model should have " + std::to_string(D.nParam) + ".";
    Rcpp::stop(errmsg);
  }
  VectorXd gg(D.nParam);
  D.df(th,gg); // calculated via side effect
  return Rcpp::wrap(-gg); // df() calculates negative gradient
}

// [[Rcpp::export]]
Rcpp::NumericVector gradF(const Eigen::VectorXd th,const Eigen::MatrixXd X,const Eigen::MatrixXd Y){
  const mdrm D(X,Y,0);
  if(th.size() != D.nParam){
    std::string errmsg = "gradF: parameter vector has ";
    errmsg += std::to_string(th.size()) + ", but the model should have " + std::to_string(D.nParam) + ".";
    Rcpp::stop(errmsg);
  }
  VectorXd fp(th.head(D.m-1));
  if(fp.minCoeff() <= 0 || fp.sum() >= 1.) Rcpp::stop("Input parameters not a distribution!");
  VectorXd g = D.gradF(th);
  return Rcpp::wrap(g);
}

// Hessian with alpha parametrization
void d2LL(const mdrm& D,VecRef theta,Ref<MatrixXd> H){
  const MatrixXd XB( D.X*unVectorize(theta.segment(D.m-1,D.p*D.q),D.q) );
  VectorXd alp(D.m);
  alp.head(D.m-1) = theta.head(D.m-1); alp(D.m-1) = 0.;
  const ArrayXXd E( ((D.U*XB.transpose()).colwise() + alp).array().exp() ); // m x n array
  const ArrayXd denoms( E.colwise().sum() );  // denoms(i) = sum_{j=1}^{m} E_ij, i=1,...,n
  const ArrayXd d2( denoms*denoms );
  const ArrayXXd EU( E.matrix().transpose()*D.U );  // n * q array
  const int pq = D.p*D.q, M = D.m-1;
  const ArrayXXd UA(D.U), XA(D.X);
  // upper block: d/a d/a
  for(int z=0;z<M;z++){
    for(int w=0;w<=z;w++){
      H(z,w) = (E.row(z)*E.row(w)/d2.transpose()).sum();
      H(w,z) = H(z,w);
    }
  }
  for(int z=0;z<M;z++){ H(z,z) -= (E.row(z)/denoms.transpose()).sum(); } //extra term on main diagonal
  // off-diagonals: d/a d/b
  ArrayXd xe(D.n);
  int a,b,c,d;
  for(int w=0;w<M;w++){
    for(int z=0;z<pq;z++){
      a = z%D.p;
      b = z/D.p;
      xe = (E.colwise()*(UA.col(b)-D.U(w,b))).colwise().sum();
      H(M+z,w) = ((XA.col(a)*E.row(w).transpose()*xe)/d2).sum();
      H(w,M+z) = H(M+z,w);
    }
  }
  // lower block: d/b d/b
  ArrayXd uue(D.n);
  for(int y=0;y<pq;y++){
    a = y%D.p; b = y/D.p;
    for(int z=0;z<pq;z++){
      c = z%D.p; d = z/D.p;
      uue = (E.colwise()*(UA.col(b)*UA.col(d))).colwise().sum();
      // for(int i=0;i<D.n;i++){ uue(i) = (D.U.col(b).array()*D.U.col(d).array()*E.col(i)).sum();}
      H(M+y,M+z) = ((XA.col(a)*XA.col(c)*(EU.col(b)*EU.col(d)-uue*denoms))/d2).sum();
      H(M+z,M+y) = H(M+y,M+z);
    }
  }
  H*=(1.0/D.n);
}

// Hessian with respect to F and beta; assume the input vector has F, not alpha
// BUT the input F is the same size as alpha, i.e. D.m - 1. We add the last jump size
void d2LLF(const mdrm& D,VecRef theta,Ref<MatrixXd> H){
  const MatrixXd XB = D.X*unVectorize(theta.tail(D.p*D.q),D.q);
  ArrayXd F(D.m);
  const int P = D.p*D.q, M = D.m-1;
  F.head(M) = theta.head(M);
  F(M) = 1. - theta.head(M).sum();
  const ArrayXXd E((XB*D.U.transpose()).array().exp()); // n x m
  const ArrayXXd Ef( E.rowwise()*F.transpose() );  // exps_{i,j} = exp(U.row(j).dot(XB.row(i)))
  const ArrayXd denoms( E.matrix()*F.matrix() );
  const ArrayXd d2( denoms*denoms );
  const ArrayXXd EU( Ef.matrix()*D.U ); // n x q array; E_{a,b} = \sum_{j=1}^m E_{aj}U_{jb}
  const ArrayXXd Edif = E.leftCols(M).colwise() - E.col(M);
  const ArrayXXd Xa = D.X.array();
  H.setZero();
  int a,b,c,d;
  const double FFm = D.uc(M)/(F(M)*F(M));
  // upper block: d2l/dF,dF
  for(int a=0;a<M;a++){
    for(int b=0;b<=a;b++){
      H(a,b) = -FFm + (Edif.col(a)*Edif.col(b)/d2).sum();
      H(b,a) = H(a,b);
    }
    H(a,a) -= D.uc(a)/(F(a)*F(a));
  }
  // corners: d2l/dF,dbeta
  ArrayXd ud(D.n),tmp(D.n);
  double t1,t2;
  for(int z=0;z<M;z++){
    for(int i=0;i<P;i++){
      a = i%D.p;  b = i/D.p;
      tmp = D.U(z,b)*E.col(z) - D.U(M,b)*E.col(M);
      t1 = (Xa.col(a)*tmp/denoms).sum();
      t2 = (Xa.col(a)*Edif.col(z)*((Ef.matrix()*D.U.col(b)).array())/d2).sum();
      //Rcpp::Rcout << "z = " << z << ", (a,b) = (" << a << "," << b << "); t1 and t2: " << t1 << ", " << t2 << std::endl;
      H(M + i,z) = t2 - t1;
      H(z,M + i) = H(M + i,z);
    }
  }
  // lower block: d2l/dbeta,dbeta
  //VectorXd uu(D.m);
  ArrayXd uu(D.m);
  for(int y=0;y<P;y++){
    a = y%D.p;  b = y/D.p;
    for(int z=0;z<=y;z++){
      c = z%D.p;  d = z/D.p;
      uu = D.U.array().col(b)*(D.U.array().col(d));
      ud = (Ef.rowwise()*uu.transpose()).rowwise().sum();
      H(M + y,M + z) = (Xa.col(a)*D.X.col(c).array()*(EU.col(b)*EU.col(d)-denoms*ud)/d2).sum();
      H(M + z,M + y) = H(M + y,M + z);
    }
  }
  H *= (1.0/D.n);  // mean
}

// [[Rcpp::export]]
Rcpp::NumericMatrix drmHess(const Eigen::VectorXd th,const Eigen::MatrixXd X,const Eigen::MatrixXd Y,bool useF = true){
  const mdrm D(X,Y,0);
  if(th.size() != D.nParam){Rcpp::stop("Incompatibly sized parameter vector.");}
  MatrixXd res(th.size(),th.size());
  if(useF){
    // ensure jump size parameters are OK:
    VectorXd tF(th.head(D.m-1));
    if(tF.minCoeff() <= 0 || tF.sum() >= 1.0) Rcpp::stop("Input parameter isn't a distribution!");
    d2LLF(D,th,res);
  } else {
    d2LL(D,th,res);
  }
  return Rcpp::wrap(res);
}

VectorXd A2F(VecRef alpha){
  const int m = alpha.size();
  ArrayXd F(m+1);
  const ArrayXd ae(alpha.array().exp());
  const double denom = ae.sum() + 1.0;
  F.head(m) = ae/(denom);
  F(m) = 1.0/denom;
  return F;
}

// get an initial (linear model) estimate of betas. We hope at least sign is correct for univariate Y.
VectorXd beta0(const mdrm& D){
  MatrixXd X1(D.n,D.p+1); // initial estimate needs X and a column of ones
  X1.col(0) = VectorXd::Constant(D.n,1.0);  // intercept
  X1.rightCols(D.p) = D.X;
  MatrixXd bhat( betaFit(D.Y,X1).bottomRows(D.p) ); // remove intercept row
  double maxAbs = std::max(-bhat.minCoeff(),bhat.maxCoeff());
  if(maxAbs > 2.) bhat *= 2./maxAbs; // scale betas to within (-1,1)
  if(D.verbosity > 1){ Rcpp::Rcout << "The initial betas:\n" << bhat << std::endl; }
  return vectorize(bhat);
}

template<int T> // T is -1 (Eigen::Dynamic) or 1 (VectorXd)
void printrng(Eigen::Matrix<double,Eigen::Dynamic,T>& M,const char* name){
  Rcpp::Rcout << "Range of " << name << ": " << M.minCoeff() << ", " << M.maxCoeff() << std::endl;
}

//[[Rcpp::export]]
Rcpp::List fitdrm(const Eigen::MatrixXd Y,const Eigen::MatrixXd X,
                  double TOL = 0.,int MAXIT = 100,int verb = 0,double maxStep=10.,
                  bool justBeta=false,std::string method = "Brent"){
  const mdrm D(X,Y,verb);
  VectorXd theta0(D.nParam);
  theta0.head(D.m - 1) = 0.5*ArrayXd::Random(D.m - 1); // randomly initialized alpha in (-0.5, 0.5) range.
  theta0.tail(D.p*D.q) = beta0(D); // initialize beta to linear approximation
  if(TOL < 1.0e-10) TOL = 1.0e-5*sqrt(D.nParam); // though TOL has a default, it's interpretation is different for the two methods
  double MLE; // holds min. attained value
  int nit = 0;
  if(method=="Brent"){
    G.init(X,Y,verb);
    funcMin res = dfpmin(theta0,drmLL,drmGrad,TOL,verb);
    theta0 = res.arg_min;
    MLE = res.min_value;
    nit = res.iterations;
  } else { // cubic interpolation method, which may throw exceptions
    try{
      dfpmin(theta0,TOL,nit,MLE,D,VectorXi::Constant(1,-1),verb);
    } catch(std::exception& _ex_){
      forward_exception_to_r(_ex_);  // a weird issue here: forward_exception_to_r is defined within Rcpp but isn't in Rcpp namespace.
    } catch(const char* errMsg){
      Rcpp::stop(errMsg);
    } catch(...){
      ::Rf_error("c++ exception (unknown reason!)"); // the scope resolution operator :: with no LHS refers to global namespace (in case it was overridden within function scope?)
    }	
  }
  const MatrixXd resid = D.Y - D.yhat(theta0);
  Rcpp::Rcout << "Covariance matrix of residuals:\n" << resid << std::endl;
  printrng(theta0,'fitted parameters');
  VectorXd Fhat = A2F(theta0.head(D.m-1));
  printrng(Fhat,"Fhat");
  MatrixXd betahat = unVectorize(theta0.tail(D.p*D.q),D.q);
  Rcpp::Rcout << "beta-hat is :\n" << betahat << std::endl;
  // intercept
  VectorXd b0 = D.U.transpose()*Fhat;
  VectorXd og(D.nParam);
  D.df(theta0,og); // return gradient
  if(justBeta){  // don't bother with calculating Hessian and standard errors
    return Rcpp::List::create(
      Named("beta") = Rcpp::wrap(betahat),
      Named("b0") = Rcpp::wrap(b0),
      Named("F") = Fhat,
      Named("alpha") = Rcpp::wrap(theta0.head(D.m-1)),
      Named("residuals") = resid,
      Named("iterations") = nit,
      Named("U") = D.U,
      Named("nUniq") = D.uc,
      Named("logLik") = -MLE
    );
  }
  // calculating covariance parameters
  double nd = static_cast<double>(D.n);
  MatrixXd Ha(D.nParam,D.nParam),Hf(D.nParam,D.nParam); // standard errors for both alpha and F
  d2LL(D,theta0,Ha); // observed information with alpha
  printrng(Ha,"Ha");
  VectorXd thF = theta0;
  thF.head(D.m-1) = Fhat.head(D.m-1);
  printrng(thF,"thF");
  d2LLF(D,thF,Hf); // observed information with F
  printrng(Hf,"Hf");
  // VectorXd vara = -Ha.inverse().diagonal();
  MatrixXd covF = -Hf.inverse()/nd;
  VectorXd varF = covF.diagonal();
  printrng(varF,"varF");
  if(any_nan(varF)){
    Rcpp::Rcout << "Bad values in varF!!\n";
    return Rcpp::List::create(varF);
  }
  VectorXd b0sd(VectorXd::Zero(D.q));
  MatrixXd HU(D.m-1,D.m-1), u1(D.U.topRows(D.m-1));
  const double f_sum = covF.topLeftCorner(D.m-1,D.m-1).sum();
  for(int j=0;j<D.q;j++){
    HU = covF.topLeftCorner(D.m-1,D.m-1).array()*(u1.col(j)*u1.col(j).transpose()).array();
    double tmp = HU.sum() + D.U(D.m-1,j)*D.U(D.m-1,j)*f_sum;
    Rcpp::Rcout << "adding " << tmp << " to b0sd at index " << j << std::endl;
    b0sd(j) = tmp;
  }
  if(varF.minCoeff() <= 0 ||
     b0sd.minCoeff() <= 0) Rcpp::stop("Error: non-singular covariance!");
  if(any_nan(b0sd)){
    Rcpp::Rcout << "Bad values in b0sd!!\n";
    return Rcpp::List::create(b0sd);
  }
  b0sd = b0sd.array().sqrt();
  // VectorXd sda = vara.array().sqrt().head(D.m-1);
  VectorXd sdF(D.m); sdF.head(D.m-1) = varF.array().sqrt().head(D.m-1);
  sdF(D.m-1) = f_sum;
  ArrayXXd sdbeta = unVectorize(varF.tail(D.p*D.q),D.q);  // these should match regardless of parametrization?
  sdbeta = sdbeta.sqrt();
  return Rcpp::List::create(
    Named("beta") = Rcpp::wrap(betahat),
    Named("sdbeta") = Rcpp::wrap(sdbeta),
    Named("b0") = Rcpp::wrap(b0),
    Named("b0sd") = Rcpp::wrap(b0sd),
    Named("F") = Fhat,
    Named("sdF") = sdF,
    Named("alpha") = Rcpp::wrap(theta0.head(D.m-1)),
    Named("vcov") = Rcpp::wrap(covF),
    Named("residuals") = resid,
    Named("iterations") = nit,
    Named("U") = D.U,
    Named("nUniq") = D.uc,
    Named("logLik") = -MLE,
    Named("gradient") = Rcpp::wrap(-og)
  );
}


//[[Rcpp::export]]
Rcpp::List drmBoot(Rcpp::NumericMatrix y,Rcpp::NumericMatrix x,int nBoot,double TOL=0,int MAXIT = 100,
                   int verb = 0,std::string method="Brent"){
  
  const MapMxd Y(Rcpp::as<MapMxd>(y));
  const MapMxd X(Rcpp::as<MapMxd>(x));
  const mdrm D(X,Y,verb);
  VectorXd theta0 = 0.5*VectorXd::Random(D.nParam);
  theta0.tail(D.p*D.q) = beta0(D); // initialize beta to linear approximation
  if(TOL < 1.0e-10) TOL = 1.0e-5*sqrt(D.nParam); // converge criterion defaults to a reasonable level but can be passed as a param (min. is 1^-10)
  double MLE;
  int nit,btry = 0,bdid = 0;
  G.init(X,Y,verb); // use Brent method for initial fit just to get parameters in neighborhood
  funcMin res = dfpmin(theta0,drmLL,drmGrad,1.e-4,verb);
  theta0 = res.arg_min;
  // save the initial fit to use as a good initial value for re-fitting the data
  const VectorXd initTh = theta0;
  VectorXi rs(D.n); // resampling index
  MatrixXd XB(D.n,D.p), YB(D.n,D.q), tbeta(D.p,D.q), tcov(D.q,D.q), yh(D.n,D.q);
  Rcpp::List bHats(nBoot), sigHats(nBoot);
  bool OKfit;
  while(bdid < nBoot){ // don't flame out on the first exception; but if the error rate is too high, don't try forever
    btry++;
    OKfit = false;
    if(btry > 2*nBoot) Rcpp::stop("Excessive error rate in bootstrap fitting attempts. Perhaps data is not scaled well?");
    rs = sample(D.n,D.n);
    for(int i=0;i<D.n;i++){
      XB.row(i) = X.row(rs(i));
      YB.row(i) = Y.row(rs(i));
    }
    mdrm db(XB,YB,verb);
    if(method == "Brent"){
      G.init(XB,YB,verb);
      res = dfpmin(initTh,drmLL,drmGrad,TOL,verb);
      theta0 = res.arg_min;
      OKfit = true;
    } else {
      theta0 = initTh; // re-set since it will get changed at each instance
      try{
        dfpmin(theta0,TOL,nit,MLE,D,VectorXi::Constant(1,-1),verb);
        OKfit = true;
      } catch(std::exception& _ex_){
        //forward_exception_to_r(_ex_);  // a weird issue here: forward_exception_to_r is defined within Rcpp but isn't in Rcpp namespace.
        continue;
      } catch(const char* errMsg){
        //Rcpp::stop(errMsg);
        continue;
      } catch(...){
        //::Rf_error("c++ exception in drmBoot (unknown reason!)"); // :: with no LHS refers to global namespace (in case it had been was overridden locally?)
        continue;
      }
    }
    if(OKfit){
      yh = db.yhat(theta0);
      bHats[bdid] = unVectorize(theta0.tail(D.p*D.q),D.q);
      sigHats[bdid++] = cov(db.Y - yh);	
    }
  }
  return Rcpp::List::create(Rcpp::Named("betas") = bHats,Rcpp::Named("sigmas") = sigHats);
}





	