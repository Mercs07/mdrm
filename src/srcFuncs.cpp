// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppEigen.h> 
#include <string>
#include <cmath>  // std::ceil

#include "util.h" // some helper functions
#include "linemins.h" // two line search methods; prefer using "Brent" for stability

using namespace Eigen;

using MapVxi = Eigen::Map<Eigen::VectorXi>;
using MapMxi = Eigen::Map<Eigen::MatrixXi>;
using MapVxd = Eigen::Map<Eigen::VectorXd>;
using MapMxd = Eigen::Map<Eigen::MatrixXd>;
using MatRef = const Eigen::Ref<const Eigen::MatrixXd>&;
using VecRef = const Eigen::Ref<const Eigen::VectorXd>&;


// Object to hold the data plus all current values
class mdrm {
public:

  const MatrixXd X, Y; // data members
  const int n, p, q, verbosity;
  VectorXi Uinx;
  const MatrixXd U;
  const VectorXd uc;    // counts of unique values
  const int m, nParam;
  
  mdrm() = delete;

  mdrm(MatRef xx,MatRef yy,int verb) : X(xx), Y(yy), n(X.rows()), p(X.cols()), q(Y.cols()),
    verbosity(verb), Uinx(n), U(uniq(Y,Uinx)), uc(Uinx.cast<double>()), m(U.rows()),
    nParam(p*q + m - 1) {
    if(n != Y.rows()){ Rcpp::stop("mdrm: Number of outcomes doesn't match number of covariates."); }
  }

  double operator()(VecRef th) const { // negative log-likelihood, scaled by sample size
    const MatrixXd XB( X*unVectorize(th.tail(p*q),q) );
    VectorXd alp(m);
    alp.head(m-1) = th.head(m-1); alp(m-1) = 0.;
    // likelihood is called often enough and fairly simple, so we can avoid O(n^2) malloc with this simple for loop
    // df() could also be re-written thusly but it's a lot more complicated since we use E in three different calcuations
    double lik = (Y.cwiseProduct(XB)).sum() + uc.dot(alp);
    ArrayXd ip(m);
    for(int i=0;i<n;++i){
        ip = U*(XB.row(i).transpose()) + alp;
        lik -= std::log(ip.exp().sum());
    }
    return (-1.0/n)*lik;
  }
  
  void df(VecRef th,Eigen::Ref<VectorXd> gg) const; // first arg: current parameters. second arg: vector which stores gradient
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

void mdrm::df(VecRef th,Eigen::Ref<Eigen::VectorXd> gg) const { // negative gradient, scaled by sample size
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

// [[Rcpp::export]]
Rcpp::NumericMatrix yhat(Rcpp::NumericMatrix yy,Rcpp::NumericMatrix xx,Rcpp::NumericVector theta){
  const MapMxd X(Rcpp::as<MapMxd>(xx));
  const MapMxd Y(Rcpp::as<MapMxd>(yy));
  // kloodgy workaround, but we don't need to worry about efficiency here
  const int nX = X.rows();
  MatrixXd Xuse(MatrixXd::Zero(Y.rows(),X.cols()));
  Xuse.topRows(nX) = X;
  const MapVxd th(Rcpp::as<MapVxd>(theta));
  const mdrm mm(Xuse,Y,0);
  if(th.size() != mm.nParam) Rcpp::stop("yhat: input parameters theta is incorrectly sized.");
  MatrixXd res = mm.yhat(th).topRows(nX);
  return Rcpp::wrap(res);
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
      tmp = (Y.col(b).array() - (Ef*U.col(b)).array()/denoms);
      res(pos) = X.col(a).dot(tmp);
    }
  }
  return res/n;
}

MatrixXd mdrm::scores(VecRef th) const { // work in progress
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
  if(F.minCoeff() < 0.) Rcpp::stop("Input parameters don't form a distribution!");
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
// BUT the input F is the same size as alpha, i.e. D.m - 1, and the last jump size is calculated
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
      H(M + i,z) = t2 - t1;
      H(z,M + i) = H(M + i,z);
    }
  }
  // lower block: d2l/dbeta,dbeta
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
  if(th.size() != D.nParam){
    std::string errmsg = "drmHess: parameter vector has ";
    errmsg += std::to_string(th.size()) + ", but the model should have " + std::to_string(D.nParam) + ".";
    Rcpp::stop(errmsg);
  }
  MatrixXd res(th.size(),th.size());
  if(useF){
    // ensure jump size parameters are OK:
    VectorXd tF(th.head(D.m-1));
    if(tF.minCoeff() <= 0 || tF.sum() >= 1.0) Rcpp::stop("Input parameter isn't a distribution! Note that last jump size is calculated");
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
  MatrixXd bhat( betaFit(D.Y,D.X,true).bottomRows(D.p) ); // remove intercept row
  double maxAbs = bhat.array().abs().maxCoeff();
  if(maxAbs > 2.) bhat *= 2./maxAbs; // betas capped at |2|
  if(D.verbosity > 1){ Rcpp::Rcout << "Initial betas:\n" << bhat << std::endl; }
  return vectorize(bhat);
}

//[[Rcpp::export]]
Rcpp::List fitdrm(Rcpp::NumericMatrix inY,Rcpp::NumericMatrix inX,Rcpp::IntegerVector zero_index,
  double TOL = 0.,int MAXIT = 100,int verb = 0,const std::string method = "Brent",bool justBeta = false,const std::string conv = "func"){
  using Rcpp::Named;
  const MapMxd X(Rcpp::as<MapMxd>(inX));
  const MapMxd Y(Rcpp::as<MapMxd>(inY));
  MapVxi zc(Rcpp::as<MapVxi>(zero_index));
  const mdrm D(X,Y,verb);
  VectorXd theta0(D.nParam);
  theta0.head(D.m - 1) = 0.25*ArrayXd::Random(D.m - 1); // randomly initialized alpha in (-0.25, 0.25) range.
  theta0.tail(D.p*D.q) = beta0(D); // initialize beta to linear approximation
  if(TOL < 1.0e-10) TOL = 1.0e-5*sqrt(D.nParam); // though TOL has a default, it's interpretation is different for the two methods
  if(zc.maxCoeff() > -1){
    zc += VectorXi::Constant(zc.size(),D.m-1); // shift args to zero out up to beta position *if* we actually want to zero any
    if(verb > 0) Rcpp::Rcout << "Zero indices @ " << zc.transpose() << "; length of parameter vector is " << D.nParam << std::endl;
  }
  funcMin opt_param;
  try { // the whole minimization should no longer throw exceptions, however we should expect the expected which is an unexpected failure
    opt_param = dfpmin(theta0,D,zc,method,verb,TOL,TOL,conv);
  } catch(std::exception& _ex_){
    //forward_exception_to_r(_ex_); // NOTE this does not unwind destructors safely, see https://github.com/RcppCore/Rcpp/issues/753
    Rcpp::stop(_ex_.what()); // "What, I say what has gone wrong?" - Foghorn Leghorn
  } catch(std::string errMsg){
    Rcpp::stop(errMsg.c_str());
  } catch(const char* errMsg){
    Rcpp::stop(errMsg);
  } catch(...){
    ::Rf_error("c++ exception (unknown reason!)");
  }
  theta0 = opt_param.arg_min;
  if(opt_param.error != dfpmin_error::NONE){
    Rcpp::CharacterVector errmsg(1);
    errmsg[0] = dfpmin_err_msg::messages[opt_param.error-1];
    return Rcpp::List::create(Named("error") = errmsg,Named("parameters") = Rcpp::wrap(theta0));
  }

  const MatrixXd resid = D.Y - D.yhat(theta0);
  VectorXd Fhat = A2F(theta0.head(D.m-1));
  MatrixXd betahat = unVectorize(theta0.tail(D.p*D.q),D.q);
  
  VectorXd b0 = D.U.transpose()*(Fhat*D.uc); // intercept
  VectorXd og(D.nParam);
  D.df(theta0,og);
  if(justBeta){  // don't bother with calculating Hessian and standard errors
    return Rcpp::List::create(
      Named("beta") = Rcpp::wrap(betahat),
      Named("b0") = Rcpp::wrap(b0),
      Named("F") = Fhat,
      Named("alpha") = Rcpp::wrap(theta0.head(D.m-1)),
      Named("residuals") = resid,
      Named("iterations") = opt_param.iterations,
      Named("U") = D.U,
      Named("nUniq") = D.uc,
      Named("logLik") = -opt_param.min_value
    );
  }
  // calculating covariance parameters
  double nd = static_cast<double>(D.n);
  //MatrixXd Ha(D.nParam,D.nParam), // standard errors for both alpha and F
  MatrixXd Hf(D.nParam,D.nParam);
  //d2LL(D,theta0,Ha); // observed information with alpha
  VectorXd thF = theta0;
  thF.head(D.m-1) = Fhat.head(D.m-1);
  d2LLF(D,thF,Hf); // observed information with F
  // VectorXd vara = -Ha.inverse().diagonal();
  MatrixXd covF = -Hf.inverse()/nd;
  VectorXd varF = covF.diagonal();
  if(any_nan(varF) || varF.minCoeff() < 0.){
    Rcpp::CharacterVector errmsg(1);
    errmsg[0] = "Bad values in varF!! Returning for diagnosis\n";
    return Rcpp::List::create(Named("error") = errmsg,
                              Named("parameters") = Rcpp::wrap(theta0),
                              Named("varF") = Rcpp::wrap(varF));
  }
  VectorXd b0sd(VectorXd::Zero(D.q));
  MatrixXd HF = covF.topLeftCorner(D.m-1,D.m-1), u1(D.U.topRows(D.m-1));
  const double f_sum = Fhat.head(D.m-1).sum();
  const VectorXd f_psum{HF.colwise().sum()}; // first part of covariance calculation
  const VectorXd f_adj{f_sum*Fhat.head(D.m-1)}; // second part of covariance calculations
  // the full m\times m covariance of F can be fortified with the differences f_psum - f_adj; this leaves the bottom right corner
  // which is simply the sum of F's covariance matrix (and it is linearly dependent, naturally)
  for(int j=0;j<D.q;j++){
    MatrixXd TT = HF.array()*(u1.col(j)*u1.col(j).transpose()).array();
    double tmp = HF.sum() + D.U(D.m-1,j)*D.U(D.m-1,j)*f_sum; // NEEDS FIXING! more covariance conundrums
    b0sd(j) = tmp;
  }
  if(any_nan(b0sd) || b0sd.minCoeff() <= 0){
    Rcpp::stop("Error: bad values in b0sd! Returning for diagnosis");
    return Rcpp::List::create(Rcpp::Named("b0var") = b0sd);
  }
  b0sd = b0sd.array().sqrt();
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
    Named("iterations") = opt_param.iterations,
    Named("U") = D.U,
    Named("nUniq") = D.uc,
    Named("logLik") = -opt_param.min_value,
    Named("gradient") = Rcpp::wrap(-og)
  );
}


//[[Rcpp::export]]
Rcpp::List drmBoot(Rcpp::NumericMatrix y,Rcpp::NumericMatrix x,int nBoot,Rcpp::IntegerVector zero_index,
  double TOL = 0,int MAXIT = 100,int verb = 0,std::string method="Brent"){
  
  const MapMxd Y(Rcpp::as<MapMxd>(y));
  const MapMxd X(Rcpp::as<MapMxd>(x));
  const MapVxi zc(Rcpp::as<MapVxi>(zero_index));
  const mdrm D(X,Y,verb);
  if(nBoot < 2) Rcpp::stop("Invalid nBoot argument");
  VectorXd theta0 = 0.5*VectorXd::Random(D.nParam);
  theta0.tail(D.p*D.q) = beta0(D); // initialize beta to linear approximation
  VectorXi zci(zc.size());
  if(zc.maxCoeff() > -1) zci = zc + VectorXi::Constant(zc.size(),D.m-1); // shift args to zero out up to beta position *if* we actually want to zero any

  if(TOL < 1.0e-10) TOL = 1.0e-5*sqrt(D.nParam); // converge criterion defaults to a reasonable level but can be passed as a param (min. is 1^-10)
  int btry = 0,bdid = 0;
  
  funcMin res = dfpmin(theta0,D,zc,"Brent",0,1e-4,1e-4); // TOL need not be too tight here - this is just to get a good, reuseable beta estimate
  theta0 = res.arg_min;

  const VectorXd b0{ theta0.tail(D.p*D.q) };
  VectorXi rs(D.n); // resampling index
  MatrixXd XB(D.n,D.p), YB(D.n,D.q), tbeta(D.p,D.q), tcov(D.q,D.q), yh(D.n,D.q);
  Rcpp::List bHats(nBoot), sigHats(nBoot);
  funcMin resB;
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
    if(zc.maxCoeff() > -1) zci = zc + VectorXi::Constant(zc.size(),db.m-1); // shift args to zero out up to beta position *if* we actually want to zero any
    VectorXd th0 = 0.25*VectorXd::Random(db.nParam);
    th0.tail(db.p*db.q) = b0;
    try{
      resB = dfpmin(th0,db,zci,method,verb,TOL,TOL);
      OKfit = true;
    } catch(std::exception& _ex_){
      forward_exception_to_r(_ex_);
    } catch(std::string errMsg){
      Rcpp::stop(errMsg);
    } catch(const char* errMsg){
      Rcpp::stop(errMsg);
    } catch(...){
      ::Rf_error("c++ exception (unknown reason!)");
    }
    if(OKfit){
      th0 = resB.arg_min;
      yh = db.yhat(th0);
      bHats[bdid] = unVectorize(th0.tail(db.p*db.q),db.q);
      sigHats[bdid++] = cov(db.Y - yh);	
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("betas") = bHats,
    Rcpp::Named("sigmas") = sigHats
  );
}





	