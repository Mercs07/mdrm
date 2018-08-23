// version replacing shitty NR vector classes with Eigen classes - should only compile in conjunction with Eigen
// not using namespace Eigen here just to clarify what's been modified

#include <Eigen/Dense>
#include <limits>
#include "util.h"

using VecRef = const Eigen::Ref<const Eigen::VectorXd>&;
using VeciRef = const Eigen::Ref<const Eigen::VectorXi>&;
using rVec = Eigen::Ref<Eigen::VectorXd>;


// calculation of an opaque 'test' value which occurs in both lnsarch
inline double calc_test(VecRef x,VecRef y){
  Eigen::ArrayXd z = x.array().abs()/y.array().abs().max(1.0);
  return z.maxCoeff();
}

const static double TOLX{std::pow(std::numeric_limits<double>::epsilon(),0.67)};

/*
 * Arguments: xold are the current parameter values, and fold is the current objective function value
 * g is the gradient at xold
 * p is the candidate search direction, but usually we need to contract it towards xold to avoid overshooting
 * x holds the updated point upon return (the goal is func(x) < func(xold))
 */
template <class T>
void line_search(VecRef xold, const double fold, VecRef g,rVec p,rVec x,double &f,const double stpmax,bool &check,T &func){
  const double alpha = 1.0e-4, p_norm = p.norm(); // alpha is min. *relative* decrease we require
  if(p_norm > stpmax) p *= stpmax/p_norm;
  const double slope = g.dot(p);
  if(slope >= 0.) throw("Roundoff problem in lnsrch."); // is it really a """roundoff""" problem?
  // allocate the obscene number of temporary variables...
  double rhs1,rhs2,tmplam,a,lambda = 1.0,alam2 = 0.0,b,disc,f2 = 0.0;
  const int maxit = 200; // provide some guard against infinite loop
  check = false;
  const double min_lambda = TOLX/calc_test(p,xold);
  
  for(int i=0;i<maxit;i++){
    x = xold + lambda*p;
    f = func(x);
    if(lambda < min_lambda){
      x = xold;
      check = true; // flag that the objective function can't be reliably increased in the given search direction, at least with this convoluted approach
      return;
    } else if(f <= fold + alpha*lambda*slope){ // what we want: sufficient function decrease
      return;
    } else {  // keep searching: calculate a polynomial interpolation to guess where a local min. might be
      if(lambda == 1.0) tmplam = -slope/(2.0*(f-fold-slope)); // first time through; quadratic approximation
      else {                                                // otherwise, do a cubic interpolation when we have sufficient data
        double l2 = lambda*lambda, al2 = alam2*alam2;
        rhs1 = f - fold - lambda*slope;
        rhs2 = f2 - fold - alam2*slope;
        a = (rhs1/l2 - rhs2/al2)/(lambda-alam2);
        b = (-alam2*rhs1/l2 + lambda*rhs2/al2)/(lambda-alam2);
        if (a == 0.0) tmplam = -slope/(2.0*b);
        else {
          disc = b*b - 3.0*a*slope;
          if (disc < 0.0) tmplam = 0.5*lambda;
          else if (b <= 0.0) tmplam = (-b + sqrt(disc))/(3.0*a);
          else tmplam = -slope/(b + sqrt(disc));
        }
        if(tmplam > 0.5*lambda) tmplam = 0.5*lambda;
      }
    }
    alam2 = lambda;
    f2 = f;
    lambda = std::max(tmplam,0.1*lambda); // limit the rate of shrinkage to 90% per round
  }
  throw("line search exceeded max. iterations");
}

// the object funcd should have overloaded operator() taking a VecRef and a method 'df' taking VecRef, Ref<VectorXd>, storing the new gradient 
// in the second argument.
// the last argument here is just for the special case when we want to force some coefficient(s) to equal zero. See setZeros() in util.h
template <class T>
void dfpmin(Eigen::Ref<Eigen::VectorXd> p, const double gtol, int &iter, double &fret,const T &funcd,
            const Eigen::VectorXi zc = Eigen::VectorXi::Constant(1,-1),int verb = 0){
  
  const int ITMAX = 200;
  const double MAX_STEP = 10.0;
  bool check, zeros = zc(0) >= 0; // do we need to futz around with setting things to zero? (if fitting a constrained model)
  if(zeros) setZeros(p,zc);
  double test,rho,k; // rho and k are added for use in BFGS update
  int n = p.size();
  Eigen::VectorXd dg(n),g(n),hdg(n),pnew(n), Hy(n);
  Eigen::VectorXd xi(-g);
  Eigen::MatrixXd hessian( Eigen::MatrixXd::Identity(n,n) );  // stores the BFGS approximation to the Hessian matrix
  double fp = funcd(p);  // objective function value
  funcd.df(p,g);
  if(zeros) setZeros(g,zc);
  double stpmax = MAX_STEP*std::max(sqrt(p.dot(g)),double(n));
  
  for(int its=0;its<ITMAX;its++){
    iter = its;
    line_search(p,fp,g,xi,pnew,fret,stpmax,check,funcd); // line search which updates pnew and fret
    if(any_nan(pnew)){
      throw("non-finite values encountered in dfpmin!");
    }
    fp = fret;
    if(zeros) setZeros(pnew,zc); // this also causes xi(zc) zero
    xi = pnew - p;
    p = pnew;
    test = calc_test(xi,p);
    if(test < TOLX) return;
    dg = g; // save the old gradient
    funcd.df(p,g); // update g
    if(zeros) setZeros(g,zc); // will cause dg to have zero at index zc also
    if(verb > 0) Rcpp::Rcout << "Iteration " << its << " log-lik = " << -funcd(p) << " and gradient norm = " << g.norm() << std::endl;
    
    test = calc_test(g,p)/std::max(1.0,std::abs(fret));
    if(test < gtol) return;
    
    dg = g - dg; // difference in gradients: dg now stores new - old
    
    // BFGS update
    Hy = dg.adjoint()*hessian.selfadjointView<Eigen::Lower>();
    rho = 1./dg.dot(xi);
    k = rho*rho*dg.dot(Hy) + rho;
    hessian.selfadjointView<Eigen::Lower>().rankUpdate(xi,Hy,-rho); // rank 2 update with distinct u and v. Note capitalization format
    hessian.selfadjointView<Eigen::Lower>().rankUpdate(xi,k); // rank 'K' update with a K-column matrix u (and alpha scalar)
    xi = -g.adjoint()*hessian.selfadjointView<Eigen::Lower>(); // replaces next 3 lines (which also fail to exploit symmetry of hessian)
  }
  throw("too many iterations in dfpmin");
}
