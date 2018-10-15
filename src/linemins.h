// line minimization routines. There are two approaches, here labeled
// 'brent_linemin' (Brent's method) and 'cubic_linemin' (using cubic approximation)
// former seems more stable for high-dimensional problems
// this version is for use with Rcpp, but the only depedency is printing traces/errors while fitting


#ifndef line_min_methods

#define line_min_methods
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "util.h"
#include <cmath> // std::abs
#include <limits>
#include <sstream>
#include <type_traits> // is_arithmetic

using Rcpp::Rcout; // if not compiling with Rcpp, can do 'using std::cout = Rcout;'
using std::endl;

using VecRef = const Eigen::Ref<const Eigen::VectorXd>&;
using MatRef = const Eigen::Ref<const Eigen::MatrixXd>&;
using refVec = Eigen::Ref<Eigen::VectorXd>;

// *****
// little helpers
// *****

// this is decidedly *not* the standard 'sign' function but is a tuft of vestigial kruft from NR
template<typename T>
inline T SIGN(T a,T b){
  static_assert(std::is_arithmetic<T>::value,"SIGN expects a single, numeric argument.");
	return b > 0. ? std::abs(a) : -std::abs(a);
}

// T is typically a double in this context: bookkeeping for functions 'brent' and 'mnbrak' below
// initial value of a is discarded while d is untouched.
template<typename T>
inline void SHIFT(T& a,T& b,T& c,const T& d){
	a=b; b=c; c=d;
}

// calculation of an opaque 'test' value which occurs in both linesearch
inline double calc_test(VecRef x,VecRef y){
  Eigen::ArrayXd z = x.array().abs()/y.array().abs().max(1.0);
  return z.maxCoeff();
}

/* defining declarations 
	These are variables which are shared between linmin and f1dim: f1dim takes a function of K args and views it as a function of 1 arg:
    f(delta) = f(theta + delta*gradient)
	the 'com' is for common.
	Both brent and mnbrak take function pointers as arguments, of the type double (*fptr)(double)
	in both cases, the actual function passed is f1dim - this is the one we modify to use VectorXd, etc.
  TODO: refactor this to a more sensible approach.
*/

static Eigen::VectorXd pcom, xicom;  // only needed for dfpmin

// the signature of this function (pointer) must match the signature of the input function pointer 'func' to brent_linemin below
// this is part of the NR kludge which needs refactoring
static double (*nrfunc)(VecRef); 

// 'x' is the contraction parameter alpha in the line search; for a fixed direction, we find an alpha which yields a sufficient decrease
inline double f1dim(double x){
	const VectorXd xt(pcom + x*xicom);
  return (*nrfunc)(xt);
}

// bracket an set of initial values with opposite signs for use in Brent's methods
void mnbrak(double *ax,double *bx,double *cx,double *fa,double *fb,double *fc,double (*func)(double));

// first 3 args are bounds attained from mnbrak
double brent(const double ax,const double bx,const double cx,const double tol,double (*f)(double),double *xmin);

// line minimization for the parameter updates. This is the part which can be very difficult to get right
void brent_linemin(Eigen::VectorXd&,Eigen::VectorXd&,double,double(*func)(VecRef));

// BFGS update of inverse Hessian approximation
void BFGS(Eigen::Ref<Eigen::MatrixXd> H,VecRef dth,VecRef dgr);

// the return value of dfpmin
struct funcMin{
  int iterations,error;
  double min_value;
  Eigen::VectorXd arg_min,gradient;
};

constexpr static double TOLX{std::pow(std::numeric_limits<double>::epsilon(),0.67)};

/* quasi-Newton Algorithm ('dfp' for Davidon-Fletcher-Powell, though the actual updating is BFGS)
 * This is the 'Brent' overload which will use brent_linemin as declared above
 *   p0 is the initial parameter value
 *   ftol controls how easy it is to declare victory and return an 'optimum'
 *   iter tracks how many iterations
 *   func and dfunc are objective function and gradient, respectively (so they should have the same signature)
 */
funcMin dfpmin(VecRef p0,double (*func)(VecRef),Eigen::VectorXd (*dfunc)(VecRef),double ftol = 0.0,int verb=0);

/*
 * Arguments: xold are the current parameter values, and fold is the current objective function value
 * g is the gradient at xold
 * p is the candidate search direction, but usually we need to contract it towards xold to avoid overshooting
 * x holds the updated point upon return (the goal is func(x) < func(xold))
 */
template <class T>
void cubic_linemin(VecRef xold, const double fold, VecRef g,refVec p,refVec x,double& f,double stpmax,bool& check,const T& func){

  constexpr static double alpha{1.0e-4}; 
  const double p_norm = p.norm(); // alpha is min. *relative* decrease we require
  if(stpmax <= 0.) stpmax = 1.0;
  if(p_norm > stpmax) p *= stpmax/p_norm;
  const double slope = g.dot(p);
  if(slope >= 0.){
    std::stringstream errMsg;
    errMsg << "Roundoff problem in cubic_linemin: slope is " << slope << "but should be negative!" << std::endl;
    throw(errMsg.str()); // is it really a """roundoff""" problem?
  }
  double rhs1,rhs2,tmplam,a,lambda = 1.0,alam2 = 0.0,b,disc,f2 = 0.0;
  const int maxit = 200; // provide some guard against infinite loop
  check = false;
  const double min_lambda = TOLX/calc_test(p,xold);
  
  for(int i=0;i<maxit;++i){
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

// the object funcd should have overloaded operator() taking a VecRef and a method 'df' taking VecRef, Ref<VectorXd>, 
// storing the new gradient in the second argument.
// ideally we could use concepts to enforce this
// 'zc' is just for the special case when we want to force some coefficient(s) to equal zero. See setZeros() in util.h
template <class T>
funcMin dfpmin(refVec p,const T& funcd, double gtol = 0.,int verb = 0,const Eigen::VectorXi zc = Eigen::VectorXi::Constant(1,-1)){
  
  constexpr int ITMAX = 200;
  constexpr double MAX_STEP = 10.0;   // this is very arbitrary and its not clear how to 'tune' it relative to the problem at hand
  bool check, zeros = zc(0) >= 0;     // do we need to futz around with setting things to zero? (if fitting a constrained model)
  if(zeros) setZeros(p,zc);
  double test, fp = funcd(p), fret;   // objective function value;
  int n = p.size(),error = 0,its;
  Eigen::VectorXd g(n),dg(n),pnew(n); // gradient, delta_gradient, new parameter values
  funcd.df(p,g);                      // initial ***negative*** gradient (since we're doing MLE via a minimization algorithm)
  Eigen::VectorXd xi{-g};             // initial search direction is -1*"gradient" which is...the actual gradient of likelihood
  Eigen::MatrixXd hessian{ Eigen::MatrixXd::Identity(n,n) };  // stores the BFGS approximation to the Hessian matrix
  
  if(zeros) setZeros(g,zc);
  double max_step = MAX_STEP*std::max(sqrt(p.dot(g)),double(n));
  Rcpp::Rcout << "Using max_step = " << max_step << std::endl;
  for(its=0; its<ITMAX; its++){
    cubic_linemin(p,fp,g,xi,pnew,fret,max_step,check,funcd); // line search which updates pnew and fret
    if(any_nan(pnew)){
      error = 1;
      break;
    }
    fp = fret;
    if(zeros) setZeros(pnew,zc); // this also causes xi(zc) zero
    xi = pnew - p;
    p = pnew;
    test = calc_test(xi,p);
    if(test < TOLX) break;
    dg = g; // save the old gradient
    funcd.df(p,g); // update g
    if(zeros) setZeros(g,zc); // will cause dg to have zero at index zc also
    if(verb > 0) Rcpp::Rcout << "Iteration " << its << " log-lik = " << -funcd(p) << " and gradient norm = " << g.norm() << std::endl;
    
    test = calc_test(g,p)/std::max(1.0,std::abs(fret));
    if(test < gtol) break;
    
    dg = g - dg; // difference in gradients: dg now stores new - old
    BFGS(hessian,xi,dg); // update (inverse) Hessian approximation
    xi = -g.adjoint()*hessian.selfadjointView<Eigen::Lower>(); // new search direction
  }
  if(its == ITMAX) throw("error state in dfpmin: too many iterations");
  if(error > 0) throw("error state in dfpmin: non-finite values");
  funcMin rv;
  rv.iterations = its;
  rv.error = 0;
  rv.arg_min = pnew;
  rv.min_value = fp;
  rv.gradient = g;
  return rv;
}

#endif // end line_min_methods