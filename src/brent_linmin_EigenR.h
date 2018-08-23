// c++ Eigen - compatible line minimization using Brent's method for the one-dimensional line search
// this may be more stable than the line search algorithm for tricky problems?
// this version is for use with Rcpp

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include "util.h"
#include <type_traits>

using namespace Eigen;
using Rcpp::Rcout;
using std::endl;

using VecRef = const Eigen::Ref<const Eigen::VectorXd>&;
using MatRef = const Eigen::Ref<const Eigen::MatrixXd>&;


template<typename T>
inline T SIGN(T a,T b){
  static_assert(std::is_arithmetic<T>::value,"SIGN expects a single, numeric argument.");
	return b > 0. ? std::abs(a) : -std::abs(a);
}

// T is always a double in this context: bookkeeping for functions 'brent' and 'mnbrak' below
template<typename T>
inline void SHFT(T& a,T& b,T& c,const T& d){
	a=b; b=c; c=d;
}

/* defining declarations 
	These are variables which are shared between linmin and f1dim: f1dim takes a function of K args and views it as a function of 1 arg: f(delta) = f(theta + delta*gradient)
	the 'com' is for common.  nrfunc gets assigned the function pointer (e.g. log-likelihood function) that we input to linmin.
	ncom is the size of pcom and xicom
	Both brent and mnbrak take function pointers as arguments, of the type double (*fptr)(double)
	in both cases, the actual function passed is f1dim - this is the one we modify to use VectorXd, etc.
	
*/

VectorXd pcom, xicom;  // global, and updated from linmin

// this is the objective function: it must only accept a parameter vector argument, so make a wrapper if the objective function is different
double (*nrfunc)(VecRef); 

inline double f1dim(double x){
	const VectorXd xt(pcom + x*xicom); // candidate line search direction: 'x' here is alpha aka the contration parameter
    return (*nrfunc)(xt);
}

// argument 'f' is always called with f1dim above
// note: compiler may complain about 'd may be un-initialized', but this isn't a concern
// except in the sense that this function is indeed implemented in a somewhat goofy manner
double brent(const double ax,const double bx,const double cx,const double tol,
             double (*f)(double),double *xmin){
  static const size_t ITMAX{100u};
  static const double CGOLD{0.3819660}, ZEPS{1.0e-10};
  const double f0 = (*f)(bx);
  double d{0.}, e{0.}, etemp;
  double u, v{bx}, w{bx}, x{bx};  // search points
  double fu, fv{f0}, fw{f0}, fx{f0}; // hold function value @ search points
  double p,q,r,tol1,tol2,xm;
  double a = ((ax < cx) ? ax : cx), b = ((ax > cx) ? ax : cx);

  for(auto iter = 0;iter < ITMAX;iter++) {
    xm = 0.5*(a + b);
    tol1 = tol*fabs(x) + ZEPS;
    tol2 = 2.0*tol1;
    if (fabs(x-xm) <= (tol2 - 0.5*(b-a))) {
      *xmin = x;
      return fx;
    }
    if (std::abs(e) > tol1){
      r = (x-w)*(fx-fv);
      q = (x-v)*(fx-fw);
      p = (x-v)*q-(x-w)*r;
      q = 2.0*(q-r);
      if (q > 0.0) p = -p;
      q = fabs(q);
      etemp = e;
      e = d;
      if(fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x)){
        d = CGOLD*(e = (x >= xm ? a-x : b-x));
      } else {
        d = p/q;
        u = x+d;
        if (u-a < tol2 || b-u < tol2)
          d = SIGN(tol1,xm-x);
      }
    } else {
      e  = (x >= xm) ? a - x : b - x;
      d = CGOLD*e;
    }
    u = (fabs(d) >= tol1 ? x + d : x + SIGN(tol1,d));
    fu = (*f)(u);
    if(fu <= fx){
      if (u >= x) a = x; else b = x;
      SHFT(v,w,x,u);
      SHFT(fv,fw,fx,fu);
    } else {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w == x) {
        v = w;
        w = u;
        fv = fw;
        fw = fu;
      } else if (fu <= fv || v == x || v == w) {
        v = u;
        fv = fu;
      }
    }
  }
  throw("Too many iterations in BRENT");
  *xmin = x; // caller of brent can access this
  return fx;
}

// note: this function is NOT safe and may, possibly, grind on forever. Watch out!!!
void mnbrak(double *ax,double *bx,double *cx,double *fa,double *fb,double *fc,double (*func)(double)){
  static const double GOLD = 1.618034, GLIMIT = 100., TINY = 1.0e-20;
  double ulim,u,r,q,fu,dum;
  *fa = (*func)(*ax);
  *fb = (*func)(*bx);
  if (*fb > *fa) {  // switch direction in this case
    SHFT(dum,*ax,*bx,dum);
    SHFT(dum,*fb,*fa,dum);
  }
  *cx = (*bx) + GOLD*(*bx - *ax);
  *fc = (*func)(*cx);
  while (*fb > *fc) { // potential for infinite loop is right here
    r = (*bx - *ax)*(*fb - *fc);
    q = (*bx - *cx)*(*fb - *fa);
    u = (*bx) - ((*bx - *cx)*q - (*bx - *ax)*r)/(2.0*SIGN(std::max(std::abs(q-r),TINY),q-r));
    ulim = (*bx) + GLIMIT*(*cx-*bx);
    if ((*bx-u)*(u-*cx) > 0.0){
      fu = (*func)(u);
      if (fu < *fc){
        *ax = *bx;
        *bx = u;
        *fa = *fb;
        *fb = fu;
        return;
      } else if (fu > *fb) {
        *cx = u;
        *fc = fu;
        return;
      }
      u = (*cx) + GOLD*(*cx - *bx);
      fu = (*func)(u);
    } else if ((*cx - u)*(u - ulim) > 0.0){
      fu = (*func)(u);
      if (fu < *fc){
        SHFT(*bx,*cx,u,*cx + GOLD*(*cx - *bx));
        SHFT(*fb,*fc,fu,(*func)(u));
      }
    } else if ((u - ulim)*(ulim - *cx) >= 0.0){
      u = ulim;
      fu = (*func)(u);
    } else {
      u = (*cx) + GOLD*(*cx - *bx);
      fu = (*func)(u);
    }
    SHFT(*ax,*bx,*cx,u);
    SHFT(*fa,*fb,*fc,fu);
  }
}

// function pointer func takes a double* as arg and returns a double. The job of f1dim is to take func and create a 1-dimensional version of it.
void linmin(VectorXd& p,VectorXd& xi,double *fret,double(*func)(VecRef)){
  const double TOL = 2.0e-4;
  double xx = 1.e-3,xmin,fx,fb,fa,bx=2.0,ax = 0.0;  // mostly for passing to subroutines
  pcom = p;  // update globals so that mnbrak and/or brent can use them
  xicom = xi;
  nrfunc = func;  // assign the input function pointer to the (global) function pointer

  mnbrak(&ax,&xx,&bx,&fa,&fx,&fb,f1dim);

  *fret = brent(ax,xx,bx,TOL,f1dim,&xmin);
  double maxxi = xi.array().abs().maxCoeff();
  if (maxxi >= 50) xmin = 1.0/maxxi; // lower bound of 0.02, hmmm....
  xi *= xmin;  // re-scale for what purpose?
  p += xi;     // update input point (contract the search direction until we have a non-negligible decrease)
}

// the return value from the below version of dfpmin
struct funcMin{
  int iterations,error;
  double min_value;
  VectorXd arg_min,gradient;
};

// Quasi-Newton Algorithm
// inputs: 
//   p is the initial parameter value
//   ftol controls how easy it is to declare victory and return an 'optimum'
//   iter tracks how many iterations
//   func and dfunc are objective function and gradient, respectively
 
 
funcMin dfpmin(VecRef p0,double (*func)(VecRef),VectorXd (*dfunc)(VecRef),double ftol = 0.0,int verb=0){
  const int ITMAX = 200, n = p0.size();
  const double EPS = 1.0e-10;
  if(ftol <= 0.0) ftol = 1.e-7*n;
  int its, psz = std::min(10,n); // psz is size of diagnostic output to print
  int error = 1;

  double rho,k,fret; // used in Hessian update calculation
  MatrixXd hessian( MatrixXd::Identity(n,n) );
  VectorXd dg(n), hdg(n), Hy(n), p(p0);
  double fp = (*func)(p);  // the starting value of the objective function
  VectorXd g = (*dfunc)(p);
  VectorXd xi = -g;  // initial candidate search direction
  for(its = 0;its < ITMAX;its++){
    if(verb > 0){
      Rcout << "\n*****\n   Iteration " << its << endl;
      if(verb > 1){ Rcpp::Rcout << "theta = " << p.head(psz).transpose() << endl; }
      Rcout << "objective function value: " << fret << endl;
      Rcout << "gradient norm: " << g.norm() << endl;
    }
    linmin(p,xi,&fret,func); // fret is the new point found by line search
    if(any_nan(p)){
      if(verb) Rcout << "Encountered non-finite values in parameter estimate!" << std::endl;
      error = 1;
      break;
    }
    // check whether 2*|newP-oldP| <= TOL*[|oldP| + |newP| + 1.0E-10] as the convergence criterion
    if (2.0*fabs(fret-fp) <= ftol*(fabs(fret)+fabs(fp)+EPS)) { // fabs() is just abs() for floats.
      if(verb > 0){
        Rcout << "\nfret - fp = " << fret - fp << ", under convergence criterion of " << ftol*(fabs(fret)+fabs(fp)+EPS) << "\n";
      }
      error = 0; // flag that we successfully converged
      break;
    }
    fp = fret;            
    dg = g;  // save the old gradient since dfunc will update it
    fret = (*func)(p);  // get new objective function value
    g = (*dfunc)(p); // get new gradient value
    dg = g - dg;   // now it really is a difference of gradients
    // the rest of this is dedicated to updating the Hessian matrix
    Hy = dg.adjoint()*hessian.selfadjointView<Eigen::Lower>();
    rho = 1./dg.dot(xi);
    k = rho*rho*dg.dot(Hy) + rho;
    hessian.selfadjointView<Eigen::Lower>().rankUpdate(xi,Hy,-rho); // rank 2 update with distinct u and v. Note capitalization format
    hessian.selfadjointView<Eigen::Lower>().rankUpdate(xi,k); // rank 'K' update with a K-column matrix u (and alpha scalar)
    xi = -g.adjoint()*hessian.selfadjointView<Eigen::Lower>();
  }
  if(error > 0) throw("exceeded iterations in dfpmin");
  funcMin R;
  R.iterations = its; R.error = error;
  R.arg_min = p; R.min_value = fret;
  R.gradient = (*dfunc)(p);  // update gradient at final point
  return R;
}