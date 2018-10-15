// line search minimization with 'Brent's method' approach
// the 'macros' SIGN and SHIFT are templated and are defined in the header
#include "linemins.h"

// update of Hessian matrix in quasi-Newton algorithm. Preserves positive definiteness
// 'dth' is change in parameters (delta theta), 'dgr' is change in gradient
void BFGS(Eigen::Ref<Eigen::MatrixXd> H,VecRef dth,VecRef dgr){
  const Eigen::VectorXd Hy(dgr.adjoint()*H.selfadjointView<Eigen::Lower>());
  const double rho = 1./dgr.dot(dth);
  const double k = rho*rho*dgr.dot(Hy) + rho;
  H.selfadjointView<Eigen::Lower>().rankUpdate(dth,Hy,-rho); // rank 2 update with distinct u and v. Note capitalization format
  H.selfadjointView<Eigen::Lower>().rankUpdate(dth,k); // rank 'K' update with a K-column matrix u (and alpha scalar)
}

// bracket an initial interval via Golden Section search
void mnbrak(double *ax,double *bx,double *cx,double *fa,double *fb,double *fc,double (*func)(double)){
  constexpr double GOLD = 1.618034, GLIMIT = 100., TINY = 1.0e-20;
  double ulim,u,r,q,fu,dum;
  *fa = (*func)(*ax);
  *fb = (*func)(*bx);
  if (*fb > *fa) {  // switch direction in this case
    SHIFT(dum,*ax,*bx,dum);
    SHIFT(dum,*fb,*fa,dum);
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
        SHIFT(*bx,*cx,u,*cx + GOLD*(*cx - *bx));
        SHIFT(*fb,*fc,fu,(*func)(u));
      }
    } else if ((u - ulim)*(ulim - *cx) >= 0.0){
      u = ulim;
      fu = (*func)(u);
    } else {
      u = (*cx) + GOLD*(*cx - *bx);
      fu = (*func)(u);
    }
    SHIFT(*ax,*bx,*cx,u);
    SHIFT(*fa,*fb,*fc,fu);
  }
}

// argument 'f' is always called with f1dim
// note: compiler may complain about 'd may be un-initialized', but this isn't a concern
// except in the sense that this function is indeed implemented in a somewhat goofy manner
// TODO: make the function pointer a template parameter so we can use a class with overloaded operator(), a lambda,
// or a std::function
double brent(const double ax,const double bx,const double cx,const double tol,
             double (*f)(double),double *xmin){
  constexpr size_t ITMAX{100u};
  constexpr double CGOLD{0.3819660}, ZEPS{1.0e-10};
  const double f0 = (*f)(bx);
  double d{0.}, e{0.}, etemp;
  double u, v{bx}, w{bx}, x{bx};  // search points
  double fu, fv{f0}, fw{f0}, fx{f0}; // hold function value @ search points
  double p,q,r,tol1,tol2,xm; // other miscellaneous bookkeeping
  double a = std::min(ax,cx), b = std::max(ax,cx);

  for(size_t iter = 0;iter < ITMAX;iter++) {
    xm = 0.5*(a + b);
    tol1 = tol*std::abs(x) + ZEPS;
    tol2 = 2.0*tol1;
    if (std::abs(x-xm) <= (tol2 - 0.5*(b-a))) {
      *xmin = x;
      return fx;
    }
    if (std::abs(e) > tol1){
      r = (x-w)*(fx-fv);
      q = (x-v)*(fx-fw);
      p = (x-v)*q-(x-w)*r;
      q = 2.0*(q-r);
      if (q > 0.0) p = -p;
      q = std::abs(q);
      etemp = e;
      e = d;
      if(std::abs(p) >= std::abs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x)){
        d = CGOLD*(e = (x >= xm ? a-x : b-x));
      } else {
        d = p/q;
        u = x + d;
        if (u-a < tol2 || b-u < tol2)
          d = SIGN(tol1,xm-x);
      }
    } else {
      e  = (x >= xm) ? a - x : b - x;
      d = CGOLD*e;
    }
    u = (std::abs(d) >= tol1 ? x + d : x + SIGN(tol1,d));
    fu = (*f)(u);
    if(fu <= fx){
      if (u >= x) a = x; else b = x;
      SHIFT(v,w,x,u);
      SHIFT(fv,fw,fx,fu);
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

// function pointer func takes a double* as arg and returns a double. The job of f1dim is to take func and create a 1-dimensional version of it.
void brent_linemin(VectorXd& p,VectorXd& xi,double *fret,double(*func)(VecRef)){
  const double TOL = 2.0e-4;
  double xx = 1.e-3,xmin,fx,fb,fa,bx=2.0,ax = 0.0;  // mostly for passing to subroutines
  pcom = p;  // update globals so that mnbrak and/or brent can use them
  xicom = xi;
  nrfunc = func;  // assign the input function pointer to the (global) function pointer

  mnbrak(&ax,&xx,&bx,&fa,&fx,&fb,f1dim);

  *fret = brent(ax,xx,bx,TOL,f1dim,&xmin);
  double max_xi = xi.array().abs().maxCoeff();
  if (max_xi >= 50) xmin = 1.0/max_xi; // lower bound of 0.02, hmmm....
  xi *= xmin;  // re-scale for what purpose?
  p += xi;     // update input point (contract the search direction until we have a non-negligible decrease)
}

funcMin dfpmin(VecRef p0,double (*func)(VecRef),Eigen::VectorXd (*dfunc)(VecRef),double ftol,int verb){
  using namespace Eigen;
  constexpr double EPS = 1.0e-10;
  const int ITMAX = 200, n = p0.size();
  if(ftol <= 0.0) ftol = 1.e-7*n;
  int its, error = 1, psz = std::min(10,n); // psz is size of diagnostic output to print

  double fret, fp = (*func)(p0); // starting value of the objective function
  MatrixXd hessian( MatrixXd::Identity(n,n) );
  VectorXd dg(n), hdg(n), p(p0);
  VectorXd g = (*dfunc)(p);
  VectorXd xi = -g;  // initial candidate search direction
  for(its = 0;its < ITMAX;its++){
    if(verb > 0){
      Rcout << "\n*****\n   Iteration " << its << endl;
      if(verb > 1){ Rcpp::Rcout << "theta = " << p.head(psz).transpose() << endl; }
      Rcout << "objective function value: " << fret << endl;
      Rcout << "gradient norm: " << g.norm() << endl;
    }
    brent_linemin(p,xi,&fret,func); // fret is the new point found by line search
    if(any_nan(p)){
      if(verb) Rcout << "Encountered non-finite values in parameter estimate!" << std::endl;
      error = 1;
      break;
    }
    // check whether 2*|newP-oldP| <= TOL*[|oldP| + |newP| + 1.0E-10] as the convergence criterion
    if (2.0*std::abs(fret-fp) <= ftol*(std::abs(fret) + std::abs(fp)+EPS)) {
      if(verb > 0){
        Rcout << "\nfret - fp = " << fret - fp << ", under convergence criterion of " << ftol*(std::abs(fret)+std::abs(fp)+EPS) << "\n";
      }
      error = 0; // flag that we successfully converged
      break;
    }
    fp = fret;
    dg = g;  // save the old gradient since dfunc will update it
    fret = (*func)(p);  // get new objective function value
    g = (*dfunc)(p); // get new gradient value
    dg = g - dg;   // now it really is a difference of gradients
    BFGS(hessian,xi,dg); // update (inverse) Hessian approximation
    xi = -g.adjoint()*hessian.selfadjointView<Eigen::Lower>(); // new search direction -f'/f''
  }
  if(error > 0) throw("exceeded iterations in dfpmin"); // should just return an error state instead of throw?
  funcMin R;
  R.iterations = its; R.error = error;
  R.arg_min = p; R.min_value = fret;
  R.gradient = (*dfunc)(p);  // update gradient at final point
  return R;
}