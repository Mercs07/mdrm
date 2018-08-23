// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// cdF
Rcpp::List cdF(Rcpp::NumericMatrix uu, Rcpp::NumericVector ff, int verbose, bool bs);
RcppExport SEXP _mdrm_cdF(SEXP uuSEXP, SEXP ffSEXP, SEXP verboseSEXP, SEXP bsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type uu(uuSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type ff(ffSEXP);
    Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< bool >::type bs(bsSEXP);
    rcpp_result_gen = Rcpp::wrap(cdF(uu, ff, verbose, bs));
    return rcpp_result_gen;
END_RCPP
}
// LL
double LL(const Eigen::VectorXd th, const Eigen::MatrixXd X, const Eigen::MatrixXd Y);
RcppExport SEXP _mdrm_LL(SEXP thSEXP, SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type th(thSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(LL(th, X, Y));
    return rcpp_result_gen;
END_RCPP
}
// LLF
double LLF(const Eigen::VectorXd th, const Eigen::MatrixXd X, const Eigen::MatrixXd Y);
RcppExport SEXP _mdrm_LLF(SEXP thSEXP, SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type th(thSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(LLF(th, X, Y));
    return rcpp_result_gen;
END_RCPP
}
// gradA
Rcpp::NumericVector gradA(const Eigen::VectorXd th, const Eigen::MatrixXd X, const Eigen::MatrixXd Y);
RcppExport SEXP _mdrm_gradA(SEXP thSEXP, SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type th(thSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(gradA(th, X, Y));
    return rcpp_result_gen;
END_RCPP
}
// gradF
Rcpp::NumericVector gradF(const Eigen::VectorXd th, const Eigen::MatrixXd X, const Eigen::MatrixXd Y);
RcppExport SEXP _mdrm_gradF(SEXP thSEXP, SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type th(thSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(gradF(th, X, Y));
    return rcpp_result_gen;
END_RCPP
}
// drmHess
Rcpp::NumericMatrix drmHess(const Eigen::VectorXd th, const Eigen::MatrixXd X, const Eigen::MatrixXd Y, bool useF);
RcppExport SEXP _mdrm_drmHess(SEXP thSEXP, SEXP XSEXP, SEXP YSEXP, SEXP useFSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type th(thSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type Y(YSEXP);
    Rcpp::traits::input_parameter< bool >::type useF(useFSEXP);
    rcpp_result_gen = Rcpp::wrap(drmHess(th, X, Y, useF));
    return rcpp_result_gen;
END_RCPP
}
// fitdrm
Rcpp::List fitdrm(const Eigen::MatrixXd Y, const Eigen::MatrixXd X, double TOL, int MAXIT, int verb, double maxStep, bool justBeta, std::string method);
RcppExport SEXP _mdrm_fitdrm(SEXP YSEXP, SEXP XSEXP, SEXP TOLSEXP, SEXP MAXITSEXP, SEXP verbSEXP, SEXP maxStepSEXP, SEXP justBetaSEXP, SEXP methodSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type TOL(TOLSEXP);
    Rcpp::traits::input_parameter< int >::type MAXIT(MAXITSEXP);
    Rcpp::traits::input_parameter< int >::type verb(verbSEXP);
    Rcpp::traits::input_parameter< double >::type maxStep(maxStepSEXP);
    Rcpp::traits::input_parameter< bool >::type justBeta(justBetaSEXP);
    Rcpp::traits::input_parameter< std::string >::type method(methodSEXP);
    rcpp_result_gen = Rcpp::wrap(fitdrm(Y, X, TOL, MAXIT, verb, maxStep, justBeta, method));
    return rcpp_result_gen;
END_RCPP
}
// drmBoot
Rcpp::List drmBoot(Rcpp::NumericMatrix y, Rcpp::NumericMatrix x, int nBoot, double TOL, int MAXIT, int verb, std::string method);
RcppExport SEXP _mdrm_drmBoot(SEXP ySEXP, SEXP xSEXP, SEXP nBootSEXP, SEXP TOLSEXP, SEXP MAXITSEXP, SEXP verbSEXP, SEXP methodSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type y(ySEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type nBoot(nBootSEXP);
    Rcpp::traits::input_parameter< double >::type TOL(TOLSEXP);
    Rcpp::traits::input_parameter< int >::type MAXIT(MAXITSEXP);
    Rcpp::traits::input_parameter< int >::type verb(verbSEXP);
    Rcpp::traits::input_parameter< std::string >::type method(methodSEXP);
    rcpp_result_gen = Rcpp::wrap(drmBoot(y, x, nBoot, TOL, MAXIT, verb, method));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_mdrm_cdF", (DL_FUNC) &_mdrm_cdF, 4},
    {"_mdrm_LL", (DL_FUNC) &_mdrm_LL, 3},
    {"_mdrm_LLF", (DL_FUNC) &_mdrm_LLF, 3},
    {"_mdrm_gradA", (DL_FUNC) &_mdrm_gradA, 3},
    {"_mdrm_gradF", (DL_FUNC) &_mdrm_gradF, 3},
    {"_mdrm_drmHess", (DL_FUNC) &_mdrm_drmHess, 4},
    {"_mdrm_fitdrm", (DL_FUNC) &_mdrm_fitdrm, 8},
    {"_mdrm_drmBoot", (DL_FUNC) &_mdrm_drmBoot, 7},
    {NULL, NULL, 0}
};

RcppExport void R_init_mdrm(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
