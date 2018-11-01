## multivariate density ratio model

Herein is an R package implementing the density ratio regression model for univariate and/or multivariate outcomes.

#### Usage

The so-called _density ratio model_ is a flexible semi-parametric regression model which functions as a useful alternative to standard GLMs when parametric model assumptions are dubious. Information about the model, background, and technical details can be found in [this article](https://www.sciencedirect.com/science/article/pii/S0047259X16301622).

This implementation is mostly intended to mirror the syntax of standard R model fitting formula interface, ex. `a_model = drm(log(Y)~X1+X2*X3,data=my_dataset)`. Likewise, most of the standard S3 methods for models (`summary`,`plot`,`vcov`,`resid`, etc.) are implemented for convenient analysis. The outcomes may be univarite or multivariate, though plotting is only specialized for univariate and bivariate outcomes.

#### Installation

The package can be installed with `devtools::install_github` or built with `R CMD INSTALL` or `install.packages`; the only possibly non-standard requirement is C++11 via `CXX_STD = CXX11`.

