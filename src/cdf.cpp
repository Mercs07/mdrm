#include <iostream>  // std::endl
#include <algorithm> // std::swap in the sorting routine
#include <cstring>   // memcpy
#include <vector>    // right now this is just used for uniq1 - could make things more primitive
#include <utility>   // std::pair
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp11)]]
#include<RcppEigen.h>

using namespace Eigen;
using VecRef = const Eigen::Ref<const Eigen::VectorXd>&;
using uipair = std::pair<unsigned int,unsigned int>;
/* 
 * A quicksort algorithm which sorts and also fills in an int* that tracks the original positions (like the order() function in R)
 * which is useful if the vector we sort is aligned with another object (a matrix, etc.)
 * the functions are inspired by http://codereview.stackexchange.com/questions/77782/quick-sort-implementation
 * but templated to make things a little more flexible
*/

template <typename T>
int partition(T *V,int *P, const int left, const int right){ //note: it is up to this function's caller to ensure that left and right are within V's range!
	const int mid = left + (right - left)/2; // avoid overflow
	const T pivot = V[mid];
	std::swap(V[left],V[mid]);  // to make things cleaner, first move the pivot to the front.
	std::swap(P[left],P[mid]);  // make sure to have the index track along with the pivot!
	int i = left + 1; // and skip the pivot when making comparisons
	int j = right;
	while(i <= j){ // until the indices overlap
		while( i <= j && V[i] <= pivot){
			i++;  // scan until the first 'wrong' value is encountered
		}
		while(i <= j && V[j] > pivot){
			j--; // scanning from the right downwards
		}
		if(i < j){
			std::swap(V[i],V[j]);
			std::swap(P[i],P[j]); // needed to track the initial positions
		}
	}
	std::swap(V[i-1],V[left]); // return the pivot value to the point where it is in order
	std::swap(P[i-1],P[left]); // keeping indices consistent with values
	return i-1; // return the location of the pivot: the recursive step splits the array here
}

// SIZE is the length of the entire array, and is not really necessary if the initial left/right mark the beginning and end of the array
template<typename T>
void qisort(T *V, int *P, const int leftInx, const int rightInx){
	if(leftInx >= rightInx){ // stopping condition
		return;
	}
	// do the swapping and return the index defining the cutpoint
	int part = partition(V,P,leftInx,rightInx);
	// recursive step(s): call sort on the two pieces (and we don't need to include the pivot in either one)
	qisort(V,P,leftInx,part-1);
	qisort(V,P,part+1,rightInx);
}

// take a given (dynamic size) Eigen vector of any supported type, sort it, and return a VectorXi map between sorted & original positions
template<typename T>
VectorXi EigenSort(Eigen::Matrix<T,Eigen::Dynamic,1>& data){
	const int L = data.size();
	T* vv = &data(0); // pointer to first element of data - we're just traversing memory already laid out in data, so no need to allocate.
	VectorXi inxs = VectorXi::LinSpaced(L,0,L-1); // initial, zero-based indices
	qisort(vv,&inxs(0),0,L-1);
	return inxs;
}

// sort the input matrix by the given column
template<typename T>
Matrix<T,Dynamic,Dynamic> sortByCol(Matrix<T,Dynamic,Dynamic> &U,int sort_column){
	if(sort_column < 0 || sort_column >= U.cols()){ return U;}
	Matrix<T,Dynamic,1> V = U.col(sort_column);
	VectorXi map = EigenSort(V);
	const int R = U.rows();
	Matrix<T,Dynamic,Dynamic> res(R,U.cols());
	for(int i=0;i<R;i++){
		res.row(i) = U.row(map(i));
	}
	return res;
}

// extract unique, sorted values from the vector
VectorXd uniq1(VectorXd &V){
	VectorXi cc = EigenSort(V);  // we don't need cc, but V is now sorted!
	const int L = V.size();
	std::vector<double> uniqC;
	uniqC.push_back(V(0));
	for(int i=1;i<L;i++){
		if(V(i) != V(i-1)) uniqC.push_back(V(i));
	}
	Eigen::Map<VectorXd> UU(&uniqC[0],uniqC.size()); // map a VectorXd to the std::vector
	return UU;
}

VectorXd cumsum(VecRef x){
  const int n = x.size();
  VectorXd res(n);
  res(0) = x(0);
  for(int i=1;i<n;i++){
    res(i) = x(i) + res(i-1);
  }
  return res;  
}


// critical assumption: x is sorted! Within a block, we can locate at least the first value to match
// instead of mindlessly iterating from the beginning each time.
template<typename T>
int bsearch(const Eigen::Matrix<T,Eigen::Dynamic,1>& x,const T z){
	const int n = x.size();
	if(x(0)==z) return 0;
	if(x(n-1)==z) return n-1;
	int l=0,h=n-1,m;
	T tmp;
	while(l<h){
		m = l + (h-l)/2;
		tmp = x(m);
		if(tmp==z) return m;
		else if(tmp > z) h = m;
		else l = m;
	}
	return -1; // did not find it!
}


// 'size' can provide a hint to the allocation if it's known ahead of time
template<class T>
std::vector<uipair> rle(const Eigen::Matrix<T,Eigen::Dynamic,1>& x,size_t size=0){
  const int n = x.size();
  std::vector<uipair> rls; // holds run lengths: index of start and length
  if(size > 0) rls.reserve(std::min(size,static_cast<size_t>(n)));
  int i=0,j=0;
  while(j<n){
    while(x(i)==x(j)){
      j++;
      if(j==n) break;
    }
    rls.emplace_back(std::make_pair(i,j-i));
    i=j;
  }
  return rls;
}

// NOTE: the matrix we pass in here is assumed to be unique! In the sense that each row shows up once
// return value is a list, with 3 elements: ux, uy, and cdf, where ux.size()==cdf.rows() and uy.size()==cdf.cols()
// [[Rcpp::export]]
Rcpp::List cdF(Rcpp::NumericMatrix uu, Rcpp::NumericVector ff,int verbose = 0,bool bs = true){

	if(uu.ncol() != 2) Rcpp::stop("cdF: this function is implemented for a 2-column matrix only. Consider passing a pair of columns of U.");
	if(uu.nrow() != ff.size()) Rcpp::stop("cdF: mismatch in input parameter sizes.");
	double *pu = &uu[0], *pf = &ff[0];
	const Eigen::Map<Eigen::MatrixXd> U(pu,uu.nrow(),uu.ncol()); // parameters: pointer to array of memory, rows, columns
	const Eigen::Map<Eigen::VectorXd> F(pf,ff.size(),1);
	const int M = U.rows();
	
	/* 
	 * We'd like to cover the smallest grid containing the data set, assuming uu has two columns (x and y)
	 * to get the estimated cdf over the Cartesian product of the data points, we need to do the following:
	 * 1) sort the data by x, the first column
	 * 2) for each unique value of x:
	 *  - calculate the cumulative jump sizes for Y|X=x
	 *  - set the row of the output cdf as the current jump sizes plus sum of previous rows
	 *  In general, for a k-dimensional cdf we'd need to keep track of a (k-1) dimensional tensor with cumulative values
	 */
	MatrixXd UF(M,3); 
	UF.block(0,0,M,2) = U;
	UF.col(2) = F;
	VectorXd U0 = UF.col(0), U1 = UF.col(1);
	VectorXi map0 = EigenSort(U0), map1 = EigenSort(U1);  // call these for the side effect of sorting each column
	MatrixXd Usort = sortByCol(UF,0);
	const VectorXd UX = uniq1(U0), UY = uniq1(U1);
	
	// forward declarations
	const int nX = UX.size(), nY = UY.size();
	std::vector<uipair> block_counts;  // first elem: index of block start. second: block size
	int k; // do NOT make this unsigned unless we're absolutely sure that an intermediate calcuation NEVER goes < 0.
	MatrixXd cdf(MatrixXd::Zero(nX,nY));
	
	if(verbose > 0) Rcpp::Rcout << "the grid has dimension " << cdf.rows() << " x " << cdf.cols() << std::endl;
	
	if(nX < M){
	  block_counts = rle(U0,static_cast<size_t>(nX));
	  // within each chunk of repeated x-values, sort the associated y-values and update Usort
	  for(auto it = block_counts.cbegin();it != block_counts.cend();++it){
	    k = it->second;
	    if(k > 1){
	      MatrixXd myB = Usort.block(it->first,0,k,3);
	      Usort.block(it->first,0,k,3) = sortByCol(myB,1);;
	    }
	  }
	} else { // no sorting needed
	  for(int i=0;i<M;i++) block_counts.emplace_back(std::make_pair(i,1));
	}
	// now the matrix should be sorted first by col0, then by col1 within col0
	// since the rows of the matrix are unique, tied X values cannot have tied Y values, 
	// so we only need to account for ties in 1 dimension
	MatrixXd YV(MatrixXd::Zero(nY,2)); // col. 0: previous cdf; col. 1: currently calculating cdf
	int end,z,currCol = 0;
	for(auto it = block_counts.cbegin();it!=block_counts.cend();++it){
	  k = static_cast<int>(it->first);
	  end = static_cast<int>(it->second);
	  YV.col(1).setZero();
	  z = 0; // position within the current block
	  int j0 = bs ? bsearch(UY,Usort(k,1)) : 0;
	  if(bs && verbose > 0) Rcpp::Rcout << "Beginning the search from position " << j0 << std::endl;
	  for(int j=j0;j<nY;j++){

	    if(UY(j) == Usort(k+z,1)){
	      YV(j,1) = Usort(k+z,2); // add the matching jump size...this is where non-uniqueness could cause a problem
	      if(++z == end) break;
	    }
	  }
	  YV.col(0) += cumsum(YV.col(1));
	  if(currCol < cdf.rows()){
	    cdf.row(currCol++) = YV.col(0).transpose(); // the 'column' which is a marginal distribution of Y|X=x is a row in the nX*nY matrix  
	  }
	}
	return Rcpp::List::create(
		Rcpp::Named("ux") = Rcpp::wrap(UX),
		Rcpp::Named("uy") = Rcpp::wrap(UY),
		Rcpp::Named("cdf") = Rcpp::wrap(cdf)
	);
}





