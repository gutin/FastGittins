#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <boost/math/special_functions/beta.hpp> 

using namespace std;
using namespace boost::math;

inline double calcindex(double beta, const double a, const double nn)
{
  double p = double(a)/double(nn);
  const double b = nn - a;
  const double r = ((double)a/(double)nn);
  const double rup = ((double)a+1.0)/((double)nn+1.0);
  const double rdn = ((double)a)/((double)nn+1.0);
  
  for(int k = 0; k < 10; ++k)
  {
    if(p > 1 || p < r || p != p) 
    {
      cerr << "Warning! returning approximation for a = " << a << " and b = " << b << endl;
      return r;
    }
    const double safe = p/(1-beta);

    const double lhs = r*(1-ibeta(a+1,b,p));
    const double expmax = lhs + p*ibeta((double)a,b,p);  
    //cout << expmax << ". lhs = " << lhs << ". rhs = " << p*ibeta((double)a,b,p) << endl;
    const double risky = r+ beta/(1-beta)*expmax;

    const double orig = risky - safe;
    const double deriv = beta/(1-beta)*ibeta(a,b,p) - 1/(1-beta);
    p -= orig/deriv;
  }
  return p;
}

void computeIndices(const int n, const double beta, vector<vector<double> >& indices)
{
  for(int nn = n-1; nn >= 2; --nn)
  {
    for(int a = 1; a <= nn-1; ++a)
    {
      double p = calcindex(beta, a, nn);
      indices[a-1][nn-a-1] = p;
    }
  }
}

int main(int argc, const char** args)
{
  if(argc != 3)
  {
    cerr << "need 2 arguments [n] [beta]" << endl;
    exit(1);
  }
  int n;
  string strnum = args[1];
  stringstream ss(strnum);
  ss >> n;

  double beta;
  string betastr = args[2];
  stringstream ssbeta(betastr);
  ssbeta >> beta;
  
  //cout << calcindex(beta,292,292+1170) << endl;
  //cout << "finding indices for size limit [" << n << "] and step size [" << step << "]" << endl; 
  vector<vector<double> > indices(n-1, vector<double>(n-1, 0));
  computeIndices(n, beta, indices);

  for(int i = 0; i < n-1; ++i)
  {
    for(int j=0; j < n-2; ++j)
    {
      cout << indices[i][j] << ", "; 
    }
    cout << indices[i][n-2] << endl;
  }
}

