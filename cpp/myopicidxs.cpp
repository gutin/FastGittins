#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <boost/math/special_functions/beta.hpp> 
#include<boost/math/distributions.hpp>
#include <cmath>
#include "helpers.hpp"

using namespace std;
using namespace boost::math;

void computeIndices(const int n, const double beta, vector<vector<double> >& indices)
{
  #pragma omp parallel for
  for(int nn = n-1; nn >= 2; --nn)
  {
    for(int a = 1; a <= nn-1; ++a)
    {
      double p = computeAGI(beta, a, nn);
      indices[a-1][nn-a-1] = p;
    }
  }
}

void computeGittinsIndices(const int n, const double step, const double beta, vector<vector<double> >& indices)
{
  vector<vector<double> > tmpindices(n-1, vector<double>(n-1, 0));
  for(int a = 1; a < n; ++a)
  {
    tmpindices[a-1][n-a-1] = ((double)a/(double)n);
  }
  for(double p=(step/2.0); p <= 1.0; p+=step)
  {
    const double safe = p/(1-beta);
    for(int nn = n-1; nn >= 2; --nn)
    {
      for(int a = 1; a <= nn-1; ++a)
      {
        const double r = ((double)a/(double)nn);
        const double risky =  r*(1 + beta*tmpindices[a][nn-a-1]) + (1-r)*beta*tmpindices[a-1][nn-a];
        if(indices[a-1][nn-a-1] == 0 && safe > risky)
        {
          indices[a-1][nn-a-1] = p - step/2.0;
        }
        tmpindices[a-1][nn-a-1] = max(safe,risky);
      }
    }
  }
  /*for(int nn = n-1; nn >= 2; --nn)
  {
    for(int a = 1; a <= nn-1; ++a)
    {
      if(a > ceil(nn/2.0))
      {
        indices[a-1][nn-a-1] = 1 - step/2.0;
      }
    }
  }*/
}

void experiment(int n, double beta, double step)
{
  vector<vector<double> > gindices(n-1, vector<double>(n-1, 0));
  const static int maxnum = 500;
  /*cout << "computing gittins indices...." << endl;
  computeGittinsIndices(n,step,beta,gindices);*/
  cout << "loading gittins indices..." << endl;
  loadFromCSV("gixs_95.csv", gindices, maxnum, maxnum);
  
  cout << "checking values..." << endl;
  for(int a = 0; a < maxnum; ++a)
  {
    for(int b = 0; b < maxnum; ++b)
    {
      const double agi1 = computeAGI(beta, a + 1, a + b + 2);
      const double agi2 = computeAGI2(beta, a + 1, b + 1);
      const double agi2a = computeGeneralAGI(beta, a + 1, b + 1, 2);
      const double agi3 = computeGeneralAGI(beta, a + 1, b + 1, 3);
      const double agi4 = computeGeneralAGI(beta, a + 1, b + 1, 4);
      double eps = 1e-6;
      if (abs(agi2a - agi2) > eps)
      {
        cerr << "difference found in agi1 comp for a: " << a << ". b: " << b << ". agi2: " << agi2 << ". agi2a: " << agi2a << endl;
      }
      if (!(agi1 > agi2 - eps && agi2 > agi3 - eps && agi3 > agi4 - eps))
      {
        cerr << "unexpected order of indices. a = " << a << ". b = " << b << ". agi1: " << agi1 << ". agi2: " << agi2 << ". agi3:" << agi3 << ". agi4:" << agi4 << endl;
        //exit(1);
        //
      }

      if (agi1 < gindices[a][b] -eps)
      {
        cerr << "unexpected. a = " << a << ". b = " << b << ". agi1: " << agi1 << ". gidx: " << gindices[a][b] << endl;
      }
      if (agi2 < gindices[a][b] -eps)
      {
        cerr << "unexpected. a = " << a << ". b = " << b << ". agi2: " << agi2 << ". gidx: " << gindices[a][b] << endl;
      }
      if (agi3 < gindices[a][b] -eps)
      {
        cerr << "unexpected. a = " << a << ". b = " << b << ". agi3: " << agi3 << ". gidx: " << gindices[a][b] << endl;
      }
      if (agi4 < gindices[a][b] -eps)
      {
        cerr << "unexpected. a = " << a << ". b = " << b << ". agi4: " << agi4 << ". gidx: " << gindices[a][b] << endl;
      }
    }
  }

  cout << "done..." << endl;
}

void calculator()
{
  while(1)
  {
    double gamma, a, b;
    cin >> gamma;
    cin >> a;
    cin >> b;
    cout << std::setprecision(20) << computeAGI(gamma,a,a+b) << endl; 
    cout << std::setprecision(20) << computeAGI2(gamma,a,b) << endl; 
    cout << std::setprecision(20) << computeGeneralAGI(gamma,a,b,3) << endl; 
  }
}


int main(int argc, const char** args)
{
  //this is bad lol
  //calculator();
  //cout << ::computeAGI2(0.99,3,3);
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

  double step = 0.0001;
  experiment(n, beta, step);

  exit(0);
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

