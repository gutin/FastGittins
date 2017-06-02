#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include "helpers.hpp"

using namespace std;

void computeIndices(const int n, const double step, const double beta, vector<vector<double> >& indices)
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
}

void writeIndicesToFile(const string& filename, const Table& indices)
{
  ofstream file;
  file.open(filename.c_str());
  file << setprecision(5) << fixed;
  for (unsigned int i = 0; i < indices.size(); ++i)
  {
    for (unsigned int j = 0; j < indices[i].size()-1; ++j)
    {
      file << indices[i][j] << ", ";
    }
    file << indices[i][indices[i].size()-1] << endl;
  }
  file.close();
}

int main(int argc, const char** args)
{
  if(argc != 4)
  {
    cerr << "need 3 arguments [n] [step] [beta]" << endl;
    exit(1);
  }
  int n;
  string strnum = args[1];
  stringstream ss(strnum);
  ss >> n;

  double step;
  string strstep = args[2];
  stringstream sstep(strstep);
  sstep >> step;

  double beta;
  string betastr = args[3];
  stringstream ssbeta(betastr);
  ssbeta >> beta;

  #pragma omp parallel
  for (int t = 1; t <= n; ++t)
  {
    double beta = 1-1.0/double(t);
    vector<vector<double> > indices(2*n-1, vector<double>(2*n-1, 0));
    computeIndices(2*n, step, beta, indices);
    stringstream filenamebuilder;
    filenamebuilder << "/dev/shm/gixs/hor_" << t << ".csv";
    writeIndicesToFile(filenamebuilder.str(), indices);
  }
  return 0;


  //cout << "finding indices for size limit [" << n << "] and step size [" << step << "]" << endl; 
  vector<vector<double> > indices(n-1, vector<double>(n-1, 0));
  computeIndices(n, step, beta, indices);
  /*for (int a = 1; a <= n-1; ++a)
  {
    for (int b = 1; b <= n-1; ++b)
    {
      double agi = computeAGI2(beta, double(a), double(b));
      double gi = indices[a-1][b-1];
      if (gi > agi)
      {
        cerr << "gittins index " << gi << " > " << agi << " for a = " << a << ", b = " << b << endl;
      }
    }
  }*/
  for(int i = 0; i < n-1; ++i)
  {
    for(int j=0; j < n-2; ++j)
    {
      cout << indices[i][j] << ", "; 
    }
    cout << indices[i][n-2] << endl;
  }
}

