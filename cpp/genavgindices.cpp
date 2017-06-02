#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

using namespace std;

void computeIndices(const int n, const double step, vector<vector<double> >& indices)
{
  const int steps = 1 + 1/step;
  //cout << "there are [" << steps << "] different values for lambda" << endl;
  vector<vector<vector<double> > >  tmpindicesold(2*(n-1), vector<vector<double > >(2*(n-1), vector<double>(steps, 0)));
  vector<vector<vector<double> > >  tmpindices(2*(n-1), vector<vector<double > >(2*(n-1), vector<double>(steps, 0)));

  for(int i = 0; i < steps; ++i)
  {
    double l = step*i;
    //cout << l << endl;
    for (int k=1; k<= 2*n-2; ++k)
    {
      for (int a=1; a <= k-1; ++a)
      {
        //cout << (a-1) << ", " << (k-a-1) << ".." << endl;
        tmpindicesold[a-1][k-a-1][i] = ((double)a/(double)k) - l*(n-1);
      }
    }
  }

  for(int t=n-2;t >= 1; --t)
  {
    for(int i = 0; i < steps; ++i)
    {
      double l = step*i;
      for (int k=1; k<= 2*n-3; ++k)
      {
        for (int nn=1;nn<=k;++nn)
        {
          for (int a=1;a<=nn-1;++a)
          {
            // so many looops .. ok...
            const double r = ((double)a)/((double) nn);
            const double v1 = r *(1 + tmpindicesold[a+1-1][nn-a-1][i]);
            const double v2 = (1-r) * tmpindicesold[a-1][nn-a+1-1][i];
            const double alt = v1+v2;
            double val;
            tmpindices[a-1][nn-a-1][i] = val = max(alt,r - l*t);       
            if(t==1 && nn<=n && val > 0)
            {
              indices[a-1][nn-a-1] = l;
            }
          }
        }
      }
    }
    tmpindicesold = tmpindices;
  }
}

int main(int argc, const char** args)
{
  if(argc != 3)
  {
    cerr << "need 3 arguments [n] [step]" << endl;
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

  //cout << "finding indices for size limit [" << n << "] and step size [" << step << "]" << endl; 
  vector<vector<double> > indices(n-1, vector<double>(n-1, 0));
  computeIndices(n, step, indices);

  for(int i = 0; i < n-1; ++i)
  {
    for(int j=0; j < n-2; ++j)
    {
      cout << indices[i][j] << ", "; 
    }
    cout << indices[i][n-2] << endl;
 }
}
