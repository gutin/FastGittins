#ifndef helpers_hpp
#define helpers_hpp

#include <vector>
#include <boost/random.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/tools/minima.hpp>
#include <fstream>
#include <sys/stat.h>

const double PI = 3.14159265358979;

using namespace std;
using namespace boost::math;
using namespace boost::random;
using namespace boost::math::tools;

typedef vector<vector<double> > Table;
typedef vector<vector<vector<double> > > Table3D;

typedef boost::variate_generator<boost::mt19937&, boost::uniform_real<> > RandUnif;
typedef boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > RandNorm;
typedef boost::variate_generator<boost::mt19937&, boost::uniform_int<> > RandIntUnif;

class RandomSampleHelper
{
public:
  RandomSampleHelper(long seedval) : rng(seedval) {}
  double sampleBeta(double alpha, double beta) 
  {
    boost::gamma_distribution<> xgamma(alpha);
    boost::gamma_distribution<> ygamma(beta);
    boost::variate_generator<boost::mt19937&, boost::gamma_distribution<> > xgen(rng, xgamma);
    boost::variate_generator<boost::mt19937&, boost::gamma_distribution<> > ygen(rng, ygamma);
    double xsample = xgen();
    double ysample = ygen();
    return xsample/(xsample+ysample);
  } 

  double sampleGaussian(double mu, double sigma)
  {
    boost::normal_distribution<> normdist(mu, sigma);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > sgen(rng, normdist);
    return sgen();
  }
private:
  boost::mt19937 rng;
};

/* courtesy of http://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c */
inline bool exists_file(const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

void loadFromCSV( const std::string& filename, std::vector< std::vector<double> >& matrix, const int maxrows, const int maxcols)
{
  if( !exists_file(filename) )
  {
    cerr << "cannot find required file " << filename << " to load." << endl;
    exit(1);
  }
  std::ifstream  file( filename.c_str() );
  std::vector<double> row;
  std::string line;
  std::string  cell;
  int rows = 0;
  while( file && rows < maxrows )
  {
    std::getline(file,line);
    std::stringstream lineStream(line);
    row.clear();
    int cols = 0;
    while( std::getline( lineStream, cell, ',' ) && cols < maxcols)
    {
      double number = strtod(cell.c_str(), NULL);
      row.push_back(number);
      ++cols;
    }

    if( !row.empty() )
    {
      matrix.push_back(row);
    }
    ++rows;
  }
}

void loadLineFromCSV( const std::string& filename, std::vector<double>& arrs, const int maxentries)
{
  std::ifstream       file( filename.c_str() );
  std::string line;
  std::string  cell;
  std::getline(file,line);
  std::stringstream lineStream(line);
  int idx = 0;
  while( std::getline( lineStream, cell, ',' ) && idx < maxentries)
  {
    double number = strtod(cell.c_str(), NULL);
    arrs[idx] = number;
    ++idx;
  }
}

double kl(double p1, double p2)
{
  return p1*log(p1/p2)+(1-p1)*log((1-p1)/(1-p2));
}

double maxkl(double n, double muh, double delta)
{
  double begin = muh + 1e-10;
  double end = 1.0-1e-10;
  double mid = (begin+end)/2.0;
  while (begin < end - 1e-10)
  {
    if (n*kl(muh, mid) > delta)
    {
      end = mid;
    }
    else
    {
      begin = mid;
    }
    mid = (begin + end)/2.0;
  }
  return mid;
} 

const int NUM_AGI_ITERS = 9;

double computeAGI(const double gamma, const double a, const double nn)
{
  double p = double(a)/double(nn);
  const double b = nn - a;
  const double r = ((double)a/(double)nn);
  
  for(int k = 0; k < NUM_AGI_ITERS; ++k)
  {
    if((p > 1 || p < r || p != p))
    {
      cerr << "Found that p = " << p << ". invalid value" << endl;
      cerr << "a = " << a << endl;
      cerr << "b = " << b << endl;
      cerr << "gamma = " << gamma << endl;
      exit(1);
    }
    const double safe = p/(1-gamma);

    const double lhs = r*(1-ibeta(a+1,b,p));
    const double expmax = lhs + p*ibeta((double)a,b,p);  
    const double risky = r+ gamma/(1-gamma)*expmax;

    const double orig = risky - safe;
    const double deriv = gamma/(1-gamma)*ibeta(a,b,p) - 1/(1-gamma);
    p -= orig/deriv;
  }
  return p;
}

double computeAGIGaussian(const double gamma, const double mu, const double sigma)
{
  double lam = mu;
  boost::math::normal_distribution<> normdist(mu,sigma);
  for(int k = 0; k < NUM_AGI_ITERS; ++k)
  {
    double plam = cdf(normdist, lam);
    double flam =  pdf(normdist, lam);
    double f = mu + gamma*sigma/(sqrt(2*PI))*exp(-pow(lam-mu,2)/(2*sigma*sigma)) + gamma*(lam-mu)*plam - lam; 
    double fp = -gamma*(lam - mu)/(sqrt(2*PI)*sigma)*exp(-pow((lam-mu),2)/(2*sigma*sigma)) + gamma*plam + gamma*(lam-mu)*flam - 1;
    lam -= f/fp;
  }
  return lam;
}

double computeDummyProblemVal(double lambda, double gamma, double a, double b, int k, bool take_max = false)
{
  double xbar = a/(a+b);
  double result = 0.0;
  if (k == 1)
  {
    result += xbar;
    result += gamma/(1.0-gamma) * (xbar * (1.0 - ibeta(a+1, b, lambda)) + lambda*ibeta(a,b,lambda));
    return take_max ? max(result,lambda/(1.0 - gamma)) : result;
  }
  result += xbar * computeDummyProblemVal(lambda, gamma, a+1, b, k-1, true);
  result += (1.0 - xbar) * computeDummyProblemVal(lambda, gamma, a, b+1, k-1, true);
  result = xbar + gamma * result;
  return take_max ? max(result,lambda/(1.0 - gamma)) : result;
}

double computeApproxGI(double gamma, double a, double b, int k, double prec = 0.0001)
{
  if (k == 1)
  {
    return computeAGI(gamma, a, a + b);
  }
  double start = a/(a+b);
  for (double lambda = start; lambda < 1.0; lambda += prec) 
  {
    double rightVal = computeDummyProblemVal(lambda, gamma, a, b, k, false);
    double leftVal = lambda / (1.0 - gamma);
    if (leftVal > rightVal)
    {
      return lambda - prec/2.0;
    }
  }
  return 1.0;
}

void computeIndices(const int a0, const int b0, const int n, const double step, const double beta, vector<vector<double> >& indices)
{
  vector<vector<double> > tmpindices(n-1, vector<double>(n-1, 0));
  for(int a = 1; a < n; ++a)
  {
    tmpindices[a-1][n-a-1] = computeAGI(beta, a0 + a - 1, n + a0 + b0 - 2);
  }
  for(double p=(step/2.0); p <= 1.0; p+=step)
  {
    const double safe = p/(1-beta);
    for(int nn = n-1; nn >= 2; --nn)
    {
      for(int a = 1; a <= nn-1; ++a)
      {
        const double r = (double(a - 1 + a0)/double(nn + a0 + b0 - 2));
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

#endif

