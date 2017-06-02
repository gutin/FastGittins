#include "policy.hpp"
#include "helpers.hpp"
#include <ctime>
#include <sys/time.h>
#include <boost/math/distributions.hpp>
#include <stdlib.h>

template <class Policy>
double timePolicy(Policy& pol, int arms, int horizon, bool gaussian, RandUnif& urandom, RandNorm& nrandom)
{

  int trials = 1;
  static double mean_rewards[] = {0.6, 0.1, 0.7, 0.45, 0.43, 0.32, 0.8, 0.93, 0.4, 0.5};  
  clock_t timing = clock();
  for (int tr= 1; tr <= trials; ++tr)
  {
    Table rewards(arms, vector<double>(horizon, 0));
    for (int a = 0; a < arms; ++a)
    {
      double mu = mean_rewards[a];
      for (int t = 0; t < horizon; ++t)
      {
        rewards[a][t] = gaussian ? (mu + nrandom()) : (urandom() < mu ? 1 : 0);
      }
    }
    for (int t = 1; t <= horizon; ++t)
    {
      int arm = pol.choosearm(t);

      pol.totalsampled[arm] += 1;
      double outcome = rewards[arm][pol.sample_index[arm]];
      pol.totalrewards[arm] += outcome;
      pol.sample_index[arm]++;
      pol.totalscore += outcome;
    }
  }
  timing = clock() - timing;
  return ((double)timing)/CLOCKS_PER_SEC;
}

int main(int argc, char** argv)
{
  timeval tv;
  gettimeofday(&tv,NULL);

  boost::mt19937 seed(tv.tv_sec);
  boost::uniform_real<> dist(0.0,1.0);
  boost::normal_distribution<> normdist(0.0,1.0);
  boost::uniform_int<> idist(1,1000000);
  RandomSampleHelper bh(tv.tv_sec);
  RandUnif urandom(seed,dist);
  RandNorm nrandom(seed,normdist);
  int horizon = 1000;
  int arms = 10;

  vector<double> uniforms(horizon, 0);
  IDSPolicy idsBer(horizon, uniforms,  arms, false, 1000);
  IDSPolicy idsGauss(horizon, uniforms, arms, true, 1000);
  InterestingPolicy nitpBer(horizon, arms, 1, 0, false);
  InterestingPolicy nitpBer3(horizon, arms, 1, 0, false, 3);
  InterestingPolicy nitpGauss(horizon, arms, 1, 0, true);
  
  ThompsonPolicy tompBer(bh, horizon, arms, false);
  ThompsonPolicy tompGauss(bh, horizon, arms, true);
  BayesUCBPolicy bucbBer(horizon, arms, false);
  BayesUCBPolicy bucbGauss(horizon, arms, true);

  cout << "TOM:" << timePolicy(tompBer, arms, horizon, false, urandom, nrandom) << endl;
  cout << "BUCB:" << timePolicy(bucbBer, arms, horizon, false, urandom, nrandom) << endl;
  cout << "IDS:" << timePolicy(idsBer, arms, horizon, false, urandom, nrandom) << endl;
  cout << "OURS-1:" << timePolicy(nitpBer, arms, horizon, false, urandom, nrandom) << endl;
  cout << "OURS-3:" <<  timePolicy(nitpBer3, arms, horizon, false, urandom, nrandom) << endl;
  
  cout << endl << "Gaussian!" << endl;
  cout << "TOM:" << timePolicy(tompGauss, arms, horizon, true, urandom, nrandom) << endl;
  cout << "BUCB:" << timePolicy(bucbGauss, arms, horizon, true, urandom, nrandom) << endl;
  cout << "IDS:" << timePolicy(idsGauss, arms, horizon, true, urandom, nrandom) << endl;
  cout << "OURS-1:" << timePolicy(nitpGauss, arms, horizon, true, urandom, nrandom) << endl;
}
