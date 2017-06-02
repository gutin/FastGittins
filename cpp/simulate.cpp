#include "simulate.hpp"

#include <iostream>
#include <sstream>
#include <boost/program_options.hpp>
#include <fstream>
#include <vector>
#include <algorithm>
#include <boost/math/distributions.hpp>
#include <ctime>
#include <string>
#include <cstdlib>
#include <sys/time.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "helpers.hpp"
#include "policy.hpp"
#include "stats.hpp"

using namespace std;
using namespace boost::math;
using namespace boost::random;

static const int MAX_ARMS = 10;
static const string WEIGHTS_FILE = "/home/gutin/research/mab/cpp/rewards.config";

void simulate(int arms, 
              unsigned long num_trials,
              unsigned long horizon,
              bool gaussian,
              double a_param,
              double b_param,
              const string& tablefile,
              const string& curvefile,
              bool p_stdout)
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
   
  RandIntUnif urandomint(seed,idist);

  #ifdef _OPENMP
  int max_threads = omp_get_max_threads();
  RandUnif* rngs[max_threads];
  RandNorm* nrngs[max_threads];
  for(int i = 0; i < max_threads; ++i)
  {
    long randomint = urandomint();
    boost::mt19937 subseed(randomint);
    rngs[i] = new RandUnif(subseed, dist);
    nrngs[i] = new RandNorm(subseed, normdist);
  }
  #endif

  Table3D indices99;
  if (!gaussian && horizon == 999)
  {
    for (unsigned int t = 1; t <= horizon+200; ++t)
    {
      Table indices;
      stringstream ss;
      ss << t;
      string number = ss.str();
      loadFromCSV("/dev/shm/gixs/hor_" + number + ".csv", indices, horizon, horizon);
      indices99.push_back(indices);
    }
  }

  #ifdef SIMMULT
  const int minalpha = 50;
  const int maxalpha = 150;
  const int alphainc = 50;
  vector<Stats> regretmult;
  vector<Stats> g_regretmult;
  int maxsteps = 1;
  for (double j = minalpha; j <= maxalpha; j += alphainc)
  {
    for (int ksteps = 1; ksteps <= maxsteps; ksteps += 2)
    {
      ostringstream os;
      os << "OGI(" << ksteps << ")-" << j;
      string name = os.str();
      regretmult.push_back(Stats(horizon, num_trials, name));
    }
    ostringstream os;
    os << "OGI-" << "-" << j;
    string name = os.str();
    //g_regretmult.push_back(Stats(horizon, num_trials, name));
  }
  #endif
  #ifdef SIMSHD
  Stats regretshd(horizon, num_trials, "OGI-1");
  #endif
  #ifdef SIMSHD2
  Stats regretshd2(horizon, num_trials, "OGI-2");
  #endif
  #ifdef SIMSHD3
  Stats regretshd3(horizon, num_trials, "OGI-3");
  #endif
  #ifdef SIMSHD5
  Stats regretshd5(horizon, num_trials, "OGI-5");
  #endif
  #ifdef SIMNEW
  Stats regretgit1(horizon, num_trials, "OGI-Gaussian");
  #endif
  #ifdef SIMGIT
  Stats regretgit(horizon, num_trials, "OGI");
  #endif
  #ifdef SIMTOM
  Stats regrettom(horizon, num_trials, "TS");
  #endif
  #ifdef SIMNAIVE
  vector<double> regretnaive(horizon, 0);
  #endif
  #ifdef SIMIDS
  Stats regretids(horizon, num_trials, "IDS");
  #endif
  #ifdef SIMIDS2
  Stats regretids2(horizon, num_trials, "IDS2");
  #endif
  #ifdef SIMBUCB
  Stats regretbucb(horizon, num_trials, "Bayes UCB");
  #endif
  #ifdef SIMUCB1
  Stats regretucb1(horizon, num_trials, "UCB1");
  #endif
  #ifdef SHOWTIMING
  clock_t timing = clock();
  #endif

  vector<double> default_mus(arms, 0);
  loadLineFromCSV(WEIGHTS_FILE, default_mus, arms);
  #pragma omp parallel
  {
    #ifdef _OPENMP
    int threadid = omp_get_thread_num();
    #endif
    
    vector<double> mus(arms, 0);
    for(int i = 0; i < arms; ++i)
      mus[i] = default_mus[i];
    #pragma omp for
    for(unsigned int trial = 0; trial < num_trials; ++trial)
    {
      double mustar = -1;
      int astar = -1;
      Table rewards(arms, vector<double>(horizon, 0));
      vector<double> uniforms(horizon, 0);
      for (int a = 0; a < arms; ++a)
      { 
        #ifdef RANDARMS
        #ifdef _OPENMP
	      double mu = gaussian ? (*nrngs[threadid])() : (*rngs[threadid])();
        #endif
        #ifndef _OPENMP
        double mu = gaussian ? nrandom() : urandom();
        #endif
        #else
        double mu = default_mus[a];
        #endif

        if (mu > mustar)
        {
          mustar = mu;
          astar = a;
        }
        
        for (unsigned long t = 0; t < horizon; ++t)
        {
          #ifdef _OPENMP
          rewards[a][t] = gaussian ? (mu + (*nrngs[threadid])()) : ((*rngs[threadid])() < mu ? 1 : 0);
          #endif
          #ifndef _OPENMP
          rewards[a][t] = gaussian ? (mu + nrandom()) : (urandom() < mu ? 1 : 0);
          #endif
        }
      }
      for (unsigned long t = 0; t < horizon; ++t)
      {
        #ifdef OPENMP
        uniforms[t] = (*rngs[threadif])();
        #else
        uniforms[t] = urandom();
        #endif
      }
      #ifdef SIMMULT
      PolicyCollection<InterestingPolicy> pcl;
      PolicyCollection<GittinsPolicy> gcl;
      for (double j = minalpha; j <= maxalpha; j += alphainc)
      {
        for (int ksteps = 1; ksteps <= maxsteps; ksteps += 2)
          pcl.pcs.push_back( shared_ptr<InterestingPolicy>( new InterestingPolicy(horizon, arms, 1, j, gaussian, ksteps) ) );
        //gcl.pcs.push_back( shared_ptr<GittinsPolicy>( new GittinsPolicy(indices99, horizon, arms, j, gaussian)));
      }
      #endif
      #ifdef SIMNAIVE
      NaivePolicy naip(horizon, arms);
      #endif
      #ifdef SIMTOM
      ThompsonPolicy tomp(bh, horizon, arms, gaussian);
      #endif
      #ifdef SIMIDS
      #ifdef _OPENMP
      IDSPolicy ids(horizon, uniforms, arms, gaussian, 10000);
      #endif
      #ifndef _OPENMP
      IDSPolicy ids(horizon, uniforms, arms, gaussian, 10000);
      #endif
      #endif
      #ifdef SIMIDS2
      IDSPolicy ids2(horizon, uniforms, arms, gaussian, 2000);
      #endif
      #ifdef SIMSHD
        #ifdef _OPENMP
      InterestingPolicy nitp(horizon, arms, a_param, b_param, gaussian);
        #else
      InterestingPolicy nitp(horizon, arms, a_param, b_param, gaussian);
        #endif
      #endif
      #ifdef SIMSHD2
        #ifdef _OPENMP
      InterestingPolicy nitp2(horizon, arms, a_param, b_param, gaussian, 2);
        #else
      InterestingPolicy nitp2(horizon, arms, a_param, b_param, gaussian, 2);
        #endif
      #endif
      #ifdef SIMSHD3
      InterestingPolicy nitp3(horizon, arms, a_param, b_param, gaussian, 3);
      #endif
      #ifdef SIMSHD5
      InterestingPolicy nitp5(horizon, arms, a_param, b_param, gaussian, 5);
      #endif
      #ifdef SIMNEW
      GittinsPolicy gitp99(indices99, horizon, arms, b_param, gaussian); 
      #endif
      #ifdef SIMGIT
      GittinsPolicy gitp(indices99, horizon, arms, b_param, gaussian); 
      #endif
      #ifdef SIMBUCB
      BayesUCBPolicy bucb(horizon, arms, gaussian);
      #endif
      #ifdef SIMUCB1
      UCB1Policy ucb1(horizon, arms); 
      #endif
      for(unsigned long t = 1; t <= horizon; ++t)
      {
        #ifdef SIMMULT
        pcl.simulate(rewards, t, mustar, regretmult);
        gcl.simulate(rewards, t, mustar, g_regretmult);
        #endif
        #ifdef SIMNAIVE
        naip.simulate(rewards, t, mustar, regretnaive);
        #endif
        #ifdef SIMTOM
        tomp.simulate(rewards, t, mustar, regrettom);
        #endif
        #ifdef SIMIDS
        ids.simulate(rewards, t, mustar, regretids);
        #endif
        #ifdef SIMIDS2
        ids2.simulate(rewards, t, mustar, regretids2);
        #endif
        #ifdef SIMSHD
        nitp.simulate(rewards, t, mustar, regretshd);
        #endif
        #ifdef SIMSHD2
        nitp2.simulate(rewards, t, mustar, regretshd2);
        #endif
        #ifdef SIMSHD3
        nitp3.simulate(rewards, t, mustar, regretshd3);
        #endif
        #ifdef SIMSHD5
        nitp5.simulate(rewards, t, mustar, regretshd5);
        #endif
        #ifdef SIMNEW
        gitp99.simulate(rewards, t, mustar, regretgit1);
        #endif
        #ifdef SIMGIT
        if (!gaussian && horizon == 1000)
          gitp.simulate(rewards, t, mustar, regretgit);
        #endif
        #ifdef SIMBUCB
        bucb.simulate(rewards, t, mustar, regretbucb);
        #endif
        #ifdef SIMUCB1
        ucb1.simulate(rewards, t, mustar, regretucb1);
        #endif
      }
    }
  }
  #ifdef SHOWTIMING
  timing = clock() - timing;
  cout << "took " << ((float)timing)/CLOCKS_PER_SEC << " seconds of cpu time" << endl;
  #endif

  vector<Stats*> algo_stats;
  #ifndef PRINT_POLICY 
  #ifdef SIMMULT
  for (unsigned int i = 0; i < regretmult.size(); ++i)
    algo_stats.push_back(&regretmult[i]);
  for (unsigned int i = 0; i < g_regretmult.size(); ++i)
    algo_stats.push_back(&g_regretmult[i]);
  #endif
  #ifdef SIMNAIVE
  printRegret(regretnaive, (double)num_trials);
  #endif
  #ifdef SIMNEW
  algo_stats.push_back(&regretgit1);
  #endif
  #ifdef SIMGIT
  if (!gaussian && horizon == 1000)
    algo_stats.push_back(&regretgit);
  #endif
  #ifdef SIMSHD
  algo_stats.push_back(&regretshd);
  #endif
  #ifdef SIMSHD2
  algo_stats.push_back(&regretshd2);
  #endif
  #ifdef SIMSHD3
  algo_stats.push_back(&regretshd3);
  #endif
  #ifdef SIMSHD5
  algo_stats.push_back(&regretshd5);
  #endif
  #ifdef SIMIDS
  algo_stats.push_back(&regretids);
  #endif
  #ifdef SIMIDS2
  algo_stats.push_back(&regretids2);
  #endif
  #ifdef SIMTOM
  algo_stats.push_back(&regrettom);
  #endif
  #ifdef SIMBUCB
  algo_stats.push_back(&regretbucb);
  #endif
  #ifdef SIMUCB1
  algo_stats.push_back(&regretucb1);
  #endif
  #endif
  int num_algos = algo_stats.size();
  for (int i = 0; i < num_algos; ++i)
  {
    Stats* stats = algo_stats[i];
    for (int t = 0; t < stats->clen-1; ++t)
    {
      cout << stats->total_regret[t] << ", ";
    }
    cout << stats->total_regret[stats->clen-1] << endl;
  }
  #ifdef _OPENMP
  for(int i = 0; i < max_threads; ++i)
  {
    delete rngs[i];
    delete nrngs[i];
  }
  #endif
}
