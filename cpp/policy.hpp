#ifndef POLICY_HPP
#define POLICY_HPP

#include <limits>
#include "helpers.hpp"
#include "stats.hpp"
#include <boost/math/distributions/geometric.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/shared_ptr.hpp>

using boost::shared_ptr;

template<class PT>
class PolicyBase
{
public:
  PolicyBase(unsigned long horizon, const int arms) : 
    totalrewards(arms, 0), sample_index(arms, 0), horizon(horizon), totalsampled(arms,0), totalscore(0), arms(arms) {}

  void simulate(const Table& rewards, const unsigned long t, const double mustar, 
    Stats& stats)
  {
    int arm = (static_cast<PT*>(this))->choosearm(t);
    totalsampled[arm] += 1;
    double outcome = rewards[arm][sample_index[arm]];
    totalrewards[arm] += outcome;
    sample_index[arm]++;
    totalscore += outcome;
    const double instantregret = (t*mustar) - totalscore;
    #pragma omp critical
    {
      stats(instantregret, t);
    }
  }
//protected:
  vector<double> totalrewards;
  vector<int> sample_index;
  unsigned long horizon;
  vector<double> totalsampled;
  double totalscore;
  const int arms;
};

template<class PT>
class PolicyCollection
{
public:
  
  void simulate(Table& rewards, const unsigned long t, const double mustar,
    vector<Stats>& totalregret)
  {
    for (unsigned int i = 0; i < pcs.size(); ++i)
    {
      pcs[i]->simulate(rewards, t, mustar, totalregret[i]); 
    }
  }

  vector<shared_ptr<PT> > pcs;
};

class ThompsonPolicy : public PolicyBase<ThompsonPolicy>
{
public:
  ThompsonPolicy(RandomSampleHelper& bh, unsigned long horizon, int arms, bool gaussian) :
    PolicyBase<ThompsonPolicy>(horizon, arms), bh(bh), gaussian(gaussian) {}
  int choosearm(int t)
  {
    int result = -1;
    double max = -std::numeric_limits<double>::max();
    for(int a = 0; a < arms; ++a)
    {
      if (totalsampled[a] == totalrewards[a])
        return a;
      const double samp = gaussian ? bh.sampleGaussian(totalrewards[a]/(1.0+totalsampled[a]), sqrt(1.0/(1.0+totalsampled[a]))) : 
        bh.sampleBeta(totalrewards[a] + 1, totalsampled[a] - totalrewards[a] + 1);
      if(samp > max)
      {
        result = a;
        max = samp;
      }
    }
    return result;
  }

private:
  RandomSampleHelper& bh;
  bool gaussian;
};

class UCB1Policy : public PolicyBase<UCB1Policy>
{
public:
  UCB1Policy(unsigned long horizon, int arms) :
    PolicyBase<UCB1Policy>(horizon, arms) {}

  int choosearm(int t)
  {
    int result = -1;
    double max = -std::numeric_limits<double>::max();
    if(t <= arms)
      return t-1;
    for(int a = 0; a < arms; ++a)
    {
      const double avg = double(totalrewards[a])/double(totalsampled[a]);
      const double ucb = avg + sqrt(2*log(double(t))/double(totalsampled[a]));
      if(ucb > max)
      {
        result = a;
        max = ucb;
      }
    }
    return result;
  }
};

class BayesUCBPolicy : public PolicyBase<BayesUCBPolicy>
{
public:
  BayesUCBPolicy(unsigned long horizon, int arms, bool gaussian) :
    PolicyBase<BayesUCBPolicy>(horizon, arms), gaussian(gaussian) {}

  int choosearm(int t)
  {
    int result = -1;
    double max = -std::numeric_limits<double>::max();
    if(t <= arms)
      return t-1;
    for(int a = 0; a < arms; ++a)
    {
      const double gamma = 1.0 - 1.0 / double(t);
      double ucb = 0;
      if (gaussian)
      {
        double mun = totalrewards[a]/(1.0+totalsampled[a]);
        double sigman = sqrt(1.0/(1.0+totalsampled[a]));
        boost::math::normal_distribution<double, policies::policy<> > normdist(mun,sigman);
        ucb = boost::math::quantile(normdist, gamma);
      }
      else 
      {
        ucb = ibeta_inv(totalrewards[a]+1, totalsampled[a]-totalrewards[a] + 1, gamma); 
      }
      if(ucb > max)
      {
        result = a;
        max = ucb;
      }
    }
    return result;
  }
private:
  bool gaussian;
};

class NaivePolicy : public PolicyBase<NaivePolicy>
{
public:
  NaivePolicy(unsigned long horizon, int arms) :
    PolicyBase<NaivePolicy>(horizon, arms) {}

  int choosearm(int t)
  {
    int result = -1;
    double max = -numeric_limits<double>::max();
    if(t <= arms)
      return t-1;
    #ifdef PRINT_POLICY
    cout << "round [" << t << "]. ";
    #endif
    for(int a = 0; a < arms; ++a)
    {
      const double avg = double(totalrewards[a])/double(totalsampled[a]);
      #ifdef PRINT_POLICY
      cout << "avg for arm [" << a << "] is [" << avg << "]. ";
      #endif
      if(avg > max)
      {
        result = a;
        max = avg;
      }
    }
    #ifdef PRINT_POLICY
    cout << "picked arm [" << result << "]" << endl;
    #endif
    return result;
  }
};

class GittinsPolicy : public PolicyBase<GittinsPolicy>
{
public:
  GittinsPolicy(const Table3D& indices, unsigned long horizon, int arms, double b_param, bool gaussian) :
    PolicyBase<GittinsPolicy>(horizon, arms), indices(indices), gaussian(gaussian), b_param(b_param)
  {
  }
  
  int choosearm(int it)
  {
    int result = -1;
    double max = -1;
    int offset = (int) b_param;
    for(int a = 0; a < arms; ++a)
    {
      double gindex = 0;
      double g = 1.0 - 1.0/(b_param + double(it));
      if (indices.size() == 0)
      {
        if (gaussian)
          gindex = approximategi2(g, totalrewards[a]/(1.0+totalsampled[a]), 1.0/sqrt(1+totalsampled[a]));
        else
          gindex = approximategi(g, totalrewards[a]+1, totalsampled[a]-totalrewards[a]+1); 
      }
      else {
        if (gaussian)
          gindex = approximategi2(g, totalrewards[a]/(1.0+totalsampled[a]), 1.0/sqrt(1+totalsampled[a]));
        else
          gindex = indices[it + offset -1][totalrewards[a]][totalsampled[a]-totalrewards[a]];
      }
      if(gindex > max)
      {
        result = a;
        max = gindex;
      }
      #ifdef PRINT_POLICY
      cout << "a" << a << " = " << long(totalrewards[a]) << ". b = " << long(totalsampled[a]-totalrewards[a]) << ". gindex = " << gindex << ". ";  
      #endif
    }
    #ifdef PRINT_POLICY
    cout << endl;
    #endif
    return result;
  }
private:
  const Table3D& indices;

  double approximategi(double g, double a, double b)
  {
    const double mu = a/(a+b);
    const double var = a*b/((a+b)*(a+b)*(a+b+1));
    const static double c = -log(g);
    return mu + sqrt(var)*psi(var/c); 
  }

  double psi(const double s)
  {
    if(s<=0.2)
      return sqrt(s/2.0);
    if(s <= 1)
      return 0.49 - 0.11/sqrt(s);
    if (s <= 5)
      return 0.63 - 0.26/sqrt(s);
    if (s <= 15)
      return 0.77 - 0.58/sqrt(s);
    return sqrt(2*log(s) - log(log(s)) - log(16*M_PI));
  }

  double approximategi2(double g, double mun, double sigman)
  {
    return g == 0.0 ? 0.0 : mun + geefunc(sigman, g);
  }

  double geefunc(double s, double gamma)
  {
    return sqrt(-log(gamma))*bfunc(-s*s/log(gamma)); 
  }

  double bfunc(double s)
  {
    if (s <= 1.0/7.0)
      return s/sqrt(2);
    if (s <= 100)
      return exp(-0.02645*(log(s)*log(s)) + 0.89106*log(s) - 0.4873);
    return sqrt(s)*sqrt(2*log(s) - log(log(s)) - log(16*PI));
  }

  bool gaussian;
  double b_param;
};

class IDSMinFunc
{
public: 
  IDSMinFunc(double di, double dj, double gi, double gj)
    : gi(gi), gj(gj), di(di), dj(dj)
  {}
  
  double operator() (double q)
  {
    return (q*di + (1-q)*dj)*(q*di + (1-q)*dj)/(q*gi + (1-q)*gj);
  }
  
  void print()
  {
    cout << "gi:" << gi << ". gj:" << gj;
    cout << "di:" << di << ". dj:" << dj;
    cout << endl;
  }
private:
  double gi, gj, di, dj;
};

class IDSPolicy : public PolicyBase<IDSPolicy>
{
  const vector<double>& uniforms;
  int POINTS; 
  double RANGE;
  double STEPSIZE;
public:
  IDSPolicy(unsigned long horizon, const vector<double>& uniforms, int arms, bool gaussian_, int bpoints) :
    PolicyBase<IDSPolicy>(horizon, arms), 
    uniforms(uniforms),
    POINTS(gaussian_ ? 8000 : bpoints), 
    fi(arms, vector<double>(POINTS, 0)),
    Fi(arms, vector<double>(POINTS, 0)),
    Qi(arms, vector<double>(POINTS, 0)),
    F(POINTS, 0),
    gaussian(gaussian_),
    lastret(-1)
  {
    RANGE = 40.0;
    STEPSIZE = gaussian ? RANGE/double(POINTS) : 1.0/double(POINTS);
    boost::math::normal_distribution<> normdist(0,1);
    for(int i = 0; i < POINTS; ++i)
    {
      double Fprod = 1;
      for(int a =0; a < arms; ++a)
      {
        double x = (gaussian ? -RANGE/2.0 : 0) + STEPSIZE*(i+0.5);
        Fi[a][i] = gaussian ? cdf(normdist, x) : ibeta(1.0,1.0,x);
        Fprod *= Fi[a][i];
        fi[a][i] = gaussian ? pdf(normdist, x) : ibeta_derivative(1.0,1.0,x);
        Qi[a][i] = gaussian ? 0 : 0.5*ibeta(1.0+1.0,1.0,x);
      }
      F[i] = Fprod;
    }
  }
  
  int choosearm(int t)
  {

    vector<double> delta(arms, 0);
    vector<double> gee(arms, 0);
    computeDeltasAndGees(delta, gee);
    
    int istar=-1,jstar=-1;
    double qstar=-1.0;
    setupRandomization(gee, delta, qstar, istar, jstar);
    int ret= uniforms[t-1] < qstar ? istar : jstar;
    assert (ret >= 0 && ret < arms);

    //remember the last arm that was pulled, so we can update relevant values of fi, Fi, etc. on the next iteration
    lastret = ret;
    return ret;
  }

  double probOptimal(int arm)
  {
    double result = 0;
    for(int i = 0; i < POINTS; ++i)
    {
      double fprod = 1;
      for (int a = 0; a < arms; ++a)
      {
        fprod *= a == arm ? 1.0 : Fi[a][i];
      }
      result += fi[arm][i]*fprod;
    }
    return STEPSIZE*result;
  }

  static void setupRandomization(const vector<double>& gee,
                          const vector<double>& delta,
                          double& qstar,
                          int& istar,
                          int& jstar)
  {
    int arms = gee.size(); 
    double opt = numeric_limits<double>::max();
    for(int j = 0; j < arms-1; ++j)
    {
      for(int i = j+1; i < arms; ++i)
      {
        IDSMinFunc minfun(delta[i], delta[j], gee[i], gee[j]); 
        std::pair<double,double> res = brent_find_minima(minfun, 0.0, 1.0, 20);
        assert((res.second < opt) || !(i == 1 && j == 0));
        if(res.second < opt)
        {
          opt = res.second;
          istar = i; jstar = j;
          qstar = res.first;
        }
      }
    }
  }

  void computeDeltasAndGees(vector<double>& delta, vector<double>& gee)
  {
    vector<boost::math::normal_distribution<> > normdists;
    vector<double> muns;
    vector<double> sigmans;
    for (int a = 0; a < arms; ++a)
    {
      double mu = totalrewards[a]/(1.0 + totalsampled[a]); 
      double sigma = sqrt(1.0/(1.0 + totalsampled[a])); 
      assert(!gaussian || abs(mu) < RANGE/2.0 - sigma);
      muns.push_back(mu);
      sigmans.push_back(sigma);
      normdists.push_back(boost::math::normal_distribution<>(mu, sigma));
    }
    vector<double> as(arms, 0);
    vector<double> bs(arms, 0); 
    vector<double> proboptimal(arms, 0);
    for(int i=0; i < arms;++i)
    {
      as[i] = totalrewards[i]+1;
      bs[i] = totalsampled[i]-totalrewards[i]+1;
    }

    if (lastret >= 0)
    {
      const boost::math::normal_distribution<>& newnormdist = normdists[lastret];
      const int newa = totalrewards[lastret]+1;
      const int newb = totalsampled[lastret] - totalrewards[lastret]+1;
      for(int i = 0; i < POINTS; ++i)
      {
        double x =  (gaussian ? -RANGE/2.0 : 0.0) + (i+0.5)*STEPSIZE;
        
        fi[lastret][i] = gaussian ? pdf(newnormdist, x) : ibeta_derivative(newa,newb,x);
        Fi[lastret][i] = gaussian ? cdf(newnormdist, x) : ibeta(newa,newb,x);
        Qi[lastret][i] = gaussian ? 0 : double(newa)/double(newa+newb)*ibeta(newa+1,newb,x);
        F[i] = 1;
        for(int a = 0; a < arms; ++a)
        {
          F[i] *= Fi[a][i];
        }
      }
    }

    for(int a =0; a < arms; ++a)
      proboptimal[a] = probOptimal(a);
    #ifndef NDEBUG
    double totalprob = 0;
    for (int a = 0; a < arms; ++a)
    {
     totalprob += proboptimal[a];
    }
    if (abs(totalprob - 1.0) >=  1e-1)
    {
      cerr << "probabilities must sum to one when in fact totalprob = " << totalprob << endl;
      for (int a = 0; a < arms; ++a)
      {
        cerr << "totalsampled[" << a << "] = " << totalsampled[a] << ". proboptimal[" << a << "] = " << proboptimal[a] << endl;
        cerr << "totalrewards[" << a << "] = " << totalrewards[a] << endl;
      }
      exit(1);
    }
    #endif
    Table Ms(arms,vector<double>(arms,0));
    for(int i = 0; i < arms;++i)
    {
      for(int j = 0; j < arms; ++j)
      {
        double sum=0;
        if(i==j)
        {  
          for(int p=0;p < POINTS; ++p)
          {
            double x = (gaussian ? - RANGE/2.0 : 0) + (p+0.5)*STEPSIZE;
            sum += x*fi[i][p]*F[p]/max(Fi[i][p],0.0000001);
          }
          sum*=STEPSIZE;
          Ms[i][j] = sum/proboptimal[i];
          assert(gaussian || (Ms[i][j] < 1 && Ms[i][j] > 0));
          continue;
        }
        for(int p =0; p < POINTS;++p)
        {
          sum +=gaussian ? fi[i][p]*F[p]*fi[j][p]/max(Fi[i][p]*Fi[j][p],0.000001) : fi[i][p]*F[p]*Qi[j][p]/max(Fi[i][p]*Fi[j][p],0.000001); 
          if(isnan(sum))
          {
            cerr << "sum is nan! probably cuz Fi["<<i<<"]["<<p<<"]=" << Fi[i][p] << ". ";
            cerr << "and Fi["<<j<<"]["<<p<<"]=" << Fi[j][p] << endl;
            cerr << "fi[" << i << "][" << p <<"]="<<fi[i][p] << endl;
            cerr << "Qi[" << j << "][" << p <<"]="<<Qi[j][p] << endl;
            cerr << "F[" << p <<"]="<< F[p] << endl;
            cerr << "the result is=" << fi[i][p]*F[p]*Qi[j][p]/max(Fi[i][p]*Fi[j][p],0.000001) << endl;
            cerr << "the denominator is=" << max(Fi[i][p]*Fi[j][p],0.000001) << endl;
            cerr << "the numerator is " << fi[i][p]*F[p]*Qi[j][p] << endl;
            exit(1);
          }
        }
        sum*=STEPSIZE;
        Ms[i][j] = gaussian ? muns[j] - sum * sigmans[j]*sigmans[j]/proboptimal[i] :  sum/proboptimal[i];
        assert(!isnan(Ms[i][j]));
      }
    }
    double rhostar=0;   
    for(int i = 0; i < arms;++i)
    {
      rhostar += proboptimal[i]*Ms[i][i];
    }
    for(int i = 0; i < arms; ++i)
    {
      delta[i] = gaussian ? rhostar - muns[i] : rhostar - as[i]/(as[i]+bs[i]);
      double geeval=0;
      for(int j = 0; j < arms;++j)
      {
        geeval += gaussian ? proboptimal[j] * (Ms[j][i] - muns[i])*(Ms[j][i] - muns[i]) :  proboptimal[j]*kl(Ms[j][i], as[i]/(as[i]+bs[i]));
        assert(!isnan(geeval));
      }
      gee[i] = geeval;
    }
  }
  
  Table fi;
  Table Fi;
  Table Qi;
  vector<double> F;
  bool gaussian;
  int lastret;
};

class GaussianIDSPolicy : public PolicyBase<GaussianIDSPolicy>
{

public:
  GaussianIDSPolicy(unsigned long horizon, RandUnif& rng, int arms) :
    PolicyBase<GaussianIDSPolicy>(horizon, arms), unif(rng), horizon(horizon)
  {
  }

private:
  RandUnif& unif;
  unsigned long horizon;
};

class InterestingPolicy : public PolicyBase<InterestingPolicy>
{
public:
  InterestingPolicy(unsigned long horizon, int arms, double a_param, double b_param, bool gaussian, int numSteps = 1) :
    PolicyBase<InterestingPolicy>(horizon, arms), a_param(a_param), b_param(b_param), gaussian(gaussian), numSteps(numSteps)
  {
  }

  int choosearm(int it)
  {
    double t = double(it);
    int result = -1;
    double max = -numeric_limits<double>::max();
    
    double  gamma_t = 1.0 - a_param/(b_param + t);

    for(int a = 0; a < arms; ++a)
    {
      #ifdef PRINT_POLICY
      cout << "discount for " << a << " is " << gamma << " ";
      #endif
      const double index = gaussian ? ::computeAGIGaussian(gamma_t, totalrewards[a]/(1+totalsampled[a]), 1.0/sqrt(1+totalsampled[a]))
         : ::computeApproxGI(gamma_t, totalrewards[a] + 1, (long)(totalsampled[a] - totalrewards[a]) + 1, numSteps);
      if(index > max)
      {
        result = a;
        max = index;
      }
      #ifdef PRINT_POLICY
      cout << "a" << a << " = " << long(totalrewards[a]) << ". b = " << long(totalsampled[a]-totalrewards[a]) << ". gindex = " << index << ". ";  
      #endif
    }
    #ifdef PRINT_POLICY
    cout << endl;
    #endif
    return result;
  }
private:
  const double a_param, b_param;
  bool gaussian;
  int numSteps;
};

#endif
