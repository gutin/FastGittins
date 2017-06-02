#include "stats.hpp"
#include <cassert>
#include <cmath>

void Stats::output() 
{
  prepare();
  for (int t = 0; t < horizon-1; ++t)
  {
    cout << mean_regret[t] << ", ";
  }
  cout << mean_regret[horizon-1] << endl;
  cout << quantile(0.25) << endl;
  cout << quantile(0.5) << endl;
  cout << quantile(0.75) << endl; 
}

double Stats::mean_final_regret() const
{
  return mean_regret[clen-1];
}

void Stats::operator() (double sample, int t)
{
  assert(1 <= t && t <= horizon);
  if ((t-1)%SKIPLEN == 0)
  {
    total_regret[(t-1)/SKIPLEN] += sample;
    total_regret_squared[(t-1)/SKIPLEN] += sample*sample;
  }
  if (t == horizon)
  {
    assert(final_reg_idx >= 0 && final_reg_idx <= trials-1);
    final_regret[final_reg_idx] = sample;
    final_reg_idx++;
  }
}

double Stats::quantile(double q) const
{
  int idx = (int) floor(q*double(trials));
  return final_regret[idx];
}

void Stats::prepare()
{
  for (int t = 0; t < clen; ++t)
  {
    mean_regret[t] = total_regret[t] / double(trials);
  }
  
  this->std_error = 0;
  for (int tr = 0; tr < trials; ++tr)
  {
    std_error += pow(mean_regret[clen-1] - final_regret[tr], 2.0);
  }
  std_error /= double(trials);
  std::sort (final_regret.begin(), final_regret.end());
}
