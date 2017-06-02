#ifndef STATS_H
#define STATS_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

const int SKIPLEN = 10000;

class Stats
{
public:
  Stats(int horizon, int trials, string name)
    : horizon(horizon),
      trials(trials),
      clen(horizon/SKIPLEN),
      total_regret(clen, 0),
      total_regret_squared(clen, 0),
      mean_regret(clen, 0),
      var_regret(clen, 0),
      final_regret(trials, 0),
      final_reg_idx(0),
      _name(name),
      std_error(0)
  {}

  void output();
  void operator()(double sample, int t);
  double mean_final_regret() const;
  const string& name() const { return _name; }
  const double mse() const { return std_error; }
  void prepare();
  double quantile(double q) const;
  const vector<double>& cumulative_regret() const { return mean_regret; }
  
  int horizon, trials, clen;
  vector<double> total_regret;
private:

  vector<double> total_regret_squared;
  vector<double> mean_regret;
  vector<double> var_regret;
  vector<double> final_regret;
  int final_reg_idx;
  string _name;
  double std_error;
};

#endif
