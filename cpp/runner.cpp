#include <iostream>
#include <sstream>
#include <boost/program_options.hpp>
#include <fstream>
#include <string>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>

#include "simulate.hpp"

using namespace std;
namespace po = boost::program_options;

int main(int argc, const char** args)
{
  unsigned long num_trials = 0;
  unsigned long horizon = 0;
  int arms = 0;
  bool gaussian = false; 
  if(argc != 5)
  {
    cerr << "need 4 arguments [num_arms] [num_trials] [horizon] [gaussian]" << endl;
    exit(1);
  }

  string strnarms = args[1];
  stringstream ssa(strnarms);
  ssa >> arms;

  string strntrials = args[2];
  stringstream ss(strntrials);
  ss >> num_trials;

  string strhor = args[3];
  stringstream ssh(strhor);
  ssh >> horizon;

  string strgauss = args[4];
  stringstream ssg(strgauss);
  ssg >> gaussian;

  simulate(arms, num_trials, horizon, gaussian, 1.0, 100.0, "", "", true);
}
