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
  
  po::options_description desc("Options");
  desc.add_options()
    ("help", "Display help message")
    ("trials", po::value<int>(), "Number of random trials")
    ("time", po::value<int>(), "Number of time periods in simulations")
    ("arms", po::value<int>(), "Number of arms used in the problem")
    ("tfile", po::value<string>(), "File to which to write the results table")
    ("cfile", po::value<string>(), "File to which to write the curves")
    //("ltot",po::value<bool>(),  "Wether the discount factor should be log(t)/t")
    //("aggressive", po::value<bool>(), "If all successes, then always pull that arm")
    //("agi_depth", po::value<int>(), "How many steps to look ahead for approximating gittins index")
    ("a_param", po::value<double>(), "The 'a' parameter used for custom discount factor")
    ("b_param", po::value<double>(), "The 'b' parameter used for custom discount factor")
    ("gaussian", "Whether or not to run the Gaussian experiment")
    ("std_out", "Whether or not to print curves to stdout")
    //("onlf", po::value<bool>(), "Use a (a+1,b) prior rather than (a,b)")
    ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, args, desc), vm);
  po::notify(vm);    

  if (vm.count("help"))
  {
    cout << desc << endl;
    exit(0);
  }

  if (!vm.count("arms") || !vm.count("trials") || !vm.count("time"))
  {
    cerr << "Missing required arguments: arms, trials and time" << endl;
    cout << desc << endl;
    exit(1);
  }

  int arms = vm["arms"].as<int>();
  unsigned long num_trials = vm["trials"].as<int>();
  unsigned long horizon = vm["time"].as<int>();
  double a_param = 1, b_param = 0;
  const bool gaussian = vm.count("gaussian");
  const bool p_stdout = vm.count("std_out");
  string tfile = "", cfile = "";
  if (vm.count("a_param"))
    a_param = vm["a_param"].as<double>();
  if (vm.count("b_param"))
    b_param = vm["b_param"].as<double>();
  if (vm.count("tfile"))
    tfile = vm["tfile"].as<string>();
  if (vm.count("cfile"))
    cfile = vm["cfile"].as<string>();

  simulate(arms, num_trials, horizon, gaussian, a_param, b_param, tfile, cfile, p_stdout);
}


