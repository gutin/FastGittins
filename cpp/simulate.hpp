#ifndef SIMULATE_HPP
#define SIMULATE_HPP

#include <string>

void simulate(int arms, 
              unsigned long num_trials,
              unsigned long horizon,
              bool gaussian,
              double a_param,
              double b_param,
              const std::string& tablefile,
              const std::string& curvefile,
              bool p_stdout);

#endif
