#include "helpers.hpp"
#include <iostream>

using namespace std;

int main()
{
  double lambda = 0.76;
  double gamma = 0.99;
  int a = 3;
  int b = 1;
  int k = 3;
  double dumVal1 = computeDummyProblemVal(lambda, gamma, a, b, k, false);
  double dumVal2 = computeDummyProb(lambda, gamma, a, b, k);
  cout << dumVal1 << " vs " << dumVal2 << endl;
}
