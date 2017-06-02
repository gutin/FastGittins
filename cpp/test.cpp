#define BOOST_TEST_MODULE dynamicAlgoTests
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "policy.hpp"

BOOST_AUTO_TEST_SUITE( agiTests )

  BOOST_AUTO_TEST_CASE( extremeCase )
  {
    InterestingPolicy agi(10000, 3, 1, 0, false); 
    agi.totalsampled[0] = 7000.0;
    agi.totalrewards[0] = 7000.0;
    agi.choosearm(7001);
  }

  BOOST_AUTO_TEST_CASE( gaussianTest )
  {
    double mu = 1.5;
    double sigma = 3.0;
    double gamma = 0.99;
    double myogi = computeAGIGaussian(gamma, mu, sigma);
    double new_ogi = mu + (1.0/(1.0 - gamma))*sigma*sqrt(2.0/3.14159265358979);
    BOOST_CHECK_CLOSE(new_ogi, myogi, 1e-2);
  }

  BOOST_AUTO_TEST_CASE( extremeCase2 )
  {
    InterestingPolicy agi(200000, 3, 1, 0, false); 
    agi.totalsampled[0] = 200000.0;
    agi.totalrewards[0] = 200000.0;
    agi.choosearm(200001);
  }

  BOOST_AUTO_TEST_CASE (basicFunction)
  {
    double a = 2;
    double b = 3;
    double gamma = 0.9;
    double val = computeDummyProblemVal(0.555, gamma, a, b, 2);
    BOOST_CHECK_CLOSE(val, 5.587903357665029, 1e-3);
  }

  BOOST_AUTO_TEST_CASE (computeSomeIndices)
  {
    double t = 7001.0;
    double gamma  = 1.0-1.0/t;

    double a = 7001.0;
    double b = 1.0;
    double agi1 = computeAGI(gamma,a,a+b);
    double mu = a/(a+b);
    BOOST_CHECK_CLOSE(agi1,mu + gamma*agi1 *ibeta(a,b,agi1) - gamma * mu * ibeta(a+1.0,b,agi1) , 1e-3);
    cout << agi1 << endl;

    a = 1.0;
    b = 1.0;
    agi1 = computeAGI(gamma,a,a+b);
    mu = a/(a+b);
    BOOST_CHECK_CLOSE(agi1,mu + gamma*agi1 *ibeta(a,b,agi1) - gamma * mu * ibeta(a+1.0,b,agi1) , 1e-3);
    cout << agi1 << endl;

    gamma = 0.9;
    a = 2;
    b = 3;
    agi1 = computeAGI(gamma,a,a+b);
    double agi2 = computeApproxGI(gamma, a, b, 2);
    double agi3 = computeApproxGI(gamma, a, b, 3);
    BOOST_REQUIRE(agi1 >= agi2);
    BOOST_REQUIRE(agi2 >= agi3);
    cout << agi1 << " vs " << agi2 << " vs " << agi3 << endl;

    gamma = 1.0 - 1.0/300000.0;
    a = 299999.0;
    b = 1.0;
    agi1 = computeAGI(gamma,a,a+b);
    mu = a/(a+b);
    BOOST_CHECK_CLOSE(agi1,mu + gamma*agi1 *ibeta(a,b,agi1) - gamma * mu * ibeta(a+1.0,b,agi1) , 1e-3);
    cout << agi1 << endl;

    gamma = 1.0 - 1.0/300000.0;
    b = 299999.0;
    a = 1.0;
    agi1 = computeAGI(gamma,a,a+b);
    mu = a/(a+b);
    BOOST_CHECK_CLOSE(agi1,mu + gamma*agi1 *ibeta(a,b,agi1) - gamma * mu * ibeta(a+1.0,b,agi1) , 1e-3);
    cout << agi1 << endl;
  }

BOOST_AUTO_TEST_SUITE_END()



BOOST_AUTO_TEST_SUITE( idsTests )

  BOOST_AUTO_TEST_CASE( setupRandomization1 )
  {
    vector<double> gee;
    vector<double> delta;
    gee.push_back(11);
    gee.push_back(3);
    delta.push_back(8);
    delta.push_back(4);

    double qstar = 0;
    int istar = -1, jstar = -1;
    IDSPolicy::setupRandomization(gee, delta, qstar, istar, jstar);

    BOOST_CHECK_CLOSE(qstar, 0.75, 1e-03);
  }

  BOOST_AUTO_TEST_CASE( setupRandomization2 )
  {
    vector<double> gee;
    vector<double> delta;
    gee.push_back(5);
    gee.push_back(11);
    gee.push_back(3);
    delta.push_back(6);
    delta.push_back(8);
    delta.push_back(5);

    double qstar = 0;
    int istar = -1, jstar = -1;
    IDSPolicy::setupRandomization(gee, delta, qstar, istar, jstar);

    BOOST_CHECK_CLOSE(qstar, 1.0/12.0, 1e-03);
    BOOST_CHECK_EQUAL(istar,2);
    BOOST_CHECK_EQUAL(jstar,1);
  }

  BOOST_AUTO_TEST_CASE( deltaAndGGaussian )
  {
    int arms = 2;
    vector<double> gee(arms, 0);
    vector<double> delta(arms, 0);

    timeval tv;
    gettimeofday(&tv,NULL);
    boost::mt19937 seed(tv.tv_sec);
    boost::uniform_real<> dist(0.0,1.0);
    RandUnif urandom(seed,dist);
    vector<double> foo;
   
    int horizon = 100;
    IDSPolicy ids(horizon, foo, arms, true, 10000);
    ids.totalsampled[0] = 1;
    ids.totalrewards[0] = 0.42;
    ids.lastret = 0;
    cout << "Heres the important part!" << endl;
    ids.computeDeltasAndGees(delta, gee);
    cout << "End of the important part!" << endl;

    BOOST_CHECK_CLOSE(delta[1], 0.6008952288515825, 3*1e-2);
    BOOST_CHECK_CLOSE(delta[0], 0.39081524936808154, 3*1e-2);
    BOOST_CHECK_CLOSE(gee[0], 0.10491394571783878, 1e-1);
    BOOST_CHECK_CLOSE(gee[1], 0.42264277241894466, 1.0);

    ids.totalsampled[1] = 1;
    ids.totalrewards[1] = 0.2;
    ids.lastret = 1;

    ids.computeDeltasAndGees(delta, gee);

    BOOST_CHECK_CLOSE(delta[0], 0.34664110475130916, 1.0);
    BOOST_CHECK_CLOSE(delta[1], 0.45628505787575235, 1.0);
    BOOST_CHECK_CLOSE(gee[0], 0.158893182657543, 1.0);
    BOOST_CHECK_CLOSE(gee[1], 0.15883345977646818, 1.0);
  }
  
  BOOST_AUTO_TEST_CASE( bernoulliDeltasAndGees )
  {
    int arms = 2;
    vector<double> gee(arms, 0);
    vector<double> delta(arms, 0);

    timeval tv;
    gettimeofday(&tv,NULL);
    boost::mt19937 seed(tv.tv_sec);
    boost::uniform_real<> dist(0.0,1.0);
    RandUnif urandom(seed,dist);
   
    vector<double> foo;
    int horizon = 100;
    IDSPolicy ids(horizon, foo, arms, false, 10000);
    ids.totalsampled[0] = 1;
    ids.totalrewards[0] = 1;
    ids.lastret = 0;
    ids.computeDeltasAndGees(delta, gee);

    BOOST_CHECK_CLOSE(delta[1], 0.25, 3*1e-2);
    BOOST_CHECK_CLOSE(delta[0], 0.08328, 1);
    BOOST_CHECK_CLOSE(gee[0], 0.030289, 1);
    BOOST_CHECK_CLOSE(gee[1], 0.06410492092853984, 1.0);
  }

  BOOST_AUTO_TEST_CASE( bernoulliDeltasAndGees_Extreme )
  {
    int arms = 2;
    vector<double> gee(arms, 0);
    vector<double> delta(arms, 0);

    timeval tv;
    gettimeofday(&tv,NULL);
    boost::mt19937 seed(tv.tv_sec);
    boost::uniform_real<> dist(0.0,1.0);
    RandUnif urandom(seed,dist);
   
    vector<double> foo;
    int horizon = 100;
    IDSPolicy ids(horizon, foo, arms, false, 10000);
    ids.totalsampled[0] = 7000;
    ids.totalrewards[0] = 7000;
    ids.lastret = 0;
    ids.computeDeltasAndGees(delta, gee);

  }
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( miscGittinsStuff )

  BOOST_AUTO_TEST_CASE( upperBoundTest )
  {
    Table indices;
    int horizon = 500;
    loadFromCSV("indices.csv", indices, horizon, horizon);
    double gamma = 0.99;
    for (int s = 0; s < horizon; ++s)
    {
      for (int f = 0; f < horizon; ++f)
      {
        const double approximation = computeAGI(gamma, s + 1, f + s + 2);
        const double secondApproximation = computeApproxGI(gamma, s + 1, f + 1, 2);

        BOOST_REQUIRE(indices[s][f] <= approximation); 
        BOOST_REQUIRE(indices[s][f] <= secondApproximation); 
        BOOST_REQUIRE(secondApproximation <= approximation + 1e-3); 
      }
    }
  }

BOOST_AUTO_TEST_SUITE_END()
