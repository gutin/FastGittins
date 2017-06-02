#include "tex_table_writer.hpp"

#include <iomanip>
#include <fstream>
#include <algorithm>

void writeTable(const vector<Stats*>& algo_stats, ofstream& tablestream)
{
  int num_algos = algo_stats.size();
  tablestream << "\\begin{tabular}{| y{50pt} |";
  for (int i = 0; i < num_algos; ++i)
    tablestream << " y{35pt} |";
  tablestream << "} \\hline" << endl;
  tablestream << "\t\\textbf{Algorithm}";
  tablestream << setprecision(2) << fixed;
  for (int i = 0; i < num_algos; ++i)
  {
    Stats* stats = algo_stats[i];
    stats->prepare();
    tablestream << " & \\textbf{" << stats->name() << "}";
  }
  tablestream << " \\\\ \\hline" << endl;
  
  tablestream << "\tMean Regret";
  for (int i = 0; i < num_algos; ++i)
  {
    Stats* stats = algo_stats[i];
    tablestream << " & " << stats->mean_final_regret();
  }
  tablestream << " \\\\ \\hline" << endl;

  tablestream << "\tMSE";
  for (int i = 0; i < num_algos; ++i)
  {
    Stats* stats = algo_stats[i];
    tablestream << " & " << stats->mse();
  }
  tablestream << " \\\\ \\hline" << endl;

  tablestream << "\t1st quartile";
  for (int i = 0; i < num_algos; ++i)
  {
    Stats* stats = algo_stats[i];
    tablestream << " & " << stats->quantile(0.25);
  }
  tablestream << " \\\\ \\hline" << endl;

  tablestream << "\tMedian";
  for (int i = 0; i < num_algos; ++i)
  {
    Stats* stats = algo_stats[i];
    tablestream << " & " << stats->quantile(0.5);
  }
  tablestream << " \\\\ \\hline" << endl;

  tablestream << "\t3rd quartile";
  for (int i = 0; i < num_algos; ++i)
  {
    Stats* stats = algo_stats[i];
    tablestream << " & " << stats->quantile(0.75);
  }
  tablestream << " \\\\ \\hline" << endl;
  tablestream << "\\end{tabular}" << endl;
}

void printTable(const vector<Stats*>& algo_stats, const string& tablefile, const string& curvefile, bool p_stdout)
{
  if (!p_stdout)
  {
  int max_num_cols = 6;
  ofstream tablestream;
  tablestream.open(tablefile.c_str());
  int num_algos = algo_stats.size();
  int algos_covered = 0;
  while (algos_covered < num_algos)
  {
    int begin = algos_covered;
    int end = std::min(algos_covered + max_num_cols, num_algos);
    vector<Stats*> smaller_stats(&algo_stats[begin], &algo_stats[end]);
    writeTable(smaller_stats, tablestream);
    algos_covered += smaller_stats.size();
  }
  tablestream.close();
  }
  ofstream curvestream;
  if (curvefile != "")
    curvestream.open(curvefile.c_str());
  for (int i = 0; i < num_algos; ++i)
  {
    Stats* stats = algo_stats[i];
    const vector<double>& cum_regret = stats->cumulative_regret();
    for (int t = 0; t < stats->clen-1; ++t)
    {
      if (curvefile != "")
        curvestream << cum_regret[t] << ", "; 
      if (p_stdout)
        cout << cum_regret[t] << ", ";
    }
    if (curvefile != "")
      curvestream << cum_regret[stats->clen-1] << endl;
    if (p_stdout)
      cout << cum_regret[stats->clen-1] << endl;
  }
  if (curvefile != "")
    curvestream.close();
}
