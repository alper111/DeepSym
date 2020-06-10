/*
 * Main program.
 *
 * Copyright 2003-2005 Carnegie Mellon University and Rutgers University
 * Copyright 2007 Håkan Younes
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "states.h"
#include "problems.h"
#include "domains.h"
#include "mdpserver.h"
#if HAVE_GETOPT_LONG
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <getopt.h>
#else
#include "port/getopt.h"
#endif
#include <chrono>
#include <cstdlib>
#include <cerrno>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <climits>
#include <iostream>
#include <fstream>
#include <stdexcept>


/* The parse function. */
extern int yyparse();
/* File to parse. */
extern FILE* yyin;

/* port to run server on */
int port;
/* Name of current file. */
std::string current_file;
/* Level of warnings. */
int warning_level;
/* Verbosity level. */
int verbosity;
/* Log directory. */
std::string log_dir;
/* Whether to log path information during execution. */
bool log_paths;

/* Program options. */
static struct option long_options[] = {
  { "port", required_argument, 0, 'P'},
  { "configuration", required_argument, 0, 'C'},
  { "turn-limit", required_argument, 0, 'L' },
  { "log-dir", required_argument, 0, 'l' },
  { "log-paths", no_argument, 0, 'p' },
  { "round-limit", required_argument, 0, 'R' },
  { "seed", required_argument, 0, 'S' },
  { "time-limit", required_argument, 0, 'T' },
  { "verbose", optional_argument, 0, 'v' },
  { "version", no_argument, 0, 'V' },
  { "warnings", optional_argument, 0, 'W' },
  { "help", no_argument, 0, 'h' },
  { 0, 0, 0, 0 }
};
static const char OPTION_STRING[] = "C:L:l:P:pR:S:T:v::VW::h";

/* Displays help. */
static void display_help() {
  std::cout << "usage: " << PACKAGE << " [options] [file ...]" << std::endl
            << "options:" << std::endl
            << "  -C c,  --configuration=c" << std::endl
            << "\t\t\tuse configuration file c" << std::endl
            << "  -L l,  --turn-limit=l\t"
            << "sets the default turn limit to l" << std::endl
            << "  -l l,  --log-dir=l\t"
            << "use l as repository for logs" << std::endl
            << "  -P p,  --port=p\t"
            << "run the simulation as a server on port p" << std::endl
            << "  -p,    --log-paths\t"
            << "logs complete execution paths if present" << std::endl
            << "  -R r,  --round-limit=r" << std::endl
            << "\t\t\tsets the default round limit to r" << std::endl
            << "  -S s,  --seed=s\t"
            << "uses s as seed for random number generator" << std::endl
            << "  -T t,  --time-limit=t\t"
            << "sets the default time limit (in milliseconds) to t"
            << std::endl
            << "  -v[n], --verbose[=n]\t"
            << "use verbosity level n;" << std::endl
            << "\t\t\t  n is a number from 0 (verbose mode off) and up;"
            << std::endl
            << "\t\t\t  default level is 1 if optional argument is left out"
            << std::endl
            << "  -V,    --version\t"
            << "display version information and exit" << std::endl
            << "  -W[n], --warnings[=n]\t"
            << "determines how warnings are treated;" << std::endl
            << "\t\t\t  0 supresses warnings; 1 displays warnings;"
            << std::endl
            << "\t\t\t  2 treats warnings as errors" << std::endl
            << "  -h     --help\t\t"
            << "display this help and exit" << std::endl
            << "  file ...\t\t"
            << "files containing domain and problem descriptions;" << std::endl
            << "\t\t\t  if none, descriptions are read from standard input"
            << std::endl
            << std::endl
            << "Report bugs to <" PACKAGE_BUGREPORT ">." << std::endl;
}


/* Displays version information. */
static void display_version() {
  std::cout << PACKAGE_STRING << std::endl
            << "Copyright 2003-2005 Carnegie Mellon University"
            << " and Rutgers University" << std::endl
            << "Copyright 2007 Håkan Younes" << std::endl;
}


/* Parses the given file, and returns true on success. */
static bool read_file(const char* name) {
  yyin = fopen(name, "r");

  if (yyin == 0) {
    std::cerr << PACKAGE << ':' << name << ": " << strerror(errno)
              << std::endl;
    return false;
  } else {
    current_file = name;
    bool success = (yyparse() == 0);
    fclose(yyin);
    return success;
  }
}


/* Selects an action for the given state. */
static const Action* action_selection(const AtomSet& atoms,
                                      const ValueMap& values,
                                      const Problem& problem) {
  ActionList actions;
  problem.enabled_actions(actions, atoms, values);
  if (actions.empty()) {
    return 0;
  } else {
    size_t i = size_t(rand()/(RAND_MAX + 1.0)*actions.size());
    return actions[i];
  }
}


/* The main program. */
int main(int argc, char* argv[]) {
  /* config file for problem specific restrictions */
  std::string config;
  /* port=0 means no server */
  port = 0;
  /* Set default verbosity. */
  verbosity = 0;
  /* Set default log directory. */
  log_dir = "logs";
  /* Do not log execution paths by default. */
  log_paths = false;
  /* Set default warning level. */
  warning_level = 1;
  /* Set default seed. */
  size_t seed = time(0);
  /* Set default time limit. */
  std::chrono::milliseconds time_limit = std::chrono::minutes(15);
  /* Set default round limit */
  int round_limit = 30;
  /* Set default turn limit. */
  int turn_limit = INT_MAX;

  /*
   * Get command line options.
   */
  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, OPTION_STRING,
                        long_options, &option_index);
    if (c == -1) {
      break;
    }
    switch (c) {
    case 'C':
      config = optarg;
      break;
    case 'P':
      port = atoi(optarg);
      break;
    case 'L':
      turn_limit = atoi(optarg);
      break;
    case 'l':
      log_dir = optarg;
      break;
    case 'p':
      log_paths = true;
      break;
    case 'R':
      round_limit = atoi(optarg);
      break;
    case 'S':
      seed = atoi(optarg);
      break;
    case 'T':
      time_limit = std::chrono::milliseconds(atol(optarg));
      break;
    case 'v':
      verbosity = (optarg != 0) ? atoi(optarg) : 1;
      break;
    case 'V':
      display_version();
      return 0;
    case 'W':
      warning_level = (optarg != 0) ? atoi(optarg) : 1;
      break;
    case 'h':
      display_help();
      return 0;
    case ':':
    default:
      std::cerr << "Try `" PACKAGE " --help' for more information."
                << std::endl;
      return -1;
    }
  }
  srand(seed);

  try {
    /*
     * Read pddl files.
     */
    if (optind < argc) {
      /*
       * Use remaining command line arguments as file names.
       */
      while (optind < argc) {
        if (!read_file(argv[optind++])) {
          return -1;
        }
      }
    } else {
      /*
       * No remaining command line argument, so read from standard input.
       */
      yyin = stdin;
      if (yyparse() != 0) {
        return -1;
      }
    }

    if (verbosity > 0) {
      std::cerr << "using seed " << seed << std::endl;
    }
    if (verbosity > 1) {
      /*
       * Display domains and problems.
       */
      std::cerr << "----------------------------------------"<< std::endl
                << "domains:" << std::endl;
      for (Domain::DomainMap::const_iterator di = Domain::begin();
           di != Domain::end(); di++) {
        std::cerr << std::endl << *(*di).second << std::endl;
      }
      std::cerr << "----------------------------------------"<< std::endl
                << "problems:" << std::endl;
      for (Problem::ProblemMap::const_iterator pi = Problem::begin();
           pi != Problem::end(); pi++) {
        std::cerr << std::endl << *(*pi).second << std::endl;
      }
      std::cerr << "----------------------------------------"<< std::endl;
    }

    if (port == 0) {
      for (Problem::ProblemMap::const_iterator pi = Problem::begin();
           pi != Problem::end(); pi++) {
        const Problem& problem = *(*pi).second;
        if (verbosity > 0) {
          std::cerr << "simulating problem `" << problem.name() << "'"
                    << std::endl;
        }
        const State* s = new State(problem);
        int time = 0;
        while (time < turn_limit && !s->goal()) {
          const Action* action =
            action_selection(s->atoms(), s->values(), problem);
          if (action == 0) {
            break;
          }
          std::cout << std::endl << time << ": " << *s << std::endl;
          const State& next_s = s->next(*action);
          delete s;
          s = &next_s;
          time++;
        }
        std::cout << std::endl << time << ": " << *s << std::endl;
        if (s->goal()) {
          std::cout << "  goal achieved" << std::endl;
        } else if (time >= turn_limit) {
          std::cout << "  turn limit reached" << std::endl;
        }
        if (!problem.constant_metric()) {
          std::cout << "  value of maximization metric is "
                    << problem.metric().value(s->values()) << std::endl;
        }
        delete s;
      }
    } else {
      if (!config.empty()) {
        std::ifstream fin(config.c_str());
        if (!fin) {
          throw std::runtime_error("cannot open configuration file `"
                                   + config + "'");
        }
        while (!fin.eof()) {
          std::string problem_name;
          fin >> problem_name;
          if (problem_name.empty()) {
            break;
          }
          Problem_CFG cfg;
          std::chrono::milliseconds::rep time_limit_value;
          fin >> time_limit_value;
          cfg.time_limit = std::chrono::milliseconds(time_limit_value);
          fin >> cfg.round_limit;
          fin >> cfg.turn_limit;
          config_map[problem_name] = cfg;
        }
      }
      return run_server(port, time_limit, round_limit, turn_limit);
    }
  } catch (const std::exception& e) {
    std::cerr << PACKAGE ": " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << PACKAGE ": fatal error" << std::endl;
    return -1;
  }

  Problem::clear();
  Domain::clear();

  return 0;
}
