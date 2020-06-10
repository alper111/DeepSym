/*
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

#include <config.h>
#include "mdpcommon.h"
#include "mdpserver.h"
#include "strxml.h"
#include "states.h"
#include "problems.h"
#include "domains.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pthread.h>
#include <sstream>

#if !HAVE_SOCKLEN_T
# if !defined(__sgi) || defined(_NO_XOPEN4)
typedef int socklen_t;
# else
typedef size_t socklen_t;
# endif
#endif

extern std::string log_dir;
extern bool log_paths;

/* Global configuration. */
CFG_map config_map;

namespace {

// A simple timer that measures elapsed time since construction.
class Timer {
 public:
  using Clock = std::chrono::steady_clock;

  // Starts the timer.
  Timer() : start_(Clock::now()) {}

  // Returns the elapsed time in milliseconds since construction.
  std::chrono::milliseconds GetElapsedMilliseconds() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - start_);
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace

/* Returns the action with the given parameters, or 0 if no such
   action exists. */
static const Action* make_action(const str_vec& params,
                                 const Problem& problem) {
  str_vec::const_iterator si = params.begin();
  Action action(*si);
  for (si++; si != params.end(); si++) {
    const Object* o = problem.terms().find_object(*si);
    if (o == 0) {
      return 0;
    }
    action.add_argument(*o);
  }
  ActionSet::const_iterator ai =problem.actions().find(&action);
  return (ai != problem.actions().end()) ? *ai : 0;
}


/* Reads the last used client id from file. */
static int read_last_id() {
  int id = 0;
  std::ifstream is("last_id");
  if (is) {
    is >> id;
  }
  return id;
}


/* Writes the given id to file as the last used client id. */
static void write_last_id(int id) {
  std::ofstream os("last_id");
  if (os) {
    os << id << std::endl;
  }
}


/* Generates a new client id. */
static int new_id() {
  static int last_id = read_last_id();
  last_id++;
  write_last_id(last_id);
  return last_id;
}


/* Writes a "bad-problem" error message to the given stream. */
void LogBadProblem(std::ostream& os, const std::string& problem_name) {
  os << "<error>bad problem \"" << problem_name << "\"</error>" << std::endl;
}


/* Writes a "session-init" message to the given stream. */
void LogSessionInit(std::ostream& os, int id, const Problem_CFG& cfg) {
  os << "<session-init>"
     << "<sessionID>" << id << "</sessionID>"
     << "<setting>"
     << "<rounds>" << cfg.round_limit << "</rounds>"
     << "<allowed-time>" << cfg.time_limit.count() << "</allowed-time>"
     << "<allowed-turns>" << cfg.turn_limit << "</allowed-turns>"
     << "</setting>"
     << "</session-init>" << std::endl;
}


/* Writes a "round-init" message to the given stream. */
void LogRoundInit(std::ostream& os, int id, int round,
                  std::chrono::milliseconds time_left, int rounds_left) {
  os << "<round-init>"
     << "<sessionID>" << id << "</sessionID>"
     << "<round>" << round << "</round>"
     << "<time-left>" << time_left.count() << "</time-left>"
     << "<rounds-left>" << rounds_left << "</rounds-left>"
     << "</round-init>" << std::endl;
}


/* Writes a action-related error message to the given stream. */
void LogActionError(std::ostream& os,
                    const std::string& reason, const str_vec& params) {
  str_vec::const_iterator si = params.begin();
  os << "<error>" << reason << " \"(" << *si;
  for (si++; si != params.end(); si++) {
    os << ' ' << *si;
  }
  os << ")\"</error>" << std::endl;
}


/* Writes an "end-round" message to the given stream. */
void LogEndRound(std::ostream& os,
                 int id, int round, const State& s,
                 std::chrono::milliseconds time_spent, int turns_used) {
  os << "<end-round>"
     << "<sessionID>" << id << "</sessionID>"
     << "<round>" << round << "</round>";
  s.printXML(os);
  if (s.goal()) {
    os << "<goal-reached/>";
  }
  os << "<time-spent>" << time_spent.count() << "</time-spent>"
     << "<turns-used>" << turns_used << "</turns-used>"
     << "</end-round>" << std::endl;
}


/* Writes an "end-session" message to the given stream. */
void LogEndSession(std::ostream& os,
                   int id, int rounds, int round_limit, int successes,
                   std::chrono::milliseconds total_time, int total_turns,
                   const Rational& total_metric) {
  os << "<end-session>"
     << "<sessionID>" << id << "</sessionID>"
     << "<rounds>" << rounds << "</rounds>"
     << "<goals>"
     << "<failed>" << round_limit - successes << "</failed>"
     << "<reached>"
     << "<successes>" << successes << "</successes>";
  if (successes > 0) {
    os << "<time-average>" << double(total_time.count())/successes
       << "</time-average>";
    os << "<turn-average>" << double(total_turns)/successes
       << "</turn-average>";
  }
  os << "</reached>"
     << "</goals>";
  if (round_limit > 0) {
    os << "<metric-average>" << total_metric.double_value()/round_limit
       << "</metric-average>";
  }
  os << "</end-session>" << std::endl;
}


/* Main procedure for server. */
static void* host_problem(void* arg) {
  std::pair<int, Problem_CFG>* p = (std::pair<int, Problem_CFG>*) arg;
  int client_socket = p->first;
  const std::chrono::milliseconds time_limit = p->second.time_limit;
  int round_limit = p->second.round_limit;
  int turn_limit = p->second.turn_limit;
  delete p;

  std::ostringstream os;

  const XMLNode* init_node = read_node(client_socket);
  if (init_node == 0 || init_node->getName() != "session-request") {
    if (init_node != 0) {
      delete init_node;
    }
    return 0;
  }

  std::string contestant_name;
  if (!init_node->dissect("name", contestant_name)
      || contestant_name.empty()) {
    delete init_node;
    return 0;
  }

  std::string problem_name;
  if (!init_node->dissect("problem", problem_name) || problem_name.empty()) {
    delete init_node;
    return 0;
  }

  delete init_node;

  int id = new_id();

  std::cout << "Contestant " << contestant_name
            << " running problem " << problem_name << "(" << id << ")"
            << std::endl;

  // Open log file for contestant_name in "append" mode.
  std::string log_file = log_dir;
  if (!log_dir.empty() && log_dir[log_dir.size() - 1] != '/') {
    log_file += '/';
  }
  log_file += contestant_name+"-"+problem_name;
  std::ofstream log_out(log_file.c_str(), std::ios::app);

  const Problem* problem = Problem::find(problem_name);
  if (problem == 0) {
    os.str("");
    LogBadProblem(os, problem_name);
    LogBadProblem(log_out, problem_name);
    if (! write(client_socket, os.str().c_str(), os.str().length()))
	EXIT_ERROR;
    return 0;
  }

  Problem_CFG cfg;
  CFG_map::const_iterator cfg_itr = config_map.find(problem_name);
  if (cfg_itr != config_map.end()) {
    cfg = cfg_itr->second;
  } else {
    cfg.round_limit = round_limit;
    cfg.time_limit = time_limit;
    cfg.turn_limit = turn_limit;
    std::cerr << "There is no config entry for requested problem "
              << problem_name << ", using default." << std::endl;
  }

  os.str("");
  LogSessionInit(os, id, cfg);
  LogSessionInit(log_out, id, cfg);
  if (! write(client_socket, os.str().c_str(), os.str().length()))
      EXIT_ERROR;

  Timer session_timer;

  Rational total_metric = 0;
  auto total_time = std::chrono::milliseconds::zero();
  int total_turns;
  int success_count = 0;

  int round = 1;
  while (round <= cfg.round_limit) {
    const XMLNode* roundReq = read_node(client_socket);
    if (roundReq == 0 || roundReq->getName() != "round-request") {
      if (roundReq != 0) {
        delete roundReq;
      }
      return 0;
    }

    if (roundReq != 0) {
      delete roundReq;
    }

    const std::chrono::milliseconds time_left = std::max(
        std::chrono::milliseconds::zero(),
        cfg.time_limit - session_timer.GetElapsedMilliseconds());

    os.str("");
    LogRoundInit(os, id, round, time_left, cfg.round_limit - round);
    LogRoundInit(log_out, id, round, time_left, cfg.round_limit - round);
    if (! write(client_socket, os.str().c_str(), os.str().length()))
	EXIT_ERROR;

    //create initial state
    const State *s = new State(*problem);

    bool running = time_left > std::chrono::milliseconds::zero();
    int turn = 1;
    Timer round_timer;
    while (running && turn <= cfg.turn_limit && !s->goal()) {
      os.str("");
      s->printXML(os);
      os << std::endl;
      if (log_paths) {
        s->printXML(log_out);
        log_out << std::endl;
      }
      if (! write(client_socket, os.str().c_str(), os.str().length()))
	  EXIT_ERROR;

      const Action *action = 0;
      const XMLNode* actnode = read_node(client_socket);
      if (actnode == 0) {
        delete s;
        std::cerr << contestant_name << " in session " << id
                  << " issued invalid XML action; connection killed."
                  << std::endl;
        return 0;
      }

      if (actnode->getName() == "done") {
        running = false;
      } else if (actnode->getName() == "act") {
        const XMLNode* actionnode = actnode->getChild("action");
        str_vec params;
        params.push_back(actionnode->getChild("name")->getText());
        for (int i=1; i<actionnode->size(); i++)
          params.push_back(actionnode->getChild(i)->getText());

        action = make_action(params, *problem);
        if (action == 0
            || !action->enabled(problem->terms(), s->atoms(), s->values())) {
          os.str("");
          if (action == 0) {
            LogActionError(os, "bad action", params);
            LogActionError(log_out, "bad action", params);
          } else {
            LogActionError(os, "disabled action", params);
            LogActionError(log_out, "disabled action", params);
          }
          if (! write(client_socket, os.str().c_str(), os.str().length()))
	      EXIT_ERROR;

          running = false;
        } else {
          if (log_paths) {
            log_out << actionnode << std::endl;
          }
        }
      } else {
        delete s;
        std::cerr << contestant_name << " in session " << id
                  << " issued invalid XML action; connection killed."
                  << std::endl;
        return 0;
      }

      delete actnode;

      if (cfg.time_limit <= session_timer.GetElapsedMilliseconds()) {
        running = false;
      }

      if (running) {
        const State& next_s = s->next(*action);
        delete s;
        s = &next_s;
        turn++;
      }
    }

    total_metric = total_metric + problem->metric().value(s->values());

    const std::chrono::milliseconds time_spent = std::max(
        std::chrono::milliseconds::zero(),
        std::min(time_left, round_timer.GetElapsedMilliseconds()));
    int turns_used = turn - 1;
    if (s->goal()) {
      total_time += time_spent;
      total_turns += turns_used;
      success_count++;
    }

    os.str("");
    LogEndRound(os, id, round, *s, time_spent, turns_used);
    LogEndRound(log_out, id, round, *s, time_spent, turns_used);
    if (! write(client_socket, os.str().c_str(), os.str().length()))
	EXIT_ERROR;

    delete s;

    round++;
  }

  os.str("");
  LogEndSession(os, id, round - 1, cfg.round_limit,
                success_count, total_time, total_turns, total_metric);
  LogEndSession(log_out, id, round - 1, cfg.round_limit,
                success_count, total_time, total_turns, total_metric);
  if (! write(client_socket, os.str().c_str(), os.str().length()))
      EXIT_ERROR;

  std::cout << "session " << id << " complete" << std::endl;

  return 0;
}


/* Runs a server. */
int run_server(int port, std::chrono::milliseconds time_limit, int round_limit,
               int turn_limit) {
  struct sockaddr_in addr;
  int server_socket;

  std::filesystem::create_directories(log_dir);
  if ((server_socket = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
    return -1;
  }

  int j = 1;
  setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, (char*) &j, sizeof(int));
  setsockopt(server_socket, IPPROTO_TCP, TCP_NODELAY, (char*) &j, sizeof(int));

  addr.sin_family=AF_INET;
  addr.sin_port=htons(port);
  //setting to zero means use first IP of machine
  memset(&addr.sin_addr, 0, sizeof(addr.sin_addr));

  if (bind(server_socket, (struct sockaddr*)&addr, sizeof(addr))) {
    std::cerr << "could not bind socket" << std::endl;
    return -1;
  }

  if (listen(server_socket, 5)) {
    std::cerr << "could not listen" << std::endl;
    return -1;
  }

  std::cout << "mdpsim is running a server on port " << port << std::endl;

  int client_socket;
  socklen_t addrlength = sizeof(addr);
  while ((client_socket = accept(server_socket, (struct sockaddr*)&addr,
                                 &addrlength)) >= 0) {
    std::pair<int, Problem_CFG>* p = new std::pair<int, Problem_CFG>();
    p->first = client_socket;
    p->second.time_limit = time_limit;
    p->second.round_limit = round_limit;
    p->second.turn_limit = turn_limit;
    pthread_attr_t t_attr;
    pthread_t t_id;
    pthread_attr_init(&t_attr);
    if (0 == pthread_create(&t_id, &t_attr, host_problem, p)) {
      pthread_detach(t_id);
    }
  }
  std::cout << client_socket << std::endl;

  close(server_socket);
  return 0;
}
