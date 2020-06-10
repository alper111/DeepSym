/* -*-C++-*- */
/*
 * MDP Server.
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
#ifndef _MDPSERVER_H
#define _MDPSERVER_H

#include <chrono>
#include <map>
#include <string>


/* Runs a server. */
int run_server(int port, std::chrono::milliseconds time_limit, int round_limit,
               int turn_limit);


/*
 * Configuration parameters for a problem.
 */
struct Problem_CFG {
  std::chrono::milliseconds time_limit;
  int round_limit;
  int turn_limit;
};


/*
 * Map from problem names to configurations.
 */
struct CFG_map : public std::map<std::string, Problem_CFG> {
};


/* Global problem configurations. */
extern CFG_map config_map;


#endif /* _MDPSERVER_H */
