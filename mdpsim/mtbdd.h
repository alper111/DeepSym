/* -*-C++-*- */
/*
 * Copyright 2004, 2005 Carnegie Mellon University
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
#ifndef MTBDD_H
#define MTBDD_H

#include <config.h>
#include "client.h"
#include <cudd.h>


/* ====================================================================== */
/* MTBDDPlanner */

/*
 * An MTBDD planner.
 */
struct MTBDDPlanner : public Planner {
  /* Constructs an MTBDD planner. */
  MTBDDPlanner(const Problem& problem, double gamma, double epsilon)
    : Planner(problem), gamma_(gamma), epsilon_(epsilon), dd_man_(0),
      mapping_(0) {}

  /* Deletes this MTBDD planner. */
  virtual ~MTBDDPlanner();

  virtual void initRound();
  virtual const Action* decideAction(const AtomSet& atoms,
                                     const ValueMap& values);
  virtual void endRound();

 private:
  /* Discount factor. */
  double gamma_;
  /* Error tolerance. */
  double epsilon_;
  /* DD manager. */
  DdManager* dd_man_;
  /* Policy. */
  DdNode* mapping_;
  /* Actions. */
  std::vector<const Action*> actions_;
  /* State variables. */
  std::map<int, const Atom*> dynamic_atoms_;
};


#endif /* MTBDD_H */
