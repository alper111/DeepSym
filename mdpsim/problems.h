/* -*-C++-*- */
/*
 * Problem descriptions.
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
#ifndef PROBLEMS_H
#define PROBLEMS_H

#include <config.h>
#include "domains.h"
#include "actions.h"
#include "effects.h"
#include "formulas.h"
#include "expressions.h"
#include "terms.h"
#include "types.h"
#include <iostream>
#include <map>
#include <string>


/* ====================================================================== */
/* Problem */

/*
 * Problem definition.
 */
struct Problem {
  /* Table of problem definitions. */
  struct ProblemMap : public std::map<std::string, const Problem*> {
  };

  /* Returns a const_iterator pointing to the first problem. */
  static ProblemMap::const_iterator begin();

  /* Returns a const_iterator pointing beyond the last problem. */
  static ProblemMap::const_iterator end();

  /* Returns a pointer to the problem with the given name, or 0 if it
     is undefined. */
  static const Problem* find(const std::string& name);

  /* Removes all defined problems. */
  static void clear();

  /* Constructs a problem. */
  Problem(const std::string& name, const Domain& domain);

  /* Deletes a problem. */
  ~Problem();

  /* Returns the name of this problem. */
  const std::string& name() const { return name_; }

  /* Returns the domain of this problem. */
  const Domain& domain() const { return *domain_; }

  /* Returns the term table of this problem. */
  TermTable& terms() { return terms_; }

  /* Returns the term table of this problem. */
  const TermTable& terms() const { return terms_; }

  /* Adds an atomic state formula to the initial conditions of this
     problem. */
  void add_init_atom(const Atom& atom);

  /* Adds a fluent value to the initial conditions of this problem. */
  void add_init_value(const Fluent& fluent, const Rational& value);

  /* Adds an initial effect for this problem. */
  void add_init_effect(const Effect& effect);

  /* Sets the goal for this problem. */
  void set_goal(const StateFormula& goal);

  /* Sets the goal reward for this problem. */
  void set_goal_reward(const Update& goal_reward);

  /* Sets the metric to maximize for this problem. */
  void set_metric(const Expression& metric, bool negate = false);

  /* Instantiates this problem. */
  void instantiate();

  /* Returns the initial atoms of this problem. */
  const AtomSet& init_atoms() const { return init_atoms_; }

  /* Returns the initial values of this problem. */
  const ValueMap& init_values() const { return init_values_; }

  /* Returns the initial effects of this problem. */
  const EffectList& init_effects() const { return init_effects_; }

  /* Returns the goal of this problem. */
  const StateFormula& goal() const { return *goal_; }

  /* Returns a pointer to the goal reward for this problem, or 0 if no
     explicit reward is associated with goal states. */
  const Update* goal_reward() const { return goal_reward_; }

  /* Returns the metric to maximize for this problem. */
  const Expression& metric() const { return *metric_; }

  /* Tests if the metric is constant. */
  bool constant_metric() const;

  /* Returns a list of instantiated actions. */
  const ActionSet& actions() const { return actions_; }

  /* Fills the given list with actions enabled in the given state. */
  void enabled_actions(ActionList& actions, const AtomSet& atoms,
                       const ValueMap& values) const;

 private:
  /* Table of defined problems. */
  static ProblemMap problems;

  /* Name of problem. */
  std::string name_;
  /* Problem domain. */
  const Domain* domain_;
  /* Problem terms. */
  TermTable terms_;
  /* Initial atoms. */
  AtomSet init_atoms_;
  /* Initial fluent values. */
  ValueMap init_values_;
  /* Initial effects. */
  EffectList init_effects_;
  /* Goal; FALSE if not a goal-directed planning problem. */
  const StateFormula* goal_;
  /* Goal reward expression. */
  const Update* goal_reward_;
  /* Metric to maximize. */
  const Expression* metric_;
  /* Instantiated actions. */
  ActionSet actions_;
};

/* Output operator for problems. */
std::ostream& operator<<(std::ostream& os, const Problem& p);


#endif /* PROBLEMS_H */
