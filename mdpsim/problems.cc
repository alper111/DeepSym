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
#include "problems.h"
#include "domains.h"
#include <typeinfo>


/* ====================================================================== */
/* Problem */

/* Table of defined problems. */
Problem::ProblemMap Problem::problems = Problem::ProblemMap();


/* Returns a const_iterator pointing to the first problem. */
Problem::ProblemMap::const_iterator Problem::begin() {
  return problems.begin();
}


/* Returns a const_iterator pointing beyond the last problem. */
Problem::ProblemMap::const_iterator Problem::end() {
  return problems.end();
}


/* Returns a pointer to the problem with the given name, or 0 if it
   is undefined. */
const Problem* Problem::find(const std::string& name) {
  ProblemMap::const_iterator pi = problems.find(name);
  return (pi != problems.end()) ? (*pi).second : 0;
}


/* Removes all defined problems. */
void Problem::clear() {
  ProblemMap::const_iterator pi = begin();
  while (pi != end()) {
    delete (*pi).second;
    pi = begin();
  }
  problems.clear();
}


/* Constructs a problem. */
Problem::Problem(const std::string& name, const Domain& domain)
  : name_(name), domain_(&domain), terms_(TermTable(domain.terms())),
    goal_(&StateFormula::FALSE), goal_reward_(0), metric_(new Value(0)) {
  RCObject::ref(goal_);
  RCObject::ref(metric_);
  const Problem* p = find(name);
  if (p != 0) {
    delete p;
  }
  problems[name] = this;
}


/* Deletes a problem. */
Problem::~Problem() {
  problems.erase(name());
  for (AtomSet::const_iterator ai = init_atoms_.begin();
       ai != init_atoms_.end(); ai++) {
    RCObject::destructive_deref(*ai);
  }
  for (ValueMap::const_iterator vi = init_values_.begin();
       vi != init_values_.end(); vi++) {
    RCObject::destructive_deref((*vi).first);
  }
  for (EffectList::const_iterator ei = init_effects_.begin();
       ei != init_effects_.end(); ei++) {
    RCObject::destructive_deref(*ei);
  }
  RCObject::destructive_deref(goal_);
  if (goal_reward_ != 0) {
    delete goal_reward_;
  }
  RCObject::destructive_deref(metric_);
  for (ActionSet::const_iterator ai = actions_.begin();
       ai != actions_.end(); ai++) {
    delete *ai;
  }
}


/* Adds an atomic state formula to the initial conditions of this
   problem. */
void Problem::add_init_atom(const Atom& atom) {
  if (init_atoms_.find(&atom) == init_atoms_.end()) {
    init_atoms_.insert(&atom);
    RCObject::ref(&atom);
  }
}


/* Adds a fluent value to the initial conditions of this problem. */
void Problem::add_init_value(const Fluent& fluent, const Rational& value) {
  if (init_values_.find(&fluent) == init_values_.end()) {
    init_values_.insert(std::make_pair(&fluent, value));
    RCObject::ref(&fluent);
  } else {
    init_values_[&fluent] = value;
  }
}


/* Adds an initial effect for this problem. */
void Problem::add_init_effect(const Effect& effect) {
  init_effects_.push_back(&effect);
  RCObject::ref(&effect);
}


/* Sets the goal for this problem. */
void Problem::set_goal(const StateFormula& goal) {
  if (&goal != goal_) {
    RCObject::ref(&goal);
    RCObject::destructive_deref(goal_);
    goal_ = &goal;
  }
}


/* Sets the goal reward for this problem. */
void Problem::set_goal_reward(const Update& goal_reward) {
  if (&goal_reward != goal_reward_) {
    delete goal_reward_;
    goal_reward_ = &goal_reward;
  }
}


/* Sets the metric to maximize for this problem. */
void Problem::set_metric(const Expression& metric, bool negate) {
  const Expression* real_metric;
  if (negate) {
    real_metric = &Subtraction::make(*new Value(0), metric);
  } else {
    real_metric = &metric;
  }
  if (real_metric != metric_) {
    RCObject::ref(real_metric);
    RCObject::destructive_deref(metric_);
    metric_ = real_metric;
  }
}


/* Instantiates this problem. */
void Problem::instantiate() {
#if 0
  set_goal(goal().instantiation(SubstitutionMap(), terms(),
                                init_atoms(), init_values()));
#endif
  if (goal_reward() != 0) {
    set_goal_reward(goal_reward()->instantiation(SubstitutionMap(),
                                                 init_values()));
  }
  set_metric(metric().instantiation(SubstitutionMap(), init_values()));
  for (ActionSchemaMap::const_iterator ai = domain().actions().begin();
       ai != domain().actions().end(); ai++) {
    (*ai).second->instantiations(actions_, terms(),
                                 init_atoms(), init_values());
  }
}


/* Tests if the metric is constant. */
bool Problem::constant_metric() const {
  return typeid(metric()) == typeid(Value);
}


/* Fills the given list with actions enabled in the given state. */
void Problem::enabled_actions(ActionList& actions, const AtomSet& atoms,
                              const ValueMap& values) const {
  for (ActionSet::const_iterator ai = this->actions().begin();
       ai != this->actions().end(); ai++) {
    if ((*ai)->enabled(terms(), atoms, values)) {
      actions.push_back(*ai);
    }
  }
}


/* Output operator for problems. */
std::ostream& operator<<(std::ostream& os, const Problem& p) {
  os << "name: " << p.name();
  os << std::endl << "domain: " << p.domain().name();
  os << std::endl << "objects:" << p.terms();
  os << std::endl << "init:";
  for (AtomSet::const_iterator ai = p.init_atoms().begin();
       ai != p.init_atoms().end(); ai++) {
    os << std::endl << "  " << **ai;
  }
  for (ValueMap::const_iterator vi = p.init_values().begin();
       vi != p.init_values().end(); vi++) {
    os << std::endl << "  (= " << *(*vi).first << ' ' << (*vi).second << ")";
  }
  for (EffectList::const_iterator ei = p.init_effects().begin();
       ei != p.init_effects().end(); ei++) {
    os << std::endl << "  " << **ei;
  }
  os << std::endl << "goal: " << p.goal();
  if (p.goal_reward() != 0) {
    os << std::endl << "goal reward: " << p.goal_reward()->expression();
  }
  os << std::endl << "metric: " << p.metric();
  os << std::endl << "actions:";
  for (ActionSet::const_iterator ai = p.actions().begin();
       ai != p.actions().end(); ai++) {
    os << std::endl << "  " << **ai;
  }
  return os;
}
