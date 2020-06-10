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
#include "states.h"


/* Verbosity level. */
extern int verbosity;


/* ====================================================================== */
/* State */

/* Constructs an initial state for the given problem. */
State::State(const Problem& problem)
  : problem_(&problem), atoms_(problem.init_atoms()),
    values_(problem.init_values()) {
  for (EffectList::const_iterator ei = problem.init_effects().begin();
       ei != problem.init_effects().end(); ei++) {
    AtomList adds;
    AtomList deletes;
    UpdateList updates;
    (*ei)->state_change(adds, deletes, updates,
                        problem.terms(), atoms_, values_);
    atoms_.insert(adds.begin(), adds.end());
    for (UpdateList::const_iterator ui = updates.begin();
         ui != updates.end(); ui++) {
      (*ui)->affect(values_);
    }
  }
  goal_ = problem.goal().holds(problem.terms(), atoms_, values_);
  if (goal()) {
    const Fluent& goal_achieved_fluent =
      Fluent::make(problem.domain().goal_achieved(), TermList());
    values_[&goal_achieved_fluent] = 1;
    if (problem.goal_reward() != 0) {
      problem.goal_reward()->affect(values_);
    }
  }
}


/* Returns a sampled successor of this state. */
const State& State::next(const Action& action) const {
  State* next_state = new State(*this);
  if (verbosity > 1) {
    std::cerr << "selected action: " << action << std::endl;
  }
  action.affect(problem().terms(), next_state->atoms_, next_state->values_);
  next_state->goal_ = problem().goal().holds(problem().terms(),
                                             next_state->atoms_,
                                             next_state->values_);
  if (next_state->goal()) {
    if (!goal()) {
      const Fluent& goal_achieved_fluent =
        Fluent::make(problem().domain().goal_achieved(), TermList());
      next_state->values_[&goal_achieved_fluent] = 1;
      if (problem().goal_reward() != 0) {
        problem().goal_reward()->affect(next_state->values_);
      }
    }
  }
  const Fluent& total_time_fluent =
    Fluent::make(problem().domain().total_time(), TermList());
  next_state->values_[&total_time_fluent] =
    next_state->values_[&total_time_fluent] + 1;
  if (verbosity > 1) {
    std::cerr << std::endl;
  }
  return *next_state;
}


/* Prints this object on the given stream. */
void State::printXML(std::ostream& os) const {
  os << "<state>";
  if (goal()) {
    os << "<is-goal/>";
  }
  for (AtomSet::const_iterator ai = atoms().begin();
       ai != atoms().end(); ai++) {
    const Atom& atom = **ai;
    if (!PredicateTable::static_predicate(atom.predicate())) {
      os << "<atom><predicate>" << atom.predicate() << "</predicate>";
      for (TermList::const_iterator ti = atom.terms().begin();
           ti != atom.terms().end(); ti++) {
        os << "<term>" << *ti << "</term>";
      }
      os << "</atom>";
    }
  }
  for (ValueMap::const_iterator vi = values().begin();
       vi != values().end(); vi++) {
    const Fluent& fluent = *vi->first;
    if (fluent.function() != problem().domain().total_time()
        && fluent.function() != problem().domain().goal_achieved()
        && !FunctionTable::static_function(fluent.function())) {
      os << "<fluent><function>" << fluent.function() << "</function>";
      for (TermList::const_iterator ti = fluent.terms().begin();
           ti != fluent.terms().end(); ti++) {
        os << "<term>" << *ti << "</term>";
      }
      os << "<value>" << vi->second << "</value>";
      os << "</fluent>";
    }
  }
  os << "</state>";
}


/* Output operator for states. */
std::ostream& operator<<(std::ostream& os, const State& s) {
  bool first = true;
  for (AtomSet::const_iterator ai = s.atoms().begin();
       ai != s.atoms().end(); ai++) {
    const Atom& atom = **ai;
    if (!PredicateTable::static_predicate(atom.predicate())) {
      if (first) {
        first = false;
      } else {
        os << ' ';
      }
      os << atom;
    }
  }
  for (ValueMap::const_iterator vi = s.values().begin();
       vi != s.values().end(); vi++) {
    const Fluent& fluent = *(*vi).first;
    if (fluent.function() != s.problem().domain().total_time()
        && fluent.function() != s.problem().domain().goal_achieved()
        && !FunctionTable::static_function(fluent.function())) {
      if (first) {
        first = false;
      } else {
        os << ' ';
      }
      os << "(= " << fluent << ' ' << (*vi).second << ")";
    }
  }
  if (s.goal()) {
    os << " <goal>";
  }
  return os;
}
