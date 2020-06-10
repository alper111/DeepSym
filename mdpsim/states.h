/* -*-C++-*- */
/*
 * States.
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
#ifndef STATES_H
#define STATES_H

#include <config.h>
#include "problems.h"
#include "actions.h"
#include "formulas.h"
#include "expressions.h"
#include <iostream>


/* ====================================================================== */
/* State */

/*
 * A state.
 */
struct State {
  /* Constructs an initial state for the given problem. */
  explicit State(const Problem& problem);

  /* Returns the problem associated with this state. */
  const Problem& problem() const { return *problem_; }

  /* Returns the set of atoms that hold in this state. */
  const AtomSet& atoms() const { return atoms_; }

  /* Returns the fluent values for this state. */
  const ValueMap& values() const { return values_; }

  /* Tests if this is a goal state. */
  bool goal() const { return goal_; }

  /* Returns a sampled successor of this state. */
  const State& next(const Action& action) const;

  /* Prints this object on the given stream in XML. */
  void printXML(std::ostream& os) const;

 private:
  /* The problem that this state is associated with. */
  const Problem* problem_;
  /* Atomic state formulas that hold in this state. */
  AtomSet atoms_;
  /* Fluent values in this state. */
  ValueMap values_;
  /* Whether this is a goal state. */
  bool goal_;
};

/* Output operator for states. */
std::ostream& operator<<(std::ostream& os, const State& s);


#endif /* STATES_H */
