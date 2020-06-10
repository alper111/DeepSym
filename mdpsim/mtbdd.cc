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
#include "mtbdd.h"
#include "problems.h"
#include "domains.h"
#include "actions.h"
#include "effects.h"
#include "formulas.h"
#include "functions.h"
#include <cudd.h>
#include <map>
#include <stdexcept>
#include <typeinfo>


/* Verbosity level. */
extern int verbosity;

/* The reward function. */
static const Function* reward_function;
/* The goal reward. */
static Rational goal_reward;
/* State variables for the current problem. */
static std::map<const Atom*, int> state_variables;
/* A mapping from variable indices to atoms. */
static std::map<int, const Atom*> dynamic_atoms;
/* DD manager. */
static DdManager* dd_man;
/* Total number of state variables. */
static int nvars;
/* Number of auxiliary variables. */
static int aux_vars;
/* Variable ordering. */
static std::vector<int> var_order;
/* Ordered variables. */
static std::vector<int> ordered_vars;
/* MTBDDs representing transition probability matrices for actions. */
static std::map<const Action*, DdNode*> action_transitions;
/* MTBDDs representing reward vectors for actions. */
static std::map<const Action*, DdNode*> action_rewards;
/* Mapping from action ids to actions used by current policy. */
static std::vector<const Action*> policy_actions;


/* ====================================================================== */
/* DD variable access. */

/*
 * Returns the BDD variable for the given state variable.
 */
static DdNode* bdd_var(int i, bool primed = false) {
  return Cudd_bddIthVar(dd_man, 2*var_order[i] + (primed ? 1 : 0));
}


/*
 * Returns the ADD variable for the given state variable.
 */
static DdNode* add_var(int i, bool primed = false) {
  return Cudd_addIthVar(dd_man, 2*var_order[i] + (primed ? 1 : 0));
}


/* ====================================================================== */
/* state_bdd */

/*
 * Returns a BDD representing the given state.
 */
static DdNode* state_bdd(DdManager* dd_man,
                         const std::map<int, const Atom*>& dynamic_atoms,
                         const AtomSet& atoms) {
  /* This is going to be the BDD representing the given state. */
  DdNode* dds = Cudd_ReadOne(dd_man);
  Cudd_Ref(dds);

  /*
   * Set Boolean state variables to the values specified by the given
   * atom set.
   */
  for (std::map<int, const Atom*>::const_reverse_iterator ai =
         dynamic_atoms.rbegin();
       ai != dynamic_atoms.rend(); ai++) {
    int i = (*ai).first;
    DdNode* ddv = bdd_var(i);
    if (atoms.find((*ai).second) == atoms.end()) {
      ddv = Cudd_Not(ddv);
    }
    DdNode* ddt = Cudd_bddAnd(dd_man, ddv, dds);
    Cudd_Ref(ddt);
    Cudd_RecursiveDeref(dd_man, dds);
    dds = ddt;
  }

  return dds;
}


/* ====================================================================== */
/* collect_state_variables */

/*
 * Collects state variables from the given formula.
 */
static void collect_state_variables(const StateFormula& formula) {
  if (formula.tautology() || formula.contradiction()) {
    /*
     * The formula is either TRUE or FALSE, so it contains no state
     * variables.
     */
    return;
  }

  const Atom* af = dynamic_cast<const Atom*>(&formula);
  if (af != 0) {
    /*
     * The formula is an atom representing a single state variable.
     */
    if (state_variables.find(af) == state_variables.end()) {
      dynamic_atoms.insert(std::make_pair(state_variables.size(), af));
      state_variables.insert(std::make_pair(af, state_variables.size()));
    }
    return;
  }

  const Negation* nf = dynamic_cast<const Negation*>(&formula);
  if (nf != 0) {
    /*
     * The state variables of a negation are the state variables of the
     * negand.
     */
    collect_state_variables(nf->negand());
    return;
  }

  const Conjunction* cf = dynamic_cast<const Conjunction*>(&formula);
  if (cf != 0) {
    /*
     * The state variables of a conjunction are the state variables of
     * the conjuncts.
     */
    for (FormulaList::const_iterator fi = cf->conjuncts().begin();
         fi != cf->conjuncts().end(); fi++) {
      collect_state_variables(**fi);
    }
    return;
  }

  const Disjunction* df = dynamic_cast<const Disjunction*>(&formula);
  if (df != 0) {
    /*
     * The state variables of a disjunction are the state variables of
     * the disjuncts.
     */
    for (FormulaList::const_iterator fi = df->disjuncts().begin();
         fi != df->disjuncts().end(); fi++) {
      collect_state_variables(**fi);
    }
    return;
  }

  /*
   * No other types of formulas should appear in fully instantiated
   * action preconditions and effect conditions.
   */
  throw std::logic_error("unexpected formula");
}


/*
 * Collects state variables from the given effect.
 */
static void collect_state_variables(const Effect& effect,
                                    const Domain& domain) {
  if (effect.empty()) {
    /*
     * This effect is empty.
     */
    return;
  }

  const UpdateEffect* ue = dynamic_cast<const UpdateEffect*>(&effect);
  if (ue != 0) {
    /*
     * Only reward updates are supported, and they do not involve any
     * state variables.
     */
    const Function& function = ue->update().fluent().function();
    if (function != domain.total_time() && function != domain.goal_achieved()
        && (reward_function == 0 || function != *reward_function)) {
      throw std::logic_error("numeric state variables not supported");
    }
    return;
  }

  const SimpleEffect* se = dynamic_cast<const SimpleEffect*>(&effect);
  if (se != 0) {
    /*
     * A simple effect involves a single state variable.
     */
    const Atom* atom = &se->atom();
    if (state_variables.find(atom) == state_variables.end()) {
      dynamic_atoms.insert(std::make_pair(state_variables.size(), atom));
      state_variables.insert(std::make_pair(atom, state_variables.size()));
    }
    return;
  }

  const ConjunctiveEffect* ce =
    dynamic_cast<const ConjunctiveEffect*>(&effect);
  if (ce != 0) {
    /*
     * The state variables of a conjunctive effect are the state
     * variables of the conjuncts.
     */
    for (EffectList::const_iterator ei = ce->conjuncts().begin();
         ei != ce->conjuncts().end(); ei++) {
      collect_state_variables(**ei, domain);
    }
    return;
  }

  const ConditionalEffect* we =
    dynamic_cast<const ConditionalEffect*>(&effect);
  if (we != 0) {
    /*
     * The state variables of a conditional effect are the state
     * variables of the condition and the effect.
     */
    collect_state_variables(we->effect(), domain);
    return;
  }

  const ProbabilisticEffect* pe =
    dynamic_cast<const ProbabilisticEffect*>(&effect);
  if (pe != 0) {
    /*
     * The state variables of a probabilistic effect are the state
     * variables of the possible effects.
     */
    for (size_t i = 0; i < pe->size(); i++) {
      collect_state_variables(pe->effect(i), domain);
    }
    return;
  }

  /*
   * No other types of effects exist.
   */
  throw std::logic_error("unexpected effect");
}


/* ====================================================================== */
/* formula_bdd */

/*
 * Constructs a BDD representing the given formula.
 */
static DdNode* formula_bdd(const StateFormula& formula, bool primed = false) {
  if (formula.tautology() || formula.contradiction()) {
    /*
     * The formula is either TRUE or FALSE, so the BDD is either
     * constant 1 or 0.
     */
    DdNode* ddf = (formula.tautology() ?
                   Cudd_ReadOne(dd_man) : Cudd_ReadLogicZero(dd_man));
    Cudd_Ref(ddf);
    return ddf;
  }

  const Atom* af = dynamic_cast<const Atom*>(&formula);
  if (af != 0) {
    /*
     * The BDD for an atom is the `current-state' (or `next-state' if
     * primed is true) DD variable for the state variable represented
     * by the atom.
     */
    DdNode* ddf;
    std::map<const Atom*, int>::const_iterator ai = state_variables.find(af);
    if (ai != state_variables.end()) {
      ddf = bdd_var((*ai).second, primed);
    } else {
      ddf = Cudd_ReadLogicZero(dd_man);
    }
    Cudd_Ref(ddf);
    return ddf;
  }

  const Negation* nf = dynamic_cast<const Negation*>(&formula);
  if (nf != 0) {
    /*
     * The BDD for a negation is the negation of the BDD for the
     * negand.
     */
    DdNode* ddn = formula_bdd(nf->negand(), primed);
    DdNode* ddf = Cudd_Not(ddn);
    Cudd_Ref(ddf);
    Cudd_RecursiveDeref(dd_man, ddn);
    return ddf;
  }

  const Conjunction* cf = dynamic_cast<const Conjunction*>(&formula);
  if (cf != 0) {
    /*
     * The BDD for a conjunction is the conjunction of the BDDs for
     * the conjuncts.
     */
    DdNode* ddf = Cudd_ReadOne(dd_man);
    Cudd_Ref(ddf);
    for (FormulaList::const_iterator fi = cf->conjuncts().begin();
         fi != cf->conjuncts().end(); fi++) {
      DdNode* ddi = formula_bdd(**fi, primed);
      DdNode* dda = Cudd_bddAnd(dd_man, ddf, ddi);
      Cudd_Ref(dda);
      Cudd_RecursiveDeref(dd_man, ddf);
      Cudd_RecursiveDeref(dd_man, ddi);
      ddf = dda;
    }
    return ddf;
  }

  const Disjunction* df = dynamic_cast<const Disjunction*>(&formula);
  if (df != 0) {
    /*
     * The BDD for a disjunction is the disjunction of the BDDs for
     * the disjuncts.
     */
    DdNode* ddf = Cudd_ReadLogicZero(dd_man);
    Cudd_Ref(ddf);
    for (FormulaList::const_iterator fi = df->disjuncts().begin();
         fi != df->disjuncts().end(); fi++) {
      DdNode* ddi = formula_bdd(**fi, primed);
      DdNode* ddo = Cudd_bddOr(dd_man, ddf, ddi);
      Cudd_Ref(ddo);
      Cudd_RecursiveDeref(dd_man, ddf);
      Cudd_RecursiveDeref(dd_man, ddi);
      ddf = ddo;
    }
    return ddf;
  }

  /*
   * No other types of formulae should appear in fully instantiated
   * action preconditions and effect conditions.
   */
  throw std::logic_error("unexpected formula");
}

/* ====================================================================== */
/* auxiliary_dbn_variables */

/*
 * Returns the ceiling of the base-2 logarithm of the given integer.
 */
static int ceil_log2(int n) {
  int result = 0;
  for (int i = n - 1; i > 0; i >>= 1) {
    result++;
  }
  return result;
}


/*
 * Calculates the number of auxiliary DBN variables that are required
 * for the given effect.
 */
static int auxiliary_dbn_variables(const Effect& effect) {
  const ConjunctiveEffect* ce =
    dynamic_cast<const ConjunctiveEffect*>(&effect);
  if (ce != 0) {
    int n = 0;
    for (EffectList::const_iterator ei = ce->conjuncts().begin();
         ei != ce->conjuncts().end(); ei++) {
      n += auxiliary_dbn_variables(**ei);
    }
    return n;
  }

  const ConditionalEffect* we =
    dynamic_cast<const ConditionalEffect*>(&effect);
  if (we != 0) {
    return auxiliary_dbn_variables(we->effect());
  }

  const ProbabilisticEffect* pe =
    dynamic_cast<const ProbabilisticEffect*>(&effect);
  if (pe != 0) {
    Rational p_rest = 1;
    size_t n = pe->size();
    for (size_t i = 0; i < n; i++) {
      p_rest = p_rest - pe->probability(i);
    }
    if (p_rest > 0) {
      n++;
    }
    n = ceil_log2(n);
    for (size_t i = 0; i < pe->size(); i++) {
      n += auxiliary_dbn_variables(pe->effect(i));
    }
    return n;
  }

  return 0;
}


/*
 * Calculates the number of auxiliary DBN variables that are required
 * for probabilistic effects of the problem's actions.
 */
static int auxiliary_dbn_variables(const Problem& problem) {
  int num_bits = 0;
  for (ActionSet::const_iterator ai = problem.actions().begin();
       ai != problem.actions().end(); ai++) {
    const Action& action = **ai;
    num_bits = std::max(num_bits, auxiliary_dbn_variables(action.effect()));
  }
  return num_bits;
}


/* ====================================================================== */
/* effect_dbn */

/*
 * Combines two CPTs.
 */
static void combine_cpts(std::map<int, DdNode*>& cpts1,
                         const std::map<int, DdNode*>& cpts2) {
  std::map<int, DdNode*>::iterator ci = cpts1.begin();
  std::map<int, DdNode*>::const_iterator cj = cpts2.begin();
  while (ci != cpts1.end() && cj != cpts2.end()) {
    int i = (*ci).first;
    if (i < (*cj).first) {
      ci++;
    } else if ((*cj).first < i) {
      cpts1.insert(*cj);
      ci = cpts1.find(i);
      cj++;
    } else {
      DdNode* ddo = Cudd_bddOr(dd_man, (*ci).second, (*cj).second);
      Cudd_Ref(ddo);
      Cudd_RecursiveDeref(dd_man, (*ci).second);
      Cudd_RecursiveDeref(dd_man, (*cj).second);
      (*ci).second = ddo;
      ci++;
      cj++;
    }
  }
  while (cj != cpts2.end()) {
    cpts1.insert(*cj);
    cj++;
  }
}


/*
 * Constructs a DBN representing the given effect.
 */
static DdNode* effect_dbn(std::map<int, DdNode*>& cpts,
                          std::map<int, DdNode*>& ncpts,
                          std::vector<DdNode*>& acpts, int& next_aux,
                          DdNode* condition_bdd, DdNode* aux_cond,
                          const Effect& effect) {
  if (condition_bdd == Cudd_ReadLogicZero(dd_man) || effect.empty()) {
    /*
     * The condition is false or the effect is empty.
     */
    DdNode* ddR = Cudd_ReadZero(dd_man);
    Cudd_Ref(ddR);
    return ddR;
  }

  const UpdateEffect* ue = dynamic_cast<const UpdateEffect*>(&effect);
  if (ue != 0) {
    /*
     * Only reward updates are supported.
     */
    const Fluent& fluent = ue->update().fluent();
    if (reward_function == 0 || fluent.function() != *reward_function) {
      throw std::logic_error("numeric state variables not supported");
    }
    ValueMap values;
    values[&fluent] = 0;
    ue->update().affect(values);
    DdNode* ddc = Cudd_BddToAdd(dd_man, condition_bdd);
    Cudd_Ref(ddc);
    DdNode* ddr = Cudd_addConst(dd_man, values[&fluent].double_value());
    Cudd_Ref(ddr);
    DdNode* ddR = Cudd_addApply(dd_man, Cudd_addTimes, ddc, ddr);
    Cudd_Ref(ddR);
    Cudd_RecursiveDeref(dd_man, ddc);
    Cudd_RecursiveDeref(dd_man, ddr);
    return ddR;
  }

  const SimpleEffect* se = dynamic_cast<const SimpleEffect*>(&effect);
  if (se != 0) {
    /*
     * Set the CPT for the affected variable, taking the condition
     * into account.
     */
    bool is_true = typeid(*se) == typeid(AddEffect);
    int v = state_variables[&se->atom()];
    Cudd_Ref(aux_cond);
    if (is_true) {
      cpts.insert(std::make_pair(v, aux_cond));
    } else {
      ncpts.insert(std::make_pair(v, aux_cond));
    }
    DdNode* ddR = Cudd_ReadZero(dd_man);
    Cudd_Ref(ddR);
    return ddR;
  }

  const ConjunctiveEffect* ce =
    dynamic_cast<const ConjunctiveEffect*>(&effect);
  if (ce != 0) {
    /*
     * Take the maximum of the CPTs for the conjuncts.
     */
    EffectList::const_iterator ei = ce->conjuncts().begin();
    DdNode* ddR = effect_dbn(cpts, ncpts, acpts, next_aux,
                             condition_bdd, aux_cond, **ei);
    for (ei++; ei != ce->conjuncts().end(); ei++) {
      std::map<int, DdNode*> c_cpts, c_ncpts;
      DdNode* ddr = effect_dbn(c_cpts, c_ncpts, acpts, next_aux,
                               condition_bdd, aux_cond, **ei);
      combine_cpts(cpts, c_cpts);
      combine_cpts(ncpts, c_ncpts);
      DdNode* dda = Cudd_addApply(dd_man, Cudd_addPlus, ddR, ddr);
      Cudd_Ref(dda);
      Cudd_RecursiveDeref(dd_man, ddR);
      Cudd_RecursiveDeref(dd_man, ddr);
      ddR = dda;
    }
    return ddR;
  }

  const ConditionalEffect* we =
    dynamic_cast<const ConditionalEffect*>(&effect);
  if (we != 0) {
    /*
     * Add to condition and recurse.
     */
    DdNode* ddf = formula_bdd(we->condition());
    DdNode* ddc = Cudd_bddAnd(dd_man, condition_bdd, ddf);
    Cudd_Ref(ddc);
    DdNode* dda = Cudd_bddAnd(dd_man, aux_cond, ddf);
    Cudd_Ref(dda);
    Cudd_RecursiveDeref(dd_man, ddf);
    DdNode* ddR = effect_dbn(cpts, ncpts, acpts, next_aux,
                             ddc, dda, we->effect());
    Cudd_RecursiveDeref(dd_man, ddc);
    Cudd_RecursiveDeref(dd_man, dda);
    return ddR;
  }

  const ProbabilisticEffect* pe =
    dynamic_cast<const ProbabilisticEffect*>(&effect);
  if (pe != 0) {
    /*
     * Handle probabilistic effect by introducing auxiliary state
     * variables.
     */
    Rational p_rest = 1;
    int n = pe->size();
    for (int i = 0; i < n; i++) {
      p_rest = p_rest - pe->probability(i);
    }
    if (p_rest > 0) {
      n++;
    }
    int low_bit = next_aux;
    int high_bit = low_bit + ceil_log2(n) - 1;
    next_aux = high_bit + 1;
    DdNode* ddP = Cudd_ReadZero(dd_man);
    Cudd_Ref(ddP);
    for (int i = 0; i < n; i++) {
      Rational p = (i + 1 == n && p_rest > 0) ? p_rest : pe->probability(i);
      DdNode* ddi = Cudd_ReadOne(dd_man);
      Cudd_Ref(ddi);
      for (int b = high_bit; b >= low_bit; b--) {
        int v = 1 << (high_bit - b);
        DdNode* ddb = bdd_var(b, true);
        if ((v & i) == 0) {
          ddb = Cudd_Not(ddb);
        }
        DdNode* dda = Cudd_bddAnd(dd_man, ddb, ddi);
        Cudd_Ref(dda);
        Cudd_RecursiveDeref(dd_man, ddi);
        ddi = dda;
      }
      DdNode* ddI = Cudd_BddToAdd(dd_man, ddi);
      Cudd_Ref(ddI);
      Cudd_RecursiveDeref(dd_man, ddi);
      DdNode* ddp = Cudd_addConst(dd_man, p.double_value());
      Cudd_Ref(ddp);
      ddi = Cudd_addApply(dd_man, Cudd_addTimes, ddI, ddp);
      Cudd_Ref(ddi);
      Cudd_RecursiveDeref(dd_man, ddI);
      Cudd_RecursiveDeref(dd_man, ddp);
      ddp = Cudd_addApply(dd_man, Cudd_addPlus, ddi, ddP);
      Cudd_Ref(ddp);
      Cudd_RecursiveDeref(dd_man, ddi);
      Cudd_RecursiveDeref(dd_man, ddP);
      ddP = ddp;
    }
    acpts.push_back(ddP);
    n = pe->size();
    DdNode* ddR = Cudd_ReadZero(dd_man);
    Cudd_Ref(ddR);
    for (int i = 0; i < n; i++) {
      DdNode* ddc = aux_cond;
      Cudd_Ref(ddc);
      for (int j = high_bit; j >= low_bit; j--) {
        int s = 1 << (high_bit - j);
        DdNode* ddb = bdd_var(j, true);
        if ((i & s) == 0) {
          ddb = Cudd_Not(ddb);
        }
        DdNode* dda = Cudd_bddAnd(dd_man, ddc, ddb);
        Cudd_Ref(dda);
        Cudd_RecursiveDeref(dd_man, ddc);
        ddc = dda;
      }
      DdNode* ddr;
      if (i == 0) {
        ddr = effect_dbn(cpts, ncpts, acpts, next_aux,
                         condition_bdd, ddc, pe->effect(i));
      } else {
        std::map<int, DdNode*> o_cpts, o_ncpts;
        ddr = effect_dbn(o_cpts, o_ncpts, acpts, next_aux,
                         condition_bdd, ddc, pe->effect(i));
        combine_cpts(cpts, o_cpts);
        combine_cpts(ncpts, o_ncpts);
      }
      Cudd_RecursiveDeref(dd_man, ddc);
      Rational p = ((i + 1 == n && p_rest > 0) ? p_rest : pe->probability(i));
      DdNode* ddp = Cudd_addConst(dd_man, p.double_value());
      Cudd_Ref(ddp);
      DdNode* ddm = Cudd_addApply(dd_man, Cudd_addTimes, ddr, ddp);
      Cudd_Ref(ddm);
      Cudd_RecursiveDeref(dd_man, ddr);
      Cudd_RecursiveDeref(dd_man, ddp);
      ddr = Cudd_addApply(dd_man, Cudd_addPlus, ddR, ddm);
      Cudd_Ref(ddr);
      Cudd_RecursiveDeref(dd_man, ddR);
      Cudd_RecursiveDeref(dd_man, ddm);
      ddR = ddr;
    }
    return ddR;
  }

  /*
   * No other types of effects exist.
   */
  throw std::logic_error("unexpected effect");
}


/* ====================================================================== */
/* action_dbn */

/*
 * Constructs a DBN for the given action.
 */
static DdNode* action_dbn(std::map<int, DdNode*>& cpts,
                          std::vector<DdNode*>& acpts, int& next_aux,
                          const Action& action) {
  std::map<int, DdNode*> ncpts;
  DdNode* ddc = Cudd_ReadOne(dd_man);
  Cudd_Ref(ddc);
  DdNode* ddR = effect_dbn(cpts, ncpts, acpts, next_aux,
                           ddc, ddc, action.effect());
  Cudd_RecursiveDeref(dd_man, ddc);
  for (int i = 0; i < nvars - aux_vars; i++) {
    std::map<int, DdNode*>::iterator ci = cpts.find(i);
    std::map<int, DdNode*>::const_iterator cj = ncpts.find(i);
    if (ci != cpts.end()) {
      if (cj != ncpts.end()) {
        DdNode* dda = Cudd_bddAnd(dd_man, (*ci).second, (*cj).second);
        Cudd_Ref(dda);
        if (dda != Cudd_ReadLogicZero(dd_man)) {
          throw std::logic_error("action `" + action.name()
                                 + "' has inconsistent effects");
        }
        Cudd_RecursiveDeref(dd_man, dda);
        DdNode* ddv = Cudd_Not(bdd_var(i));
        DdNode* ddo = Cudd_bddOr(dd_man, ddv, (*cj).second);
        Cudd_Ref(ddo);
        Cudd_RecursiveDeref(dd_man, (*cj).second);
        DdNode* ddn = Cudd_Not(ddo);
        Cudd_Ref(ddn);
        Cudd_RecursiveDeref(dd_man, ddo);
        ddo = Cudd_bddOr(dd_man, ddn, (*ci).second);
        Cudd_Ref(ddo);
        Cudd_RecursiveDeref(dd_man, ddn);
        Cudd_RecursiveDeref(dd_man, (*ci).second);
        (*ci).second = ddo;
      } else {
        DdNode* ddv = bdd_var(i);
        DdNode* ddo = Cudd_bddOr(dd_man, ddv, (*ci).second);
        Cudd_Ref(ddo);
        Cudd_RecursiveDeref(dd_man, (*ci).second);
        (*ci).second = ddo;
      }
    } else if (cj != ncpts.end()) {
      DdNode* ddv = Cudd_Not(bdd_var(i));
      DdNode* ddo = Cudd_bddOr(dd_man, ddv, (*cj).second);
      Cudd_Ref(ddo);
      Cudd_RecursiveDeref(dd_man, (*cj).second);
      DdNode* ddn = Cudd_Not(ddo);
      Cudd_Ref(ddn);
      Cudd_RecursiveDeref(dd_man, ddo);
      cpts.insert(std::make_pair(i, ddn));
    } else {
      DdNode* ddv = bdd_var(i);
      Cudd_Ref(ddv);
      cpts.insert(std::make_pair(i, ddv));
    }
  }
  return ddR;
}


/* ====================================================================== */
/* Value iteration. */

/*
 * Returns a policy for the current problem generated using value
 * iteration.
 */
static DdNode* value_iteration(const Problem& problem,
                               DdNode* ddng, DdNode** col_variables,
                               double gamma, double epsilon) {
  if (verbosity > 0) {
    std::cout << "Value iteration";
  }
  /*
   * Precompute variable permutations.
   */
  int* row_to_col = new int[2*nvars];
  int* col_to_row = new int[2*nvars];
  for (int i = 0; i < nvars; i++) {
    row_to_col[2*i] = 2*i + 1;
    row_to_col[2*i + 1] = 2*i + 1;
    col_to_row[2*i] = 2*i;
    col_to_row[2*i + 1] = 2*i;
  }

  /*
   * Construct action conditions, action value filters, and initial policy.
   */
  std::map<const Action*, DdNode*> filters;
  std::map<const Action*, DdNode*> policy;
  for (std::map<const Action*, DdNode*>::const_iterator ai =
         action_transitions.begin();
       ai != action_transitions.end(); ai++) {
    DdNode* ddc = formula_bdd((*ai).first->precondition());
    DdNode* ddt = Cudd_bddAnd(dd_man, ddc, ddng);
    Cudd_Ref(ddt);
    Cudd_RecursiveDeref(dd_man, ddc);
    DdNode* ddn = Cudd_Not(ddt);
    Cudd_Ref(ddn);
    Cudd_RecursiveDeref(dd_man, ddt);
    ddc = Cudd_BddToAdd(dd_man, ddn);
    Cudd_Ref(ddc);
    Cudd_RecursiveDeref(dd_man, ddn);
    ddn = Cudd_ReadPlusInfinity(dd_man);
    Cudd_Ref(ddn);
    ddt = Cudd_addApply(dd_man, Cudd_addTimes, ddc, ddn);
    Cudd_Ref(ddt);
    Cudd_RecursiveDeref(dd_man, ddc);
    Cudd_RecursiveDeref(dd_man, ddn);
    filters.insert(std::make_pair((*ai).first, ddt));
    DdNode* ddp = Cudd_ReadLogicZero(dd_man);
    Cudd_Ref(ddp);
    policy.insert(std::make_pair((*ai).first, ddp));
  }

  /*
   * Iterate until value function converges.
   */
  DdNode* ddg = Cudd_addConst(dd_man, gamma);
  Cudd_Ref(ddg);
  DdNode* ddV = Cudd_ReadZero(dd_man);
  Cudd_Ref(ddV);
  if (verbosity > 2) {
    std::cout << std::endl << "V 0:" << std::endl;
    Cudd_PrintDebug(dd_man, ddV, 2*nvars, 2);
  }
  double tolerance = epsilon*(1.0 - gamma)/(2.0*gamma);
  bool done = false;
  size_t iters = 0;
  while (!done) {
    iters++;
    if (verbosity == 1) {
      if (iters % 1000 == 0) {
        std::cout << ':';
      } else if (iters % 100 == 0) {
        std::cout << '.';
      }
    }
    DdNode* ddVp = Cudd_addPermute(dd_man, ddV, row_to_col);
    Cudd_Ref(ddVp);
    DdNode* ddM = Cudd_ReadZero(dd_man);
    Cudd_Ref(ddM);
    for (std::map<const Action*, DdNode*>::const_iterator ai =
           action_transitions.begin();
         ai != action_transitions.end(); ai++) {
      DdNode* dds = Cudd_addMatrixMultiply(dd_man, (*ai).second, ddVp,
                                           col_variables, nvars - aux_vars);
      Cudd_Ref(dds);
      DdNode* ddp = Cudd_addApply(dd_man, Cudd_addTimes, ddg, dds);
      Cudd_Ref(ddp);
      Cudd_RecursiveDeref(dd_man, dds);
      dds = Cudd_addApply(dd_man, Cudd_addPlus,
                          action_rewards[(*ai).first], ddp);
      Cudd_Ref(dds);
      Cudd_RecursiveDeref(dd_man, ddp);
      ddp = Cudd_addPermute(dd_man, dds, col_to_row);
      Cudd_Ref(ddp);
      Cudd_RecursiveDeref(dd_man, dds);
      DdNode* ddf = Cudd_addApply(dd_man, Cudd_addMinus, ddp,
                                  filters[(*ai).first]);
      Cudd_Ref(ddf);
      Cudd_RecursiveDeref(dd_man, ddp);
      ddp = ddf;
      if (verbosity > 3) {
        std::cout << std::endl << "value of action " << *(*ai).first << ':'
                  << std::endl;
        Cudd_PrintDebug(dd_man, ddp, 2*nvars, 2);
      }
      DdNode* ddm = Cudd_addApply(dd_man, Cudd_addMinus, ddp, ddM);
      Cudd_Ref(ddm);
      DdNode*& dde = policy[(*ai).first];
      Cudd_RecursiveDeref(dd_man, dde);
      dde = Cudd_addBddStrictThreshold(dd_man, ddm, 0);
      Cudd_Ref(dde);
      Cudd_RecursiveDeref(dd_man, ddm);
      DdNode* ddn = Cudd_Not(dde);
      Cudd_Ref(ddn);
      if (ddn != Cudd_ReadOne(dd_man)) {
        for (std::map<const Action*, DdNode*>::const_iterator aj =
               policy.begin();
             (*aj).first != (*ai).first; aj++) {
          DdNode*& ddj = policy[(*aj).first];
          DdNode* dda = Cudd_bddAnd(dd_man, ddn, ddj);
          Cudd_Ref(dda);
          Cudd_RecursiveDeref(dd_man, ddj);
          ddj = dda;
        }
      }
      Cudd_RecursiveDeref(dd_man, ddn);
      ddm = Cudd_addApply(dd_man, Cudd_addMaximum, ddp, ddM);
      Cudd_Ref(ddm);
      Cudd_RecursiveDeref(dd_man, ddM);
      ddM = ddm;
      if (verbosity > 3) {
        std::cout << "current max values:" << std::endl;
        Cudd_PrintDebug(dd_man, ddM, 2*nvars, 2);
      }
      Cudd_RecursiveDeref(dd_man, ddp);
    }
    Cudd_RecursiveDeref(dd_man, ddVp);
    ddVp = ddM;
    if (verbosity > 2) {
      std::cout << "V " << iters << ':' << std::endl;
      Cudd_PrintDebug(dd_man, ddVp, 2*nvars, 2);
    }
    done = (Cudd_EqualSupNorm(dd_man, ddV, ddVp, tolerance, 0) == 1);
    Cudd_RecursiveDeref(dd_man, ddV);
    ddV = ddVp;
  }
  if (verbosity == 1) {
    std::cout << ' ' << iters << " iterations." << std::endl;
  }
  Cudd_RecursiveDeref(dd_man, ddV);
  Cudd_RecursiveDeref(dd_man, ddg);
  for (std::map<const Action*, DdNode*>::const_iterator ai = filters.begin();
       ai != filters.end(); ai++) {
    Cudd_RecursiveDeref(dd_man, (*ai).second);
  }
  delete row_to_col;
  delete col_to_row;

  /*
   * Construct single policy MTBDD.
   */
  DdNode* ddP = Cudd_ReadZero(dd_man);
  Cudd_Ref(ddP);
  for (std::map<const Action*, DdNode*>::const_iterator ai = policy.begin();
       ai != policy.end(); ai++) {
    if (verbosity > 5) {
      std::cout << "condition for " << *(*ai).first << ':' << std::endl;
      Cudd_PrintDebug(dd_man, (*ai).second, 2*nvars, 2);
    }
    if ((*ai).second != Cudd_ReadLogicZero(dd_man)) {
      policy_actions.push_back((*ai).first);
      DdNode* ddp = Cudd_BddToAdd(dd_man, (*ai).second);
      Cudd_Ref(ddp);
      DdNode* ddi = Cudd_addConst(dd_man, policy_actions.size());
      Cudd_Ref(ddi);
      DdNode* ddt = Cudd_addApply(dd_man, Cudd_addTimes, ddi, ddp);
      Cudd_Ref(ddt);
      Cudd_RecursiveDeref(dd_man, ddi);
      Cudd_RecursiveDeref(dd_man, ddp);
      ddp = Cudd_addApply(dd_man, Cudd_addPlus, ddt, ddP);
      Cudd_Ref(ddp);
      Cudd_RecursiveDeref(dd_man, ddt);
      Cudd_RecursiveDeref(dd_man, ddP);
      ddP = ddp;
    }
    Cudd_RecursiveDeref(dd_man, (*ai).second);
  }
  return ddP;
}


/* ====================================================================== */
/* solve_problem */

/* Solves the given problem. */
static DdNode* solve_problem(const Problem& problem,
                             double gamma, double epsilon) {
  /*
   * Extract the reward function.
   */
  reward_function = problem.domain().functions().find_function("reward");
  if (reward_function != 0) {
    if (problem.goal_reward() != 0) {
      ValueMap values;
      const Fluent& fluent = problem.goal_reward()->fluent();
      values[&fluent] = 0;
      problem.goal_reward()->affect(values);
      goal_reward = values[&fluent];
    } else {
      goal_reward = 0;
    }
  } else {
    goal_reward = 1;
  }

  /*
   * Collect state variables and assign indices to them.
   */
  const StateFormula& inst_goal =
    problem.goal().instantiation(SubstitutionMap(), problem.terms(),
                                 problem.init_atoms(), problem.init_values(),
                                 false);
  RCObject::ref(&inst_goal);
  collect_state_variables(inst_goal);
  for (ActionSet::const_iterator ai = problem.actions().begin();
       ai != problem.actions().end(); ai++) {
    const Action& action = **ai;
    collect_state_variables(action.effect(), problem.domain());
  }
  for (EffectList::const_iterator ei = problem.init_effects().begin();
       ei != problem.init_effects().end(); ei++) {
    collect_state_variables(**ei, problem.domain());
  }
  nvars = state_variables.size();
  aux_vars = auxiliary_dbn_variables(problem);
  nvars += aux_vars;
  ordered_vars.resize(nvars);
  for (int i = 0; i < nvars; i++) {
    if (i < nvars - aux_vars) {
      var_order.push_back(i + aux_vars);
      ordered_vars[i + aux_vars] = i;
    } else {
      var_order.push_back(i - (nvars - aux_vars));
      ordered_vars[i - (nvars - aux_vars)] = i;
    }
  }
  if (verbosity > 0) {
    std::cout << std::endl << "Number of state variables: " << nvars
              << std::endl;
    if (verbosity > 1) {
      for (std::map<int, const Atom*>::const_iterator vi =
             dynamic_atoms.begin();
           vi != dynamic_atoms.end(); vi++) {
        std::cout << (*vi).first << '\t' << var_order[(*vi).first] << '\t'
                  << *(*vi).second << std::endl;
      }
      for (int i = nvars - aux_vars; i < nvars; i++) {
        std::cout << i << '\t' << var_order[i] << "\tauxiliary variable"
                  << std::endl;
      }
    }
  }

  /*
   * Initialize CUDD.
   */
  dd_man = Cudd_Init(2*nvars, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0);

  /*
   * Construct a BDD representing goal states.
   */
  DdNode* ddg = formula_bdd(inst_goal);
  RCObject::destructive_deref(&inst_goal);
  if (verbosity > 1) {
    std::cout << std::endl << "Goal state BDD:" << std::endl;
    Cudd_PrintDebug(dd_man, ddg, 2*nvars, 2);
  }

  /*
   * Collect column variables and compute their cube.
   */
  DdNode** col_variables = new DdNode*[nvars];
  for (int i = 0; i < nvars; i++) {
    col_variables[i] = add_var(i, true);
    Cudd_Ref(col_variables[i]);
  }
  DdNode* aux_cube = Cudd_addComputeCube(dd_man,
                                         col_variables + (nvars - aux_vars),
                                         0, aux_vars);
  Cudd_Ref(aux_cube);

  DdNode* ddng = Cudd_Not(ddg);
  Cudd_Ref(ddng);
  DdNode* ddgr;
  if (goal_reward == 0) {
    ddgr = Cudd_ReadZero(dd_man);
    Cudd_Ref(ddgr);
  } else {
    int* row_to_col = new int[2*nvars];
    for (int i = 0; i < nvars; i++) {
      row_to_col[2*i] = 2*i + 1;
      row_to_col[2*i + 1] = 2*i + 1;
    }
    DdNode* ddgp = Cudd_bddPermute(dd_man, ddg, row_to_col);
    Cudd_Ref(ddgp);
    delete row_to_col;
    DdNode* ddt = Cudd_BddToAdd(dd_man, ddgp);
    Cudd_Ref(ddt);
    Cudd_RecursiveDeref(dd_man, ddgp);
    DdNode* ddr = Cudd_addConst(dd_man, goal_reward.double_value());
    Cudd_Ref(ddr);
    ddgr = Cudd_addApply(dd_man, Cudd_addTimes, ddt, ddr);
    Cudd_Ref(ddgr);
    Cudd_RecursiveDeref(dd_man, ddt);
    Cudd_RecursiveDeref(dd_man, ddr);
  }
  Cudd_RecursiveDeref(dd_man, ddg);
  for (ActionSet::const_iterator ai = problem.actions().begin();
       ai != problem.actions().end(); ai++) {
    const Action& action = **ai;
    if (verbosity > 3) {
      std::cout << "processing action " << action << "..." << std::endl;
    }
    std::map<int, DdNode*> cpts;
    std::vector<DdNode*> acpts;
    int next_aux = nvars - aux_vars;
    DdNode* ddR = action_dbn(cpts, acpts, next_aux, action);
    DdNode* ddP = Cudd_ReadOne(dd_man);
    Cudd_Ref(ddP);
    if (verbosity > 3) {
      std::cout << std::endl << "DBN for " << action << ':' << std::endl;
    }
    const std::vector<int>& ov = ordered_vars;
    for (std::vector<int>::const_reverse_iterator oi = ov.rbegin();
         oi != ov.rend(); oi++) {
      std::map<int, DdNode*>::const_iterator ci = cpts.find(*oi);
      if (ci != cpts.end()) {
        if (verbosity > 3) {
          std::cout << "CPT for " << *dynamic_atoms[(*ci).first] << ':'
                    << std::endl;
          Cudd_PrintDebug(dd_man, (*ci).second, 2*nvars, 2);
        }
        DdNode* ddi = (*ci).second;
        ddi = Cudd_BddToAdd(dd_man, ddi);
        Cudd_Ref(ddi);
        DdNode* ddv = add_var((*ci).first, true);
        Cudd_Ref(ddv);
        DdNode* ddnv = Cudd_addCmpl(dd_man, ddv);
        Cudd_Ref(ddnv);
        DdNode* ddp = Cudd_addApply(dd_man, Cudd_addTimes, ddv, ddi);
        Cudd_Ref(ddp);
        Cudd_RecursiveDeref(dd_man, ddv);
        DdNode* dd1 = Cudd_ReadOne(dd_man);
        Cudd_Ref(dd1);
        DdNode* ddn = Cudd_addApply(dd_man, Cudd_addMinus, dd1, ddi);
        Cudd_Ref(ddn);
        Cudd_RecursiveDeref(dd_man, dd1);
        Cudd_RecursiveDeref(dd_man, ddi);
        Cudd_RecursiveDeref(dd_man, (*ci).second);
        DdNode* ddq = Cudd_addApply(dd_man, Cudd_addTimes, ddnv, ddn);
        Cudd_Ref(ddq);
        Cudd_RecursiveDeref(dd_man, ddnv);
        Cudd_RecursiveDeref(dd_man, ddn);
        DdNode* dds = Cudd_addApply(dd_man, Cudd_addPlus, ddp, ddq);
        Cudd_Ref(dds);
        Cudd_RecursiveDeref(dd_man, ddp);
        Cudd_RecursiveDeref(dd_man, ddq);
        ddp = Cudd_addApply(dd_man, Cudd_addTimes, dds, ddP);
        Cudd_Ref(ddp);
        Cudd_RecursiveDeref(dd_man, dds);
        Cudd_RecursiveDeref(dd_man, ddP);
        ddP = ddp;
      }
    }
    for (std::vector<DdNode*>::reverse_iterator ai = acpts.rbegin();
         ai != acpts.rend(); ai++) {
      DdNode* ddA = *ai;
      if (verbosity > 3) {
        std::cout << "CPT for auxiliary variables:" << std::endl;
        Cudd_PrintDebug(dd_man, ddA, 2*nvars, 2);
      }
      DdNode* ddm = Cudd_addApply(dd_man, Cudd_addTimes, ddA, ddP);
      Cudd_Ref(ddm);
      Cudd_RecursiveDeref(dd_man, ddA);
      Cudd_RecursiveDeref(dd_man, ddP);
      ddP = ddm;
    }
    if (next_aux > nvars - aux_vars) {
      int cube_start = nvars - aux_vars;
      int cube_end = next_aux - 1;
      DdNode** aux_variables = new DdNode*[cube_end - cube_start + 1];
      for (int i = cube_start; i <= cube_end; i++) {
        aux_variables[i - cube_start] = add_var(i, true);
        Cudd_Ref(aux_variables[i - cube_start]);
      }
      DdNode* aux_cube = Cudd_addComputeCube(dd_man, aux_variables, 0,
                                             cube_end - cube_start + 1);
      Cudd_Ref(aux_cube);
      for (int i = cube_start; i <= cube_end; i++) {
        Cudd_RecursiveDeref(dd_man, aux_variables[i - cube_start]);
      }
      delete aux_variables;
      DdNode* ddm = Cudd_addExistAbstract(dd_man, ddP, aux_cube);
      Cudd_Ref(ddm);
      Cudd_RecursiveDeref(dd_man, ddP);
      Cudd_RecursiveDeref(dd_man, aux_cube);
      ddP = ddm;
    }
    if (goal_reward != 0) {
      DdNode* dde = Cudd_addMatrixMultiply(dd_man, ddP, ddgr,
                                           col_variables, nvars - aux_vars);
      Cudd_Ref(dde);
      DdNode* ddt = Cudd_addApply(dd_man, Cudd_addPlus, dde, ddR);
      Cudd_Ref(ddt);
      Cudd_RecursiveDeref(dd_man, dde);
      Cudd_RecursiveDeref(dd_man, ddR);
      ddR = ddt;
    }
    if (verbosity > 2) {
      std::cout << std::endl << "Probability matrix for " << action << ':'
                << std::endl;
      Cudd_PrintDebug(dd_man, ddP, 2*nvars, 2);
      std::cout << std::endl << "Reward vector for " << action << ':'
                << std::endl;
      Cudd_PrintDebug(dd_man, ddR, 2*nvars, 2);
    }
    action_transitions.insert(std::make_pair(&action, ddP));
    action_rewards.insert(std::make_pair(&action, ddR));
  }
  Cudd_RecursiveDeref(dd_man, ddgr);
  Cudd_RecursiveDeref(dd_man, aux_cube);

  DdNode* ddP = value_iteration(problem, ddng, col_variables, gamma, epsilon);
  Cudd_RecursiveDeref(dd_man, ddng);
  for (int i = 0; i < nvars; i++) {
    Cudd_RecursiveDeref(dd_man, col_variables[i]);
  }
  delete col_variables;
  if (verbosity > 1) {
    std::cout << std::endl << "Policy:" << std::endl;
    Cudd_PrintDebug(dd_man, ddP, 2*nvars, 2);
    std::cout << "0\t<quit>" << std::endl;
    for (size_t i = 0; i < policy_actions.size(); i++) {
      std::cout << i + 1 << '\t' << *policy_actions[i] << std::endl;
    }
  }

  /*
   * Clean up.
   */
  state_variables.clear();
  for (std::map<const Action*, DdNode*>::const_iterator ai =
         action_transitions.begin();
       ai != action_transitions.end(); ai++) {
    Cudd_RecursiveDeref(dd_man, (*ai).second);
  }
  action_transitions.clear();
  for (std::map<const Action*, DdNode*>::const_iterator ai =
         action_rewards.begin();
       ai != action_rewards.end(); ai++) {
    Cudd_RecursiveDeref(dd_man, (*ai).second);
  }
  action_rewards.clear();

  return ddP;
}


/* ====================================================================== */
/* MTBDDPlanner */

/* Deletes this MTBDD planner. */
MTBDDPlanner::~MTBDDPlanner() {
  if (dd_man_ != 0) {
    if (mapping_ != 0) {
      Cudd_RecursiveDeref(dd_man_, mapping_);
    }
    Cudd_DebugCheck(dd_man_);
    int unrel = Cudd_CheckZeroRef(dd_man_);
    if (unrel != 0) {
      std::cerr << unrel << " unreleased DDs" << std::endl;
    }
    Cudd_Quit(dd_man_);
  }
}  


void MTBDDPlanner::initRound() {
  if (dd_man_ == 0) {
    mapping_ = solve_problem(_problem, gamma_, epsilon_);
    dd_man_ = dd_man;
    actions_ = policy_actions;
    policy_actions.clear();
    dynamic_atoms_ = dynamic_atoms;
    dynamic_atoms.clear();
  }
}


const Action* MTBDDPlanner::decideAction(const AtomSet& atoms,
                                         const ValueMap& values) {
  DdNode* dds = state_bdd(dd_man_, dynamic_atoms_, atoms);
  DdNode* ddS = Cudd_BddToAdd(dd_man_, dds);
  Cudd_Ref(ddS);
  Cudd_RecursiveDeref(dd_man_, dds);
  DdNode* dda = Cudd_addEvalConst(dd_man_, ddS, mapping_);
  Cudd_Ref(dda);
  Cudd_RecursiveDeref(dd_man_, ddS);
  size_t id = int(Cudd_V(dda) + 0.5);
  Cudd_RecursiveDeref(dd_man_, dda);
  return (id > 0) ? actions_[id - 1] : 0;
}


void MTBDDPlanner::endRound() {
}
