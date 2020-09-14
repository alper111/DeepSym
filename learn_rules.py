import os
import argparse
import yaml
from sklearn.tree import DecisionTreeClassifier
import torch
import pickle
import numpy as np
import utils


parser = argparse.ArgumentParser("learn pddl rules from decision tree.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))

save_name = os.path.join(opts["save"], "domain.pddl")
if os.path.exists(save_name):
    os.remove(save_name)

category = torch.load(os.path.join(opts["save"], "category.pt"))
label = torch.load(os.path.join(opts["save"], "label.pt"))
effect_names = np.load(os.path.join(opts["save"], "effect_names.npy"))
K = len(effect_names)

tree = DecisionTreeClassifier()
tree.fit(category, label)
file = open(os.path.join(opts["save"], "tree.pkl"), "wb")
pickle.dump(tree, file)
file.close()

CODE_DIM = 2
obj_names = {}
for i in range(2**CODE_DIM):
    category = utils.decimal_to_binary(i, length=CODE_DIM)
    obj_names[category] = "objtype{}".format(i)

file_loc = os.path.join(opts["save"], "domain.pddl")
if os.path.exists(file_loc):
    os.remove(file_loc)

pddl_code = utils.tree_to_code(tree, ["f%d" % i for i in range(K)], effect_names, obj_names)
pretext = "(define (domain stack)\n"
pretext += "\t(:requirements :typing :negative-preconditions :probabilistic-effects :conditional-effects :disjunctive-preconditions)\n"
pretext += "\t(:predicates"

for i in range(K):
    pretext += "\n\t\t(%s) " % effect_names[i]
pretext += "(base) \n\t\t(pickloc ?x)\n\t\t(instack ?x)\n\t\t(stackloc ?x)\n\t\t(relation0 ?x ?y)\n\t\t(relation1 ?x ?y)"
for i in range(2**CODE_DIM):
    pretext += "\n\t\t(" + obj_names[utils.decimal_to_binary(i, length=CODE_DIM)] + " ?x)"
for i in range(7):
    pretext += "\n\t\t(H%d)" % i
for i in range(7):
    pretext += "\n\t\t(S%d)" % i
pretext += "\n\t)"
print(pretext, file=open(file_loc, "a"))

action_template = "\t(:action stack%d\n\t\t:parameters (?below ?above)"
for i, (precond, effect) in enumerate(pddl_code):
    print(action_template % i, file=open(file_loc, "a"))
    print("\t\t"+precond, file=open(file_loc, "a"))
    print("\t\t"+effect, file=open(file_loc, "a"))
    print("\t)", file=open(file_loc, "a"))
for i in range(6):
    print("\t(:action increase-height%d" % (i+1), file=open(file_loc, "a"))
    print("\t\t:precondition (and (stacked) (H%d))" % i, file=open(file_loc, "a"))
    print("\t\t:effect (and (not (H%d)) (H%d) (not (stacked)))\n\t)" % (i, i+1), file=open(file_loc, "a"))
for i in range(6):
    print("\t(:action increase-stack%d" % (i+1), file=open(file_loc, "a"))
    print("\t\t:precondition (and (inserted) (S%d))" % i, file=open(file_loc, "a"))
    print("\t\t:effect (and (not (S%d)) (S%d) (not (inserted)))\n\t)" % (i, i+1), file=open(file_loc, "a"))
print("\t(:action makebase", file=open(file_loc, "a"))
print("\t\t:parameters (?obj)", file=open(file_loc, "a"))
print("\t\t:precondition (not (base))", file=open(file_loc, "a"))
print("\t\t:effect (and (base) (stacked) (inserted) (not (pickloc ?obj)) (stackloc ?obj))", file=open(file_loc, "a"))
print("\t)", file=open(file_loc, "a"))
print(")", file=open(os.path.join(opts["save"], "domain.pddl"), "a"))
