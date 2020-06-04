import argparse
import os
from sklearn.tree import DecisionTreeClassifier
import torch
import pickle
import numpy as np
import utils


parser = argparse.ArgumentParser("learn pddl rules from decision tree.")
parser.add_argument("-ckpt", help="model path", type=str, required=True)
args = parser.parse_args()

save_name = os.path.join(args.ckpt, "domain.pddl")
if os.path.exists(save_name):
    os.remove(save_name)

category = torch.load(os.path.join(args.ckpt, "category.pt"))
label = torch.load(os.path.join(args.ckpt, "label.pt"))
effect_names = np.load(os.path.join(args.ckpt, "effect_names.npy"))
K = len(effect_names)

tree = DecisionTreeClassifier()
tree.fit(category, label)
file = open(os.path.join(args.ckpt, "tree.pkl"), "wb")
pickle.dump(tree, file)
file.close()

# obj_names = {(-1, 1): "hollow", (1, -1): "stable", (1, 1): "sphere", (-1, -1): "cylinder"}
CODE_DIM = 2
obj_names = {}
for i in range(2**CODE_DIM):
    category = utils.decimal_to_binary(i, length=CODE_DIM)
    obj_names[category] = "objtype{}".format(i)

pddl_code = utils.tree_to_code(tree, ["f%d" % i for i in range(K)], effect_names, obj_names)
pretext = "(define (domain stack)\n"
pretext += "\t(:requirements :typing :negative-preconditions :probabilistic-effects :conditional-effects)\n"
pretext += "\t(:predicates"

for i in range(K):
    pretext += "\n\t\t(%s) " % effect_names[i]
pretext += "\n\t\t(pickloc ?x)\n\t\t(instack ?x)\n\t\t(stackloc ?x)\n\t\t(larger ?x ?y)"
pretext += "\n\t\t(hollow ?x)\n\t\t(stable ?x)\n\t\t(sphere ?x)\n\t\t(cylinder ?x)"
for i in range(7):
    pretext += "\n\t\t(H%d)" % i
for i in range(7):
    pretext += "\n\t\t(S%d)" % i
pretext += "\n\t)"
print(pretext, file=open(os.path.join(args.ckpt, "domain.pddl"), "a"))

action_template = "\t(:action act%d\n\t\t:parameters (?above ?below)"
for i, (precond, effect) in enumerate(pddl_code):
    print(action_template % i, file=open(os.path.join(args.ckpt, "domain.pddl"), "a"))
    print("\t\t"+precond, file=open(os.path.join(args.ckpt, "domain.pddl"), "a"))
    print("\t\t"+effect, file=open(os.path.join(args.ckpt, "domain.pddl"), "a"))
    print("\t)", file=open(os.path.join(args.ckpt, "domain.pddl"), "a"))
for i in range(6):
    print("\t(:action gainheight%d" % (i+1), file=open(os.path.join(args.ckpt, "domain.pddl"), "a"))
    print("\t\t:precondition (and (stacked) (H%d))" % i, file=open(os.path.join(args.ckpt, "domain.pddl"), "a"))
    print("\t\t:effect (and (not (H%d)) (H%d) (not (stacked)))\n\t)" % (i, i+1), file=open(os.path.join(args.ckpt, "domain.pddl"), "a"))
for i in range(6):
    print("\t(:action gainstack%d" % (i+1), file=open(os.path.join(args.ckpt, "domain.pddl"), "a"))
    print("\t\t:precondition (and (inserted) (S%d))" % i, file=open(os.path.join(args.ckpt, "domain.pddl"), "a"))
    print("\t\t:effect (and (not (S%d)) (S%d) (not (inserted)))\n\t)" % (i, i+1), file=open(os.path.join(args.ckpt, "domain.pddl"), "a"))

print(")", file=open(os.path.join(args.ckpt, "domain.pddl"), "a"))
