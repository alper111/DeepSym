from sklearn.tree import DecisionTreeClassifier
import torch
import utils
import argparse
import os

parser = argparse.ArgumentParser("learn pddl rules from decision tree.")
parser.add_argument("-load", help="object path", type=str, required=True)
args = parser.parse_args()


codes_second = torch.load(os.path.join(args.load, "codes_second.torch"))
codes_first = torch.load(os.path.join(args.load, "objcodes_second.torch"))
codes = torch.cat([codes_first, codes_second], dim=-1)
effects = torch.load("data/effects_2.torch")
effects = effects.abs()
eff_mu = effects.mean(dim=0)
eff_std = effects.std(dim=0)
effects = (effects - eff_mu) / (eff_std + 1e-6)

# need a mechanism to select the number K
K = 6
centroids, assigns, mse, _ = utils.kmeans(effects, k=K)
centroids = centroids * (eff_std + 1e-6) + eff_mu
for i, c_i in enumerate(centroids):
    print("Centroid %d: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f" %
          (i, c_i[0], c_i[1], c_i[2], c_i[3], c_i[4], c_i[5]))

print("Which one is the stack effect?")
print(">>>", end="")
stack_idx = int(input())

tree = DecisionTreeClassifier()
tree.fit(codes.numpy(), assigns.numpy())

pddl_code = utils.tree_to_code(tree, ["f%d" % i for i in range(K)], stack_idx)
pretext = "(define (domain stack)\n\t(:requirements :fluents :typing :negative-preconditions :probabilistic-effects :conditional-effects)\n"
pretext += "\t(:types cube cylinder cylinderside sphere hollow)\n"
pretext += "\t(:predicates "
for i in range(codes.shape[1]):
    pretext += "(f%d) " % i
for i in range(K):
    if i == stack_idx:
        pretext += "(stack_eff) "
    else:
        pretext += "(eff%d) " % i
pretext += "(pickloc ?x) (instack ?x) (stackloc ?x) (compared))\n"
pretext += "\t(:functions (height ?x))"
print(pretext, file=open("save/domain.pddl", "a"))

action_template = "\t(:action act%d\n\t\t:parameters (?above ?below)"
for i, (precond, effect) in enumerate(pddl_code):
    print(action_template % i, file=open("save/domain.pddl", "a"))
    print("\t\t"+precond, file=open("save/domain.pddl", "a"))
    print("\t\t"+effect, file=open("save/domain.pddl", "a"))
    print("\t)", file=open("save/domain.pddl", "a"))
print(")", file=open("save/domain.pddl", "a"))
