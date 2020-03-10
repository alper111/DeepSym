from sklearn.tree import DecisionTreeClassifier
import torch
import utils
import argparse
import os

parser = argparse.ArgumentParser("learn pddl rules from decision tree.")
parser.add_argument("-load", help="object path", type=str)
args = parser.parse_args()


codes_second = torch.load(os.path.join(args.load, "codes_second.torch"))
codes_first = torch.load(os.path.join(args.load, "objcodes_second.torch"))
codes = torch.cat([codes_first, codes_second], dim=-1)
effects = torch.load("data/effects_2.torch")
eff_mu = effects.mean(dim=0)
eff_std = effects.std(dim=0)
effects = (effects - eff_mu) / (eff_std + 1e-6)
effects = effects.abs()

# need a mechanism to select the number K
K = 6
centroids, assigns, mse, _ = utils.kmeans(effects, k=K, epsilon=1e-5)
tree = DecisionTreeClassifier()
tree.fit(codes.numpy(), assigns.numpy())

pddl_code = utils.tree_to_code(tree, ["f%d" % i for i in range(K)])
pretext = "(define (domain stack)\n\t(:requirements :negative-preconditions :probabilistic-effects)\n"
pretext += "\t(:predicates "
for i in range(codes.shape[1]):
    pretext += "(f%d) " % i
for i in range(K):
    pretext += "(eff%d) " % i
pretext += "(pickloc ?x) (instack ?x) (stackloc ?x))"
print(pretext, file=open("save/domain.pddl", "a"))

action_template = "\t(:action act%d\n\t\t:parameters (?above ?below)"
for i, (precond, effect) in enumerate(pddl_code):
    print(action_template % i, file=open("save/domain.pddl", "a"))
    print("\t\t"+precond, file=open("save/domain.pddl", "a"))
    print("\t\t"+effect, file=open("save/domain.pddl", "a"))
    print("\t)", file=open("save/domain.pddl", "a"))
print(")", file=open("save/domain.pddl", "a"))
