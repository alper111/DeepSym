import os
import argparse
import yaml

parser = argparse.ArgumentParser("Parse plan.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()
opts = yaml.safe_load(open(args.opts, "r"))

file = open(os.path.join(opts["save"], "planresult.txt"))
lines = file.readlines()
file.close()
plans = []
current_plan = []
is_plan = False
for line in lines:
    if line[:5] == "    +":
        is_plan = True
        if line[6] == "s":
            current_plan.append(line[:-1])
    elif is_plan:
        is_plan = False
        if line[1] == "r":
            current_plan.append(line[:-1])
            plans.append(current_plan.copy())
        current_plan = []
file_loc = os.path.join(opts["save"], "plan.txt")
if os.path.exists(file_loc):
    os.remove(file_loc)
print("plan probability: %.2f" % (len(plans)/100))
print("plan probability: %.2f" % (len(plans)/100), file=open(file_loc, "a"))
if len(plans) > 0:
    plan = plans[0].copy()
    print(plan)
    for token in plan[:-1]:
        token = token[6:-1].split(" ")
        print("stack " + token[1].upper() + " " + token[2].upper(), file=open(file_loc, "a"))
else:
    print("not found.", file=open(file_loc, "a"))
