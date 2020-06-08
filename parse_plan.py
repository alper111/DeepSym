import os
import argparse

parser = argparse.ArgumentParser("Parse plan.")
parser.add_argument("-ckpt", help="load path", type=str, required=True)
args = parser.parse_args()

file = open(os.path.join(args.ckpt, "planresult.txt"))
lines = file.readlines()
file.close()
plans = []
current_plan = []
is_plan = False
for line in lines:
    if line[:5] == "    +":
        is_plan = True
        if line[6] == "a":
            current_plan.append(line[:-1])
    elif is_plan:
        is_plan = False
        if line[1] == "r":
            current_plan.append(line[:-1])
            plans.append(current_plan.copy())
        current_plan = []

file_loc = os.path.join(args.ckpt, "plan.txt")
if os.path.exists(file_loc):
    os.remove(file_loc)
print("Plan:", file=open(file_loc, "a"))
if len(plans) > 0:
    print("Probability: " + str(len(plans)/100), file=open(file_loc, "a"))
    plan = plans[0].copy()
    for token in plan[:-1]:
        token = token[6:-1]
        print("("+token+")", file=open(file_loc, "a"))
else:
    print("not found.", file=open(file_loc, "a"))
