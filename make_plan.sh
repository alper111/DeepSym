#!/bin/bash
# get save location
loc="$(grep save: $1 | sed 's/^.*: //')"

# transform image to pddl problem
python recognize.py -opts "$1" -goal "$2"
# concatenate domain and problem file
cat "$loc/domain.pddl" > "$loc/temp.pddl"
cat "$loc/problem.pddl" >> "$loc/temp.pddl"
# open PDDL server
./mdpsim/mdpsim --port=2322 -R 100 --time-limit=10000 "$loc/temp.pddl" & _pid="$!"
# save mdpsim pid
echo "$_pid" > server.pid
# run planner and save temporary result to planresult.txt
# to see other planning options, run the planner without any argument
# e.g. ./mini-gpt/planner
./mini-gpt/planner -v 100 -h ff localhost:2322 "$loc/temp.pddl" dom1 > "$loc/planresult.txt"
# kill mdpsim server
kill -9 "$(cat server.pid)"
# remove auxilliary files
rm server.pid
rm -r logs
# parse the plan result
# see the plan in {savepath}/plan.txt
python parse_plan.py -opts "$1"
cat "$loc/plan.txt" >> "$loc/objects.txt"
rm "$loc/planresult.txt"
rm "$loc/plan.txt"
mv "$loc/objects.txt" "$loc/plan.txt"
