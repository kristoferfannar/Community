#!/bin/bash
# Call with:
# `nohup ./runner.sh &
# to run in detached mode
#
# This allows you to let the script continue execution
# after you've disconnected from the SSH session
#
# cat results.log >>old.results.log
# echo -n "" >results.log
#
echo "Starting at $(date)" >results.log
python3 -u teams/team_2/training/train.py >>results.log
echo "Finished  at $(date)" >>results.log
