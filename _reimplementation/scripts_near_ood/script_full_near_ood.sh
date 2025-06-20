#!/bin/bash
set -e

./exec_near_ood.sh
echo "Tests finished"
./script_auto_pp_near_ood.sh
echo "Eval finished"