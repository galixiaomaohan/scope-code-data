#!/bin/bash
# run37_wait.sh -- gen37 self-sustaining launcher (decoupled from the terminal after setsid): execs the full chain directly; the GPU2 gate wait lives inside the chain.
#   usage: setsid nohup bash run37_wait.sh >/dev/null 2>&1 &
set -u
R="${SCOPE_RESEARCH_ROOT:-.}"
G37=$R/scratch/ccp/gen37
mkdir -p $G37/logs
echo "[$(date '+%F %T')] WAIT_DRIVER start -> chain all" >> $G37/logs/wait.log
exec bash $G37/run37_chain.sh all >> $G37/logs/chain_driver.log 2>&1
