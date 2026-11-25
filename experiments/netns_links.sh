#!/usr/bin/env bash
# Kernel-shaped links for the live harness (run with sudo on Linux, WSL2 included). One network namespace per party,
# each joined to the host by a veth pair; the host side carries the party's downlink shaping, the namespace side its
# uplink shaping (tc netem: delay, jitter, loss; tbf-like rate through netem's rate option).
#   sudo bash experiments/netns_links.sh up <parties> <down_mbit> <up_mbit> <delay_ms> [loss_pct] [jitter_ms]
#   sudo bash experiments/netns_links.sh down <parties>
# Then: sudo -E FEDPROOF_ROUTER_BIND=0.0.0.0 python3 experiments/run.py E6 --netns
set -euo pipefail
cmd=$1; n=$2
if [ "$cmd" = up ]; then
  down=$3; up=$4; delay=$5; loss=${6:-0}; jit=${7:-0}
  for i in $(seq 0 $((n-1))); do
    ns=fpp$i; h=vh$i; p=vp$i
    ip netns add $ns 2>/dev/null || true
    if ! ip netns exec $ns ip link show $p >/dev/null 2>&1; then   # already set up by an earlier call: only reshape
      ip link del $h 2>/dev/null || true
      ip link add $h type veth peer name $p
      ip link set $p netns $ns
    fi
    ip addr add 10.201.$i.1/24 dev $h 2>/dev/null || true
    ip link set $h up
    ip netns exec $ns ip addr add 10.201.$i.2/24 dev $p 2>/dev/null || true
    ip netns exec $ns ip link set $p up
    ip netns exec $ns ip link set lo up
    tc qdisc replace dev $h root netem delay ${delay}ms ${jit}ms loss ${loss}% rate ${down}mbit limit 100000
    ip netns exec $ns tc qdisc replace dev $p root netem delay ${delay}ms ${jit}ms loss ${loss}% rate ${up}mbit limit 100000
  done
  echo "links up for $n parties: down ${down} Mbit/s, up ${up} Mbit/s, ${delay} ms one way, loss ${loss}%"
else
  for i in $(seq 0 $((n-1))); do ip netns del fpp$i 2>/dev/null || true; ip link del vh$i 2>/dev/null || true; done
  echo "links removed"
fi
