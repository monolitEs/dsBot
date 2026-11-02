#!/usr/bin/env bash
set -Eeuo pipefail

install -d -m 0700 -o 10001 -g 10001 /data

nfq_pid=
bot_pid=
firewalls=()
cleanup() {
    trap - EXIT TERM INT
    if [[ -n "$bot_pid" ]]; then kill -TERM -- "-$bot_pid" 2>/dev/null || true; fi
    if [[ -n "$nfq_pid" ]]; then kill -TERM "$nfq_pid" 2>/dev/null || true; fi
    for firewall in "${firewalls[@]}"; do
        "$firewall" -w -t mangle -D OUTPUT -j DSBOT_ZAPRET 2>/dev/null || true
        "$firewall" -w -t mangle -F DSBOT_ZAPRET 2>/dev/null || true
        "$firewall" -w -t mangle -X DSBOT_ZAPRET 2>/dev/null || true
    done
    if [[ -n "$bot_pid" ]]; then wait "$bot_pid" 2>/dev/null || true; fi
    if [[ -n "$nfq_pid" ]]; then wait "$nfq_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 0' TERM INT

if [[ $# -eq 0 ]]; then echo 'No bot command supplied' >&2; exit 2; fi
mapfile -t nfq_options < /etc/zapret/nfqws.conf
nfqws "${nfq_options[@]}" &
nfq_pid=$!
queue_ready=false
for attempt in {1..50}; do
    if ! kill -0 "$nfq_pid" 2>/dev/null; then
        echo 'nfqws exited before its queue was ready' >&2
        exit 1
    fi
    if [[ -r /proc/net/netfilter/nfnetlink_queue ]] && \
        awk '$1 == 200 { ready=1 } END { exit !ready }' /proc/net/netfilter/nfnetlink_queue; then
        queue_ready=true
        break
    fi
    sleep 0.1
done
if [[ "$queue_ready" != true ]]; then echo 'NFQUEUE 200 did not become ready' >&2; exit 1; fi

# Same port selection as the supplied ALT13, including its local GameFilter.
for firewall in iptables ip6tables; do
    "$firewall" -w -t mangle -N DSBOT_ZAPRET
    firewalls+=("$firewall")
    "$firewall" -w -t mangle -A DSBOT_ZAPRET \
        -m mark --mark 0x40000000/0x40000000 -j RETURN
    "$firewall" -w -t mangle -A DSBOT_ZAPRET -p tcp -m multiport \
        --dports 80,443,2053,2083,2087,2096,8443,13000:14000 \
        -j NFQUEUE --queue-num 200
    "$firewall" -w -t mangle -A DSBOT_ZAPRET -p udp -m multiport \
        --dports 443,19294:19344,50000:50100,12 -j NFQUEUE --queue-num 200
    "$firewall" -w -t mangle -A OUTPUT -j DSBOT_ZAPRET
done

echo 'ALT13 nfqws and IPv4/IPv6 rules ready; starting bot as uid 10001'
setsid setpriv --reuid=10001 --regid=10001 --clear-groups \
    --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs "$@" &
bot_pid=$!
status=0
wait -n "$nfq_pid" "$bot_pid" || status=$?
echo "A supervised process exited (status $status); stopping container" >&2
if [[ "$status" -eq 0 ]]; then status=1; fi
exit "$status"
