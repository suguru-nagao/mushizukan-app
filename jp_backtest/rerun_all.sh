#!/bin/bash
# 再開できる再計算。各段階が終わると rerun_done/<名前> を作り、あるものは飛ばす。
cd /home/user/mushizukan-app/jp_backtest
step() { n=$1; shift; [ -f rerun_done/$n ] && return; echo "[$n] start $(date +%H:%M)"; "$@" && touch rerun_done/$n; echo "[$n] end $(date +%H:%M)"; }
bg() { n=$1; shift; ( step $n "$@" ) & }
bg ev_buy bash -c "python3 build_events_w.py > rerun_ev_buy.log 2>&1"
step ev_sell bash -c "SELL=1 python3 build_events_w.py > rerun_ev_sell.log 2>&1"
wait
bg pool2 bash -c "python3 pool_compare_run.py > rerun_pool2.log 2>&1"
step pool3 bash -c "python3 stock_pool3.py > rerun_pool3.log 2>&1"
wait
bg sell bash -c "SELL=1 python3 sell_check.py > rerun_sell.log 2>&1"
bg rank_buy bash -c "python3 weekly_rank_build.py > rerun_rank_buy.log 2>&1"
step rank_sell bash -c "SELL=1 python3 weekly_rank_build.py > rerun_rank_sell.log 2>&1"
wait
step long_zoneW python3 long_run.py zone:W > rerun_long2.log 2>&1
step long_M python3 long_run.py candle:M chart:M tech:M zone:M > rerun_long3.log 2>&1
step pool4 bash -c "python3 stock_pool4.py > rerun_pool4.log 2>&1"
echo ALLDONE
