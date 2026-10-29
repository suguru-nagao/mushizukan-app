#!/bin/bash
# 長期データの清掃(clean_long.py)のあと、長期の結果と銘柄別シグナルの検証をすべて作り直す
cd /home/user/mushizukan-app/jp_backtest
echo "[1] long_run" ; python3 long_run.py candle:W chart:W tech:W zone:W candle:M chart:M tech:M zone:M > rerun_long.log 2>&1
echo "[2] events" ; python3 build_events_w.py > rerun_ev_buy.log 2>&1 &
SELL=1 python3 build_events_w.py > rerun_ev_sell.log 2>&1
wait
echo "[3] pools" ; python3 pool_compare_run.py > rerun_pool2.log 2>&1 &
python3 stock_pool3.py > rerun_pool3.log 2>&1
wait
SELL=1 python3 sell_check.py > rerun_sell.log 2>&1 &
python3 stock_pool4.py > rerun_pool4.log 2>&1
wait
echo "[4] rank" ; python3 weekly_rank_build.py > rerun_rank_buy.log 2>&1 ; SELL=1 python3 weekly_rank_build.py > rerun_rank_sell.log 2>&1
echo ALLDONE
