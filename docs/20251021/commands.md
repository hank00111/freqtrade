.\.venv\Scripts\Activate.ps1; freqtrade backtesting --strategy RLStrategy4Action --config user_data/config_dqn_gpu.json --freqaimodel ReinforcementLearner4Action --timerange 20230101-20251019 --export trades

.\.venv\Scripts\Activate.ps1; freqtrade download-data --exchange binance --pairs BTC/USDT:USDT ETH/USDT:USDT --timeframe 5m 15m 1h 4h --timerange 20221101-20251019 --trading-mode futures --prepend

 .\.venv\Scripts\Activate.ps1;  tensorboard --logdir user_data/models/  