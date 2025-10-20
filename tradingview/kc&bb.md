# 肯特納通道 (Keltner Channel)

- 主要技術指標: Keltner Channel
- Timeframe: 30 分鐘
- 兩個 Keltner Channel 指標:
  - KC_3.75: length 50, multiplier 3.75,source = close,bands style= Range, atr length= 10
  - KC_2.75: length 50, multiplier 2.75,source = close,bands style= Range, atr length= 10
- 做多進場條件:
  - 1. 價格觸碰 KC_3.75 下軌
  - 2. 實體 K 棒脫離 KC_2.75 跟 KC_3.75 下軌之間的區間
- 做空進場條件:
  - 1. 價格觸碰 KC_3.75 上軌
  - 2. 實體 K 棒脫離 KC_2.75 跟 KC_3.75 上軌之間的區間
