# HW1: PM2.5 Prediction
## Data Description
本次作業使用豐原站的觀測記錄，分成 train set 跟 test set，train set 是豐原站每個月的前 20 天所有資料。test set 則是從豐原站剩下的資料中取樣出來。

* train.csv: 每個月前 20 天的完整資料。

* test.csv : 從剩下的資料當中取樣出連續的 10 小時為一筆，前九小時的所有觀測數據當作 feature，第十小時的 PM2.5 當作 answer。一共取出 240 筆不重複的 test data，請根據 feature 預測這 240 筆的 PM2.5。

Data 含有 18 項觀測數據 AMB_TEMP, CH4, CO, NHMC, NO, NO2, NOx, O3, PM10, PM2.5, RAINFALL, RH, SO2, THC, WD_HR, WIND_DIREC, WIND_SPEED, WS_HR

## 作業範例
https://colab.research.google.com/drive/1okOuo9B0f_7J3nonbNc017DPzriOnI_k?usp=sharing
