# Science Exhibition - 銷售預測專案

這是一個基於機器學習的銷售數量預測系統，結合了多種先進的預測模型來提高預測準確度。

## 專案概述

本專案使用 XGBoost 和 LSTM 神經網路的集成學習方法，針對產品銷售數據進行時間序列預測。系統能夠處理不同類型的產品（主要產品和冷門產品），並提供未來 7 天的銷售量預測。

## 專案結構

```
Project/
├── main.py                     # 主程式入口
├── src/                        # 核心程式碼模組
│   ├── data_preprocessing.py   # 數據預處理
│   ├── feature_engineering.py # 特徵工程
│   ├── model_training_lstm.py  # LSTM 模型訓練
│   ├── model_training_xgb.py   # XGBoost 模型訓練
│   ├── model_ensemble.py       # 模型集成
│   ├── sequence_preparation.py # 序列數據準備
│   └── visualization.py        # 視覺化模組
├── analysis_data/              # 數據分析腳本
│   ├── analysis_autocorrelation.py
│   ├── analysis_correlation.py
│   ├── analysis_feature_importance_rf.py
│   ├── evaluation_random_forest.py
│   ├── feature_engineering_lag.py
│   └── pca.py
├── models/                     # 訓練好的模型
│   ├── multi_store_lstm.keras
│   └── multi_store_xgb.pkl
└── outputs/                    # 輸出圖表和結果
    ├── lstm_learning_curve.png
    ├── lstm_learning_curve_optimized.png
    ├── mse_improvement_stages.png
    ├── quantity_distribution.png
    ├── quantity_distribution_final.png
    ├── residual_plot.png
    └── residual_plot_optimized.png
```

## 核心功能

### 1. 數據預處理 (data_preprocessing.py)
- 清理和整理銷售數據
- 將產品分類為「主要產品」和「冷門產品」（基於數據點數量）
- 處理缺失值和異常值
- 時間序列數據排序和格式化

### 2. 特徵工程 (feature_engineering.py)
建立多種預測特徵：
- **時間特徵**：年、月、週數、星期幾
- **循環編碼**：星期幾的 sin/cos 轉換
- **產品特徵**：產品 ID、產品與節假日/天氣的交互作用
- **滾動統計特徵**：14天、30天、60天的滾動均值和總和
- **其他特徵**：價格、節假日標記、週末標記、極端天氣標記

### 3. 模型訓練

#### LSTM 神經網路 (model_training_lstm.py)
- **架構**：單向 LSTM 網路
  - 第一層：96個單元，ReLU激活函數
  - 第二層：48個單元，ReLU激活函數
  - Dropout 層用於防止過擬合
  - 全連接層：32個單元 + L2正則化
- **特色**：自定義損失函數，對高銷量預測給予更大懲罰
- **訓練參數**：Adam優化器，學習率 3e-4，50個週期

#### XGBoost 模型 (model_training_xgb.py)
- 梯度提升決策樹模型
- 處理結構化特徵數據
- 快速訓練和預測

### 4. 模型集成 (model_ensemble.py)
- **動態權重計算**：基於多個性能指標
  - 均方誤差 (MSE)
  - 平均絕對誤差 (MAE)
  - R² 分數
  - 預測穩定性（預測變異度）
- **加權預測融合**：`集成預測 = (α_xgb × XGB預測 + α_lstm × LSTM預測)`
- **冷門產品處理**：使用歷史平均值進行預測

### 5. 視覺化 (visualization.py)
生成各種分析圖表：
- 學習曲線
- 殘差圖
- 數量分布圖
- MSE 改善階段圖

## 執行流程

1. **數據預處理**：清理和分類產品數據
2. **特徵工程**：創建時間、產品和統計特徵
3. **數據分割和標準化**：準備訓練、驗證和測試集
4. **XGBoost 訓練**：訓練梯度提升模型
5. **LSTM 訓練**：訓練深度學習序列模型
6. **模型集成**：結合兩個模型的預測結果
7. **視覺化**：生成性能圖表和指標

## 技術特點

- **混合模型架構**：結合傳統機器學習和深度學習的優勢
- **智能權重分配**：根據模型性能動態調整集成權重
- **產品分層處理**：針對不同數據量的產品採用不同策略
- **豐富的特徵工程**：涵蓋時間、產品、統計等多維度特徵
- **完整的評估體系**：多種性能指標和視覺化分析

## 使用方法

```bash
python main.py
```

執行主程式將自動完成整個預測流程，包括數據處理、模型訓練、集成預測和結果視覺化。

## 依賴套件

- pandas
- numpy
- scikit-learn
- xgboost
- tensorflow/keras
- matplotlib
- seaborn

## 輸出結果

- 訓練好的模型文件 (models/)
- 性能評估圖表 (outputs/)
- 預測結果和性能指標
## 授權

本專案採用MIT授權條款。
---
*此專案展示了現代機器學習在銷售預測領域的應用，通過多模型集成和深度特徵工程實現了高精度的時間序列預測。*
