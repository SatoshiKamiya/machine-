# Kaggler手法研究


##  Kaggler
Nazim Cherpanov  
https://www.kaggle.com/nazimcherpanov

##  予測精度
.96298

##  モデル
- LightGBM
- XGBoost

##  処理の流れ
- トレーニングデータとテストデータ、オリジナルデータを読み込む
- トレーニングデータとオリジナルデータを合体させる（91226 rows × 12 columns）
- ターゲットの値になにがあり、割合を表示
- nullチェック
- データ型チェック
- objectをラベルエンコーディングする
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 


##  ポイント
- target = 'loan_status'　ターゲットカラムを変数に格納している（good）
- ワンホットで相関がみられない場合、ラベルエンコーディングで相関も見てみる
- 
- 
- 
- 

##  課題
- オリジナルデータの実態が分からん
- target = 'loan_status'　ターゲットカラムを変数に格納している
- 
- 
- 
- 
- 

## 関連サイト
- https://www.kaggle.com/code/nazimcherpanov/0-96298-loan-approval-prediction#8.-Final-Model-Training