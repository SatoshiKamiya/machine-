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
- 欠損値補完する（平均値）
- 年齢に対しヒストグラムにプロット
- person_home_ownershipをグラフで可視化してターゲットの割合を可視化
- 箱ひげ図でグラフプロット　

Data Preprocessing & Cleaning（データの前処理とクリーニング）
■DataCleanerクラスを定義　
 - fit_label_encoders：objectクラスでラベルエンコーディングするカラムを選択する処理
 - transform_labels：fit_label_encodersで選択したカラムを実際にエンコーディングする処理
 - impute_numeric：float, intクラスの欠損値補完　K近傍法で行っている

- DataCleanerクラスを利用したデータクリーニング開始
- トレーニングデータはfit_label_encoders処理をする
- トレーニングデータ＆テストデータをラベルエンコーディングする
- ラベルエンコーディング後、両データを欠損値補完する（impute_numeric）




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