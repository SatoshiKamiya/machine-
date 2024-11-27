# Kaggler手法研究


##  Kaggler
Nazim Cherpanov  
https://www.kaggle.com/nazimcherpanov


## 関連サイト
- https://www.kaggle.com/code/nazimcherpanov/0-96298-loan-approval-prediction#8.-Final-Model-Training

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
- 箱ひげ図による外れ値を除去


特徴量の作成（特徴量エンジニアリング）  
■比率　
 - income_to_age：収入/年齢　person_income/person_age【相関係数： 0.1】
 - rate_to_loan：ローン利率/ローン金額　←　借入の条件やリスクを間接的に反映（重要らしい）　loan_int_rate/loan_amnt【相関係数：0.11】
 - loan_to_employment：ローン金額/勤続年数　loan_amnt/person_emp_length【相関係数：0.1】
 - rate_to_credit_history： ローン利率/信用履歴の期間　loan_int_rate/cb_person_cred_hist_length【相関係数：0.0】
 - age_to_credit_history：年齢/信用履歴の期間　person_age/cb_person_cred_hist_length【相関係数： 0.87】　←　相関がめちゃ高い(箱ひげ図も形が似ている)
 - income_to_loan：収入/ローン金額　person_income/loan_amnt【相関係数： 0.3】
 - loan_to_income：ローン金額/年収　loan_amnt/person_income【相関係数： 0.3】
 - rate_to_age：ローン利率/年齢　loan_int_rate/person_age【相関係数：0.0】
 - 

■非線形変換
 - age_squared：person_age　2乗
 - log_income：person_income　対数
 - age_cubed：person_age　3乗
 - log_loan_amnt：loan_amnt　対数
 - 
 - 

■積
- 
- 
- 
- 

■三角関数
- age_sin：年齢を周期表現している　100で割ると値が-1~1までに収まる
- age_cos：上記のcos版
- 
- 

■その他
●グループ化
- rate_to_grade：ローンの各グレード（A～G）の内でローン利率の平均（各グレードをgroupy）
- normalized_loan_amount：ローンを必要とする理由内でローン金額のZスコア（理由をgroupy）

●high and low
- high_loan_to_income：元カラム「loan_percent_income」が0.5以上　← 0 or 1の値をとる
- is_new_credit_user：信用履歴の期間が1年以内　← 0 or 1の値をとる
- high_interest_rate：ローン利率のうち平均値以上の値　← 0 or 1の値をとる

●その他条件分岐
- intent_home_match：ローンを必要とする理由が「HOMEIMPROVEMENT（改築）」且つ持ち家　← 0 or 1の値をとる
- creditworthiness_score：信用スコアをあわらし「 所得/ローン金額×金利×信用履歴」
- high_risk_flag：ローン収入の割合が0.4以上且つ、ローン率が平均より高い且つ、ローン返済失敗あり　← 0 or 1の値をとる
- stability_score：勤続年数×年収/ローン金額×信用履歴の期間

●分位数
- income_home_mismatch：person_incomeで分位数0.8以上且つ賃貸　← 0 or 1の値をとる
- high_loan_amount：ローン金額で分位数0.75以上　← 0 or 1の値をとる
- age_income_mismatch：30歳未満且つ年収が分位数0.9以上　← 0 or 1の値をとる
- 
- 
-
##  ポイント
- target = 'loan_status'　ターゲットカラムを変数に格納している（good）
- ワンホットで相関がみられない場合、ラベルエンコーディングで相関も見てみる
- ターゲットと比較して相関は低かったが、異なる特徴量同士で相関が高い場合新たな新たな特徴量として生成してもよい
- 特徴量エンジニアリングで、新たな特徴量を生成するため「比率」「非線形変換」・・・・等あり
- 
- 

##  特徴量エンジニアリング
■比率
- 相関係数が近い者同士で組み合わせる
- 箱ひげ図で似たような形の者同士で組み合わせる
- 経験則で組み合わせる（業界の常識）
- 分子と分母を入れ替えて2つの特徴量を作ることは基本的に意味はない
- 入れ替えても解釈的に意味をなせば、やる価値はあり（迷ったらやるのも手か？）
- 
- 
- 
■非線形変換（対数、平方根、2乗など）
- 見分け方：散布図や残差プロットなどで可視化　
- 見分け方：特徴量（X軸）とターゲット（Y軸）の散布図（ペアプロット）を描く
- 見分け方：カイ二乗検定 ←カテゴリカルデータ（object型のデータで今回でいうperson_home_ownershipを指す）
- 全体の単位のオーダーを合わせるという意味でも利用する　大きい値はLOG、小さい値はべき乗
- 
- 
- 
- 

##  ROC曲線
- モデルの精度を確認するための手法としてROC曲線を利用
- 縦軸：TPR、横軸：FPRとおく
- TPR：予測が正しいと判断＆実際に正しい（TP）/実際に正しい数
- FPR：予測が正しいと判断＆実際は間違え（FP）/実際に間違っている数
- AUC（Area Under the Curve）ROC曲線内の面積（1日回ほど正確）
- AUC値（0.6〜0.7）: 要改善
- AUC値（0.8〜0.9）: 非常に良いモデル。信頼性が高い。　
- AUC値（0.9〜0.1）: 優れたモデル ←オーバーフィッティングの可能性もあり

##  モデル：XGBoost
- XGBClassifierの引数に値を入れていく
- eval_metric：テストデータの評価指標を指定　logloss=負の対数尤度　mse=二乗平均平方誤差  
  loglossの場合、2値分類問題や確率に基づく予測を評価する際に使用される
- tree_method：使用するツリーメソッド　	auto/exact/approx/hist/gpu_hist  
  GPU使ってなっかたら「gpu_hist」はやめたほうがよい
- 

##  モデル出力結果
特徴量全部のせ  
------------LightGBM  
Cross-validation scores: [0.94181065 0.95372336 0.95210018 0.95497332 0.95018653]  
Mean cross-validation score: 0.9505588108007071  
------------XGBoost  
Cross-validation scores: [0.93430151 0.94590789 0.94607957 0.948851   0.93913414]  
Mean cross-validation score: 0.9428548212254286  
- 


##  課題
- オリジナルデータの実態が分からん
- target = 'loan_status'　ターゲットカラムを変数に格納している
- データクリーン以降処理がなくても変わらない！？（特徴量増やしても変わらない・・・・・）


----------------------------------------------------------------------------------------------------------  
  
# Kaggler手法研究

##  Kaggler
Zongao Bian  
https://www.kaggle.com/zongaobian


## 関連サイト
- https://www.kaggle.com/code/zongaobian/loan-approval-tutorial-on-xgboost

##  予測精度
.

##  モデル
- XGBoost

##  処理の流れ
- 欠損値チェック
- 'person_emp_length'の欠損値を中央値で埋める
- 'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'をラベルエンコーディング
- X = トレーニングデータからidとローンステータスを抜く
- y = トレーニングデータのローンステータスのみ
- X_test = テストデータのidを抜く
- train_test_splitでモデル学習用データを生成
- XGBClassifierでハイパーパラメータ設定
- モデル学習
- AUC平均0.92ほど
- グリッドサーチによるハイパーパラメータ調整
- 

##  ポイント
- 


##  特徴量エンジニアリング
- 

##  ROC曲線
- 

##  モデル出力結果
- 

##  課題
- 

----------------------------------------------------------------------------------------------------------  
  
# Kaggler手法研究

##  Kaggler
Zongao Bian  
https://www.kaggle.com/zongaobian


## 関連サイト
- https://www.kaggle.com/code/zongaobian/loan-approval-tutorial-on-xgboost

##  予測精度
.

##  モデル
- XGBoost

##  処理の流れ
- 欠損値チェック
- 'person_emp_length'の欠損値を中央値で埋める
- 'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'をラベルエンコーディング
- X = トレーニングデータからidとローンステータスを抜く
- y = トレーニングデータのローンステータスのみ
- X_test = テストデータのidを抜く
- train_test_splitでモデル学習用データを生成
- XGBClassifierでハイパーパラメータ設定
- モデル学習
- AUC平均0.92ほど
- グリッドサーチによるハイパーパラメータ調整
- 
- 
- 
- 
- 
- 
- 
- 


##  ポイント
- 
- 
- 
- 
- 
- 
- 
- 


##  特徴量エンジニアリング
- 
- 
- 
- 
- 
- 
- 
- 



##  ROC曲線
- 
- 
- 
- 
- 
- 
- 
- 

##  モデル：XGBoost
- 
- 
- 
- 
- 
- 
- 
- 


##  モデル出力結果
- 
- 
- 
- 
- 
- 
- 
- 



##  課題
- 
- 
- 
- 
- 
- 
- 
- 
