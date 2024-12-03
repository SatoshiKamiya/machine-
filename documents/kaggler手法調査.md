
----------------------------------------------------------------------------------------------------------  
  
# Kaggler手法研究

##  Kaggler
Competition Notebook  
https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction

## 関連サイト
【日本語】JPX LightGBM Demo  
https://www.kaggle.com/code/ikeppyo/jpx-lightgbm-demo/notebook

##  予測精度


##  モデル
- LightGBM

##  処理の流れ
⬜︎ファイル読み込み
- read_files関数を定義
- train_files、supplemental_files配下のファイル全て（モデルのトレーニングデータにあたる）を読み込む
- 各ファイルのデータは変数prices, options, financials, trades, secondary_pricesとして得られる
- 
- 

⬜︎データの結合
- merge_data関数を定義
- ファイル読み込みで行った結果得られたファイルデータを元に結合する
- 引数はprices, options, financials, trades, secondary_prices＋stock_list
- ベースはprices　base_df
- base_dfにstock_listを水平結合（left join）する　← 正確には_stock_list
- tradesとfinancialsも結合可能だが非活性になっていた
  
⬜︎adjust_priceの実装
- 引数：stock_price (型 DateFrame)
- 戻り値：stock_price内にAdjustedCloseを追加したDateFrame
- 引数とし渡されたstock_priceのDateカラムをformat="%Y-%m-%d"に変換する
（generate_adjusted_close関数の定義）
- AdjustedCloseの生成　SecuritiesCodeとDateを使ってソートする
- SecuritiesCodeに対しgroupbyしてgenerate_adjusted_close関数を実行（インデックスを削除）
- Dateをインデックスとして再定義する

⬜︎generate_adjusted_closeを定義する（adjust_price関数の中にある関数を定義する）
- 引数：df (型 DateFrame) ← groupbyした銘柄
- 戻り値：stock_price内にAdjustedCloseを追加したDateFrame（各groupbyした銘柄）
- Dateカラムを元に並び替え（古い順）
- CumulativeAdjustmentFactorカラムを新規生成し、累積積を割り当てる（株式分割対策）
- AdjustedCloseカラムにCumulativeAdjustmentFactorと終値(close)を掛け合わせた値を格納←端数処理も行う
- Dateカラムを元に並び替え（最新順）
- AdjustedClose列の値に0がある場合、np.nanを代入
- 欠損値に対してfill()関数を利用してNaNを代入
- 


⬜︎特徴量エンジニアリング
● def calc_change_rate_base(column_name, periods):
- 引数1：column_name（AdjustedCloseを渡していた）
- 引数2：periods 営業日を渡す想定 ([3,9]を渡していた)
- pct_changeで変化率を算出
- stock_prices.csv内になるAdjustedClose（新しく生成した特徴量）の変化率(priod=[3,9])を記録
- 上記の値を"{column_name}_change_rate_{period}"カラムに記録していく

● def calc_volatility_base(column_name, periods):
- 引数1：column_name（AdjustedCloseを渡していた）
- 引数2：periods 営業日を渡す想定 ([3,9]を渡していた)
- stock_prices.csv内になるAdjustedClose（新しく生成した特徴量）を対数に変換して差分(priod=[3,9])の標準偏差を算出
- 上記の値を"{column_name}_volatility_{period}"カラムに記録していく
- 
- 

● def calc_moving_average_rate_base(column_name, periods):
- 引数1：column_name（Volumeを渡していた）
- 引数2：periods 営業日を渡す想定 ([3,9]を渡していた)
- 移動平均値と現在値の比率を導出し、項目として追加する関数
- 
- 

● def calc_target_shift2(price):
- 引数：price.csvのDataframe型
- stock_prices.csv内にあるCloseの変化率を記録
- 上記の値を"Target_shift2"カラムに記録していく
- 終値の変動率を生成し、項目として追加する関数
- 

● def add_columns_per_code(price, functions):
- 引数1：price（base_df:merge_data関数で処理したDataFrame）
- 引数2：functions 配列化した各関数 (calc_change_rate_base, calc_volatility_base, calc_moving_average_rate_base, calc_target_shift2)
- 内部関数funcにより各関数を実行する関数を定義 ※1
- priceソート（"SecuritiesCode", "Date"）
- SecuritiesCodeでgroupbyしてfunc実行 ※1
- reset_indexでインデックス振り直し
- 

● def add_columns_per_day(base_df):
- 引数1：price（base_df:merge_data関数で処理したDataFrame）
- (終値 - 開始値) / 終値 を新規カラム  「diff_rate1」に格納
- (最高値 - 最低値) / 終値 を新規カラム「diff_rate2」に格納
- 

● def generate_features(base_df):
- 特徴量生成のコア関数
- calc_change_rate_base関数利用
- calc_volatility_base関数利用
- calc_moving_average_rate_base関数利用
- calc_target_shift2関数利用
- add_columns_per_code 上記の関数をまとめて実行
- add_columns_per_day 関数実行
- add_column_names変数を定義　追加したカラム名をリスト化
- 

● def select_features(feature_df, add_column_names, is_train):
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
- 
● def preprocessor(base_df, is_train=True):
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
- 
- 


##  ポイント
- groupbyとapplyによってgroupbyしたグループ内をまとめて処理できる
- とapplyはセットであり、applayに関数名をセットするだけで、その関数に引数を渡さなくてもよい
- 引数はgroupbyされたデータを自動で渡している
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
