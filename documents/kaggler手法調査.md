
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

● def generate_features(base_df):　←　☆１
- 特徴量生成のコア関数
- calc_change_rate_base関数利用
- calc_volatility_base関数利用
- calc_moving_average_rate_base関数利用
- calc_target_shift2関数利用
- add_columns_per_code 上記の関数をまとめて実行
- add_columns_per_day 関数実行
- add_column_names変数を定義　追加したカラム名をリスト化
- 

● def select_features(feature_df, add_column_names, is_train):　←　☆２
- 引数1：feature_df - 特徴量生成後に得られたbase_df（DataFrame型）
- 引数2：add_column_names - 特徴量生成後に得られた新規生成カラムラベルのリスト
- 引数3：is_train - トレーニングモードか否か（デフォルトTrue）
- 基本項目のラベルとして'RowId', 'Date', 'SecuritiesCode'を定義　←ラベルリスト★1
- 引数2のadd_column_namesをソート　アルファベット順又は数値順になる　←　ラベルリスト★2
- カテゴリ系の特徴量ラベルとして'NewMarketSegment', '33SectorCode', '17SectorCode'を定義　←　ラベルリスト★3
- 目的変数ラベルとして'Target'を定義　←　ラベルリスト★4
- feat_cols 特徴量として★2+★3のラベルリストを定義★5
- feature_df内にカラムラベルを指定した項目を選択し絞込　←　★1 + ★5 + ★4 
- feature_df内のカテゴリ系★3をcategory型へ変換（カテゴリデータを数値で管理　← enumのようなもの）
- 分岐処理　トレーニングモードの場合：欠損値NANのあるレコードを削除
- 分岐処理　推論モードの場合：欠損値補完
- 
● def preprocessor(base_df, is_train=True):
- 引数1：base_df - 特徴量生成後に得られたbase_df（DataFrame型）
- 引数2：is_train - トレーニングモードか否か（デフォルトTrue）
- 特徴量生成 generate_features（☆１）関数の実行
- 特徴量選択 select_features（☆２）関数の実行


下準備はここまで＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿


⬜︎学習
● def add_rank(df, col_name="pred"):
- 引数1：df - 特徴量生成後に得られたbase_df（DataFrame型）
- 引数2：col_name="pred" - トレーニングモードか否か（デフォルトTrue）
- Dateでgroupbyでまとめてcol_nameでランク付けする　その結果をRankカラムへ格納する
-Rankカラムの値をint型へ変更する
- 
- 

● calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
- 引数1：df: pd.DataFrame　推論結果のデータ
- 引数2：portfolio_size: int = 200　ポートフォリオサイズ（順位を上位または下位200位に設定するからか？）
- 引数3：toprank_weight_ratio: float = 2) -> float　重み2~1の奴
●_calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):　←　内部関数
- 引数1：df: pd.DataFrame　推論結果のデータ
- 引数2：portfolio_size: int = 200　ポートフォリオサイズ（順位を上位または下位200位に設定するからか？）
- 引数3：toprank_weight_ratio: float = 2) -> float　重み2~1の奴
- 引数は外部関数と同じ
- assert文でrankカラム内の最小値が0、最大値が199であることを確認
- weightsの定義　等差数列の作成　2~1の間で200個の要素のリスト生成
- purchase変数 Rankカラムでソートして、taegetカラムを選択し上位200までを抽出しweightsを
  かけてすべて足す（アダマール積）さらにその値をweightsの平均で割る
- purchase半数 変数 
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
