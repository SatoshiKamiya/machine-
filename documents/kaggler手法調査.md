
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
- 
- 
- 
- 



##  ポイント
- groupbyとapplyによってgroupbyしたグループ内をまとめて処理できる
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
