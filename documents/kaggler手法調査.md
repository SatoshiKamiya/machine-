
----------------------------------------------------------------------------------------------------------  
  
# Kaggler手法研究

##  Kaggler
Abdallah Wagih Ibrahim

## 関連サイト
NLP - Transformers
https://www.kaggle.com/code/abdallahwagih/nlp-transformers

##  予測精度


##  モデル
- transformers

##  処理の流れ
■MultiHeadAttentionクラスの定義
■Encoder blockクラスの定義
■Encoderクラスの定義
■Decoder blockクラスの定義
■Decoderクラスの定義
■Transformerクラスの定義

~Fine-tuning~
■データ読込
- データの読み込み　
- データチェック　Id、Review、Ratingを確認
- ターゲットの分布を調べる　棒グラフ（x軸：rate　y軸：数）で1~5の評価でどのくらい数があるか表示
■前処理
- Hugging Face データセット オブジェクトと互換性を持たせるための処理
- Ratingの値を変更　全体に-1する
- train_test_splitメソッドを利用してX_train, X_valid, _, _を取得
- 一旦セーブ　new_train.csv、new_valid.csv
■トレーニング前準備
- load_datasetでnew_train.csv、new_valid.csvを呼ぶ
- トークナイザーの指定　モデルを指定すれば、AutoTokenizerが自動でトークナイザーを指定してくれる
- モデル　事前トレーニング済みの distilbert-base-uncased モデルを使用
- GPUにコネクト
- モデルをダウンロード
- 評価基準（精度チェック）MAEを利用
■トレーニング
- ハイパーパラメータの調整
- transformersのTrainerを使ってトレーニングする
- 検証データセットを利用して予測
- 
■評価
- 混同行列をプロット
- 
- 


##  ポイント
- トークンナイザーとは文章中の単語をトークン化（一定の規則に基づいた数字インデックスに変換）
- 
- 
- 

----------------------------------------------------------------------------------------------------------  
  
# Kaggler手法研究

##  Kaggler
Ivan Shingel

## 関連サイト
Sentiment Analysis for Company Reviews
https://www.kaggle.com/code/ivanshingel/sentiment-analysis-for-company-reviews

##  予測精度


##  モデル
- LogisticRegression

##  処理の流れ


■データ読込
- データの読み込み　
- データチェック　Id、Review、Ratingを確認
- ターゲットの分布を調べる　棒グラフ（x軸：rate　y軸：数）で1~5の評価でどのくらい数があるか表示
■前処理
- Hugging Face データセット オブジェクトと互換性を持たせるための処理
- Ratingの値を変更　全体に-1する
- train_test_splitメソッドを利用してX_train, X_valid, _, _を取得
- 一旦セーブ　new_train.csv、new_valid.csv
■トレーニング前準備
- load_datasetでnew_train.csv、new_valid.csvを呼ぶ
- トークナイザーの指定　モデルを指定すれば、AutoTokenizerが自動でトークナイザーを指定してくれる
- モデル　事前トレーニング済みの distilbert-base-uncased モデルを使用
- GPUにコネクト
- モデルをダウンロード
- 評価基準（精度チェック）MAEを利用
■トレーニング
- ハイパーパラメータの調整
- transformersのTrainerを使ってトレーニングする
- 検証データセットを利用して予測
- 
■評価
- 混同行列をプロット
- 
- 


##  ポイント
- 