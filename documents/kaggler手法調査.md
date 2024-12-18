
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
■データチェック
- データの読み込み　
- describe
- head
- nullチェック
- ターゲットの値チェック　何があるか確認
- ターゲットの値の割合　円グラフで確認

■ワードクラウド生成
- 英語をストップワードとして指定
- Review内の文章を半角文字列を使って連結
- ワードクラウド生成
- 確かに単語を見るとそれっぽいことが書かれている印象

■各文章整形
- 各単語をすべて小文字化
- htmlタグを削除
- 絵文字を文字に変換　笑顔絵文字を「笑顔」に変換する等
- テキストから単語と数字を抽出
- ストップワードを削除
- 正規表現で単語を削除（@～、数字、記号、URL、1文字等）
- 
- 

■特徴量エンジニアリング
- 1,2：Negative、3：Neutral、4,5：Postivie
- 新規特徴量で割合チェック
- ratingと新規特徴量をワンホットエンコーディングする
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



##  メモ
- ストップワード：前置詞等単語そのものに意味を持たない単語のこと　日本語では「が、は、で」とか
- ワードクラウド：文章内の単語を視覚化する。頻出する単語は大きく表示される
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