import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style='darkgrid', font_scale=1.6)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import os
os.environ["WANDB_DISABLED"] = "true"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import datasets
from datasets import load_dataset


def main():
    print("main処理開始")
    # 変数の定義
    model_name = "distilbert-base-uncased"  # モデルの指定
    model_name_out = f"{model_name}-finetuned-emotion"
    batch_size = 32  # バッチサイズ
    num_labels = 5  # らべる


    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sub = pd.read_csv("data/sample_submission.csv")

    print("トレーニング　行列数")
    print(train.shape)
    print("test　行列数")
    print(test.shape)
    print("sub　行列数")
    print(sub.shape)
    print("トレーニング　head")
    print(train.head(5))

    # 棒グラフ
    # plt.figure(figsize=(10,4))
    # sns.countplot(data=train, x="Rating")
    # plt.title("Target distribution")
    # plt.show()

    # Map labels
    train["Rating"] = train["Rating"] - 1 #rating=1~5 →　0~4へ変更

    # ターゲット（rating）を取り除いたトレーニングデータ（8:2）
    X_train, X_valid, _, _ = train_test_split(train, train["Rating"], test_size=0.2, shuffle=True, random_state=0)
    
    # カラム名を変更　Review → text、Rating → label
    # 変更したデータをX_train「new_train.csv」、X_valid「new_valid.csv」、test「new_test.csv」に保存する
    X_train[["Review","Rating"]].rename(columns={"Review":"text", "Rating":"label"}).to_csv("data/new_train.csv", index=False)
    X_valid[["Review","Rating"]].rename(columns={"Review":"text", "Rating":"label"}).to_csv("data/new_valid.csv", index=False)
    test[["Review"]].rename(columns={"Review":"text"}).to_csv("data/new_test.csv", index=False)

    # データセット生成 先ほど作ったcsvファイルのデータを呼び出す
    train_ds = load_dataset("csv", data_files={"train": "data/new_train.csv"})
    valid_ds = load_dataset("csv", data_files={"valid": "data/new_valid.csv"})
    test_ds = load_dataset("csv", data_files={"test": "data/new_test.csv"})

    # データセット内のデータを処理しやすいように再定義
    # textをString型、labelをラベル付け（ラベルエンコーディング）する
    train_ds = train_ds.cast(datasets.Features({"text": datasets.Value("string"), "label": datasets.ClassLabel(num_classes=5)}))
    valid_ds = valid_ds.cast(datasets.Features({"text": datasets.Value("string"), "label": datasets.ClassLabel(num_classes=5)}))

    # Print summary
    print("データセットチェック")
    print(train_ds)
    print(valid_ds)
    print(test_ds)

    print("train_ds　型チェック")
    print(type(train_ds))
    print("train_ds　head")
    print(train_ds['train'][:5])

    # トークナイザーの呼び出し（自動）
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize batch
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)
    
    

    train_encoded = train_ds.map(tokenize, batched=True, batch_size=batch_size)
    valid_encoded = valid_ds.map(tokenize, batched=True, batch_size=batch_size)
    test_encoded = test_ds.map(tokenize, batched=True, batch_size=batch_size)

    # GPUが使えたらGPUを使う
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Download model
  
    model = (AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device))
    

    logging_steps = len(train_encoded["train"]) 
    

    # Training hyper-parameters
    training_args = TrainingArguments(output_dir=model_name,
                                      num_train_epochs=3,
                                      learning_rate=2e-5,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      weight_decay=0.01,
                                      evaluation_strategy="epoch",
                                      disable_tqdm=False,
                                      logging_steps=logging_steps,
                                      push_to_hub=False, 
                                      log_level="error")

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        mae = mean_absolute_error(labels, preds)
        return {"mae": mae, "accuracy": acc, "f1": f1}
    

    trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=train_encoded["train"],
                  eval_dataset=valid_encoded["valid"],
                  tokenizer=tokenizer)
    

    # Train model
    trainer.train()

    # Predictions on validation set
    valid_preds = trainer.predict(valid_encoded["valid"])
    valid_preds = np.argmax(valid_preds.predictions, axis=1)
    
    # Ground truth labels
    y_valid = np.array(valid_ds["valid"]["label"])
    labels = train_ds["train"].features["label"].names


    # Plot confusion matrix
    def plot_confusion_matrix(y_preds, y_true, labels):
        cm = confusion_matrix(y_true, y_preds, normalize="true")
        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
        plt.title("Normalized confusion matrix")
        ax.grid(False)
        plt.show()
    
    plot_confusion_matrix(valid_preds, y_valid, labels)
    
    # Test set predictions
    preds = trainer.predict(test_encoded["test"])
    test_preds = np.argmax(preds.predictions, axis=1)




   

if __name__ == "__main__":
    main()
