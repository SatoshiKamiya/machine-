import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.imputation.mice import MICEData
from sklearn.ensemble import RandomForestRegressor
from enum_collection import Rounding
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from constant import Constant
from sklearn.preprocessing import StandardScaler


# csvファイル処理クラス
class CsvHandler:
    # 色配列
    base_color = "blue"

    # コンストラクタ
    def __init__(self, path):
        print("CsvHandler __init__ path:", path)
        # 表示制限解除
        pd.set_option("display.max_rows", None)
        # pd.set_option("display.width", None)

        # csvファイル保存処理
        self.path = path

        if os.path.exists(path):
            # オリジナルデータ変数
            self._original_csv_data = pd.read_csv(path)
            # 変更に対応するデータ変数
            self._csv_data = self._original_csv_data
        else:
            print(f"ファイルが見つかりません: {path}")

    # データ取得
    # インスタンス
    def get_data_instance(self):
        return self._csv_data

    def get_original_data_instance(self):
        return self._original_csv_data

    # 1つのカラム取得
    def choose_column_data_instance(self, column_name):
        return self._original_csv_data[column_name]

    # 複数カラム取得
    def choose_columns_data_instance(self, column_names):
        return self._original_csv_data[column_names]

    # データチェック
    # データ取得
    def get_record(self, record_count):
        print("CsvHandler get_record")
        result = self._csv_data.head(record_count)
        print(result)

    # データ取得（カラム指定）
    def get_specification_record(self, labels, display_num=10):
        print("CsvHandler get_record")
        result = self._csv_data[labels].head(display_num)
        print(result)

    # データチェック
    def get_table_info(self):
        print("CsvHandler get_table_info")
        result = self._csv_data.info
        print(result)

    # カラム表示
    def get_column_label(self):
        print("CsvHandler get_column_label")
        result = self._csv_data.columns
        print(result)

    # 行列数
    def get_matrix_num(self):
        print("CsvHandler get_matrix_num")
        result = self._csv_data.shape
        print(result)

    # カラム値と利用回数を取得
    def get_value_and_count(self, columns):
        print("CsvHandler get_value_and_count")
        for column in self._csv_data[columns]:
            print(f"Column: {column}")
            result = self._csv_data[column].value_counts()

            print("CsvHandler get_value_and_count result")
            print("\n")
            print(result)
            print("\n")

    # 総データ数取得チェック
    def get_records_count(self):
        result = self._csv_data.shape[0]
        print("CsvHandler get_records_count:", result)
        return result

    # 各カラムの欠損値数チェック
    def get_data_isnull(self):
        print("CsvHandler get_data_describe")
        result = self._csv_data.isnull().sum()
        print(result)

    # 各カラムの型チェック
    def get_column_type(self):
        print("CsvHandler get_column_type")
        result = pd.DataFrame(self._csv_data)
        print(result.dtypes)

    # path取得
    def get_Path(self):
        return self.path

    # データ初期化
    def reset_data(self):
        self._csv_data = self._original_csv_data

    # カラムを指定して欠損値のある行を指定
    def get_drop_records_number(self, column):
        result = self._csv_data[self._csv_data[column].isnull()].index
        print("CsvHandler get_drop_records_number result:", result)
        return result

    # 行数を指定してそのレコードを取得
    def get_assignment_records(self, column, record_numbers):
        result = self._csv_data[column].loc[record_numbers]
        print("CsvHandler get_assignment_records result:", result)

    # 値変換
    # ラベルエンコーディング（文字列→数値変換）
    def change_text_to_int(self, column_name):
        print("CsvHandler change_text_to_int")
        # カラム内の値をリスト化
        unique_values = self._csv_data[column_name].unique().tolist()
        print("change_text_to_int unique_values=", unique_values)
        self._csv_data[column_name] = self._csv_data[column_name].map(
            lambda x: unique_values.index(x) if x in unique_values else x
        )

    # 標準化（平均0、分散1にスケーリング） ←これ使っておけば問題なさそう
    def change_standardize(self, column_names):
        print("CsvHandler change_standardize")
        df = self._csv_data
        df[column_names] = StandardScaler().fit_transform(df[column_names])
        # 結果確認
        print(df[column_names].head())

    # 正規化（最小値0、最大値1スケーリング）

    # 欠損値補間
    # 行削除
    def drop_records(self, records_array):
        print("CsvHandler drop_records")
        self._csv_data.drop(records_array, axis=0, inplace=True)

    # 行削除 範囲
    def drop_records_area(self, start, end):
        print("CsvHandler drop_records")
        self._csv_data.drop(self._csv_data.index[start:end], axis=0, inplace=True)

    # 列削除
    def drop_columns(self, columns_array):
        print("CsvHandler drop_columns")
        self._csv_data.drop(columns_array, axis=1, inplace=True)

    # 平均値取得（カラム指定）
    def average_value(self, column_name):
        print("CsvHandler average_value")
        result = self._csv_data[column_name].mean()
        print(result)

    # 平均値補間（カラム指定）
    def average_value_interpolation(self, column_name):
        print("CsvHandler average_value_interpolation")
        result = self._csv_data[column_name].fillna(self._csv_data[column_name].mean())
        print(result)

    # 中央値取得（カラム指定）
    def median_value(self, column_name):
        print("CsvHandler median_value")
        result = self._csv_data[column_name].median()
        print(result)

    # 中央値補間（カラム指定）
    def median_value_interpolation(self, column_name):
        print("CsvHandler median_value_interpolation")
        result = self._csv_data[column_name].fillna(
            self._csv_data[column_name].median()
        )
        print(result)

    # 最頻値取得（カラム指定）
    def mode_value(self, column_name):
        print("CsvHandler mode_value")
        result = self._csv_data[column_name].mode()[0]
        print(result)

    # 最頻値補間（カラム指定）
    def mode_value_interpolation(self, column_name):
        print("CsvHandler mode_value_interpolation")
        result = self._csv_data[column_name].fillna(
            self._csv_data[column_name].mode()[0]
        )
        print(result)

    #    #小数点切り捨て（カラム指定）　←　平均と補完と組み合わせられないので要調査
    #     def decimal_point_truncation(self, column_name):
    #         print("CsvHandler decimal_point_truncation")
    #         result = self._csv_data[column_name].fillna(0).astype('int64')
    #         # result = np.floor(self._csv_data[column_name])
    #         print(result)

    # 端数処理（カラム指定）
    def rounding_process(self, column_name, rounding_type, decimal_point_position):
        print("CsvHandler rounding_process")
        # 四捨五入処理
        if rounding_type == Rounding.ROUNDINGUP:
            print("CsvHandler rounding_process ROUNDINGUP　四捨五入")
            if 0 < decimal_point_position:
                self._csv_data[column_name] = self._csv_data[column_name].apply(
                    lambda x: round(x, decimal_point_position)
                )
            else:
                self._csv_data[column_name] = (
                    self._csv_data[column_name].round().astype(int)
                )
        # 丸め処理
        elif rounding_type == Rounding.TRUNCATE:
            print("CsvHandler rounding_process TRUNCATE 少数切り捨て")
            if 0 < decimal_point_position:
                digit_adjustment_num = 10**decimal_point_position
                self._csv_data[column_name] = self._csv_data[column_name].apply(
                    lambda x: np.floor(x * digit_adjustment_num) / digit_adjustment_num
                )
            else:
                self._csv_data[column_name] = self._csv_data[column_name].astype(int)
        result = self._csv_data[column_name]
        print(result)

    # 相関係数
    # 1対1相関係数表示
    def correlation_coefficient_oto(self, column_name):
        corr_matrix = self._csv_data.select_dtypes(include="number").corr()
        result = corr_matrix[column_name]
        print(result)

    # 1対多相関係数表示
    def correlation_coefficient_otm(self):
        result = self._csv_data.select_dtypes(include="number").corr()
        print(result)
        # グラフ表示
        plt.figure(figsize=(8, 6))
        sns.heatmap(result, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.show()

    # 多重代入法
    # MICE
    def multiple_imputation_MICE(self, column_names):
        csv_data_selected = self._csv_data[column_names]

        mice_data = MICEData(csv_data_selected)
        for i in range(10):
            mice_data.update_all()
        result = mice_data.data
        print(result)

    # ランダムフォレスト
    def random_forest(self, column_array):
        print("CsvHandler ModelProcesrandom_forest column_array:", column_array)
        # age_df = self.data_instance.get_record(10)

        # 推定に使用する項目を指定
        age_df = self._csv_data[column_array]

        # ラベル特徴量をワンホットエンコーディング
        age_df = pd.get_dummies(age_df)

        # 学習データとテストデータに分離し、numpyに変換
        known_age = age_df[age_df.Age.notnull()].values
        unknown_age = age_df[age_df.Age.isnull()].values

        # 学習データをX, yに分離
        X = known_age[:, 1:]
        y = known_age[:, 0]

        # ランダムフォレストで推定モデルを構築
        rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
        rfr.fit(X, y)

        # 推定モデルを使って、テストデータのAgeを予測し、補完
        predictedAges = rfr.predict(unknown_age[:, 1::])
        self._csv_data.loc[(self._csv_data.Age.isnull()), "Age"] = predictedAges
        self.get_specification_record("Age", 890)

        # 年齢別生存曲線と死亡曲線
        # facet = sns.FacetGrid(self._csv_data[0:890], hue="Survived",aspect=2)
        # facet.map(sns.kdeplot,'Age',shade= True)
        # facet.set(xlim=(0, self._csv_data.loc[0:890,'Age'].max()))
        # facet.add_legend()
        # plt.show()

    # 勾配ブースティング
    # point_column:補完対象カラム
    # column_array:補完対象外で利用するカラム
    def gradient_boosting(self, point_column, column_array):
        print("CsvHandler gradient_boosting")
        model = HistGradientBoostingRegressor()
        df = self._csv_data
        # 欠損していないデータを抽出（学習データ）
        train_data = df[df[point_column].notnull()]
        # print("gradient_boosting train_data=", train_data)

        # 欠損データを抽出（テストデータ）
        test_data = df[df[point_column].isnull()]
        # print("gradient_boosting test_data=", test_data)
        # 特徴量とターゲット変数を指定（ここでは'Age'がターゲット）
        features = column_array
        X_train = train_data[features]
        # print("gradient_boosting X_train=", X_train)

        y_train = train_data[point_column]
        # print("gradient_boosting y_train=", y_train)

        # モデルの学習
        model.fit(X_train, y_train)

        # 欠損値のあるデータ（test_data）から特徴量を取得
        X_test = test_data[features]

        # 欠損しているAgeを予測
        predicted_ages = model.predict(X_test)

        # 予測した値で欠損値を補完
        df.loc[df[point_column].isnull(), point_column] = predicted_ages

        # 結果を確認
        # print(
        #     "gradient_boosting 欠損値なし確認（0ならOK）",
        #     df[point_column].isnull().sum(),
        # )  # 欠損値が0になっているか確認

    # 外れ値チェック

    # 可視化
    # ヒストグラム
    def show_all_column_hist(self):
        self._csv_data.hist(figsize=(10, 10), bins=30)
        plt.show()

    # ヒストグラム（指定）
    def show_part_column_hist(self, columns):
        self._csv_data[columns].hist(figsize=(10, 10), bins=30)
        plt.show()

    # ヒストグラム（カラム＋カラム内データ指定）
    def show_part_column_recrods_hist(self, column_name, records):
        # ヒストグラムの重ね描き
        plt.figure(figsize=(10, 10))
        # Aカラムのヒストグラム
        plt.hist(
            self._csv_data[column_name],
            bins=30,
            alpha=0.5,
            label="base",
            color=Constant.BASE_COLOR,
        )
        plt.hist(
            self._csv_data[column_name].loc[records],
            bins=30,
            alpha=0.5,
            label=column_name,
            color="orange",
        )
        plt.legend()
        # グラフを表示
        plt.show()

    # 異なるモデルを比較するヒストグラム
    def show_part_column_recrods_hist(
        self, column_name, models_results01, models_results02, records
    ):
        # ヒストグラムの重ね描き
        plt.figure(figsize=(10, 10))
        # 基準のヒストグラム
        plt.hist(
            self._csv_data[column_name],
            bins=30,
            alpha=0.5,
            label="base",
            color=Constant.BASE_COLOR,
        )

        plt.hist(
            models_results01[column_name].loc[records],
            bins=30,
            alpha=0.5,
            label=column_name,
            color=Constant.COLORS[0],
        )

        plt.hist(
            models_results02[column_name].loc[records],
            bins=30,
            alpha=0.5,
            label=column_name,
            color=Constant.COLORS[1],
        )

        # for index, model in models_results:
        #     plt.hist(
        #         model[column_name].loc[records],
        #         bins=30,
        #         alpha=0.5,
        #         label=column_name,
        #         color=Constant.COLORS[index],
        #     )
        plt.legend()
        # グラフを表示
        plt.show()

    # カーネル密度推定（KDE）
    def show_kds(self):
        sns.kdeplot(self._csv_data, shade=True)
        plt.show()

    # ペアプロット）
    def show_pair_plot(self):
        sns.pairplot(self._csv_data, diag_kind="kde")
        plt.show()

    # ボックスプロット（箱ひげ図）
    def show_block_plot(self, column_names):
        df = self._csv_data
        plt.figure(figsize=(8, 6))
        plt.boxplot([df[col] for col in column_names], labels=column_names)
        plt.title("Box Plot")
        plt.ylabel("Values")
        plt.show()
