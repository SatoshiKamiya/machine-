import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from csv_handler import CsvHandler


def main():
    print("main処理開始")
    csvHandler = CsvHandler("data/train.csv")
    header = csvHandler.checkHead()
    column = csvHandler.checkColumn()
    print(f"header: {header}")
    print(f"column: {column}")


if __name__ == "__main__":
    main()
