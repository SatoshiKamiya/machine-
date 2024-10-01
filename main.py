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
    csvHandler.get_record(20)
    csvHandler.get_all_missing_values_count()
   


if __name__ == "__main__":
    main()
