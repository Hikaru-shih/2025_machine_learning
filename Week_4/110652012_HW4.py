import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample
from math import sqrt

# ---------- Config ----------
XML_PATH = "O-A0038-003.xml"
N_COLS = 67
N_ROWS = 120
LON0 = 120.00
LAT0 = 21.88
RES = 0.03
INVALID = -999.0

# ---------- Parsing ----------
def floats_from_text(text):
    nums = re.findall(r"-?\d+\.\d+|-?\d+", text)
    return [float(x) for x in nums]

def parse_xml_to_grid(path):
    txt = Path(path).read_text(encoding="utf-8")
    nums = floats_from_text(txt)
    total = N_COLS * N_ROWS
    for i in range(0, max(1, len(nums)-total+1)):
        chunk = nums[i:i+total]
        if len(chunk) == total:
            arr = np.array(chunk).reshape((N_ROWS, N_COLS))
            return arr
    raise ValueError("無法在 XML 中找到符合大小的數值陣列。")

def grid_to_points(grid):
    rows, cols = grid.shape
    lon_list, lat_list, val_list = [], [], []
    for r in range(rows):
        for c in range(cols):
            lon = LON0 + c * RES
            lat = LAT0 + r * RES
            val = grid[r, c]
            lon_list.append(lon)
            lat_list.append(lat)
            val_list.append(val)
    return pd.DataFrame({"lon": lon_list, "lat": lat_list, "value": val_list})

# ---------- Build datasets ----------
def build_datasets(xml_path=XML_PATH):
    grid = parse_xml_to_grid(xml_path)
    df = grid_to_points(grid)
    # Classification dataset
    df_clf = df.copy()
    df_clf["label"] = df_clf["value"].apply(lambda x: 0 if x == INVALID else 1)
    df_clf = df_clf[["lon", "lat", "label"]]
    # Regression dataset (drop invalid)
    df_reg = df[df["value"] != INVALID].copy()
    df_reg = df_reg[["lon", "lat", "value"]]
    return df_clf, df_reg

# ---------- Train & Evaluate ----------
def train_and_evaluate(df_clf, df_reg):
    # ----- Classification -----
    df_majority = df_clf[df_clf.label == 0]
    df_minority = df_clf[df_clf.label == 1]

    # 取兩類的最小數量來平衡
    min_size = min(len(df_majority), len(df_minority))
    df_majority_down = resample(df_majority, replace=False, n_samples=min_size, random_state=42)
    df_minority_down = resample(df_minority, replace=False, n_samples=min_size, random_state=42)
    df_balanced = pd.concat([df_majority_down, df_minority_down])

    Xc = df_balanced[["lon", "lat"]].values
    yc = df_balanced["label"].values

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc, yc, test_size=0.2, random_state=42, stratify=yc
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(Xc_train, yc_train)
    yc_pred = clf.predict(Xc_test)
    acc = accuracy_score(yc_test, yc_pred)
    prec = precision_score(yc_test, yc_pred)
    rec = recall_score(yc_test, yc_pred)
    f1 = f1_score(yc_test, yc_pred)

    print("Classification results (balanced):")
    print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # ----- Regression -----
    Xr = df_reg[["lon", "lat"]].values
    yr = df_reg["value"].values
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )
    reg = KNeighborsRegressor(n_neighbors=5)
    reg.fit(Xr_train, yr_train)
    yr_pred = reg.predict(Xr_test)
    mae = mean_absolute_error(yr_test, yr_pred)
    rmse = sqrt(mean_squared_error(yr_test, yr_pred))
    r2 = r2_score(yr_test, yr_pred)
    print("Regression results (KNN):")
    print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    return clf, reg, (Xc_test, yc_test, yc_pred), (Xr_test, yr_test, yr_pred), df_reg

# ---------- Plot ----------
def plot_results(clf_test_tuple, reg_test_tuple, df_reg):
    Xc_test, yc_test, yc_pred = clf_test_tuple
    Xr_test, yr_test, yr_pred = reg_test_tuple

    # Classification results
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title("Classification True Labels")
    plt.scatter(Xc_test[:,0], Xc_test[:,1], c=yc_test, s=8, cmap="coolwarm")
    plt.subplot(1,2,2)
    plt.title("Classification Predicted Labels")
    plt.scatter(Xc_test[:,0], Xc_test[:,1], c=yc_pred, s=8, cmap="coolwarm")
    plt.tight_layout()
    plt.savefig("classification.png", dpi=300)
    plt.close()

    # Regression scatter plot
    plt.figure(figsize=(6,5))
    plt.title("Regression: Actual vs Predicted")
    plt.scatter(yr_test, yr_pred, s=8)
    plt.plot([min(yr_test), max(yr_test)], [min(yr_test), max(yr_test)], "r--")
    plt.xlabel("actual"); plt.ylabel("predicted")
    plt.savefig("regression.png", dpi=300)
    plt.close()

    # Heatmap of valid data points
    plt.figure(figsize=(8,6))
    plt.title("Valid Temperature Data Distribution")
    sc = plt.scatter(df_reg["lon"], df_reg["lat"], c=df_reg["value"], s=10, cmap="coolwarm")
    plt.colorbar(sc, label="Temperature")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.savefig("heatmap.png", dpi=300)
    plt.close()

# ---------- Main ----------
if __name__ == "__main__":
    df_clf, df_reg = build_datasets(XML_PATH)
    clf, reg, clf_t, reg_t, df_reg = train_and_evaluate(df_clf, df_reg)
    plot_results(clf_t, reg_t, df_reg)
git pull upstream project
