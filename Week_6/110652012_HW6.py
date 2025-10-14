# week_6/week6_main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from math import pi, log
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from math import sqrt

# ======= 與你 HW4 一致的設定（請確認 XML 檔相對路徑） =======
XML_PATH = "O-A0038-003.xml"
N_COLS = 67
N_ROWS = 120
LON0 = 120.00
LAT0 = 21.88
RES = 0.03
INVALID = -999.0

# ---------- 從文字抽數字（沿用 HW4 想法） ----------
import re
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
    raise ValueError("Cannot find a proper array in XML.")

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

def build_datasets(xml_path=XML_PATH):
    grid = parse_xml_to_grid(xml_path)
    df = grid_to_points(grid)
    # classification: label = 0 (INVALID) / 1 (valid)
    df_clf = df.copy()
    df_clf["label"] = (df_clf["value"] != INVALID).astype(int)
    df_clf = df_clf[["lon", "lat", "label"]]
    # regression: drop invalid
    df_reg = df[df["value"] != INVALID][["lon", "lat", "value"]].copy()
    return df_clf, df_reg

class GDA:
    """
    General GDA (class-conditional Gaussian with class-specific Sigma_k).
    Features: R^d (這裡 d=2).
    """
    def fit(self, X, y):
        X0 = X[y==0]
        X1 = X[y==1]
        self.phi = X1.shape[0] / X.shape[0] 
        self.mu0 = X0.mean(axis=0)
        self.mu1 = X1.mean(axis=0)
        S0 = (X0 - self.mu0).T @ (X0 - self.mu0) / X0.shape[0]
        S1 = (X1 - self.mu1).T @ (X1 - self.mu1) / X1.shape[0]
        self.S0 = S0
        self.S1 = S1
        self.S0_inv = np.linalg.inv(S0)
        self.S1_inv = np.linalg.inv(S1)
        self.logdetS0 = np.log(np.linalg.det(S0))
        self.logdetS1 = np.log(np.linalg.det(S1))
        return self

    def _log_gauss(self, X, mu, S_inv, logdetS):
        diff = X - mu
        quad = np.einsum('bi,ij,bj->b', diff, S_inv, diff)
        d = X.shape[1]
        return -0.5 * (d*np.log(2*pi) + logdetS + quad)

    def predict_proba(self, X):
        logp1 = self._log_gauss(X, self.mu1, self.S1_inv, self.logdetS1) + np.log(self.phi + 1e-12)
        logp0 = self._log_gauss(X, self.mu0, self.S0_inv, self.logdetS0) + np.log(1 - self.phi + 1e-12)
        m = np.maximum(logp0, logp1)
        den = m + np.log(np.exp(logp0 - m) + np.exp(logp1 - m))
        p1 = np.exp(logp1 - den)
        return np.c_[1-p1, p1]

    def predict(self, X):
        proba = self.predict_proba(X)[:,1]
        return (proba >= 0.5).astype(int)

# ============= 使用你 HW4 的模型做 piecewise =============
def train_week4_models(df_clf, df_reg):
    # RF 分類（與你作法一致，但這裡不做 downsample，保持簡潔）
    Xc = df_clf[["lon","lat"]].values
    yc = df_clf["label"].values
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(Xc_tr, yc_tr)
    acc = accuracy_score(yc_te, rf.predict(Xc_te))

    # KNN 回歸（只在 valid 值上）
    Xr = df_reg[["lon","lat"]].values
    yr = df_reg["value"].values
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(Xr_tr, yr_tr)
    rmse = sqrt(mean_squared_error(yr_te, knn.predict(Xr_te)))

    return rf, knn, acc, rmse

def piecewise_h(C_model, R_model, X):
    c = C_model.predict(X)
    out = np.full((X.shape[0],), INVALID, dtype=float)
    mask = (c == 1)
    if mask.any():
        out[mask] = R_model.predict(X[mask])
    return out

# ============= 視覺化：GDA 決策邊界 & piecewise 行為 =============
def plot_decision_boundary(clf, X, y, path):
    x1_min, x1_max = X[:,0].min()-0.01, X[:,0].max()+0.01
    x2_min, x2_max = X[:,1].min()-0.01, X[:,1].max()+0.01
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 400),
                           np.linspace(x2_min, x2_max, 400))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    zz = clf.predict(grid).reshape(xx1.shape)

    plt.figure(figsize=(6,5))
    plt.contourf(xx1, xx2, zz, alpha=0.25, levels=[-0.1,0.5,1.1])
    plt.scatter(X[:,0], X[:,1], c=y, s=8, edgecolors='none')
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("GDA Decision Boundary (0=INVALID, 1=valid)")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_piecewise_sample(C_model, R_model, df_all, path):
    # 在整個網格上演示 h(x) 的行為
    X = df_all[["lon","lat"]].values
    h = piecewise_h(C_model, R_model, X)
    plt.figure(figsize=(7,6))
    sc = plt.scatter(df_all["lon"], df_all["lat"], c=h, s=6)
    plt.colorbar(sc, label="h(x)  (valid -> regression value; invalid -> -999)")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("Piecewise h(x) over the grid")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def main():
    df_clf, df_reg = build_datasets(XML_PATH)
    Xc = df_clf[["lon","lat"]].values
    yc = df_clf["label"].values
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)

    gda = GDA().fit(Xc_tr, yc_tr)
    yhat = gda.predict(Xc_te)
    gda_acc = accuracy_score(yc_te, yhat)
    print(f"[GDA] test accuracy = {gda_acc:.4f}")
    Path("figures").mkdir(exist_ok=True, parents=True)
    plot_decision_boundary(gda, Xc, yc, "figures/gda_boundary.png")

    rf, knn, rf_acc, knn_rmse = train_week4_models(df_clf, df_reg)
    print(f"[Week4 models] RF acc = {rf_acc:.4f}, KNN RMSE = {knn_rmse:.4f}")

    df_all = grid_to_points(parse_xml_to_grid(XML_PATH))
    plot_piecewise_sample(rf, knn, df_all, "figures/regression_piecewise.png")

if __name__ == "__main__":
    main()
