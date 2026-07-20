"""
特征工程模块

LIBS 光谱特征分三层:
  1. 统计特征   — 均值/方差/偏度/峰度/熵/分位数/导数
  2. 谱线特征   — 关键元素谱线积分（绝对值 & 相对归一化）
  3. 物理比值   — 可燃元素/灰分比，H/C 比等

三层拼接后进行 PCA 降维 + 标准化，作为 Ridge 模型的输入。
"""

import numpy as np
from scipy.stats import skew, kurtosis, entropy as scipy_entropy
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import KEY_LINES, N_PCA_MAX, RANDOM_STATE


# ── 谱线积分 ──────────────────────────────────────────────────────────────────

def line_integral(wl, inten, center_nm, halfwin_nm):
    """
    对 [center-halfwin, center+halfwin] 窗口内的强度求和。
    用于提取特定元素的发射谱线强度。
    """
    mask = (wl >= center_nm - halfwin_nm) & (wl <= center_nm + halfwin_nm)
    return float(inten[mask].sum()) if mask.sum() > 0 else 0.0

# ── Asymmetric Least Squares Smoothing ───────────────────────────────────────

def baseline_als(y, lam=1e6, p=0.005, niter=10):
    """
    Memory-optimized Asymmetric Least Squares baseline correction.
    
    Parameters:
    -----------
    y: 1D numpy array
        Raw spectrum array
    lam: float
        Smoothness parameter (try 1e5 to 1e8)
    p: float
        Asymmetry parameter (try 0.001 to 0.01)
    niter: int
        Number of iterations for the solver
        
    Returns:
    --------
    z: 1D numpy array
        Fitted baseline
    """
    L = len(y)
    
    # Build the 2nd-order difference matrix natively in sparse format
    # Diagonals: [1, -2, 1], Offsets: [0, 1, 2]. Shape: (L-2, L)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L-2, L))
    H = lam * D.T.dot(D)
    w = np.ones(L)
    
    for i in range(niter):
        W = sparse.diags(w, 0, shape=(L, L))
        Z = (W + H).tocsc()
        z = spsolve(Z, w * y)
        # Update weights asymmetrically
        w = p * (y > z) + (1 - p) * (y < z)
        
    return z

# ── 单条光谱特征 ──────────────────────────────────────────────────────────────

def spectrum_features(wl, inten, baseline_correction=False, als_param=(1e6, 0.005)):
    """
    从一条光谱提取三层特征。

    返回:
        inorm  (D,)   : 强度归一化（用于 PCA）
        stats  (17,)  : 统计特征
        labs   (11,)  : 谱线绝对积分
        lrel   (11,)  : 谱线相对积分
        rats   (4,)   : 物理比值
    """
    if baseline_correction:
        baseline = baseline_als(inten, lam=als_param[0], p=als_param[1])    # Baseline correction
        inten = inten - baseline
    total      = inten.sum() + 1e-8
    inten_norm = inten / total          # 归一化强度（消除激光能量波动）
    #inten_norm = (inten - inten.mean()) / (inten.std() + 1e-8)  # SNV normalization
    deriv      = np.diff(inten_norm)    # 一阶差分
    deriv2     = np.diff(inten_norm, 2) # 二阶差分

    # 统计特征: 描述整体光谱形状
    stats = np.array([
        inten.mean(), inten.std(), inten.max(), np.log1p(total),
        inten_norm.mean(), inten_norm.std(),
        float(skew(inten_norm)),
        float(kurtosis(inten_norm)),
        float(scipy_entropy(inten_norm + 1e-12)),
        np.percentile(inten, 10), np.percentile(inten, 25),
        np.percentile(inten, 75), np.percentile(inten, 90),
        deriv.std(),  float(np.abs(deriv).mean()),
        deriv2.std(), float(np.abs(deriv2).mean()),
    ], dtype=np.float32)

    # 谱线特征: 物理意义最强的部分
    line_names = list(KEY_LINES.keys())
    labs = np.array(
        [line_integral(wl, inten, c, h) for _, (c, h) in KEY_LINES.items()],
        dtype=np.float32
    )
    lrel = labs / total   # 相对强度（消除光谱总能量差异）

    # 物理比值: 可燃元素 vs 灰分元素
    # 索引: C=0, H=1, O=2, N=3 | Ca=4, Ca2=5, Mg=6, Al=7, Si=8, Fe=9, Na=10
    comb_sum = labs[[0, 1, 2, 3]].sum()                        # 可燃元素
    ash_sum  = labs[[4, 5, 6, 7, 8, 9, 10]].sum() + 1e-8       # 灰分元素
    #acid_sum = labs[[7, 8]].sum() + 1e-8         # Acidic elements: Si, Al
    #base_sum = labs[[4, 5, 6, 9, 10]].sum()     # Basic elements: Ca, Ca2, Mg, Fe, Na
    rats = np.array([
        comb_sum / ash_sum,                 # 可燃/灰分 (发热量代理)
        labs[1] / ash_sum,                  # H/灰分
        labs[0] / ash_sum,                  # C/灰分
        labs[1] / (labs[0] + 1e-8),        # H/C
        #labs[2] / (labs[0] + 1e-8),        # O/C Ratio
        #base_sum / acid_sum,               # Basic to Acid Ratio
        #(labs[1] + labs[2]) / (labs[0] + 1e-8)  # Volatility Proxy (H + O) / C
    ], dtype=np.float32)

    return inten_norm, stats, labs, lrel, rats


# ── Anomaly detection ────────────────────────────────────────────────────────────────

def detect_spectral_anomalies(inorm_mat, batch_ids, similarity_threshold=0.92, sigma_cutoff=2.5):
    """
    Computes Cosine Similarity of each shot relative to its batch mean.
    
    Parameters:
    -----------
    inorm_mat : np.ndarray (N_shots, N_pixels)
        Normalized spectral matrix from compute_features()
    batch_ids : np.ndarray or list (N_shots,)
        Batch/group IDs corresponding to each shot
    similarity_threshold : float
        Absolute minimum cosine similarity cutoff (e.g., 0.90 to 0.95)
    sigma_cutoff : float
        Z-score threshold for within-batch outlier rejection
        
    Returns:
    --------
    valid_mask : np.ndarray (N_shots,) of bool
        True for clean shots, False for anomalies
    similarities : np.ndarray (N_shots,) of float
        Cosine similarity scores for each shot
    """
    # L2-normalize vectors for fast dot-product cosine similarity
    norms = np.linalg.norm(inorm_mat, axis=1, keepdims=True) + 1e-8
    unit_mat = inorm_mat / norms
    
    similarities = np.zeros(len(inorm_mat), dtype=np.float32)
    valid_mask = np.ones(len(inorm_mat), dtype=bool)
    
    # Compute similarity relative to EACH batch's unique mean
    unique_batches = np.unique(batch_ids)
    for b_id in unique_batches:
        b_mask = (batch_ids == b_id)
        b_shots = unit_mat[b_mask]
        
        # Calculate mean spectrum vector for this batch
        b_mean = b_shots.mean(axis=0, keepdims=True)
        b_mean /= (np.linalg.norm(b_mean) + 1e-8)
        
        # Cosine similarity = dot product of unit vectors
        b_sims = (b_shots * b_mean).sum(axis=1)
        similarities[b_mask] = b_sims
        
        # Method A: Absolute threshold
        mask_abs = b_sims >= similarity_threshold
        
        # Method B: Relative Z-score threshold within the batch
        if len(b_sims) > 3:
            mean_sim, std_sim = b_sims.mean(), b_sims.std() + 1e-8
            mask_z = (b_sims >= mean_sim - sigma_cutoff * std_sim)
        else:
            mask_z = True
            
        valid_mask[b_mask] = mask_abs & mask_z
        
    return valid_mask, similarities

# ── 批量特征计算 ──────────────────────────────────────────────────────────────

def compute_features(data, fit = True, baseline_correction=False, als_param=(1e6, 0.005)):
    """
    对 data_dict 中所有光谱计算特征，原地填充 stats/labs/lrel/rats 字段。
    同时返回归一化光谱矩阵（用于 PCA）。
    """
    raw_spectra = data['spectra']
    inorms, stats_list, labs_list, lrel_list, rats_list = [], [], [], [], []

    for wl, inten in raw_spectra:
        inorm, stats, labs, lrel, rats = spectrum_features(wl, inten, 
                                                           baseline_correction=baseline_correction,
                                                           als_param=als_param)
        inorms.append(inorm)
        stats_list.append(stats)
        labs_list.append(labs)
        lrel_list.append(lrel)
        rats_list.append(rats)

    # 裁剪到最短公共波长（不同仪器/文件行数可能略有差异）
    min_len = min(len(s) for s in inorms)
    inorm_mat = np.array([s[:min_len] for s in inorms], dtype=np.float32)

    if fit:
        # Filter out invalid spectral shots
        valid_mask, _ = detect_spectral_anomalies(inorm_mat, data['groups'])
        inorm_mat = inorm_mat[valid_mask]

        # Match the rest of the data with the spectral data
        data['stats'] = np.array(stats_list)[valid_mask]
        data['labs']  = np.array(labs_list)[valid_mask]
        data['lrel']  = np.array(lrel_list)[valid_mask]
        data['rats']  = np.array(rats_list)[valid_mask]
        data["targets"] = data["targets"][valid_mask]
        data["aux"] = data["aux"][valid_mask]
        data["groups"] = data["groups"][valid_mask]
    else:
        data['stats'] = np.array(stats_list)
        data['labs']  = np.array(labs_list)
        data['lrel']  = np.array(lrel_list)
        data['rats']  = np.array(rats_list)

    return inorm_mat



# ── PCA + 拼接 ────────────────────────────────────────────────────────────────

def build_feature_matrix(data, n_batches,
                         scaler_spec=None, pca=None, scaler_hand=None, fit=True,
                         baseline_correction=False, als_param=(1e6, 0.005)):
    """
    构建最终特征矩阵: [PCA(归一化光谱)] + [手工特征]。

    fit=True  : 用训练数据拟合 scaler/PCA，返回 (X, scaler_spec, pca, scaler_hand)
    fit=False : 用已拟合的 scaler/PCA 变换测试数据，返回 X

    参数:
        n_batches: 批次数，用于限制 PCA 维度（不能超过样本数-1）
        als_param: ALS 基线校正参数
    """
    inorm_mat = compute_features(data, fit=fit,
                                 baseline_correction=baseline_correction, 
                                 als_param=als_param)

    if fit:
        scaler_spec = StandardScaler()
        spec_scaled = scaler_spec.fit_transform(inorm_mat)
        n_pca = min(N_PCA_MAX, n_batches - 1, inorm_mat.shape[0] - 1)
        pca = PCA(n_components=n_pca, random_state=RANDOM_STATE)
        spec_pca = pca.fit_transform(spec_scaled)
    else:
        spec_pca = pca.transform(scaler_spec.transform(inorm_mat))

    # 拼接手工特征
    hand_feats = np.hstack([data['stats'], data['labs'], data['lrel'], data['rats']])
    X = np.hstack([spec_pca, hand_feats])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if fit:
        scaler_hand = StandardScaler()
        X = scaler_hand.fit_transform(X)
        return X, scaler_spec, pca, scaler_hand
    else:
        return scaler_hand.transform(X)
