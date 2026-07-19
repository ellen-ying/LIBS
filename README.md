# LIBS 煤炭发热量预测 — Baseline 教程

**任务**: 通过激光诱导击穿光谱（LIBS）预测煤炭发热量（kcal/kg）  
**评估**: 线上 RMSE（越低越好）  
**当前版本**: V7  本地 CV-RMSE ≈ 176

---

## 项目结构

```
LIBS/
├── README.md          # 本文件
├── requirements.txt   # Python 依赖
├── config.py          # 所有超参数 & 路径（改这里即可）
├── train.py           # 一键运行入口
│
├── src/
│   ├── data.py        # 数据加载（光谱读取 + 标签解析）
│   ├── features.py    # 特征工程（谱线积分 + 统计量 + PCA）
│   ├── model.py       # 两阶段 Ridge 训练 + 推理
│   └── submit.py      # 打包 submit.zip（含方案介绍）
│
├── train_data/        # 训练集（已解压）
├── test_data/         # 测试集（已解压）
├── submit_sample/     # 提交格式样例
├── output/            # 生成文件（submit.csv / submit.zip）
│
└── _archive/          # 历史版本（参考用）
    ├── baseline_original.py
    └── train_v2.py ~ train_v7.py
```

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 训练 + 推理 + 打包
python train.py

# 提交 output/submit.zip 到平台
```

---

## 方法说明

### 核心思路: 两阶段回归

```
LIBS 光谱 (7305维, 196-813nm)
        │
        ▼  Stage 1: 光谱 → 辅助指标
   全水分 / 灰分 / 氢 / 硫  (OOF 预测)
        │
        ▼  Stage 2: [光谱特征 + 辅助指标] → 发热量
   发热量 Q (kcal/kg)
```

**为什么两阶段?**  
辅助指标（灰分↑ → 发热量↓，氢↑ → 发热量↑）与目标强相关，
先预测辅助指标相当于把物理先验注入模型。

### 特征设计

| 特征 | 维度 | 关键思路 |
|---|---|---|
| PCA 降维光谱 | ≤30 | 整体光谱形状 |
| 统计特征 | 17 | 偏度/峰度/熵反映谱线丰富程度 |
| 谱线积分 | 11×2 | C(247.86nm)/H(656.3nm)/灰分元素 |
| 物理比值 | 4 | 可燃/灰分、H/C 等 |

### 防过拟合策略

1. **正则化**: Ridge，alpha ∈ {1, 10, 50, 100, 500, 1000, 5000, 10000}（去掉 0.01/0.1）
2. **GroupKFold CV**: 以批次为单位分组，防止同批次光谱泄露
3. **均值收缩**: 批次数 ≤ 10 时，预测值向煤种均值收缩
4. **OOF**: Stage2 输入的辅助指标来自 Out-of-Fold 预测

---

## 调参建议

修改 `config.py` 中的参数:

```python
# 增大 N_PCA_MAX 可捕获更多光谱变化（过多会过拟合）
N_PCA_MAX = 30

# 扩大 ALPHAS 范围（加入更大值）可加强正则化
ALPHAS = [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]

# 调整收缩阈值
SMALL_BATCH_THRESHOLD = 10
```

---

## 进一步优化方向

- [ ] LightGBM 替换 Stage2 Ridge（捕获非线性）
- [ ] 批次内光谱异常检测（去除污染光谱）
- [ ] Boltzmann 图估算等离子体温度（更深物理特征）
- [ ] 多版本集成（Ensemble V6 + V7 预测）

---

## Baseline performance for documentation

| Model | 赵固一矿豫焦末煤 | 赵固二矿中煤矿 | 中马矿中煤矿 | 九里山矿中煤矿 | 煤场混煤 | Global CV-RMSE |
|---|---|---|---|---|---|---|
| Baseline | 159.50 ± 68.94 | raw=82.45  shrunk=121.81  w=1.00 | raw=223.69  shrunk=255.83  w=0.65 | 148.67 ± 57.60 | 157.62 ± 41.31 | 168.69 |
| Spectra data with baseline correction | 182.09 ± 100.87 | raw=96.24  shrunk=123.32  w=1.00 | raw=245.83  shrunk=312.03  w=0.80 | 130.10 ± 38.25 | 186.46 ± 45.91 | 186.80 | 
