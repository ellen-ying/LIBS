"""
提交文件打包模块

生成符合赛题要求的 submit.zip，内部结构:
    submit/
    ├── submit.csv     预测结果
    └── README.md      方案介绍（每次提交自动附上）
"""

import os
import zipfile
import pandas as pd
from datetime import datetime

from config import SUBMIT_TEMPLATE, OUTPUT_DIR, SUBMISSION_VERSION, SUBMISSION_AUTHOR


# ── README 模板 ───────────────────────────────────────────────────────────────

def build_readme(cv_results: dict, global_cv_rmse: float) -> str:
    """
    生成方案介绍 Markdown，自动填入当次的 CV-RMSE 结果。

    参数:
        cv_results     : {煤种名: cv_rmse}
        global_cv_rmse : 加权平均 CV-RMSE
    """
    table_rows = "\n".join(
        f"| {ct} | {rmse:.2f} |"
        for ct, rmse in cv_results.items()
    )

    return f"""# LIBS 煤炭发热量预测 — 方案介绍

**版本**: {SUBMISSION_VERSION}
**作者**: {SUBMISSION_AUTHOR}
**提交时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 方法概述

基于 LIBS（激光诱导击穿光谱）光谱数据，预测煤炭发热量（kcal/kg）。

### 两阶段回归框架

```
LIBS 光谱 (7305维)
      │
      ▼
  Stage 1: Ridge 回归 × 4
      │  预测辅助指标: 全水分 / 灰分 / 氢 / 硫
      │  (OOF 预测，防止数据泄露)
      ▼
  Stage 2: Ridge 回归
      │  输入: [光谱特征] + [预测辅助指标]
      ▼
  发热量 Q (kcal/kg)
```

### 特征工程

| 特征类型 | 维度 | 说明 |
|---|---|---|
| PCA 降维光谱 | 最多30维 | 归一化强度 → StandardScaler → PCA |
| 统计特征 | 17维 | 均值/方差/偏度/峰度/熵/分位数/导数统计 |
| 谱线积分 (绝对) | 11维 | C/H/O/N/Ca/Ca2/Mg/Al/Si/Fe/Na |
| 谱线积分 (相对) | 11维 | 谱线积分 / 总强度 |
| 物理比值 | 4维 | 可燃/灰分、H/灰分、C/灰分、H/C |

### 正则化策略

- Ridge 正则化候选值: [1, 10, 50, 100, 500, 1000, 5000, 10000]
  （去掉 0.01/0.1，避免小批次煤种过拟合）
- 批次数 ≤ 10 的煤种: 预测值向煤种均值收缩（OOF 搜索最优权重）
- CV 策略: 按批次分组，LOOCV（≤10批次）或 GroupKFold-5（>10批次）

---

## 本次提交 CV-RMSE

| 煤种 | CV-RMSE |
|---|---|
{table_rows}
| **全局** | **{global_cv_rmse:.2f}** |

---

## 改进方向

- [ ] 更丰富的物理谱线特征（温度估算、Boltzmann 图）
- [ ] 非线性模型（LightGBM / XGBoost）替换 Stage2 Ridge
- [ ] 多模型集成（Stacking）
- [ ] 更细粒度的批次内光谱异常检测
"""


# ── 打包函数 ──────────────────────────────────────────────────────────────────

def pack_submission(all_preds: dict, cv_results: dict, global_cv_rmse: float):
    """
    生成 submit.csv 和 submit.zip（含 README.md）。

    参数:
        all_preds      : {批次名: 预测发热量}
        cv_results     : {煤种名: cv_rmse}
        global_cv_rmse : 全局 CV-RMSE
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 读取提交模板（保证名称顺序正确）
    template = pd.read_csv(SUBMIT_TEMPLATE, encoding='utf-8')
    template['预测发热量_MJ_KG'] = template['名称'].map(all_preds)

    missing = template[template['预测发热量_MJ_KG'].isna()]['名称'].tolist()
    if missing:
        print(f"  警告: {len(missing)} 个批次缺少预测，用均值填充: {missing}")
        fallback = sum(all_preds.values()) / len(all_preds)
        template['预测发热量_MJ_KG'] = template['预测发热量_MJ_KG'].fillna(fallback)

    # ── 写 CSV
    csv_path = os.path.join(OUTPUT_DIR, "submit.csv")
    template.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n  ✓ {csv_path}")
    print(template.to_string(index=False))

    # ── 写 README
    readme_content = build_readme(cv_results, global_cv_rmse)
    readme_path    = os.path.join(OUTPUT_DIR, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    # ── 打包 ZIP
    zip_path = os.path.join(OUTPUT_DIR, "submit.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path,     arcname="submit/submit.csv")
        zf.write(readme_path,  arcname="submit/README.md")

    print(f"  ✓ {zip_path}")
    print(f"     内含: submit/submit.csv  +  submit/README.md")
    return zip_path
