# Cutout Lab · 伪透明背景抠图

> A local research prototype for converting AI-generated fake checkerboard transparency into RGBA PNGs.

许多生成图片看似拥有透明背景，实际上只是 RGB 图片中画出了棋盘格。Cutout Lab 在本机运行两阶段模型：先检测伪透明背景，再预测连续 alpha 通道并导出真正透明的 PNG。

> **研究原型 / Research prototype.** 此项目用于研究和演示，不应被视为对任意图片均可靠的生产级抠图系统。

## 功能

- 本地 Streamlit 界面：上传、预览、下载，无图片上传至云端。
- **Auto**：Stage A 判定伪透明背景后，自动进入 Stage B。
- **强制抠图**：跳过分类器，直接预测连续 alpha。
- **保留原图**：不修改现有 alpha。
- 在下载前显示 alpha 与中性棋盘格合成预览。
- Apple Silicon 上自动使用 PyTorch MPS；其他环境回退 CPU。

## 模型流程

```text
image → Stage A pseudo-transparency classifier
      ├─ ordinary → preserve original alpha
      └─ fake     → Stage B continuous alpha matting → RGBA PNG
```

Stage B 输出连续 alpha（0–255），而非二值遮罩。模型仅恢复 alpha；最终 RGBA 使用输入 RGB。

## 当前结果

严格隔离的合成 test split（756 对）上，Stage B alpha-only 模型的结果为：

| Metric | Result |
| --- | ---: |
| Full alpha MAE | 0.01170 |
| Boundary MAE | 0.04145 |
| Soft-alpha MAE | 0.02444 |

详细结果见 `matting_reports/test_metrics.json`（本仓库默认不包含生成的评估资产）。

## 快速开始

需要 Python 3.13+。建议使用虚拟环境：

```bash
git clone https://github.com/wwjjames/fake-alpha-cutout.git
cd fake-alpha-cutout
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
streamlit run app.py
```

首次运行下载并校验两个公开模型权重；之后可离线使用。权重由 GitHub Release 提供，而不是存入 Git 历史。

## V1 限制

- Stage A 对未见过的伪透明风格不稳定；Auto 模式必须保留手动覆盖。
- Stage B 的训练背景主要是规则棋盘格。对不连续格子、长条插入、局部色偏、波纹/扭转的伪透明背景，可能无法可靠识别或恢复 alpha。
- 主体内部的暗色或高对比纹理可能被错误预测为透明。
- 真实网格、金属线、细小孔洞会有漏检/误检和厚度偏差。
- 半透明物的 RGB 仍含原始背景污染；本版本不会恢复真实前景颜色。
- 输入会等比例缩放至 Stage B 的 512×512 推理画布，再还原 alpha 至原图尺寸；超高分辨率细节有限。

## V2 方向

1. 扩展训练背景：不规则分区、长方形插入、局部颜色变化、波纹和透视/扭转棋盘格。
2. 训练 Stage A V2，使伪透明风格覆盖与 Stage B 一致，并引入不确定性处理。
3. 引入 RGB 去污染/前景颜色恢复分支，改善玻璃、薄纱等半透明区域。
4. 增加高分辨率或分块推理，并持续评估真实无标签伪透明图。

## 发布内容与数据政策

- **公开**：代码、MIT 许可证、模型权重（通过 GitHub Release）。
- **不公开**：训练前景、合成数据集、原始测试资产及其派生评估图片。
- 公开 Release 应包含 `stage_a_gridnet_baseline.pth` 和 `stage_b_alpha_v1_best.pt`，版本号应为 `v0.1.0`。

## License

Code and released model weights are available under the [MIT License](LICENSE). Training and evaluation data are excluded from this repository.
