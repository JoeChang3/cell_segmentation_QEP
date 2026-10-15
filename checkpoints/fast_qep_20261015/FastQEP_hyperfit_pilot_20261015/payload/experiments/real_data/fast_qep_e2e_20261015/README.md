# Fast-QEP：只替换重建器的原版 R pipeline 对比

本包的状态是**执行准备完成**，不是已获得新的整图分割结果。
已有两个固定真实 tile 的采样诊断通过；接下来用两张开发图共 25 个原始 tile
比较 Fast-GP 与 Fast-QEP q=1.5。全部方案见 `payload/experiments/real_data/fast_qep_e2e_20261015/PROTOCOL.md`。

## 在 MacBook 上执行

1. 下载并解压 `FastQEP_R_pipeline_20261015.zip` 到 Downloads。
2. 打开之前成功运行项目的 Python 环境，在终端执行：

```bash
cd ~/Downloads/FastQEP_R_pipeline_20261015
caffeinate -i python run.py --run
```

这条命令先做环境和原 R 基线预检，全部通过才开始整图采样。
不需要把文件复制进原 Cell_Seg_QEP 仓库，也不需要覆盖旧文件。
原 R 参考仓库默认仍位于：

`/Users/zchan/eclipse-workspace/cell_segmentation_original`

如果路径不同，在命令后加 `--reference /实际路径/cell_segmentation_original`。
如需只跑预检，使用 `python run.py --preflight`。通过后用同一 `--run` 命令继续，
会核验并复用预检结果。`caffeinate` 只是让 Mac 在运行期间保持唤醒。

## 需要什么

- Python >=3.10；NumPy、SciPy、pandas、Pillow、threadpoolctl。
- R 4.5.1；EBImage 4.52.0、RobustGaSP 0.6.8、magick 2.9.1、pracma 2.4.6。
- 原 R 参考仓库的已验证 commit 与源码。运行包没有复制该仓库。
- 至少 8 GiB 空闲磁盘；顺序运行、单 BLAS 线程。

脚本会列出缺失项，不会自动安装依赖或改系统环境。若旧 R 用户库
`~/Library/R/arm64/4.5/library` 存在，且没有显式设置 `R_LIBS_USER`，
只会在本次 R 子进程中使用它。R 包底层 C/C++ 仍由真实 R 包执行。

云端两块 tile 耗时外推约 80 分钟，加上预检和文件输出；Mac 耗时可能不同。
每块采样上限 15 分钟，累计采样上限 2 小时。达到上限或诊断失败就停止，
不会自动增加样本、换种子或忽略失败。

## 会比较什么

- 主比较：Fast-GP q=2 与 Fast-QEP q=1.5，二者均接原 RobustGaSP → EBImage → cleanup。
- GP 平滑强度诊断对照：已有 proposal 规则确定的 Gaussian mean；不调标签/AP。
- 四条链各自均值的分割：检查微小重建变化是否引起不稳定的阈值跳动，不挑最好一条。
- Raw/Fast-GP 历史基线：在新采样前必须复现。

仅两张开发图，不构成泛化证据。AP 使用项目冻结的 TP/(TP+FP+FN) 定义。
任何结果都不能自动证明 QEP 独有优势；同时看 AP75、merge/split、旧正确实例损失
以及 Gaussian 对照和 Monte Carlo 稳定性。

## 输出和报错

结果目录：`payload/results/fast_qep_e2e_20261015/`

- `run.log`：完整终端输出。报错时请上传这个文件和 `last_stop.json`，不用截图。
- `environment.json`：Python/R 版本和环境问题。
- `preflight/R_anchor_checks.json`：R 基线核验。
- `tiles/`：逐 tile 诊断、样本块均值和完整检查点。
- 完成后：`REPORT.md`、`metrics.csv`、`comparisons.json`、`MC_stability.json`。

如在 tile 之间中断，再运行同一命令会验证并跳过已通过的 tile。
如果某个 tile 在运行中被中断或失败，脚本会保留现场并拒绝自动重跑该 tile。
请把日志发回分析，不要删除失败记录或自行改阈值。`Ctrl+C` 可停止运行。
这里不提供自动续接半条 MCMC 链，避免混入不可审计的样本。

技术说明：运行包使用源文件 SHA-256 固定代码和数据；原指标函数保持原文，
通过 AST 加载以避免无关 Torch/skimage 依赖。四个历史分割结果用于核验，
绝不会作为新 Fast-QEP 结果。`--local-checks` 是不执行 R 的开发者验证入口，
不能替代 Mac 上的真实 R 预检。
