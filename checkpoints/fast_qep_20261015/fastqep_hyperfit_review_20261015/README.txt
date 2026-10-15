Fast-QEP 超参数学习：第一阶段核查包（2026-10-15）

已完成：针对当前有限 tile、协方差匹配的 qED 先验 + 独立高斯噪声模型，
推导边际似然梯度、实现 Kronecker 梯度，并通过 15 项数值检查。
q 固定为 1.5 或 2；没有开始新的真实图像实验，没有新的 AP 结果。

本压缩包包含：
- FastQEP_hyperparameter_feasibility_20261015.pdf：推导、结果、公平比较与下一步条件。
- verify_hyperfit.py：可独立运行的数学核验程序，不是上一版真实图像 runner 的替代品。
- validation.json：本次验证的结果、容差、环境与代码 SHA-256。
- source_provenance.json：原 R 源码、冻结审计、用户结果包的来源记录。
- requirements.txt：验证程序所需库；不要求你更改已跑通的研究环境。
- make_report.py：PDF 生成脚本，另需 reportlab 与 DejaVu 字体。
- SHA256SUMS.txt：交付文件校验值。

可选复核（通常现有 gpytorch_arm 环境已经满足依赖）：
  conda activate gpytorch_arm
  cd 解压后的 fastqep_hyperfit_review_20261015 目录
  python verify_hyperfit.py --output validation_local.json

此命令不读取细胞图像、不调用 R、不运行分割，通常几秒即可完成。
无需覆盖已有 validation.json，也不要用此包覆盖原来的 FastQEP_R_pipeline 目录。

本阶段结论：数学上可行，可以进入受预算限制的实现试验；尚未证明比 GP 更好。
必须加入重新拟合的 Fast-GP，以及 GP 使用 QEP 所学参数的对照，避免把优化器
改进误判成 QEP 优势。真实图像上的边际似然优化、采样梯度稳定性与运行成本尚待检验。

下一步所需：把你 Mac 上的 FastQEP_R_pipeline_20261015.zip 上传到当前对话。
当前审查结果包不含原 sampler 源码，不能据此声称已经接入或修改了运行程序。
恢复源码后，先核对采样器与数值保护，再定稿参数范围和更新准入条件。
原始首块图像上的试验及整图分割尚未启动。
