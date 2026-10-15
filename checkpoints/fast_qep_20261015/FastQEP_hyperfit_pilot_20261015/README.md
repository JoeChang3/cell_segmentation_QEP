# Fast-QEP 超参数学习：原始首块图像诊断

这是独立的新目录，不要覆盖上一版 FastQEP_R_pipeline_20261015。
上一版的 53 个冻结文件保留原样；新的根目录 run.py 是本次入口。

## 目前做到哪里

已接入正确的边际似然梯度、用于梯度的固定 Stein 控制变量、原始矩 Kronecker
计算及 m/s/nu 的 ECM 候选更新。已经验证 q=2 退回解析 GP，以及 q=1.5 的独立积分。

下一阶段先问：在重新拟合的 GP 参数附近，QEP 是否还有超出 Monte Carlo 噪声的
参数调整方向？这不是整图分割实验，也还不是完整 QEP 超参数优化。

## 运行环境

使用上次成功的 gpytorch_arm。只需 Python >=3.10、numpy、scipy、Pillow、
threadpoolctl。本阶段不调用 R；后续端到端分割仍必须使用真实 R 下游。
不会自动安装或升级包。

## 第一步：仅检查（现在就可以做）

```bash
conda activate gpytorch_arm
cd ~/Downloads/FastQEP_hyperfit_pilot_20261015
python run.py --check
```

检查包括小规模采样、梯度积分和归档 R 首块重建一致性；不拟合真实图像。
完成后会打印一个 *_review.zip 的绝对路径，可以直接上传复核。

## 第二步：首块诊断（新实验，确认方案后执行）

```bash
caffeinate -i python run.py --pilot
```

只使用 nuclei 和 whole-cell 两张 development 图像的原始第一块。
先以两个固定初值重新拟合解析 GP；再在原 R 参数与重拟合 GP 参数两处，
分别运行 q=1.5 的四条链，估计参数梯度、采样误差和两种重建的差异。

不读取标注进行优化、不计算 AP、不运行 watershed、不优化 q。
ECM 更新只记录为候选，不把一次 noisy 更新冒充已经收敛的 QEP 拟合。
每个采样目标最多 900 秒，GP 拟合每数据集最多 300 秒，总试验上限 7200 秒。
这是预算上限，不是保证能收敛的耗时估计。失败会停止并保留诊断。

输出：payload/results/fast_qep_hyperfit_pilot_20261015/。
完成或失败后，入口都会打包 *_review.zip 并打印路径。请把该包上传。
不要删除失败结果重跑，也不要选择最好的一条链。

检查和试验输出目录分开。已有目录会阻止覆盖；不自动续跑失败目标。
详见 payload/experiments/real_data/fast_qep_hyperfit_20261015/PROTOCOL.md。

## 结果如何解释

- 梯度方向明确：支持进入有误差控制的 QEP 拟合，不等于分割会改善。
- 梯度无法与采样误差区分：当前点处证据不足，不等于证明所有 QEP 最优解都无效。
- 数值检查失败：属于计算问题，不能当作模型的负结果。
- GP 改善但 QEP 没有额外收益：不能宣称 QEP 特异性优势。

代码不修改你的 Cell_Seg_QEP checkout，也不覆盖前面的结果。
