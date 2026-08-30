# GNN 与 Q-CAST 在线配对基准

- 场景数：2
- 配对 episode 数：40
- 业务质量结论：**GNN 吞吐显著更优**
- 规划耗时结论：**Q-CAST 决策显著更快**
- 可执行性硬门槛：通过

判定顺序是先比较完成请求数；只有每个配对 episode 的完成数都相同时，才使用删失完成延迟判优。规划耗时单独报告，不与业务质量混合成一个分数。

## 分场景结果

| 场景 | 样本数 | GNN 完成数 | Q-CAST 完成数 | 完成数优势及 95% CI | GNN 延迟 | Q-CAST 延迟 | 业务质量结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| waxman192 | 20 | 40.250 | 38.600 | 1.650 [0.300, 3.000] | 866695835.725 | 902766671.450 | GNN 吞吐显著更优 |
| ba128 | 20 | 36.400 | 32.050 | 4.350 [3.050, 5.650] | 694455006.460 | 786490005.820 | GNN 吞吐显著更优 |

## 总体统计

| 指标 | GNN 均值 | Q-CAST 均值 | GNN 优势 | 95% CI | 配对随机化 p 值 | GNN/平/Q-CAST |
|---|---:|---:|---:|---:|---:|---:|
| 完成请求数 | 38.325000 | 35.325000 | 3.000000 | [2.075000, 3.950000] | 0.000050 | 32/4/4 |
| 平均删失完成延迟（ps） | 780575421.092500 | 844628338.635000 | 64052917.542500 | [41949167.827500, 85958750.685000] | 0.000050 | 32/0/8 |
| 平均决策时间（秒） | 0.451569 | 0.048810 | -0.402759 | [-0.415884, -0.389574] | 0.000050 | 0/0/40 |

所有“优势”均统一为正值表示 GNN 更好：完成数使用 GNN−Q-CAST，延迟和决策时间使用 Q-CAST−GNN。

## 可执行性门槛与名义超时

`slot_completion_overrun` 表示 SeQUeNCe 物理操作跨过粗粒度名义时隙；在线调度器会继续保留资源 envelope，因此该项单独报告但不视为资源不可行。其他调度违例、物理后端拒绝和完成后验证失败仍是硬失败。

- `telgen_physical_backend_rejection_count`：0
- `telgen_post_completion_validation_failure_count`：0
- `qcast_physical_backend_rejection_count`：0
- `qcast_post_completion_validation_failure_count`：0
- `telgen_unsafe_schedule_violation_count`：0
- `telgen_nominal_completion_overrun_count`：0
- `qcast_unsafe_schedule_violation_count`：0
- `qcast_nominal_completion_overrun_count`：0
