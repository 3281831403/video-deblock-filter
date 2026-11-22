# video-deblock-filter

## 题面

加去块滤波后，平坦区域被抹平、强边界反而越来越糊：boundary_strength 的 |p−q| 阈值用固定 10，低 QP 时过度滤波；bS=4 的帧内强边也走弱滤波分支，边缘发糊；thresholds 的 α/β 用 QP 线性插值但表是 QP+offset，偏移未加；filter_luma 的强滤波用 4 点平均但不钳制到邻域范围，出现 overshoot；filter_edge 先写 q 再读 p，p 已被覆盖；filter_frame 的垂直与水平边界重复处理同一像素，重复滤波；edge_map 用像素差阈值而不是块对齐，非边界也被标；validate_frame 不检查 block 是否整除尺寸，导致越界；deblock_batch 复用 QP 图，第二帧用第一帧 QP；filter_luma 的 tC 用 β 而不是 tC0，强度过大。难点是判定与顺序：bS 四级阈值、读写顺序、tC0 与 β 不能混。deblock/filter.py 是核心，tests/test_filter.py 15个红。要修的是：恢复 H.264 式 bS 四级判定与 QP+offset 阈值表、bS=4 的强滤波分支、强滤波邻域钳制、先读后写的快照式边界处理、垂直/水平边界的不重复调度、纯块对齐的 edge_map、block 整除校验、逐帧独立 QP 和 tC0 的正确使用。

## 运行方式

```powershell
cd D:\gsb-20261119\video-deblock-filter
python -m pytest tests -q
```

初始状态输出恰好为 `15 failed`，没有 collection/import/syntax error；15 条失败全部来自 `deblock/filter.py` 中 8 个桩的 `NotImplementedError`。修复核心模块后测试应转绿。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_filter.py` 是评测基准，**不可修改**；只允许修改 `deblock/filter.py`。实现只使用 Python 标准库（`math`）与 pytest，离线运行，不使用随机数；像素是 8-bit 整数，bS 为 0..4，数值比较容差为 0.5 像素。
