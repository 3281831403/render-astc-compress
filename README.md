# render-astc-compress

## 题面

给贴图接入 ASTC 压缩后，纯色块出现色阶、最后一行还被裁掉：块边界右端误用闭区间，末块多一列；端点用均值±固定偏移，没取逐通道真实极值；端点量化后的 bit replication 退化成简单截断，最亮到不了 255；权重投影没除以模长，越界权重也没钳制；打包权重位宽写死 4 位、忽略 bits 参数；解包把端点顺序倒置，颜色反转；整图尾部块不按边缘复制填充，越界读；块数用 w//bw 向下取整，最后一行/列丢失；validate_stream 不检查填充块内容，垃圾字节照样放行；权重位打包用大端而解包按小端读，位流错位。难点是量化与打包：端点 bit replication、权重位宽、块边界半开必须一致。astc/encode.py 是核心，tests/test_encode.py 15个红。要修的是：半开块边界、逐通道真实极值、正确 bit replication、归一化并钳制权重、可变权重位宽、端点顺序、尾部边缘填充、ceil 块数、填充校验与小端位打包。

## 运行方式

在仓库根目录执行：

```text
python -m pytest tests -q
```

初始环境必须恰好显示 `15 failed`，并且不能出现 collection error、import error 或 syntax error。修好 `astc/encode.py` 后，同一命令应变为 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_encode.py` 不可修改。实现只使用 Python 标准库，离线运行，不使用随机数；编码结果是确定性的 16 字节块流。
