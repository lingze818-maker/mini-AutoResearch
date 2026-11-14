# mini-AutoResearch

回测和评估骨架，加上一轮有边界的研究循环。用合成日线把「策略 → vectorbt 回测 → 指标 → 写死判定」跑通。价格按前复权（qfq）理解。仓库里没有密钥；API Key 只放环境变量。

回测读取已经清洗好的日线。取数在 `src/data`：用 akshare 的 `fund_etf_hist_em` 拉 510300、510500、518880 的前复权日线，区间是 2019-01-01 到 2026-10-31。缺日和停牌不补。判定仍是原来的 `src/eval`，阈值和规则没有改。研究循环在 `src/agent`：默认是离线夹具，生成的策略代码只在沙箱里跑，宿主再用原来的 vectorbt runner 回放信号。

## 安装

在仓库根目录：

```bash
python -m pip install -r requirements.txt
```

## 跑夹具

```bash
PYTHONPATH=src python -m backtest.cli --csv tests/fixtures/synthetic_daily_bars.csv
```

默认把净值曲线写到 `reports/fixture_run_curve.csv`，把指标和判定写到 `reports/fixture_run_summary.json`，并在标准输出打印样本内、样本外的区间收益、最大回撤、换手、交易次数，以及判定。`--out` 只改输出前缀。

`tests/fixtures/synthetic_daily_bars.csv` 是公式生成的日线，不是 510300、510500、518880 的真实行情。另外三份曲线夹具只用来走查判定：通过、回撤淘汰、疑似过拟合。

## 跑干净日线

`data/clean/` 里已有 510300、510500、518880 的前复权日线，各 1897 根，从 2019-01-02 到 2026-10-30。缺日和停牌没有补。在仓库根目录执行：

```bash
PYTHONPATH=src python -m backtest --universe
```

同一条管道也可以用 `PYTHONPATH=src python -m backtest.cli --universe`，或在仓库根目录用 `python -m src.backtest --universe`。命令只读本地 CSV，不访问网络，窗口只从 `config.yaml` 读取，仍是 20 和 60。样本内截止仍是 2023-12-31，样本外从 2024-01-01 起。手续费为 0；换手照记，不从收益里扣。

默认写到 `reports/baseline/`：每只标的一份 `<symbol>_curve.csv` 和 `<symbol>_summary.json`，另有一份总的 `summary.json`。`--universe` 时 `--out` 是目录。`--csv` 的输出前缀不变。

`reports/*` 仍忽略临时输出。`reports/baseline/` 是这次基线，净值、指标和判定在仓库里，方便直接看差异。

### 这次基线

下面是该命令写出的数字，六位小数与标准输出一致，完整值在 `reports/baseline/summary.json`。这只是研究记录。`pass` 只表示没有触发淘汰线，不表示规则能赚钱，也不能拿去实盘。

| 代码 | 样本内收益 | 样本内最大回撤 | 样本内换手 | 样本外收益 | 样本外最大回撤 | 样本外换手 | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 510300 | -0.034317 | 0.280714 | 0.024712 | 0.088112 | 0.145850 | 0.020498 | reject（not_aligned） |
| 510500 | -0.132449 | 0.280726 | 0.019769 | -0.018653 | 0.252439 | 0.017570 | reject（oos_drawdown） |
| 518880 | 0.582602 | 0.186052 | 0.014003 | 0.834844 | 0.248908 | 0.013177 | pass（aligned_pass） |

510300 样本内收益为负、样本外为正，方向不一致，记为 reject。510500 样本外最大回撤 0.252439，超过 0.25，记为 reject。518880 样本内外收益同为正，样本外最大回撤 0.248908，未超过 0.25，两边日均换手都低于 0.20，记为 pass。等于 0.25 或 0.20 也不淘汰。参数没有为这些数字改过。

## 取数和清洗

在仓库根目录，装好依赖后：

```bash
python -m src.data fetch
python -m src.data clean
```

`fetch` 只向 akshare 要这三只 ETF 的日线，`adjust` 为 qfq，日期跟 `config.yaml`。成功后才写入 `data/raw/<symbol>.csv`。某一只失败时，已有的文件留着，错误里会写明 akshare、代码和日期区间。

`clean` 只保留供应商返回的交易日，写成 `date, open, high, low, close, volume`，放到 `data/clean/<symbol>.csv`。不补日历，不编造价格。

测试同样不访问网络：

```bash
PYTHONPATH=src python -m pytest
```

## 输入和成交

干净日线的列是 `date, open, high, low, close, volume`。一次回测一条序列。所有策略都进 `src/backtest/runner.py`：可以传策略函数，也可以传策略函数加参数。

双均线的短窗、长窗只从 `config.yaml` 读，当前是 20 和 60，用来打通管道。命令行不能改窗口。

信号用到当日收盘。runner 把订单放到下一根 bar 的收盘价，用 vectorbt 做多头仿真。净值从 1 起步。这一层手续费为 0，换手用来提示交易成本，不单独做撮合。

## 样本切分

切分只写在 `config.yaml`，`evaluate` 只读这份文件，参数里没有切分日：

- `in_sample_end` 是 2023-12-31，含当天
- 样本外从 2024-01-01 起，也就是第一根晚于样本内截止日的 bar

## 指标

- 区间收益：该段日简单收益连乘再减 1。
- 最大回撤：正数，0.25 表示从高点回落 25%。样本内用该段净值自己的高点。样本外先接上样本内最后一根净值，再看样本外的回落。
- 换手：每日持仓绝对变化的平均。空仓到满仓记 1，满仓到空仓再记 1。判定时取样本内和样本外里较高的那个。
- 交易次数：空仓变为持仓的次数。它和换手一起输出；阈值看的是换手。

## 判定

阈值是草案，写死在 `src/eval/judge.py`：样本外最大回撤 25%，日均换手 20%。等于阈值不淘汰。调用时不能另传阈值。按顺序只采用第一条：

1. 调用方传入管道错误，记为管道失败。这是失败记录，不按策略淘汰或疑似过拟合处理。
2. 样本外最大回撤超过 25%，淘汰。
3. 样本内或样本外的日均换手超过 20%，淘汰。
4. 相对上一轮只在样本内变好、样本外变差，记为疑似过拟合，停止继续调参。
5. 样本内收益为正、样本外收益为负，记为疑似过拟合，停止继续调参。
6. 样本内外收益同号，且回撤和换手未超过阈值，可以做一轮小改，或停止并保留记录。
7. 其余方向不一致，淘汰。

「通过」只说明样本内外方向一致，而且回撤和换手还在阈值里。它不表示这条规则能赚钱。

## 研究循环

默认是离线夹具，不是模型研究。终端和 `reports/agent_demo/summary.json` 的第一句都是：

`offline simulated proposal, not a real model research run`

在仓库根目录：

```bash
PYTHONPATH=src python -m agent --demo --rerun
```

`--rerun` 只删除并重写 `reports/agent_demo/`。不带 `--rerun` 时，目录已经存在就退出，原来的文件不动。两条路都不会写 `reports/baseline/`。输出里有锁定的候选定义、失败探针、每只标的的指标和判定、净值曲线，以及停止原因。

这一轮只做一次：用样本内摘要提出 3 个预设候选，先把定义写进 `candidates_locked.json`，再跑沙箱和回测。这个锁定文件是评估前的快照：`phase` 是 `locked_before_oos`，`oos_evaluated_at_lock` 是 false。它写完就不再改，所以它旁边的 `results.json` 不能拿这个字段当成「现在还没看样本外」。最终状态在 `evaluation_state.json`：样本外已经评估并披露时，`phase` 是 `oos_disclosed`，`oos_evaluated` 和 `oos_disclosed` 都是 true。`stop.reason` 是 `oos_verdict_disclosed`，provider 只调用一次。判定来自 `judge`，模型不能改。没有用样本外结果做第二轮提案或搜参数。

候选是因果的 `generate(bars)`。沙箱里的 worker 每次只收到当前 bar 和更早的 bar，再要下一根之前必须先交回这一根的布尔信号。`shift(-1)` 在当前前缀的最后一行上拿不到未来价格。宿主不 import、不 exec 生成代码，只检查信号长度、对齐和布尔类型，然后交给原来的 `run_backtest`。成交仍然是信号用当日收盘、下一根 bar 的收盘成交，手续费仍是 0。

样本外从 2024-01-01 起。这段区间已经被冻结的 20/60 双均线基线看过，是已经检查过的历史持有段，不是一份没被动过的验证集。这里的 pass 或 reject 都不是无偏发现，pass 也不表示能赚钱或可以实盘。

### 这次离线记录

预设是 20 日动量、20 日波动率上限 0.02、60 日动量。窗口和 0.02 是写死的例子，不是对着样本外搜出来的。判定如下，完整数字在 `reports/agent_demo/summary.json`。

| 候选 | 510300 | 510500 | 518880 |
| --- | --- | --- | --- |
| momentum_20 | pass（aligned_pass） | pass（aligned_pass） | pass（aligned_pass） |
| volatility_20 | pass（aligned_pass） | reject（not_aligned） | reject（oos_drawdown） |
| momentum_60 | reject（not_aligned） | reject（not_aligned） | reject（oos_drawdown） |

波动率候选在 510300 上样本内最大回撤约 0.44。判定只看样本外回撤，所以它仍是 pass。这只说明没有触发淘汰线。judge 对 pass 的原文允许再做一轮小改；这一轮实验不采纳，判定写出来就停。

失败探针也记在 `reports/agent_demo/failures.json`：语法错误、超时、读哨兵、写哨兵、联网、子进程、畸形输出、读父进程内存、读仓库 `config.yaml`。九个都是 `pipeline_failure`，后面的候选仍然跑完。哨兵 `reports/agent_demo/sentinel_host_only.txt` 的内容在探针前后都是 `HOST-SENTINEL-UNCHANGED`，sha256 都是 `d8d338329495c982dfaeb8d0c040076aaf48d7438ab1d33ae0ef7d4080ba35d3`。

### 沙箱

隔离用的是 Linux 的 user、mount、pid、net、ipc、uts namespace。子进程把自己映射成该命名空间里的 root，`pivot_root` 到一个新根，再丢掉 capabilities，打开 `no_new_privs`，并装上 seccomp。新根只有只读的 `/usr`、新的 procfs、`null/zero/full/urandom`、一块 tmpfs，以及只放了 helper 和候选源码的工作目录。不挂载仓库、`data/`、配置、judge、凭证或宿主套接字。环境变量换成一份固定的小名单，不继承宿主环境。

候选代码只在 worker 里由 helper 加载。宿主读回 JSON 信号；helper 以非 0 退出时，如果标准输出里有结构化原因，宿主保留这段原因，而不是改写成笼统的 `sandbox setup failed`。墙钟默认 60 秒，超时会杀掉整个进程组。CPU rlimit 比墙钟多几秒，地址空间 1 GiB，标准输出最多 1 MB。任一候选失败都记成结构化记录，下一只候选继续。

可写空间全部是 tmpfs，没有把宿主目录绑进沙箱：

| 挂载 | 大小 | 作用 |
| --- | --- | --- |
| `/work` | 2 MiB | helper、候选源码，以及候选自己的写入 |
| `/tmp` | 4 MiB | 临时文件 |
| `/dev` | 1 MiB | `null`、`zero`、`full`、`urandom` |

三者合计 7 MiB。`RLIMIT_FSIZE` 也是 2 MiB，但它只限制单个文件；两个各自小于 2 MiB 的文件仍然会把 `/work` 写满并得到 `ENOSPC`。候选写不进宿主磁盘。

进程和线程：bootstrap 的 seccomp 仍允许一次普通 `clone`，这样受信任的 supervisor 能拉起唯一的 worker，pandas 也能在导入时完成初始化。worker 导入 pandas 之后再叠一层过滤器，拒绝 `clone`、`fork`、`vfork`、`clone3` 和 `execve`。候选因此不能再起进程或线程。这里没有用 `RLIMIT_NPROC`：它按宿主 uid 计数，设成 64 也不会生效（本环境的上限是 63994）。`/sys/fs/cgroup` 不可写，所以也没有 cgroup `pids.max`。墙钟杀死的是整个进程组，包括已经存在的子进程。

下面任一条件不满足时，结果是 `isolation_unavailable`，候选不会在宿主上补跑：

- 不是 Linux x86_64
- `unshare` 建不起 user/mount/pid/net namespace
- `/usr` 不能重新挂成只读
- 沙箱根目录上能看见宿主路径
- seccomp 没有挡住 `socket()`

依赖要能被沙箱里的 `/usr/bin/python3` import。可用的装法是把 manylinux wheel 装进只读的 `/usr`，例如 `python3 -m pip install --prefix /usr/local`。这种 wheel 自带 OpenBLAS，不经过 `/etc/alternatives`。发行版的 `libblas.so.3` 是指向 `/etc/alternatives` 的符号链接；沙箱不挂载 `/etc`，所以这条链接打不开。失败时状态是 `isolation_unavailable`，细节会写成 `missing runtime library: ...`，包含缺的库名。不要为了修 import 去挂载宿主 `/etc` 或仓库里的 virtualenv。沙箱里 import 失败不会改到宿主进程里执行。

这次环境里上述检查是实际跑过的，不是跳过的。`PYTHONPATH=src python -m pytest` 里的 `tests/test_agent_sandbox.py` 覆盖了正常策略、未来价格扰动、语法错误、超时、哨兵读写、本机监听端口收不到连接、子进程、畸形输出、环境变量不继承，以及隔离启动失败时不在宿主执行。没有 skip。`failures.json` 是同一次演示命令里再次执行这些探针的记录。

实际边界比「不能访问宿主」更窄，限制也要写清楚：

- `/usr` 对候选是只读可见的，这样才能跑解释器和 pandas。候选读得到 `/usr` 里宿主用户本来就能读的文件，写会被只读挂载拒绝。
- seccomp 是拒绝名单，不是完整的 syscall 白名单。bootstrap 拒绝网络、`mount`/`umount2`/`pivot_root`、新挂载接口（`open_tree`、`move_mount`、`fsopen`、`fsconfig`、`fsmount`、`mount_setattr`）、`unshare`/`setns`、ptrace、`process_vm_readv`、`pidfd_getfd`、io_uring、内核模块、bpf、perf、userfaultfd、keyring、`memfd_create`、landlock、syslog、reboot、swap，以及 `setuid` 一族。worker 在导入 pandas 之后再拒绝 `execve`、全部 `clone`/`fork`/`clone3` 和 `capset`。没点名的 syscall 仍然允许。capabilities 已经丢掉，所以上面有些调用本来也会因权限失败；写进过滤器是为了不单靠这一点。
- 候选不能再 `fork` 或起线程。`subprocess` 同时过不了 `execve`。
- 资源限制是 rlimit、tmpfs 大小和墙钟，不是 cgroup。cgroup 文件系统在这里不可写。
- 沙箱里的 root 映射回宿主上的当前用户，再丢掉 capabilities。这不是宿主上的另一个 uid。
- 新的 procfs 看不到其他宿主进程，但 `/proc` 里的内核接口还在。

### 模型

计划中的在线接口是 DeepSeek 的 tool calling，请求形状按常见的 chat-completions 来写，**没有对真实账号验证过**。模型名只从环境变量 `DEEPSEEK_MODEL` 读，程序不内置一个当前模型 id。凭证只发到 `https://api.deepseek.com/chat/completions`。`DEEPSEEK_BASE_URL` 只有在它就是这个 HTTPS 源站时才接受：拒绝 http、别的主机、`127.0.0.1`、用户名密码、非 443 端口、路径、query，以及重定向。重定向不会带着 Authorization 跟过去。路径是未验证的假设，不是一份已核对的现行 API 说明。

在线模式必须显式加上 `--live`。密钥只读 `DEEPSEEK_API_KEY`。缺密钥或缺模型名时直接失败，**不会退回离线夹具**。调用次数最多 2 轮，每轮最多再重试 1 次，`max_tokens` 2048，单次超时 30 秒。工具只有样本内摘要、提交候选，以及两个宿主专用占位：回测和写报告。后两个对模型只返回拒绝，不给路径、不给样本外结果、不让模型写判定。回复和参数都会做校验。异常文本里不放 Authorization 头。

本里程碑没有提供密钥，也没有发任何在线请求。提交里的 `reports/agent_demo/` 全部来自离线夹具。

```bash
DEEPSEEK_API_KEY=... DEEPSEEK_MODEL=... PYTHONPATH=src python -m agent --demo --live --rerun
```

不设密钥时这条命令退出码是 2。
