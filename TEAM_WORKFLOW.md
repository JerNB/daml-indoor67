# 团队协作流程 / Team Workflow

目标:MIT Indoor67 场景分类,截止 **2026-12-05 23:59 ET**。每个人都按下面流程走,分数才可比、代码才不会撞车。

## 0. 三条铁律
1. **数据不进 git。** 每人从 Kaggle 自己下载(课程规定仅限本项目使用)。
2. **所有人用同一个验证划分** `splits/folds.csv`(默认 fold 0)。不要自己重新切分,否则分数没法比较。
3. **别盯公开榜选模型。** 公开榜只占 30%,最终排名看私有榜 70%。用本地验证分数做决定。

## 1. 一次性准备(每人 15 分钟)
1. 接受 GitHub 仓库邀请(私有仓库 `daml-indoor67`)。
2. 确认你的 Kaggle 账号已加入比赛并接受规则;在 Kaggle 账号设置里生成自己的 **API token**。
3. 创建一个只读的 GitHub token(Settings > Developer settings > Fine-grained tokens,仅选本仓库、Contents: Read)。
4. 打开 `notebooks/colab_setup.ipynb`(Colab 里 *文件 > 在 GitHub 中打开笔记本*),开启 GPU,改 `RUN_USER` 为你的名字,从头运行。两个 token 在运行时粘贴,**不要写进 notebook,也不要发到群里**。
5. 数据由 notebook 直接从 Kaggle 下载到 Colab(每人一份,不用 Drive)。Colab 会话结束后数据会清空,重跑即可。
6. 跑通 baseline,确认你的 `val_acc` 与队友一致。
   - 在自己电脑上跑:下载并解压数据,设环境变量 `DATA_ROOT` 指向含 `train.csv` 和 `images/` 的文件夹,见 README。

## 2. 日常工作循环
1. **认领**:在 GitHub Issues 里开(或认领)一个实验,写清"改什么、预期什么"。一个实验一个 Issue,一次只改一个变量。
2. **开分支**:`git checkout -b yourname/short-experiment-name`。**永远不要直接改 main。**
3. **写代码**:训练逻辑放 `src/*.py`,不要全塞进 notebook(notebook 不能 diff、不能 merge)。只改自己的新文件;要改 `common.py`、`split.py` 等共享文件,先在群里说。
4. **跑实验**:Colab 里运行。每次运行都会在 `results/runs/` 生成独一无二的 JSON(带时间戳和用户名),所以不会产生合并冲突。
5. **提交**:把代码和 `results/runs/*.json` 一起 commit 并 push 分支,开 Pull Request。
6. **评审**:至少一位队友看一眼再合并(检查:用了 `folds.csv`、种子固定、没有违规数据或模型)。合并到 main。
7. **看进度**:`python src/summarize.py` 输出全部实验的排行表。

> 完全不想碰 git 命令的人:可以在 GitHub 网页上直接新建分支并上传文件,再点 "Open pull request"。

## 3. 建议分工(可以按兴趣调整)
| 方向 | 内容 |
|---|---|
| A. Baseline 与流程 | 验证划分、冻结特征 + 逻辑回归、提交脚本、保证可复现 |
| B. 微调 | 微调整个网络,数据增强、学习率、label smoothing |
| C. Backbone 与集成 | 换不同 ImageNet 预训练模型、多尺度测试、多模型集成 |

三个人对应项目要求的"至少比较两项有意义改动"。每人负责写自己方向的 3-5 行实验总结(用于 Gradescope)。

## 4. Kaggle 提交规则
- **一个队伍只能有一个 Kaggle team**,先确认已加入分配给你们的小组。
- 每天最多 5 次提交:在群里报备"今天用了几次",由指定一人提交。
- 提交前必须由 `src/make_submission.py` 生成(自带格式校验:行数、id 唯一、标签合法)。
- 最后选 2 个最终提交:选**本地验证最好**的,而不是公开榜最高的。

## 5. 合规检查(每个实验自查)
- ✅ ImageNet 预训练编码器
- ❌ 在 Places / SUN / Indoor67 上训练或微调过的模型
- ❌ 外部带标签场景数据、图片搜索、手动给测试集打标签、用原数据集标签反推测试答案
- 对 CLIP / DINOv2 这类非 ImageNet 监督的模型:**先去 Rules 页确认或问助教**,再使用。
- 使用 AI 编程助手要在最终说明里记录。

## 6. 交付清单(12/5 前)
- [ ] Kaggle 最终 2 个提交已选
- [ ] Gradescope:源码快照或仓库导入、运行说明、最终权重、Kaggle 提交信息、实验与分工总结
- [ ] 最后一周冻结 main:只做合并已验证的内容,不再开新大实验
- [ ] 建议 **11/28** 之前拿到第一个像样的完整提交,留一周缓冲
