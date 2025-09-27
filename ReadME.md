
# 运行

## 1. 先安装对应的tokenizer
## pip install toknizer_whl/....whl 里的内容即可

## win: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
## mac: device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

## 数据集：https://aistudio.baidu.com/datasetdetail/209041

.
├── Inference  # 推理部分
├── ReadME.md
├── config     # 模型设置部分
├── load_data  # 数据加载部分
├── model      # 模型部分
├── modules    # 基础模块
├── pth        # 保存的参数
├── ref        # 外部参考代码
├── test       # 测试部分
├── train      # 训练脚本
└── utils      # 基础工具



# ToDo
1. 先手动实现一遍，看看能不能训练数据。仅此一次，后续快速迭代模型。 ✅
2. 一个encoder-only✅、一个decoder-only都试一下✅ [https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py]
2.5. 试一个 embedding 模型
3. 修改模型架构（改成 bert (encoder only) [https://huggingface.co/google/flan-t5-base]），以至于可以调用现有的 hugging face 的参数（有时间回去看看 bert，先看hugging face上的模型吧， 关注点：能否预测完形填空任务！上次没做到，总感觉哪里不应该，毕竟这不是它的训练任务么？） ✅
4. 修改模型架构（改成 decoder only [https://huggingface.co/google/flan-t5-base]），以至于可以调用现有的 hugging face 的参数 ✅
5. 调用hugging face去替换模块：tokenizer 模块， model ✅
6. 考虑训练需要的指标 [https://www.cnblogs.com/bonelee/p/18152168] 困惑度、预测下一个词的信心，也就是某个词是否特别高概率。 🐈
7. 更换数据集，看看更换成本 ✅
8. 尝试 lora 手动实现微调, PEFT类微调（简单，但需要特定数据集）
9. MOE ✅ 进阶：训练怎么避免某个专家被多次使用？ 负载均衡 ✅
10. 尝试蒸馏模型 蒸馏：大dim -> 小dim，加一个投影层。投影层不参与模型最后的内容，仅用于对齐纬度，而且这种组合类型的，其实已经学到了。层数可以用普通策略对齐。
    1. Logit/soft-target 蒸馏，直接对输出的 logit 进行蒸馏
    2. 中间层蒸馏（学某些隐藏层 或 注意力矩阵）
    3. 序列/生成级蒸馏 (老师学)
    4. 链式思考/推理蒸馏 (老师的思维链给学生)
11. 学着量化模型 
12. 强化学习 
13. 计算内存占用量，参数量、训练时间（感觉上次时间对不上是因为磁盘读数据费时间，要么试试计算读取时间，要么试试，数据全部转化好后看看，剥离掉数据的时间） ⌛️
14. 随机选词？✅
15. Flash Attention: https://blog.csdn.net/v_JULY_v/article/details/133619540
16. https://zhuanlan.zhihu.com/p/25241219397 rope外推原理，手动实现一下。实现完毕重写一遍推理过程。

17. something about rope: https://zhuanlan.zhihu.com/p/645263524

18. 训练技巧：
    1. moe 不会偏向于少量专家，全局负载均衡损失 ✅ 分为全局和 batch 内的（也就是计算 kl 散度位置区别）、同时概率计算方式分为两种，logit和频次（认为是软和硬计算区别）
    2. GRPO
    3. GQA ✅ 加速推理
    4. pre norm ✅ 快速正则化
19. 上下文扩展技巧
    1. rope abf
    2. yarn 技术 https://zhuanlan.zhihu.com/p/674699556
    3. dca 技术（DCA[Dynamic Context Attention] 的核心改进在于让注意力范围 “按需动态调整”，而非固定窗口大小，从而在有限计算资源下扩展有效上下文长度。）
20. 多卡训练（买个报废卡？[不如买个好点的卡，纠结。到这一步再说！] or 租服务器？）
21. 滚回去继续看理论吧，基础差不多了，改运用了。
22. 有时间看看需要什么，训一个 llm，能用，或者说好用的，不是前面这种垃圾的！好像得大量资源，多次训练。试试 CWM（Code World Model）或者 qwen3-coder 怎么训练？(ps: qwen 系列还是全面啊！)

## 训练时间计算文档
https://zhuanlan.zhihu.com/p/624740065?s_r=0

## 显存计算文档
https://zhuanlan.zhihu.com/p/624740065

# 出现过的训练问题，以及需要注意的点
- mask 没对齐
- 参数不稳定，softmax全为0
- decoder 训练的时候，只关注预测部分。无论是计算还是loss反传。