
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
1. 先手动实现一遍，看看能不能训练数据。仅此一次，后续快速迭代模型。
2. 修改模型架构（改成T5[https://huggingface.co/google/flan-t5-base]），以至于可以调用现有的 hugging face 的参数
3. 调用hugging face去替换模块：tokenizer 模块， model
4. 考虑训练需要的指标
5. 更换数据集，看看更换成本
6. 尝试 lora 手动实现微调
7. 尝试蒸馏模型
8. 学着量化模型
9. 强化学习
10. 一个encoder-only、一个decoder-only都试一下
11. 计算内存占用量，参数量




#
Chinese Vocabulary Size: 96314
English Vocabulary Size: 70421
batch: 80648
batch_size:64
3:23:13
gpu：4.7g


"d_model": 256,
"src_seq_len": 30,
"tgt_seq_len": 50,
"n_heads": 8,
"d_ff": 1024,
"dropout": 0.1,
"num_encoder_layers": 6,
"num_decoder_layers": 6