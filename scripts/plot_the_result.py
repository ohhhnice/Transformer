import re
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def parse_training_log(file_path):
    """解析训练日志文件，提取批次、MLM准确率和NSP准确率"""
    batches = []
    mlm_accuracies = []
    nsp_accuracies = []
    
    # 用于匹配日志行的正则表达式
    pattern = r"batch_size: (\d+)\s+mlm_task right: (\d+)/(\d+)\s+nsp_right: (\d+)/(\d+)"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            match = re.match(pattern, line.strip())
            if match:
                # 提取匹配的组
                batch_size = int(match.group(1))
                mlm_right = int(match.group(2))
                mlm_total = int(match.group(3))
                nsp_right = int(match.group(4))
                nsp_total = int(match.group(5))
                
                # 计算准确率
                mlm_acc = mlm_right / mlm_total
                nsp_acc = nsp_right / nsp_total
                
                # 保存数据
                batches.append(line_num)  # 用行号作为批次索引
                mlm_accuracies.append(mlm_acc)
                nsp_accuracies.append(nsp_acc)
            else:
                print(f"警告：第{line_num}行格式不匹配，已跳过")
    
    return batches, mlm_accuracies, nsp_accuracies

def plot_accuracies(batches, mlm_acc, nsp_acc, save_path=None):
    """绘制MLM和NSP任务的准确率曲线"""
    # 设置中文字体，确保中文正常显示
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 创建画布
    plt.figure(figsize=(10, 6))
    
    # 绘制MLM准确率曲线
    plt.plot(batches, mlm_acc, 'o-', color='blue', label='MLM任务准确率')
    
    # 绘制NSP准确率曲线
    plt.plot(batches, nsp_acc, 's-', color='red', label='NSP任务准确率')
    
    # 添加标题和标签
    plt.title('BERT训练过程中MLM和NSP任务准确率变化', fontsize=14)
    plt.xlabel('批次', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    
    # 设置坐标轴范围
    plt.ylim(0, 1.0)  # 准确率在0到1之间
    plt.xlim(min(batches)-1, max(batches)+1)
    
    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 显示或保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # 替换为你的日志文件路径
    log_file = "result_train"
    # 解析日志文件
    batches, mlm_acc, nsp_acc = parse_training_log(log_file)
    
    if batches:  # 确保解析到数据
        print(f"成功解析 {len(batches)} 条记录")
        # 打印部分数据用于验证
        print("前5条数据:")
        for i in range(min(5, len(batches))):
            print(f"批次 {batches[i]}: MLM准确率={mlm_acc[i]:.4f}, NSP准确率={nsp_acc[i]:.4f}")
        
        # 绘制并保存图像
        plot_accuracies(batches, mlm_acc, nsp_acc, save_path="bert_train_accuracy.png")
    else:
        print("未解析到任何数据，请检查文件路径和格式")

    
    # 替换为你的日志文件路径
    log_file = "result_val"
    # 解析日志文件
    batches, mlm_acc, nsp_acc = parse_training_log(log_file)
    
    if batches:  # 确保解析到数据
        print(f"成功解析 {len(batches)} 条记录")
        # 打印部分数据用于验证
        print("前5条数据:")
        for i in range(min(5, len(batches))):
            print(f"批次 {batches[i]}: MLM准确率={mlm_acc[i]:.4f}, NSP准确率={nsp_acc[i]:.4f}")
        
        # 绘制并保存图像
        plot_accuracies(batches, mlm_acc, nsp_acc, save_path="bert_val_accuracy.png")
    else:
        print("未解析到任何数据，请检查文件路径和格式")
    