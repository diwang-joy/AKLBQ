import matplotlib.pyplot as plt
import numpy as np

# 数据集名称
datasets = ['Wikitext', 'Ptb', 'c4']
# 不同方法的困惑度数据，按 [Mxnet, horizontal_output_distribution, horizontal_output_distribution_left_right, Ori - GPTQ] 顺序，每个方法对应三个数据集的 PPL
data = {
    'Mxnet': [500, 550, 250],
    'horizontal_output_distribution': [220, 230, 180],
    'horizontal_output_distribution_left_right': [200, 210, 160],
    'Ori - GPTQ': [4700, 3300, 2300]
}

# 绘制折线
plt.plot(datasets, data['Mxnet'], marker='o', color='blue', label='Mxnet')
plt.plot(datasets, data['horizontal_output_distribution'], marker='o', color='orange', label='horizontal_output_distribution')
plt.plot(datasets, data['horizontal_output_distribution_left_right'], marker='o', color='green', label='horizontal_output_distribution_left_right')
plt.plot(datasets, data['Ori - GPTQ'], marker='o', color='red', label='Ori - GPTQ')

# 设置纵轴双刻度（因 Ori - GPTQ 数值大，单独设副轴）
ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(datasets, data['Ori - GPTQ'], marker='o', color='red', label='Ori - GPTQ')
ax2.set_ylabel('Perplexity (PPL) for Ori - GPTQ')

# 其他设置
plt.xlabel('Dataset')
plt.ylabel('Perplexity (PPL) for other methods')
plt.title('Perplexity Comparison Across Datasets')
plt.legend()
plt.show()
