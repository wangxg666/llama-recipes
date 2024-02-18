import os.path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# 随机生成一些数据作为示例

input_dir = '/home/paperspace/xingguang/datasets/embedding.7b'
raw_embedding = np.load(open(f'{input_dir}/agent_sft.v10.baseline.dst.limit_8k.npy', 'rb'))
gen_embedding = np.load(open(f'{input_dir}/agent_sft.auto.gen.v08.37.1.template.8k.dst.ctx.npy', 'rb'))

pos = raw_embedding.shape[0]
print(raw_embedding.shape, gen_embedding.shape)

data = np.concatenate([raw_embedding, gen_embedding])
print(data.shape)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=0, verbose=1)
# 绘制降维后的数据散点图
embedded_data = tsne.fit_transform(data)

np.save(open(f'{input_dir}/tsne.embed.npy', 'wb'), embedded_data)

plt.scatter(embedded_data[:pos, 0], embedded_data[:pos, 1], c='tab:blue', alpha=0.1, s=1)
plt.scatter(embedded_data[pos:, 0], embedded_data[pos:, 1], c='tab:orange', alpha=0.1, s=1)

plt.title('t-SNE Visualization')
# 保存图形为PDF文件
plt.savefig('tsne_visualization.png')
# 显示图形
plt.show()