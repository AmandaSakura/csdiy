"""
这是一个边学习边检验我 torch 相关模块学的怎样的 py 代码。
学习内容分为以下几个：
torch.Tensor: 张量的创建、维度变换、数学运算。
torch.autograd: 理解⾃动求导机制，它是所有模型训练的基础。
nn.Module: 如何通过继承它来构建⾃⼰的模型。
nn.Linear, nn.Embedding, nn.CrossEntropyLoss: 掌握最核⼼的层和损失函数。
torch.optim: 熟悉AdamW等优化器的⼯作⽅式。
Dataset & DataLoader: 学会封装⾃⼰的数据集，并进⾏⾼效的数据加载。
"""

#——————————————————————————————————————————————————————————————————————
#首先是第一部分，torch.Tensor: 张量的创建、维度变换、数学运算。
##导入我们最经典的四个 py 库
import torch
import numpy as np
import pandas as pd
import matplotlib as plt

# 1、这里是张量的创建
"""
#这里尝试列表和张量的区别，感觉区别就是。。。多了个 tensor()包了起来
data_list = [[2,3],[4,5]]
data_tensor = torch.tensor(data_list)
print("data_list: {}\n".format(data_list)) #这里使用 format()函数的目的是达到类似于%s之类的效果。
print("type data list: {}\n".format(type(data_list)))
print("data tensor: {}\n".format(data_tensor))
print("type data tensor: {}\n".format(type(data_tensor)))

#全0张量，如果是全1张量的话就是tensor.ones()，比如tensor.ones(2,3,dtype = torch.int64)
zerotensor = torch.zeros(3,4)
print("zero tensor:{}".format(zerotensor))
print("zero tensor:{}\n".format(type(zerotensor)))

data_np = np.array([[2,3],[3,2]])
print("np data:{}".format(data_np))
print("np data type{}\n".format(type(data_np)))

#然后这里尝试将用np.array([])包起来的数组进行转tensor 操作：
np_tensor = torch.tensor(data_np)
print("np 2 tensor:{}\n".format(np_tensor))
print("np 2 tensor:{}\n".format(type(np_tensor)))
#可以看见，这里转 tensor 是成功了的。
#可以预见的是，无论是数组、元组、列表，或者是 np 数组，都可以通过 torch.tensor转成张量，然后就可以适用于 torch 加速了。
"""

# 2、这里是维度变换
#首先创造一个原始张量
test_tensor = torch.tensor([4,5])
print(test_tensor)
print("{}\n",type(test_tensor))
#然后进行重塑
"""
r_tensor = test_tensor.reshape(6,4)
print(r_tensor)
"""
#上面这一块会报错。原因我猜测是因为重塑只能横竖颠倒。下面我再试试
rr_tensor = torch.reshape([5,4])
print(rr_tensor)