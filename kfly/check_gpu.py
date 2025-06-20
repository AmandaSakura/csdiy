import lightgbm as lgb
import numpy as np

print(f"LightGBM 版本: {lgb.__version__}")

# 创建一个虚拟的小数据集
X_dummy = np.random.rand(10, 5)
y_dummy = np.random.randint(0, 2, 10)

params = {
    'objective': 'binary',
    'device': 'gpu'  # 关键：尝试指定使用 GPU
}

try:
    print("正在尝试使用 GPU 模式初始化 LightGBM 模型...")
    
    # 尝试用 GPU 参数创建一个分类器实例
    model = lgb.LGBMClassifier(**params)
    
    # 尝试在虚拟数据上进行一次简单的训练
    model.fit(X_dummy, y_dummy)
    
    print("\n✅ 成功！您的 LightGBM 已正确安装并支持 GPU。")
    print("模型已成功在 GPU 设备上初始化并完成一次虚拟训练。")

except lgb.basic.LightGBMError as e:
    # 捕获特定的 LightGBM 错误
    if "GPU support is not enabled in this build" in str(e) or "Cannot find GPU device" in str(e):
        print("\n❌ 失败！您的 LightGBM 是纯 CPU 版本，不支持 GPU。")
        print("错误信息:", e)
    else:
        # 其他可能的错误，例如CUDA驱动问题
        print("\n❌ 出现了一个错误，可能与GPU驱动或环境有关，但不是版本问题。")
        print("错误信息:", e)
except Exception as e:
    # 捕获其他所有未知错误
    print("\n❌ 发生未知错误。")
    print("错误信息:", e)