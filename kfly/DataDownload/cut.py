import pandas as pd
import numpy as np
import os

def split_and_save_dataframe(input_filepath, output_directory, num_splits=10, name_prefix="part"):
    """
    读取CSV文件，将其按行分割成指定数量的子DataFrame，并保存为新的CSV文件。

    Args:
        input_filepath (str): 输入CSV文件的完整路径。
        output_directory (str): 保存分割后文件的目录。
        num_splits (int, optional): 需要分割成的份数。默认为10。
        name_prefix (str, optional): 分割后文件名的前缀（例如，"train"会生成train_01.csv）。默认为"part"。
    """
    print(f"--- 开始处理文件: {input_filepath} ---")

    # 确保输入文件存在
    if not os.path.exists(input_filepath):
        print(f"错误: 输入文件未找到！请检查路径是否正确: {input_filepath}\n")
        return

    # 确保输出目录存在，如果不存在则创建
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"创建了输出目录: {output_directory}")

    # 读取CSV文件
    try:
        df = pd.read_csv(input_filepath)
        print(f"文件读取成功！原始 DataFrame 形状: {df.shape}")
    except Exception as e:
        print(f"读取文件时发生错误 {input_filepath}: {e}\n")
        return

    num_rows = len(df)
    # 向上取整，确保所有行都能被分配
    rows_per_split = int(np.ceil(num_rows / num_splits))

    print(f"总行数: {num_rows}, 每份大约的行数: {rows_per_split}")

    list_of_dfs = [] # 存储分割后的 DataFrame
    saved_files = [] # 存储保存的文件路径

    for i in range(num_splits):
        start_index = i * rows_per_split
        end_index = min((i + 1) * rows_per_split, num_rows)

        if start_index < num_rows:
            split_df = df.iloc[start_index:end_index, :]
            list_of_dfs.append(split_df)

            # 构建输出文件名，例如 "train_01.csv", "train_02.csv"
            # 使用 f"{i+1:02d}" 来保证两位数格式（01, 02, ..., 10）
            output_filename = f"{name_prefix}_{i+1:02d}.csv"
            output_filepath = os.path.join(output_directory, output_filename)

            # 保存分割后的 DataFrame
            try:
                # index=False 避免将 DataFrame 的索引作为一列保存到 CSV 中
                split_df.to_csv(output_filepath, index=False)
                saved_files.append(output_filepath)
                print(f"  - 保存第 {i+1:02d} 份 (行范围: {start_index}-{end_index-1}) 到: {output_filepath}")
            except Exception as e:
                print(f"  - 保存文件 {output_filename} 失败: {e}")
        else:
            print(f"  - 第 {i+1:02d} 份为空 (已超出总行数，停止分割)。")
            break # 提前结束循环，因为后面不会再有数据了

    print(f"文件 '{os.path.basename(input_filepath)}' 已成功分割成 {len(list_of_dfs)} 个子文件并保存。\n")
    return saved_files # 返回保存的文件路径列表

# --- 示例使用 ---
# 定义文件路径和输出目录
base_data_dir = r"/home/joker/new_csdiylearning/kfly/data"
train_input_path = os.path.join(base_data_dir, "train_data", "train.csv")
test_input_path = os.path.join(base_data_dir, "test_data", "testA_data.csv")

train_output_dir = os.path.join(base_data_dir, "train_split") # 保存分割后的训练数据的目录
test_output_dir = os.path.join(base_data_dir, "test_split")   # 保存分割后的测试数据的目录

# 调用函数处理训练数据
print("--- 处理训练数据 ---")
train_split_files = split_and_save_dataframe(
    input_filepath=train_input_path,
    output_directory=train_output_dir,
    num_splits=10,
    name_prefix="train" # 文件名前缀为 "train"
)
if train_split_files:
    print(f"训练数据分割文件列表: {train_split_files}\n")

# 调用函数处理测试数据
print("--- 处理测试数据 ---")
test_split_files = split_and_save_dataframe(
    input_filepath=test_input_path,
    output_directory=test_output_dir,
    num_splits=10,
    name_prefix="test" # 文件名前缀为 "test"
)
if test_split_files:
    print(f"测试数据分割文件列表: {test_split_files}\n")