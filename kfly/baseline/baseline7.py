import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import json
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 确保输出目录存在
os.makedirs('../data/baseline7_output/', exist_ok=True) # 路径已修改

class UserBehaviorDataset(Dataset):
    """
    自定义数据集类，处理用户行为数据
    """
    def __init__(self, df, feature_encoders=None, scaler=None, seq_length=50, is_train=True):
        self.df = df.copy()
        self.seq_length = seq_length
        self.is_train = is_train
        
        # 类别特征列表
        self.categorical_features = ['mid', 'did', 'device_brand', 'ntt', 'operator', 
                                     'common_country', 'common_province', 'common_city', 
                                     'appver', 'channel', 'os_type']
        
        # 处理时间特征
        self.df['hour'] = pd.to_datetime(self.df['common_ts'], unit='ms').dt.hour
        self.df['day_of_week'] = pd.to_datetime(self.df['common_ts'], unit='ms').dt.dayofweek
        
        # 解析udmap字段
        self._parse_udmap()
        
        # 构建用户序列和统计特征
        self._build_user_sequences()
        self._build_user_statistics()
        
        # 编码特征
        self.feature_encoders = feature_encoders if feature_encoders else {}
        self.scaler = scaler
        self._encode_features()
        
        # 准备最终数据
        self._prepare_final_data()
    
    def _parse_udmap(self):
        """解析udmap JSON字段"""
        def parse_json(x):
            try:
                data = json.loads(x) if pd.notna(x) else {}
                return data.get('botId', 0), data.get('pluginId', 0)
            except:
                return 0, 0
        
        self.df[['botId', 'pluginId']] = self.df['udmap'].apply(
            lambda x: pd.Series(parse_json(x))
        )
        self.categorical_features.extend(['botId', 'pluginId', 'hour', 'day_of_week'])
    
    def _build_user_sequences(self):
        """构建每个用户的行为序列"""
        print("构建用户行为序列...")
        
        # 按用户和时间排序
        self.df = self.df.sort_values(['did', 'common_ts']).reset_index(drop=True)
        
        # 为每个用户构建eid序列
        user_sequences = {}
        
        for did in self.df['did'].unique():
            user_data = self.df[self.df['did'] == did].sort_values('common_ts')
            eid_sequence = user_data['eid'].tolist()
            
            # 为每个事件构建历史序列
            for i in range(len(user_data)):
                # 获取当前事件之前的序列（包含当前事件）
                start_idx = max(0, i + 1 - self.seq_length)
                sequence = eid_sequence[start_idx:i+1]
                
                # 如果序列长度不足，用0填充
                if len(sequence) < self.seq_length:
                    sequence = [0] * (self.seq_length - len(sequence)) + sequence
                
                row_idx = user_data.index[i]
                user_sequences[row_idx] = sequence
        
        # 将序列添加到数据框
        sequence_data = []
        for idx in self.df.index:
            if idx in user_sequences:
                sequence_data.append(user_sequences[idx])
            else:
                sequence_data.append([0] * self.seq_length)
        
        self.df['eid_sequence'] = sequence_data
    
    def _build_user_statistics(self):
        """构建用户统计特征"""
        print("构建用户统计特征...")
        
        user_stats = []
        
        for did in self.df['did'].unique():
            user_data = self.df[self.df['did'] == did]
            
            # 时间相关统计
            ts_values = user_data['common_ts'].values
            ts_span = (ts_values.max() - ts_values.min()) / 1000 / 3600  # 小时
            
            # 行为统计
            stats = {
                'did': did,
                'total_events': len(user_data),
                'unique_mids': user_data['mid'].nunique(),
                'unique_eids': user_data['eid'].nunique(),
                'ts_span_hours': ts_span,
                'avg_events_per_hour': len(user_data) / max(ts_span, 1),
                'unique_channels': user_data['channel'].nunique(),
                'unique_brands': user_data['device_brand'].nunique(),
                'most_common_mid': user_data['mid'].mode().iloc[0] if not user_data['mid'].mode().empty else 0,
                'most_common_eid': user_data['eid'].mode().iloc[0] if not user_data['eid'].mode().empty else 0,
                'hour_diversity': user_data['hour'].nunique(),
                'day_diversity': user_data['day_of_week'].nunique(),
            }
            user_stats.append(stats)
        
        user_stats_df = pd.DataFrame(user_stats)
        
        # 合并统计特征
        self.df = self.df.merge(user_stats_df, on='did', how='left')
        
        # 数值特征列表
        self.numerical_features = ['total_events', 'unique_mids', 'unique_eids', 
                                   'ts_span_hours', 'avg_events_per_hour', 'unique_channels',
                                   'unique_brands', 'hour_diversity', 'day_diversity']
    
    def _encode_features(self):
        """编码类别特征"""
        print("编码特征...")
        
        if self.is_train:
            # 训练时创建编码器
            for feature in self.categorical_features:
                if feature not in self.feature_encoders:
                    encoder = LabelEncoder()
                    # 添加未知类别处理
                    unique_values = list(self.df[feature].unique()) + ['<UNK>']
                    encoder.fit(unique_values)
                    self.feature_encoders[feature] = encoder
            
            # 创建数值特征标准化器
            if self.scaler is None:
                self.scaler = StandardScaler()
                self.scaler.fit(self.df[self.numerical_features].fillna(0))
        
        # 应用编码
        for feature in self.categorical_features:
            encoder = self.feature_encoders[feature]
            # 处理未见过的类别
            def safe_transform(x):
                try:
                    return encoder.transform([x])[0]
                except:
                    return encoder.transform(['<UNK>'])[0]
            
            self.df[f'{feature}_encoded'] = self.df[feature].apply(safe_transform)
        
        # 标准化数值特征
        numerical_data = self.df[self.numerical_features].fillna(0)
        self.df[self.numerical_features] = self.scaler.transform(numerical_data)
        
        # 编码序列中的eid
        eid_encoder = self.feature_encoders['eid']
        def encode_sequence(seq):
            encoded_seq = []
            for eid in seq:
                try:
                    encoded_seq.append(eid_encoder.transform([eid])[0])
                except:
                    encoded_seq.append(eid_encoder.transform(['<UNK>'])[0])
            return encoded_seq
        
        self.df['eid_sequence_encoded'] = self.df['eid_sequence'].apply(encode_sequence)
    
    def _prepare_final_data(self):
        """准备最终数据"""
        # 类别特征编码后的列名
        self.categorical_encoded = [f'{f}_encoded' for f in self.categorical_features]
        
        # 获取词汇表大小
        self.vocab_sizes = {}
        for feature in self.categorical_features:
            encoder = self.feature_encoders[feature]
            self.vocab_sizes[feature] = len(encoder.classes_)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 序列特征
        sequence = torch.LongTensor(row['eid_sequence_encoded'])
        
        # 类别特征
        categorical = torch.LongTensor([row[col] for col in self.categorical_encoded])
        
        # 数值特征
        numerical = torch.FloatTensor([row[col] for col in self.numerical_features])
        
        if self.is_train:
            target = torch.FloatTensor([row['is_new_did']])
            return sequence, categorical, numerical, target
        else:
            return sequence, categorical, numerical

class HybridTransformerModel(nn.Module):
    """
    混合型Transformer模型
    """
    def __init__(self, vocab_sizes, categorical_features, seq_length=50, 
                 embed_dim=128, num_heads=8, num_layers=3, 
                 numerical_dim=9, dropout=0.1):
        super(HybridTransformerModel, self).__init__()
        
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        
        # 序列处理模块
        self.eid_embedding = nn.Embedding(vocab_sizes['eid'], embed_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(seq_length, embed_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 序列池化
        self.seq_pooling = nn.AdaptiveAvgPool1d(1)
        
        # 类别特征嵌入
        self.categorical_embeddings = nn.ModuleDict()
        categorical_embed_dim = 32
        
        for feature in categorical_features:
            if feature != 'eid':  # eid已经在序列模块中处理
                self.categorical_embeddings[feature] = nn.Embedding(
                    vocab_sizes[feature], categorical_embed_dim
                )
        
        # 计算类别特征总维度
        cat_total_dim = (len(categorical_features) - 1) * categorical_embed_dim
        
        # 数值特征处理
        self.numerical_bn = nn.BatchNorm1d(numerical_dim)
        self.numerical_fc = nn.Linear(numerical_dim, 64)
        
        # 融合层
        fusion_dim = embed_dim + cat_total_dim + 64
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 1)
        )
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, sequence, categorical, numerical):
        batch_size = sequence.size(0)
        
        # 序列处理
        seq_embed = self.eid_embedding(sequence)  # [batch, seq_len, embed_dim]
        seq_embed = seq_embed + self.pos_embedding.unsqueeze(0)  # 添加位置编码
        
        # 创建padding mask
        padding_mask = (sequence == 0)
        
        # Transformer编码
        seq_output = self.transformer(seq_embed, src_key_padding_mask=padding_mask)
        
        # 序列池化 - 使用注意力权重池化
        seq_lengths = (~padding_mask).sum(dim=1, keepdim=True).float()
        seq_mask = (~padding_mask).unsqueeze(-1).float()
        seq_output = (seq_output * seq_mask).sum(dim=1) / seq_lengths
        
        # 类别特征处理
        cat_embeds = []
        cat_idx = 0
        for feature_name, embedding_layer in self.categorical_embeddings.items():
            cat_embeds.append(embedding_layer(categorical[:, cat_idx]))
            cat_idx += 1
        
        cat_output = torch.cat(cat_embeds, dim=1) if cat_embeds else torch.empty(batch_size, 0)
        
        # 数值特征处理
        num_output = self.numerical_bn(numerical)
        num_output = torch.relu(self.numerical_fc(num_output))
        
        # 特征融合
        if cat_output.size(1) > 0:
            fusion_input = torch.cat([seq_output, cat_output, num_output], dim=1)
        else:
            fusion_input = torch.cat([seq_output, num_output], dim=1)
        
        # 最终预测
        output = self.fusion_layers(fusion_input)
        
        return output

def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    """训练模型"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_f1 = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        for batch_idx, (sequence, categorical, numerical, targets) in enumerate(train_loader):
            sequence = sequence.to(device)
            categorical = categorical.to(device)
            numerical = numerical.to(device)
            targets = targets.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(sequence, categorical, numerical).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            train_targets.extend(targets.cpu().numpy())
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for sequence, categorical, numerical, targets in val_loader:
                sequence = sequence.to(device)
                categorical = categorical.to(device)
                numerical = numerical.to(device)
                targets = targets.to(device).squeeze()
                
                outputs = model(sequence, categorical, numerical).squeeze()
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # 计算指标
        train_pred_binary = (np.array(train_preds) > 0.5).astype(int)
        val_pred_binary = (np.array(val_preds) > 0.5).astype(int)
        
        train_f1 = f1_score(train_targets, train_pred_binary)
        val_f1 = f1_score(val_targets, val_pred_binary)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val F1: {val_f1:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 50)
        
        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, '../data/baseline7_output/best_model.pth') # 路径已修改
            print(f'New best model saved with F1: {best_f1:.4f}')
        
        scheduler.step(val_f1)
    
    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_f1

def predict_and_submit(model, test_loader, test_df, device):
    """预测并生成提交文件"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for sequence, categorical, numerical in test_loader:
            sequence = sequence.to(device)
            categorical = categorical.to(device)
            numerical = numerical.to(device)
            
            outputs = model(sequence, categorical, numerical).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            predictions.extend(probs)
    
    # 生成提交文件
    submission_df = pd.DataFrame({
        'did': test_df['did'].values,
        'is_new_did': (np.array(predictions) > 0.5).astype(int)
    })
    
    # 确保did顺序与原始测试文件一致
    original_order = test_df['did'].tolist()
    submission_df = submission_df.set_index('did').reindex(original_order).reset_index()
    
    submission_df.to_csv('../data/baseline7_output/submission.csv', index=False) # 路径已修改
    print("提交文件已保存到: ../data/baseline7_output/submission.csv") # 路径已修改
    
    return predictions

def main():
    """主函数"""
    print("开始训练混合型Transformer模型...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载训练数据...")
    train_df = pd.read_csv('../data/train_data/train.csv') # 路径已修改
    print(f"训练数据形状: {train_df.shape}")
    
    print("加载测试数据...")
    test_df = pd.read_csv('../data/test_data/testA_data.csv') # 路径已修改
    print(f"测试数据形状: {test_df.shape}")
    
    # 创建训练数据集
    print("创建训练数据集...")
    full_dataset = UserBehaviorDataset(train_df, is_train=True)
    
    # 分割训练和验证集
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), 
        test_size=0.2, 
        random_state=42,
        stratify=train_df['is_new_did']
    )
    
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # 创建数据加载器
    batch_size = 128
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 创建测试数据集
    print("创建测试数据集...")
    test_dataset = UserBehaviorDataset(
        test_df, 
        feature_encoders=full_dataset.feature_encoders,
        scaler=full_dataset.scaler,
        is_train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 创建模型
    print("创建模型...")
    model = HybridTransformerModel(
        vocab_sizes=full_dataset.vocab_sizes,
        categorical_features=full_dataset.categorical_features,
        seq_length=50,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        numerical_dim=len(full_dataset.numerical_features),
        dropout=0.1
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 训练模型
    print("开始训练...")
    trained_model, best_f1 = train_model(
        model, train_loader, val_loader, device, 
        epochs=20, lr=1e-3
    )
    
    print(f"训练完成！最佳验证F1分数: {best_f1:.4f}")
    
    # 预测并生成提交文件
    print("生成预测结果...")
    predictions = predict_and_submit(trained_model, test_loader, test_df, device)
    
    print("所有任务完成！")
    print(f"模型权重已保存到: ../data/baseline7_output/best_model.pth") # 路径已修改
    print(f"提交文件已保存到: ../data/baseline7_output/submission.csv") # 路径已修改

if __name__ == "__main__":
    main()