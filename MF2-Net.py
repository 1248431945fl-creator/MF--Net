import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from scipy.io import loadmat
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


# ==================== 数据预处理 ====================
def load_data():
    """加载数据"""
    audio_data = loadmat('./audio_features.mat')
    train_data_ = np.load('./MODMApcc.npy')

    # 假设train_data_是EEG数据，形状为[n_samples, 128, 128]
    # 假设audio_data['stft_data']是语音数据，形状为[n_samples, 64, 640]
    eeg_data = train_data_
    audio_features = audio_data['stft_data']
    labels = audio_data['label'].T

    # 确保维度一致
    n_samples = min(len(eeg_data), len(audio_features))
    eeg_data = eeg_data[:n_samples]
    audio_features = audio_features[:n_samples]
    labels = labels[:n_samples].flatten()

    print(f"Data loaded: EEG shape: {eeg_data.shape}, Audio shape: {audio_features.shape}, Labels: {labels.shape}")

    return eeg_data, audio_features, labels


def preprocess_data(eeg_data, audio_features, labels, num_folds=4):
    """数据预处理和划分"""
    # 归一化
    eeg_data = (eeg_data - eeg_data.mean()) / (eeg_data.std() + 1e-8)
    audio_features = (audio_features - audio_features.mean()) / (audio_features.std() + 1e-8)

    # 创建折叠划分
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    folds = []

    for train_idx, test_idx in kf.split(eeg_data):
        fold = {
            'train': {
                'eeg': eeg_data[train_idx],
                'audio': audio_features[train_idx],
                'labels': labels[train_idx]
            },
            'test': {
                'eeg': eeg_data[test_idx],
                'audio': audio_features[test_idx],
                'labels': labels[test_idx]
            }
        }
        folds.append(fold)

    return folds


# ==================== 模型组件 ====================
class BaseLearner(nn.Module):
    """基学习器（每个模态一个）"""

    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super(BaseLearner, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class FeatureExtractor(nn.Module):
    """特征提取器"""

    def __init__(self, modality_type='eeg'):
        super(FeatureExtractor, self).__init__()
        self.modality_type = modality_type

        if modality_type == 'eeg':
            # EEG特征提取器 - 简化版本
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            # 输入128x128，经过2次pooling后为32x32
            self.fc1 = nn.Linear(16 * 32 * 32, 128)
        elif modality_type == 'audio':
            # 语音特征提取器 - 简化版本
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            # 64x640经过3次pooling后为8x80
            self.fc1 = nn.Linear(32 * 8 * 80, 128)

        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # 添加通道维度

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        if self.modality_type == 'audio':
            x = F.relu(self.conv3(x))
            x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class FuzzyMeasureLearner(nn.Module):
    """模糊度量学习器 - 自适应版本"""

    def __init__(self, num_base_learners=2, num_classes=2):
        super(FuzzyMeasureLearner, self).__init__()
        self.num_base_learners = num_base_learners
        self.num_classes = num_classes

        # 动态确定输入维度
        # 输入维度 = num_base_learners * (num_classes + 1) = 2 * (2 + 1) = 6
        input_dim = num_base_learners * (num_classes + 1)

        # 简单MLP学习模糊度量
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_base_learners * num_classes)
        )

    def forward(self, x):
        output = self.mlp(x)
        output = output.view(-1, self.num_base_learners, self.num_classes)
        output = torch.sigmoid(output)  # 确保在0-1之间
        return output


# ==================== MF²-Net 主模型 ====================
class MF2Net(nn.Module):
    def __init__(self, num_classes=2):
        super(MF2Net, self).__init__()

        # 特征提取器
        self.eeg_extractor = FeatureExtractor('eeg')
        self.audio_extractor = FeatureExtractor('audio')

        # 基学习器
        self.eeg_learner = BaseLearner(64, num_classes=num_classes)
        self.audio_learner = BaseLearner(64, num_classes=num_classes)

        # 模糊度量学习器
        self.fuzzy_learner = FuzzyMeasureLearner(
            num_base_learners=2,
            num_classes=num_classes
        )

        self.num_classes = num_classes

    def compute_simple_features(self, predictions, labels):
        """计算简单的特征 - 修复维度问题"""
        batch_size = predictions.shape[0]
        num_learners = predictions.shape[1]

        features = []
        for b in range(batch_size):
            sample_features = []
            for l in range(num_learners):
                # 预测类别
                pred_class = torch.argmax(predictions[b, l])
                correct = (pred_class == labels[b]).float()

                # 特征：正确与否 + 两个类别的概率
                probs = predictions[b, l]  # [num_classes]

                # 拼接特征：正确与否 + 概率
                feature = torch.cat([correct.unsqueeze(0), probs])
                sample_features.append(feature)

            # 拼接两个基学习器的特征
            # 每个基学习器特征维度 = 1 (正确与否) + num_classes (概率) = 3
            # 两个基学习器总维度 = 2 * 3 = 6
            sample_features = torch.cat(sample_features)
            features.append(sample_features)

        features = torch.stack(features)  # [batch, 6]
        return features

    def forward(self, eeg_data, audio_data, labels=None, mode='train'):
        # 特征提取
        eeg_features = self.eeg_extractor(eeg_data)
        audio_features = self.audio_extractor(audio_data)

        # 基学习器预测
        eeg_pred = self.eeg_learner(eeg_features)
        audio_pred = self.audio_learner(audio_features)

        # 堆叠预测结果 [batch, num_learners, num_classes]
        predictions = torch.stack([eeg_pred, audio_pred], dim=1)

        if mode == 'train' and labels is not None:
            # 计算特征
            features = self.compute_simple_features(predictions, labels)

            # 检查特征维度
            # print(f"Features shape: {features.shape}")  # 调试用

            # 学习模糊度量
            fuzzy_measures = self.fuzzy_learner(features)  # [batch, 2, 2]

            # 简单加权融合（使用学习到的模糊度量作为权重）
            # 取每个基学习器在所有类别上的平均权重
            weights = fuzzy_measures.mean(dim=2, keepdim=True)  # [batch, 2, 1]
            weights = F.softmax(weights, dim=1)

            final_predictions = torch.sum(predictions * weights, dim=1)

            return final_predictions, fuzzy_measures

        else:
            # 测试模式：简单平均融合
            final_predictions = torch.mean(predictions, dim=1)
            return final_predictions


# ==================== 训练和评估 ====================
def train_model(model, train_data, val_data, epochs=30, lr=0.001):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 准备训练数据
    train_eeg = torch.tensor(train_data['eeg'], dtype=torch.float32)
    train_audio = torch.tensor(train_data['audio'], dtype=torch.float32)
    train_labels = torch.tensor(train_data['labels'], dtype=torch.long)

    # 准备验证数据
    val_eeg = torch.tensor(val_data['eeg'], dtype=torch.float32)
    val_audio = torch.tensor(val_data['audio'], dtype=torch.float32)
    val_labels = val_data['labels']

    best_acc = 0
    train_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()

        # 随机打乱训练数据
        indices = torch.randperm(len(train_labels))
        batch_size = min(16, len(indices))
        total_loss = 0
        num_batches = 0

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:min(i + batch_size, len(indices))]

            batch_eeg = train_eeg[batch_indices].to(device)
            batch_audio = train_audio[batch_indices].to(device)
            batch_labels = train_labels[batch_indices].to(device)

            # 前向传播
            predictions, _ = model(batch_eeg, batch_audio, batch_labels, mode='train')

            # 计算损失
            loss = F.cross_entropy(predictions, batch_labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_loss)
        scheduler.step()

        # 验证
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                # 分批验证
                val_predictions = []
                val_batch_size = min(32, len(val_eeg))

                for i in range(0, len(val_eeg), val_batch_size):
                    batch_eeg = val_eeg[i:i + val_batch_size].to(device)
                    batch_audio = val_audio[i:i + val_batch_size].to(device)

                    predictions = model(batch_eeg, batch_audio, mode='eval')
                    val_predictions.append(predictions.cpu())

                if val_predictions:
                    val_predictions = torch.cat(val_predictions, dim=0)
                    pred_classes = torch.argmax(val_predictions, dim=1).numpy()

                    accuracy = accuracy_score(val_labels, pred_classes)
                    val_accs.append(accuracy)

                    if accuracy > best_acc:
                        best_acc = accuracy
                        torch.save(model.state_dict(), 'best_mf2net.pth')

                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {accuracy:.4f}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, No validation data")

    return model, train_losses, val_accs


def evaluate_model(model, test_data):
    """评估模型"""
    device = next(model.parameters()).device

    model.eval()

    # 准备数据
    test_eeg = torch.tensor(test_data['eeg'], dtype=torch.float32)
    test_audio = torch.tensor(test_data['audio'], dtype=torch.float32)
    test_labels = test_data['labels']

    with torch.no_grad():
        # 分批预测
        test_predictions = []
        batch_size = min(32, len(test_eeg))

        for i in range(0, len(test_eeg), batch_size):
            batch_eeg = test_eeg[i:i + batch_size].to(device)
            batch_audio = test_audio[i:i + batch_size].to(device)

            predictions = model(batch_eeg, batch_audio, mode='eval')
            test_predictions.append(predictions.cpu())

        if test_predictions:
            test_predictions = torch.cat(test_predictions, dim=0)
            pred_classes = torch.argmax(test_predictions, dim=1).numpy()
        else:
            pred_classes = np.array([])

    if len(pred_classes) > 0 and len(test_labels) > 0:
        accuracy = accuracy_score(test_labels, pred_classes)
        recall = recall_score(test_labels, pred_classes, average='binary', zero_division=0)
        precision = precision_score(test_labels, pred_classes, average='binary', zero_division=0)
        f1 = f1_score(test_labels, pred_classes, average='binary', zero_division=0)

        print(f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")

        # 混淆矩阵
        cm = confusion_matrix(test_labels, pred_classes)
        print("Confusion Matrix:")
        print(cm)

        return accuracy, cm
    else:
        print("No predictions or labels to evaluate")
        return 0.0, np.zeros((2, 2))


# ==================== 简化的元学习 ====================
def create_few_shot_tasks(data, num_tasks=5, num_shots=5):
    """创建小样本任务"""
    tasks = []

    # 获取标签
    labels = data['labels']

    # 分离正负样本索引
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]

    for _ in range(num_tasks):
        # 从每个类别中随机选择样本
        if len(positive_indices) >= 2 * num_shots and len(negative_indices) >= 2 * num_shots:
            pos_selected = np.random.choice(positive_indices, 2 * num_shots, replace=False)
            neg_selected = np.random.choice(negative_indices, 2 * num_shots, replace=False)

            # 前num_shots作为支持集，后num_shots作为查询集
            support_indices = np.concatenate([pos_selected[:num_shots], neg_selected[:num_shots]])
            query_indices = np.concatenate([pos_selected[num_shots:], neg_selected[num_shots:]])

            def extract_batch(idx):
                return {
                    'eeg': torch.tensor(data['eeg'][idx], dtype=torch.float32),
                    'audio': torch.tensor(data['audio'][idx], dtype=torch.float32),
                    'labels': torch.tensor(data['labels'][idx], dtype=torch.long)
                }

            tasks.append({
                'support': extract_batch(support_indices),
                'query': extract_batch(query_indices)
            })

    return tasks


def meta_train_step(model, tasks, device):
    """执行一步元训练"""
    if not tasks:
        return 0.0

    # 保存原始参数
    original_state = {k: v.clone() for k, v in model.fuzzy_learner.state_dict().items()}

    task_losses = []

    for task in tasks:
        # 创建模型副本
        fast_model = copy.deepcopy(model)
        fast_model.to(device)

        # 将任务数据移动到设备
        support_data = {k: v.to(device) for k, v in task['support'].items()}

        # 检查支持集是否有足够的数据
        if len(support_data['labels']) == 0:
            continue

        # 在支持集上训练
        fast_model.train()
        inner_optimizer = torch.optim.SGD(fast_model.fuzzy_learner.parameters(), lr=0.01)

        for _ in range(3):
            predictions, _ = fast_model(
                support_data['eeg'],
                support_data['audio'],
                support_data['labels'],
                mode='train'
            )
            loss = F.cross_entropy(predictions, support_data['labels'])

            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        # 在查询集上评估
        fast_model.eval()
        query_data = {k: v.to(device) for k, v in task['query'].items()}

        if len(query_data['labels']) == 0:
            continue

        with torch.no_grad():
            predictions, _ = fast_model(
                query_data['eeg'],
                query_data['audio'],
                query_data['labels'],
                mode='train'
            )
            task_loss = F.cross_entropy(predictions, query_data['labels'])
            task_losses.append(task_loss.item())

    # 恢复原始参数
    model.fuzzy_learner.load_state_dict(original_state)

    return np.mean(task_losses) if task_losses else 0.0


# ==================== 主函数 ====================
def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载数据
    print("Loading data...")
    eeg_data, audio_features, labels = load_data()

    # 预处理和划分
    folds = preprocess_data(eeg_data, audio_features, labels, num_folds=4)

    # 交叉验证
    fold_accuracies = []

    for fold_idx, fold in enumerate(folds[:2]):  # 只运行前两个折叠以节省时间
        print(f"\n{'=' * 50}")
        print(f"Fold {fold_idx + 1}/{len(folds[:2])}")
        print('=' * 50)

        # 创建模型
        model = MF2Net(num_classes=2)

        # 获取设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model = model.to(device)

        # 元训练
        print("Meta-training...")
        tasks = create_few_shot_tasks(fold['train'], num_tasks=3, num_shots=3)

        if tasks:
            for meta_epoch in range(5):
                meta_loss = meta_train_step(model, tasks, device)
                if (meta_epoch + 1) % 2 == 0:
                    print(f"Meta-epoch {meta_epoch + 1}, Loss: {meta_loss:.4f}")
        else:
            print("Not enough data for meta-training")

        # 常规训练
        print("\nRegular training...")
        trained_model, train_losses, val_accs = train_model(
            model, fold['train'], fold['test'], epochs=20, lr=0.001
        )

        # 最终评估
        print("\nFinal evaluation:")
        accuracy, cm = evaluate_model(trained_model, fold['test'])
        fold_accuracies.append(accuracy)

    # 打印平均性能
    if fold_accuracies:
        print(f"\n{'=' * 50}")
        print(f"Cross-validation results:")
        print(f"Average Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
        print(f"Fold Accuracies: {fold_accuracies}")

    # 可视化训练过程
    if fold_accuracies and 'train_losses' in locals():
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        if val_accs:
            epochs = range(5, len(train_losses) + 1, 5)
            if len(epochs) > len(val_accs):
                epochs = epochs[:len(val_accs)]
            elif len(epochs) < len(val_accs):
                val_accs = val_accs[:len(epochs)]
            plt.plot(epochs, val_accs)
            plt.title('Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')

        plt.tight_layout()
        plt.savefig('training_curve.png', dpi=300)
        plt.show()


if __name__ == "__main__":
    main()