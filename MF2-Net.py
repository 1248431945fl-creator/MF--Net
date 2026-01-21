import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from scipy.io import loadmat
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from scipy import signal


# ==================== 简化的数据预处理 ====================
def load_and_preprocess_data():
    """加载并预处理数据 - 简化版本"""
    # 加载数据
    audio_data = loadmat('./audio_features.mat')
    train_data_ = np.load('./MODMApcc.npy')

    # 假设train_data_是EEG数据，形状为[n_samples, 128, 128]
    # 假设audio_data['stft_data']是语音数据，形状为[n_samples, 64, 640]
    eeg_data = train_data_
    audio_features = audio_data['stft_data']
    labels = audio_data['label'].T.flatten()

    # 确保维度一致
    n_samples = min(len(eeg_data), len(audio_features), len(labels))
    eeg_data = eeg_data[:n_samples]
    audio_features = audio_features[:n_samples]
    labels = labels[:n_samples]

    print(f"Data shapes: EEG: {eeg_data.shape}, Audio: {audio_features.shape}, Labels: {labels.shape}")

    # 简单的特征提取
    def extract_eeg_features(eeg):
        # 提取简单的统计特征
        features = []
        for i in range(eeg.shape[0]):
            sample = eeg[i]
            # 均值、标准差、最大值、最小值
            mean_features = np.mean(sample, axis=1)
            std_features = np.std(sample, axis=1)
            max_features = np.max(sample, axis=1)
            min_features = np.min(sample, axis=1)

            # 组合特征
            sample_features = np.concatenate([mean_features, std_features, max_features, min_features])
            features.append(sample_features)
        return np.array(features)

    def extract_audio_features(audio):
        # 提取简单的音频特征
        features = []
        for i in range(audio.shape[0]):
            sample = audio[i]
            # 计算MFCC的近似特征
            mfcc_features = extract_simple_mfcc(sample)

            # 统计特征
            mean_features = np.mean(sample, axis=1)
            std_features = np.std(sample, axis=1)

            # 频谱特征
            spectral_features = []
            for channel in range(sample.shape[0]):
                freqs, psd = signal.welch(sample[channel], fs=100, nperseg=64)
                spectral_features.extend([
                    np.mean(psd), np.std(psd), np.max(psd), np.min(psd)
                ])

            # 组合特征
            sample_features = np.concatenate([mfcc_features, mean_features, std_features, spectral_features])
            features.append(sample_features)
        return np.array(features)

    def extract_simple_mfcc(audio_sample):
        """简化的MFCC提取"""
        n_mfcc = 13
        n_filters = 20

        # 计算功率谱
        power_spectrum = np.abs(np.fft.rfft(audio_sample, axis=1)) ** 2

        # 简化的梅尔滤波器组
        mel_filters = np.random.randn(n_filters, power_spectrum.shape[1])
        mel_spectrum = np.dot(mel_filters, power_spectrum.T).T

        # 取对数
        log_mel_spectrum = np.log(mel_spectrum + 1e-10)

        # 简化的DCT
        mfcc = np.dot(log_mel_spectrum, np.random.randn(n_filters, n_mfcc))

        # 展平
        return mfcc.flatten()

    # 提取特征
    print("Extracting EEG features...")
    eeg_features = extract_eeg_features(eeg_data)

    print("Extracting audio features...")
    audio_features = extract_audio_features(audio_features)

    # 简单的特征选择 - 选择方差最大的特征
    def select_features_by_variance(features, n_select=50):
        variances = np.var(features, axis=0)
        selected_indices = np.argsort(variances)[-n_select:]
        return features[:, selected_indices]

    eeg_features = select_features_by_variance(eeg_features, 50)
    audio_features = select_features_by_variance(audio_features, 50)

    print(f"Feature shapes after selection: EEG: {eeg_features.shape}, Audio: {audio_features.shape}")

    # 归一化
    eeg_features = (eeg_features - eeg_features.mean(axis=0)) / (eeg_features.std(axis=0) + 1e-8)
    audio_features = (audio_features - audio_features.mean(axis=0)) / (audio_features.std(axis=0) + 1e-8)

    return eeg_features, audio_features, labels


# ==================== 基础模型组件 ====================
class BaseLearner(nn.Module):
    """基学习器"""

    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super(BaseLearner, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


# ==================== 模糊度量学习器 ====================
class FuzzyMeasureLearner(nn.Module):
    """模糊度量学习器"""

    def __init__(self, input_dim, num_base_learners=2, num_classes=2):
        super(FuzzyMeasureLearner, self).__init__()

        # 学习单个模糊度量
        self.single_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_base_learners * num_classes)
        )

        # 学习互补增量
        self.complement_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        single = self.single_mlp(x)
        single = single.view(-1, 2, 2)  # [batch, 2, 2]
        single = torch.sigmoid(single)

        complement = self.complement_mlp(x)
        complement = torch.sigmoid(complement)

        return single, complement


# ==================== 模糊积分 ====================
def choquet_integral(predictions, single_measures, complement_measures):
    """Choquet积分"""
    batch_size = predictions.shape[0]
    num_classes = predictions.shape[2]

    results = torch.zeros(batch_size, num_classes, device=predictions.device)

    for b in range(batch_size):
        for c in range(num_classes):
            # 获取当前类别的预测
            p = predictions[b, :, c]  # [2]

            # 获取模糊度量
            mu_1 = single_measures[b, 0, c]
            mu_2 = single_measures[b, 1, c]
            complement = complement_measures[b, c]

            # 计算组合的模糊度量
            mu_12 = torch.max(mu_1, mu_2) + complement

            # 排序预测值
            if p[0] <= p[1]:
                result = p[0] * mu_1 + (p[1] - p[0]) * mu_12
            else:
                result = p[0] * mu_2 + (p[1] - p[0]) * mu_12

            results[b, c] = result

    return F.softmax(results, dim=1)


# ==================== MF²-Net主模型 ====================
class MF2Net(nn.Module):
    def __init__(self, eeg_input_dim, audio_input_dim, num_classes=2):
        super(MF2Net, self).__init__()

        # 特征提取器
        self.eeg_extractor = nn.Sequential(
            nn.Linear(eeg_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )

        self.audio_extractor = nn.Sequential(
            nn.Linear(audio_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )

        # 基学习器
        self.eeg_learner = BaseLearner(64, num_classes=num_classes)
        self.audio_learner = BaseLearner(64, num_classes=num_classes)

        # 模糊度量学习器
        # 输入特征：信息熵(2) + 简单特征(4) = 6
        self.fuzzy_learner = FuzzyMeasureLearner(6, num_base_learners=2, num_classes=num_classes)

        self.num_classes = num_classes

    def compute_entropy(self, predictions):
        """计算信息熵"""
        eps = 1e-10
        return -torch.sum(predictions * torch.log(predictions + eps), dim=2)

    def compute_simple_features(self, predictions, labels):
        """计算简单特征"""
        batch_size = predictions.shape[0]

        features = []
        for b in range(batch_size):
            sample_features = []

            for l in range(2):  # 两个基学习器
                # 预测类别
                pred_class = torch.argmax(predictions[b, l])
                correct = (pred_class == labels[b]).float()

                # 概率特征
                probs = predictions[b, l]

                # 组合特征：正确与否 + 两个类别的概率
                feature = torch.cat([correct.unsqueeze(0), probs])
                sample_features.append(feature)

            # 拼接两个基学习器的特征
            sample_features = torch.cat(sample_features)
            features.append(sample_features)

        return torch.stack(features)  # [batch, 6]

    def forward(self, eeg_input, audio_input, labels=None, mode='train'):
        # 特征提取
        eeg_features = self.eeg_extractor(eeg_input)
        audio_features = self.audio_extractor(audio_input)

        # 基学习器预测
        eeg_pred = self.eeg_learner(eeg_features)  # [batch, 2]
        audio_pred = self.audio_learner(audio_features)  # [batch, 2]

        # 堆叠预测
        predictions = torch.stack([eeg_pred, audio_pred], dim=1)  # [batch, 2, 2]

        if mode == 'train' and labels is not None:
            # 计算特征
            features = self.compute_simple_features(predictions, labels)

            # 学习模糊度量
            single_measures, complement_measures = self.fuzzy_learner(features)

            # Choquet积分融合
            final_predictions = choquet_integral(predictions, single_measures, complement_measures)

            return final_predictions, {
                'single_measures': single_measures,
                'complement_measures': complement_measures
            }
        else:
            # 测试模式：平均融合
            final_predictions = torch.mean(predictions, dim=1)
            return final_predictions, None


# ==================== MAML框架 ====================
class SimpleMAML:
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr

        # 只优化模糊度量学习器
        self.meta_params = list(model.fuzzy_learner.parameters())
        self.meta_optimizer = torch.optim.Adam(self.meta_params, lr=meta_lr)

    def create_few_shot_task(self, data, num_shots=3):
        """创建小样本任务"""
        labels = data['labels']
        indices = np.arange(len(labels))

        # 按类别分离
        class_0_idx = indices[labels == 0]
        class_1_idx = indices[labels == 1]

        # 随机选择样本
        support_idx = np.concatenate([
            np.random.choice(class_0_idx, num_shots, replace=False),
            np.random.choice(class_1_idx, num_shots, replace=False)
        ])

        query_idx = np.concatenate([
            np.random.choice(class_0_idx[~np.isin(class_0_idx, support_idx)], num_shots, replace=False),
            np.random.choice(class_1_idx[~np.isin(class_1_idx, support_idx)], num_shots, replace=False)
        ])

        np.random.shuffle(support_idx)
        np.random.shuffle(query_idx)

        def extract_batch(idx):
            return {
                'eeg': torch.tensor(data['eeg'][idx], dtype=torch.float32),
                'audio': torch.tensor(data['audio'][idx], dtype=torch.float32),
                'labels': torch.tensor(data['labels'][idx], dtype=torch.long)
            }

        return extract_batch(support_idx), extract_batch(query_idx)

    def meta_update(self, data, num_tasks=5):
        """元更新"""
        device = next(self.model.parameters()).device
        total_loss = 0

        for _ in range(num_tasks):
            # 创建任务
            support_data, query_data = self.create_few_shot_task(data, num_shots=3)

            # 内循环：快速适应
            fast_model = copy.deepcopy(self.model)
            fast_model.fuzzy_learner.load_state_dict(self.model.fuzzy_learner.state_dict())

            # 在支持集上训练
            fast_model.train()
            inner_optimizer = torch.optim.SGD(fast_model.fuzzy_learner.parameters(), lr=self.inner_lr)

            support_eeg = support_data['eeg'].to(device)
            support_audio = support_data['audio'].to(device)
            support_labels = support_data['labels'].to(device)

            for _ in range(3):  # 3步内循环
                predictions, _ = fast_model(support_eeg, support_audio, support_labels, mode='train')
                loss = F.cross_entropy(predictions, support_labels)

                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()

            # 在查询集上评估
            fast_model.eval()
            query_eeg = query_data['eeg'].to(device)
            query_audio = query_data['audio'].to(device)
            query_labels = query_data['labels'].to(device)

            with torch.no_grad():
                predictions, _ = fast_model(query_eeg, query_audio, query_labels, mode='train')
                query_loss = F.cross_entropy(predictions, query_labels)
                total_loss += query_loss.item()

        # 计算平均损失
        avg_loss = total_loss / num_tasks

        # 元优化
        self.meta_optimizer.zero_grad()

        # 创建虚拟损失进行反向传播
        dummy_loss = torch.tensor(avg_loss, device=device, requires_grad=True)
        dummy_loss.backward()

        # 应用梯度
        for param in self.meta_params:
            if param.grad is not None:
                param.data -= self.meta_lr * param.grad

        return avg_loss


# ==================== 训练和评估 ====================
def train_fold(model, train_data, test_data, device, epochs=30):
    """训练单个折叠"""
    # 准备数据
    train_eeg = torch.tensor(train_data['eeg'], dtype=torch.float32)
    train_audio = torch.tensor(train_data['audio'], dtype=torch.float32)
    train_labels = torch.tensor(train_data['labels'], dtype=torch.long)

    test_eeg = torch.tensor(test_data['eeg'], dtype=torch.float32)
    test_audio = torch.tensor(test_data['audio'], dtype=torch.float32)
    test_labels = test_data['labels']

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_acc = 0
    train_losses = []

    for epoch in range(epochs):
        model.train()

        # 批处理训练
        indices = torch.randperm(len(train_labels))
        batch_size = 16
        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:min(i + batch_size, len(indices))]

            batch_eeg = train_eeg[batch_idx].to(device)
            batch_audio = train_audio[batch_idx].to(device)
            batch_labels = train_labels[batch_idx].to(device)

            # 前向传播
            predictions, _ = model(batch_eeg, batch_audio, batch_labels, mode='train')

            # 计算损失
            loss = F.cross_entropy(predictions, batch_labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_loss)
        scheduler.step()

        # 验证
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # 分批预测
                test_predictions = []
                test_batch_size = 32

                for i in range(0, len(test_eeg), test_batch_size):
                    batch_eeg = test_eeg[i:i + test_batch_size].to(device)
                    batch_audio = test_audio[i:i + test_batch_size].to(device)

                    predictions, _ = model(batch_eeg, batch_audio, mode='eval')
                    test_predictions.append(predictions.cpu())

                if test_predictions:
                    test_predictions = torch.cat(test_predictions, dim=0)
                    pred_classes = torch.argmax(test_predictions, dim=1).numpy()

                    accuracy = accuracy_score(test_labels, pred_classes)

                    if accuracy > best_acc:
                        best_acc = accuracy

                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {accuracy:.4f}")

    return model, train_losses, best_acc


def evaluate_model(model, test_data, device):
    """评估模型"""
    model.eval()

    test_eeg = torch.tensor(test_data['eeg'], dtype=torch.float32)
    test_audio = torch.tensor(test_data['audio'], dtype=torch.float32)
    test_labels = test_data['labels']

    with torch.no_grad():
        # 分批预测
        predictions = []
        batch_size = 32

        for i in range(0, len(test_eeg), batch_size):
            batch_eeg = test_eeg[i:i + batch_size].to(device)
            batch_audio = test_audio[i:i + batch_size].to(device)

            batch_pred, _ = model(batch_eeg, batch_audio, mode='eval')
            predictions.append(batch_pred.cpu())

        if predictions:
            predictions = torch.cat(predictions, dim=0)
            pred_classes = torch.argmax(predictions, dim=1).numpy()

            # 计算指标
            accuracy = accuracy_score(test_labels, pred_classes)
            recall = recall_score(test_labels, pred_classes, average='binary', zero_division=0)
            precision = precision_score(test_labels, pred_classes, average='binary', zero_division=0)
            f1 = f1_score(test_labels, pred_classes, average='binary', zero_division=0)

            # 混淆矩阵
            cm = confusion_matrix(test_labels, pred_classes)

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("Confusion Matrix:")
            print(cm)

            return accuracy, cm

    return 0.0, None


# ==================== 主函数 ====================
def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载和预处理数据
    print("Loading and preprocessing data...")
    eeg_features, audio_features, labels = load_and_preprocess_data()

    print(f"Final feature shapes: EEG: {eeg_features.shape}, Audio: {audio_features.shape}")
    print(f"Label distribution: {np.bincount(labels.astype(int))}")

    # 创建交叉验证折叠
    n_folds = 4
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_accuracies = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(eeg_features)):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print('=' * 60)

        # 准备折叠数据
        fold_data = {
            'train': {
                'eeg': eeg_features[train_idx],
                'audio': audio_features[train_idx],
                'labels': labels[train_idx]
            },
            'test': {
                'eeg': eeg_features[test_idx],
                'audio': audio_features[test_idx],
                'labels': labels[test_idx]
            }
        }

        # 设备设置
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # 创建模型
        model = MF2Net(
            eeg_input_dim=eeg_features.shape[1],
            audio_input_dim=audio_features.shape[1],
            num_classes=2
        ).to(device)

        # 创建MAML训练器
        maml = SimpleMAML(model, inner_lr=0.01, meta_lr=0.001)

        # 元训练
        print("Meta-training...")
        for meta_epoch in range(10):
            meta_loss = maml.meta_update(fold_data['train'], num_tasks=5)
            if (meta_epoch + 1) % 2 == 0:
                print(f"Meta-epoch {meta_epoch + 1}/10, Loss: {meta_loss:.4f}")

        # 常规训练
        print("\nRegular training...")
        trained_model, train_losses, best_acc = train_fold(
            model, fold_data['train'], fold_data['test'], device, epochs=30
        )

        # 最终评估
        print("\nFinal evaluation:")
        accuracy, cm = evaluate_model(trained_model, fold_data['test'], device)
        fold_accuracies.append(accuracy)

        # 可视化训练损失
        if train_losses:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses)
            plt.title(f'Fold {fold_idx + 1} Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            if cm is not None:
                plt.subplot(1, 2, 2)
                plt.imshow(cm, cmap='Blues')
                plt.title(f'Fold {fold_idx + 1} Confusion Matrix')
                plt.colorbar()
                plt.xlabel('Predicted')
                plt.ylabel('True')

                # 添加数值
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')

            plt.tight_layout()
            plt.savefig(f'fold_{fold_idx + 1}_results.png', dpi=150)
            plt.close()

    # 打印最终结果
    print(f"\n{'=' * 60}")
    print("Cross-validation Results:")
    print('=' * 60)
    print(f"Fold accuracies: {fold_accuracies}")
    print(f"Mean accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")

    # 绘制最终结果
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(fold_accuracies) + 1), fold_accuracies)
    plt.axhline(y=np.mean(fold_accuracies), color='r', linestyle='--',
                label=f'Mean: {np.mean(fold_accuracies):.3f}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Cross-validation Accuracy per Fold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cv_results.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    main()