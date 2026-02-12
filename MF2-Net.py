import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy import signal
import warnings

warnings.filterwarnings('ignore')

# ==================== 设备设置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== fMRI 特征提取（基于功能连接矩阵） ====================
def extract_fmri_features(fmri_data):
    """
    输入: fmri_data 形状 (n_samples, 90, 90) —— 功能连接矩阵
    输出: 特征矩阵 (n_samples, n_features)
    对每个脑区（90个）提取：
        - 该脑区与其他脑区连接强度的均值、标准差、最大值、最小值
    额外全局特征：
        - 全矩阵均值、标准差、最大值、最小值
        - 网络密度（阈值>0.3 的边比例）
    """
    n_samples = fmri_data.shape[0]
    n_rois = fmri_data.shape[1]  # 90
    features = []

    for i in range(n_samples):
        mat = fmri_data[i]  # (90,90)
        # 每个ROI的特征
        roi_feats = []
        for r in range(n_rois):
            row = mat[r, :]
            roi_feats.extend([
                np.mean(row),
                np.std(row),
                np.max(row),
                np.min(row)
            ])
        # 全局特征
        triu_vals = mat[np.triu_indices_from(mat, k=1)]  # 上三角元素（无对角线）
        global_mean = np.mean(triu_vals)
        global_std = np.std(triu_vals)
        global_max = np.max(triu_vals)
        global_min = np.min(triu_vals)
        # 网络密度：连接强度 > 0.3 的比例（可根据数据调整阈值）
        density = np.sum(triu_vals > 0.3) / len(triu_vals) if len(triu_vals) > 0 else 0

        sample_feats = roi_feats + [global_mean, global_std, global_max, global_min, density]
        features.append(sample_feats)

    return np.array(features)


# ==================== 语音特征提取（基于Mel谱图） ====================
def extract_speech_features(speech_data):
    """
    输入: speech_data 形状 (n_samples, 64, 640) —— Mel谱图 (频带×时间)
    输出: 特征矩阵 (n_samples, n_features)
    对每个频带（64个）提取时间维度的：
        - 均值、标准差、最大值、最小值
    额外全局特征：
        - 所有频带所有帧的均值、标准差
        - 谱通量（平均帧间差异）
    """
    n_samples = speech_data.shape[0]
    n_mels = speech_data.shape[1]  # 64
    n_time = speech_data.shape[2]  # 640
    features = []

    for i in range(n_samples):
        spec = speech_data[i]  # (64,640)
        # 每个频带的统计量
        mel_feats = []
        for m in range(n_mels):
            band = spec[m, :]
            mel_feats.extend([
                np.mean(band),
                np.std(band),
                np.max(band),
                np.min(band)
            ])
        # 全局统计
        flat_spec = spec.flatten()
        global_mean = np.mean(flat_spec)
        global_std = np.std(flat_spec)
        # 谱通量：相邻帧差分的绝对值平均
        flux = np.mean(np.abs(np.diff(spec, axis=1)))

        sample_feats = mel_feats + [global_mean, global_std, flux]
        features.append(sample_feats)

    return np.array(features)


# ==================== 数据加载与完整预处理 ====================
def load_and_preprocess_data():
    print("=" * 60)
    print("数据加载与预处理")
    print("=" * 60)

    # ----- 1. fMRI 数据：1015,90,90 → 2030,90,90 -----
    fmri_flat = np.load('./fMri_pcc.npy')  # (1015, 90, 90)
    fmri_5 = fmri_flat.reshape(203, 5, 90, 90)  # (203,5,90,90)
    fmri_10 = np.repeat(fmri_5, repeats=2, axis=1)  # (203,10,90,90)
    fmri_segments = fmri_10.reshape(2030, 90, 90)  # (2030,90,90)
    print(f"[fMRI] {fmri_flat.shape} → {fmri_segments.shape}")

    # ----- 2. 语音数据：203,10,64,640 → 2030,64,640 -----
    audio_10 = np.load('stft_data_fixed_10/all_stfts_64x640.npy')  # (203,10,64,640)
    audio_segments = audio_10.reshape(2030, 64, 640)  # (2030,64,640)
    print(f"[语音] {audio_10.shape} → {audio_segments.shape}")

    # ----- 3. 标签：203 → 2030（每个样本的10段共享同一标签）-----
    path_brain = '/home/idal-01/code/IBGNN-master/datasets/HIV.mat'
    f_brain = loadmat(path_brain)
    labels_203 = f_brain['label'].squeeze()  # (203,)
    labels_2030 = np.repeat(labels_203, repeats=10)  # (2030,)
    print(f"[标签] {labels_203.shape} → {labels_2030.shape}")

    # ----- 4. 特征提取 -----
    print("\n提取 fMRI 特征...")
    fmri_feats = extract_fmri_features(fmri_segments)  # (2030, n_fmri)
    print(f"fMRI 特征维度: {fmri_feats.shape}")

    print("提取 语音 特征...")
    speech_feats = extract_speech_features(audio_segments)  # (2030, n_speech)
    print(f"语音 特征维度: {speech_feats.shape}")

    # ----- 5. 简单特征选择（方差阈值）-----
    def variance_threshold(feats, threshold=0.01):
        var = np.var(feats, axis=0)
        keep = var > threshold
        if np.sum(keep) == 0:
            # 若全部低于阈值，保留方差最大的50个
            keep = np.argsort(var)[-50:]
        return feats[:, keep], keep

    fmri_feats, _ = variance_threshold(fmri_feats, threshold=0.01)
    speech_feats, _ = variance_threshold(speech_feats, threshold=0.01)
    print(f"方差筛选后 fMRI: {fmri_feats.shape}, 语音: {speech_feats.shape}")

    # ----- 6. 标准化（Z-score）-----
    scaler_fmri = StandardScaler()
    scaler_speech = StandardScaler()
    fmri_feats = scaler_fmri.fit_transform(fmri_feats)
    speech_feats = scaler_speech.fit_transform(speech_feats)

    print(f"\n最终特征维度: fMRI {fmri_feats.shape}, 语音 {speech_feats.shape}")
    print(f"标签分布: {np.bincount(labels_2030)}")
    print("=" * 60)

    return fmri_feats, speech_feats, labels_2030


# ==================== 域划分（基于分层K折） ====================
def assign_domains(labels, n_domains=6, min_samples_per_class=6):
    n_samples = len(labels)
    classes = np.unique(labels)
    required_per_class = min_samples_per_class * n_domains
    actual_per_class = [np.sum(labels == c) for c in classes]

    if any(actual < required_per_class for actual in actual_per_class):
        max_domains = min([actual // min_samples_per_class for actual in actual_per_class])
        n_domains = max(1, max_domains)
        print(f"调整域数量为 {n_domains} (每类样本不足)")

    skf = StratifiedKFold(n_splits=n_domains, shuffle=True, random_state=42)
    domains = np.zeros(n_samples, dtype=int)
    for fold, (_, test_idx) in enumerate(skf.split(np.zeros(n_samples), labels)):
        domains[test_idx] = fold
    return domains


# ==================== 任务生成器（支持2模态，2-way） ====================
class TaskGenerator:
    def __init__(self, feat1, feat2, labels, domains, n_way=2, n_shot=3, n_query=3):
        self.feat1 = feat1
        self.feat2 = feat2
        self.labels = labels
        self.domains = domains
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.classes = np.unique(labels)
        assert len(self.classes) == n_way

    def _sample_task_from_domain(self, domain_id):
        idx = np.where(self.domains == domain_id)[0]
        class_idx = []
        for c in self.classes:
            c_idx = idx[self.labels[idx] == c]
            if len(c_idx) < self.n_shot + self.n_query:
                return None
            class_idx.append(c_idx)

        support_idx, query_idx = [], []
        for c in range(self.n_way):
            perm = np.random.permutation(class_idx[c])
            support_idx.extend(perm[:self.n_shot])
            query_idx.extend(perm[self.n_shot:self.n_shot + self.n_query])

        np.random.shuffle(support_idx)
        np.random.shuffle(query_idx)
        return {
            'support': {
                'feat1': self.feat1[support_idx],
                'feat2': self.feat2[support_idx],
                'labels': self.labels[support_idx]
            },
            'query': {
                'feat1': self.feat1[query_idx],
                'feat2': self.feat2[query_idx],
                'labels': self.labels[query_idx]
            }
        }

    def generate_tasks(self, domain_list, n_tasks_per_domain=1):
        tasks = []
        for d in domain_list:
            for _ in range(n_tasks_per_domain):
                task = self._sample_task_from_domain(d)
                if task is not None:
                    tasks.append(task)
        if len(tasks) == 0:
            print(f"警告: 域 {domain_list} 无法生成任何任务，请检查域内样本数。")
        return tasks


# ==================== 基学习器（SVM） ====================
def train_base_learners(support_data):
    svm1 = SVC(probability=True, random_state=42)
    svm2 = SVC(probability=True, random_state=42)
    svm1.fit(support_data['feat1'], support_data['labels'])
    svm2.fit(support_data['feat2'], support_data['labels'])
    prob1 = svm1.predict_proba(support_data['feat1'])
    prob2 = svm2.predict_proba(support_data['feat2'])
    return [svm1, svm2], [prob1, prob2]


def predict_base_learners(base_learners, data):
    probs = []
    for bl, x in zip(base_learners, [data['feat1'], data['feat2']]):
        p = bl.predict_proba(x)
        probs.append(p)
    return probs


# ==================== 模糊度量特征构建 ====================
def compute_confusion_matrix(labels, pred_probs):
    pred_classes = np.argmax(pred_probs, axis=1)
    cm = confusion_matrix(labels, pred_classes, labels=[0, 1])
    return cm.flatten()


def compute_entropy(probs):
    eps = 1e-12
    return -np.sum(probs * np.log2(probs + eps), axis=1)


def build_fuzzy_features(cm1, cm2, ent1, ent2):
    domain_feat = np.concatenate([cm1, cm2])  # 4+4 = 8
    adapt_feat = np.array([ent1, ent2])  # 2
    return np.concatenate([domain_feat, adapt_feat])  # 10


# ==================== 模糊度量学习器 ====================
class FuzzyMeasureLearner(nn.Module):
    def __init__(self, input_dim, num_base=2, num_classes=2):
        super().__init__()
        self.num_base = num_base
        self.num_classes = num_classes
        self.m1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_base * num_classes)
        )
        self.m2 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        single = self.m1(x).view(-1, self.num_base, self.num_classes)
        single = torch.sigmoid(single)
        full_inc = torch.sigmoid(self.m2(x))
        return single, full_inc


# ==================== Choquet积分（二基学习器专用） ====================
def choquet_integral_binary(probs, single_measures, full_increment):
    batch_size = probs.shape[0]
    results = torch.zeros(batch_size, 2, device=probs.device)
    for b in range(batch_size):
        for c in range(2):
            p0 = probs[b, 0, c]
            p1 = probs[b, 1, c]
            mu1 = single_measures[b, 0, c]
            mu2 = single_measures[b, 1, c]
            inc = full_increment[b, c]
            mu12 = torch.max(mu1, mu2) + inc
            mu12 = torch.clamp(mu12, max=1.0)
            if p0 <= p1:
                result = p0 * mu1 + (p1 - p0) * mu12
            else:
                result = p1 * mu2 + (p0 - p1) * mu12
            results[b, c] = result
    return F.softmax(results, dim=1)


# ==================== MF²-Net主模型 ====================
class MF2Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fuzzy_learner = FuzzyMeasureLearner(input_dim, num_base=2, num_classes=2)

    def forward(self, probs, fuzzy_features):
        single, inc = self.fuzzy_learner(fuzzy_features)
        return choquet_integral_binary(probs, single, inc)


# ==================== MAML元学习器 ====================
class MAML:
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    def inner_update(self, support_probs, support_features, support_labels, steps=3, lambda_reg=0.01):
        fast_weights = {name: param.clone().requires_grad_(True)
                        for name, param in self.model.named_parameters()}
        for _ in range(steps):
            pred = self._forward_with_weights(support_probs, support_features, fast_weights)
            loss = F.cross_entropy(pred, support_labels)
            reg_loss = sum(torch.norm(p, p=2) ** 2 for p in fast_weights.values())
            loss += lambda_reg * reg_loss
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True, allow_unused=False)
            fast_weights = {name: (param - self.inner_lr * grad).requires_grad_(True)
                            for (name, param), grad in zip(fast_weights.items(), grads)}
        return fast_weights

    def _forward_with_weights(self, probs, features, weights):
        x = features
        # m1
        h1 = F.linear(x, weights['fuzzy_learner.m1.0.weight'], weights['fuzzy_learner.m1.0.bias'])
        h1 = F.relu(h1)
        h1 = F.linear(h1, weights['fuzzy_learner.m1.2.weight'], weights['fuzzy_learner.m1.2.bias'])
        single = torch.sigmoid(h1).view(-1, 2, 2)
        # m2
        h2 = F.linear(x, weights['fuzzy_learner.m2.0.weight'], weights['fuzzy_learner.m2.0.bias'])
        h2 = F.relu(h2)
        h2 = F.linear(h2, weights['fuzzy_learner.m2.2.weight'], weights['fuzzy_learner.m2.2.bias'])
        inc = torch.sigmoid(h2)
        return choquet_integral_binary(probs, single, inc)

    def meta_train_step(self, task_batch):
        if len(task_batch) == 0:
            return 0.0
        task_losses = []
        for task in task_batch:
            # 数据移至device
            sup_feat1 = torch.tensor(task['support']['feat1'], dtype=torch.float32, device=device)
            sup_feat2 = torch.tensor(task['support']['feat2'], dtype=torch.float32, device=device)
            sup_labels = torch.tensor(task['support']['labels'], dtype=torch.long, device=device)
            qry_feat1 = torch.tensor(task['query']['feat1'], dtype=torch.float32, device=device)
            qry_feat2 = torch.tensor(task['query']['feat2'], dtype=torch.float32, device=device)
            qry_labels = torch.tensor(task['query']['labels'], dtype=torch.long, device=device)

            # 基学习器
            base_learners, _ = train_base_learners(task['support'])
            sup_probs = predict_base_learners(base_learners, task['support'])
            qry_probs = predict_base_learners(base_learners, task['query'])
            sup_probs_t = torch.tensor(np.stack(sup_probs, axis=1), dtype=torch.float32, device=device)
            qry_probs_t = torch.tensor(np.stack(qry_probs, axis=1), dtype=torch.float32, device=device)

            # 模糊度量特征（基于支持集计算域信息）
            cm1 = compute_confusion_matrix(task['support']['labels'], sup_probs[0])
            cm2 = compute_confusion_matrix(task['support']['labels'], sup_probs[1])
            ent1_sup = compute_entropy(sup_probs[0])
            ent2_sup = compute_entropy(sup_probs[1])
            sup_feat = np.array([build_fuzzy_features(cm1, cm2, ent1_sup[i], ent2_sup[i])
                                 for i in range(len(ent1_sup))])
            ent1_qry = compute_entropy(qry_probs[0])
            ent2_qry = compute_entropy(qry_probs[1])
            qry_feat = np.array([build_fuzzy_features(cm1, cm2, ent1_qry[i], ent2_qry[i])
                                 for i in range(len(ent1_qry))])

            sup_feat_t = torch.tensor(sup_feat, dtype=torch.float32, device=device)
            qry_feat_t = torch.tensor(qry_feat, dtype=torch.float32, device=device)

            # 内循环
            fast_weights = self.inner_update(sup_probs_t, sup_feat_t, sup_labels, steps=3, lambda_reg=0.01)

            # 外循环损失
            qry_pred = self._forward_with_weights(qry_probs_t, qry_feat_t, fast_weights)
            loss = F.cross_entropy(qry_pred, qry_labels)
            task_losses.append(loss)

        self.meta_optimizer.zero_grad()
        meta_loss = torch.stack(task_losses).sum()
        meta_loss.backward()
        self.meta_optimizer.step()
        return meta_loss.item() / len(task_batch)


# ==================== 元测试 ====================
def meta_test(model, task, inner_lr=0.01, fine_tune_steps=5, lambda_reg=0.01):
    sup_feat1 = torch.tensor(task['support']['feat1'], dtype=torch.float32, device=device)
    sup_feat2 = torch.tensor(task['support']['feat2'], dtype=torch.float32, device=device)
    sup_labels = torch.tensor(task['support']['labels'], dtype=torch.long, device=device)
    qry_feat1 = torch.tensor(task['query']['feat1'], dtype=torch.float32, device=device)
    qry_feat2 = torch.tensor(task['query']['feat2'], dtype=torch.float32, device=device)
    qry_labels = task['query']['labels']

    base_learners, _ = train_base_learners(task['support'])
    sup_probs = predict_base_learners(base_learners, task['support'])
    qry_probs = predict_base_learners(base_learners, task['query'])
    sup_probs_t = torch.tensor(np.stack(sup_probs, axis=1), dtype=torch.float32, device=device)
    qry_probs_t = torch.tensor(np.stack(qry_probs, axis=1), dtype=torch.float32, device=device)

    cm1 = compute_confusion_matrix(task['support']['labels'], sup_probs[0])
    cm2 = compute_confusion_matrix(task['support']['labels'], sup_probs[1])
    ent1_sup = compute_entropy(sup_probs[0])
    ent2_sup = compute_entropy(sup_probs[1])
    sup_feat = np.array([build_fuzzy_features(cm1, cm2, ent1_sup[i], ent2_sup[i])
                         for i in range(len(ent1_sup))])
    ent1_qry = compute_entropy(qry_probs[0])
    ent2_qry = compute_entropy(qry_probs[1])
    qry_feat = np.array([build_fuzzy_features(cm1, cm2, ent1_qry[i], ent2_qry[i])
                         for i in range(len(ent1_qry))])

    sup_feat_t = torch.tensor(sup_feat, dtype=torch.float32, device=device)
    qry_feat_t = torch.tensor(qry_feat, dtype=torch.float32, device=device)

    model_copy = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model_copy.parameters(), lr=inner_lr)
    model_copy.train()
    for _ in range(fine_tune_steps):
        pred = model_copy(sup_probs_t, sup_feat_t)
        loss = F.cross_entropy(pred, sup_labels)
        reg_loss = sum(torch.norm(p, p=2) ** 2 for p in model_copy.parameters())
        loss += lambda_reg * reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model_copy.eval()
    with torch.no_grad():
        pred = model_copy(qry_probs_t, qry_feat_t)
        pred_classes = torch.argmax(pred, dim=1).cpu().numpy()
        acc = accuracy_score(qry_labels, pred_classes)
        recall = recall_score(qry_labels, pred_classes, zero_division=0)
        precision = precision_score(qry_labels, pred_classes, zero_division=0)
        f1 = f1_score(qry_labels, pred_classes, zero_division=0)
    return acc, recall, precision, f1


# ==================== 主训练流程 ====================
def main():
    print("=" * 60)
    print("加载并预处理数据...")
    fmri_feats, speech_feats, labels = load_and_preprocess_data()

    # 域划分（保证每域每类至少6个样本）
    domains = assign_domains(labels, n_domains=6, min_samples_per_class=6)
    unique_domains = np.unique(domains)
    print(f"域编号: {unique_domains}, 各域样本数: {np.bincount(domains)}")

    # 划分元训练/测试域（80%训练，20%测试）
    np.random.seed(42)
    np.random.shuffle(unique_domains)
    n_train_domains = int(0.8 * len(unique_domains))
    train_domains = unique_domains[:n_train_domains]
    test_domains = unique_domains[n_train_domains:]
    print(f"元训练域: {train_domains}")
    print(f"元测试域: {test_domains}")

    # 任务生成器
    task_gen = TaskGenerator(fmri_feats, speech_feats, labels, domains,
                             n_way=2, n_shot=3, n_query=3)

    # 模型初始化
    input_dim = 8 + 2  # 两个混淆矩阵(4+4) + 两个信息熵(2) = 10
    model = MF2Net(input_dim).to(device)

    # MAML元训练
    maml = MAML(model, inner_lr=0.01, meta_lr=0.001)
    n_meta_epochs = 30
    meta_batch_size = 4

    print("\n" + "=" * 60)
    print("元训练...")
    for epoch in range(n_meta_epochs):
        tasks = task_gen.generate_tasks(train_domains, n_tasks_per_domain=2)
        if len(tasks) == 0:
            print(f"Epoch {epoch + 1}: 无任务生成，跳过")
            continue
        batch_size = min(meta_batch_size, len(tasks))
        selected = np.random.choice(len(tasks), batch_size, replace=False)
        task_batch = [tasks[i] for i in selected]
        loss = maml.meta_train_step(task_batch)
        if (epoch + 1) % 5 == 0:
            print(f"元Epoch {epoch + 1}/{n_meta_epochs}, Loss: {loss:.4f}")

    # 元测试
    print("\n" + "=" * 60)
    print("元测试（未见过的域）...")
    test_tasks = task_gen.generate_tasks(test_domains, n_tasks_per_domain=5)
    if len(test_tasks) == 0:
        print("无法生成测试任务，退出。")
        return
    acc_list, rec_list, prec_list, f1_list = [], [], [], []
    for task in test_tasks:
        acc, rec, prec, f1 = meta_test(model, task, inner_lr=0.01, fine_tune_steps=5, lambda_reg=0.01)
        acc_list.append(acc)
        rec_list.append(rec)
        prec_list.append(prec)
        f1_list.append(f1)

    print("\n===== 元测试结果 =====")
    print(f"Accuracy:  {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"Recall:    {np.mean(rec_list):.4f} ± {np.std(rec_list):.4f}")
    print(f"Precision: {np.mean(prec_list):.4f} ± {np.std(prec_list):.4f}")
    print(f"F1 Score:  {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'MF2Net_fmri_speech.pth')
    print("\n模型已保存。")


if __name__ == "__main__":
    main()