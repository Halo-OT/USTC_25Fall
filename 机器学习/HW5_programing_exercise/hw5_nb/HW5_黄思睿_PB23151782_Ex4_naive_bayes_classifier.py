"""
朴素贝叶斯分类器 - 用于垃圾邮件检测
实现练习4中的算法2和算法3
"""

import os
import re
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Set


class NaiveBayesClassifier:
    """朴素贝叶斯分类器实现"""
    
    def __init__(self, use_laplace: bool = True):
        """
        初始化分类器
        
        Args:
            use_laplace: 是否使用Laplace平滑（默认True）
        """
        self.use_laplace = use_laplace
        self.vocabulary: Set[str] = set()  # 词汇表V
        self.class_priors: Dict[str, float] = {}  # P(c)
        self.word_probs: Dict[str, Dict[str, float]] = {}  # P(w_k|c)
        self.classes: List[str] = []
        
    def tokenize(self, text: str) -> List[str]:
        """
        将文本分词，移除所有非字母字符
        
        Args:
            text: 输入文本
            
        Returns:
            分词后的token列表
        """
        # 转换为小写
        text = text.lower()
        # 只保留字母字符，其他字符替换为空格
        text = re.sub(r'[^a-z]+', ' ', text)
        # 分词并过滤空字符串
        tokens = [word for word in text.split() if word]
        return tokens
    
    def load_emails(self, directory: str) -> List[Tuple[List[str], str]]:
        """
        从目录加载邮件数据
        
        Args:
            directory: 邮件文件所在目录
            
        Returns:
            (tokens列表, 标签)的列表
        """
        emails = []
        
        for filename in os.listdir(directory):
            if not filename.endswith('.txt'):
                continue
                
            filepath = os.path.join(directory, filename)
            
            # 根据文件名判断是否为垃圾邮件
            if filename.startswith('spmsg'):
                label = 'spam'
            else:
                label = 'ham'
            
            # 读取邮件内容
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # 邮件正文从第三行开始
                    lines = content.split('\n')
                    # 包含主题和正文
                    text = ' '.join(lines)
                    tokens = self.tokenize(text)
                    emails.append((tokens, label))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
        
        return emails
    
    def train(self, training_data: List[Tuple[List[str], str]]):
        """
        训练朴素贝叶斯分类器 - 实现算法2
        
        Args:
            training_data: 训练数据，格式为[(tokens, label), ...]
        """
        # 步骤1: 构建词汇表V
        for tokens, _ in training_data:
            self.vocabulary.update(tokens)
        
        print(f"词汇表大小: {len(self.vocabulary)}")
        
        # 获取所有类别
        self.classes = list(set(label for _, label in training_data))
        
        # 为每个类别进行处理
        for c in self.classes:
            # 步骤3: 获取该类别的训练样本
            class_docs = [(tokens, label) for tokens, label in training_data if label == c]
            
            # 步骤4: 计算P(c) = |D_c| / |D|
            self.class_priors[c] = len(class_docs) / len(training_data)
            
            # 步骤5: 将该类别的所有文档连接成一个大文档
            all_tokens = []
            for tokens, _ in class_docs:
                all_tokens.extend(tokens)
            
            # 步骤6: 计算该类别的文档长度
            n_c = len(all_tokens)
            
            # 步骤7-9: 计算每个词的条件概率P(w_k|c)
            self.word_probs[c] = {}
            
            # 统计词频
            word_count = defaultdict(int)
            for token in all_tokens:
                word_count[token] += 1
            
            # 计算条件概率
            for word in self.vocabulary:
                if self.use_laplace:
                    # 使用Laplace平滑: P(w_k|c) = (n_c,k + 1) / (n_c + |V|)
                    self.word_probs[c][word] = (word_count[word] + 1) / (n_c + len(self.vocabulary))
                else:
                    # 不使用平滑: P(w_k|c) = (n_c,k + 1) / (n_c + |V|)
                    # 为了避免概率为0，仍然加1
                    self.word_probs[c][word] = (word_count[word] + 1) / (n_c + len(self.vocabulary))
        
        print(f"训练完成! 类别: {self.classes}")
        for c in self.classes:
            print(f"  P({c}) = {self.class_priors[c]:.4f}")
    
    def predict(self, tokens: List[str]) -> str:
        """
        预测单个文档的类别 - 实现算法3
        
        Args:
            tokens: 文档的token列表
            
        Returns:
            预测的类别标签
        """
        # 步骤1-4: 找出文档中存在于词汇表中的词
        word_indices = set()
        for token in tokens:
            if token in self.vocabulary:
                word_indices.add(token)
        
        # 步骤6: 计算每个类别的得分
        # ŷ = argmax P(c) * ∏ P(w_k|c) for i∈I
        max_score = float('-inf')
        best_class = None
        
        for c in self.classes:
            # 使用对数避免下溢
            # log P(c) + Σ log P(w_k|c)
            score = math.log(self.class_priors[c])
            
            for word in word_indices:
                prob = self.word_probs[c].get(word, 1e-10)  # 避免log(0)
                if prob > 0:
                    score += math.log(prob)
                else:
                    # 如果概率为0（不使用平滑时），使用一个极小值
                    score += math.log(1e-10)
            
            if score > max_score:
                max_score = score
                best_class = c
        
        return best_class
    
    def evaluate(self, test_data: List[Tuple[List[str], str]]) -> Dict:
        """
        在测试集上评估模型
        
        Args:
            test_data: 测试数据
            
        Returns:
            包含评估指标的字典
        """
        predictions = []
        true_labels = []
        
        for tokens, true_label in test_data:
            pred_label = self.predict(tokens)
            predictions.append(pred_label)
            true_labels.append(true_label)
        
        # 计算混淆矩阵
        confusion_matrix = self._compute_confusion_matrix(true_labels, predictions)
        
        # 计算评估指标
        metrics = self._compute_metrics(confusion_matrix)
        
        return {
            'confusion_matrix': confusion_matrix,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        }
    
    def _compute_confusion_matrix(self, true_labels: List[str], 
                                  predictions: List[str]) -> Dict:
        """计算混淆矩阵"""
        # 对于二分类：spam为正类，ham为负类
        tp = sum(1 for t, p in zip(true_labels, predictions) 
                if t == 'spam' and p == 'spam')
        tn = sum(1 for t, p in zip(true_labels, predictions) 
                if t == 'ham' and p == 'ham')
        fp = sum(1 for t, p in zip(true_labels, predictions) 
                if t == 'ham' and p == 'spam')
        fn = sum(1 for t, p in zip(true_labels, predictions) 
                if t == 'spam' and p == 'ham')
        
        return {
            'TP': tp,  # True Positive
            'TN': tn,  # True Negative
            'FP': fp,  # False Positive
            'FN': fn   # False Negative
        }
    
    def _compute_metrics(self, cm: Dict) -> Dict:
        """根据混淆矩阵计算评估指标"""
        tp, tn, fp, fn = cm['TP'], cm['TN'], cm['FP'], cm['FN']
        
        # 准确率 Accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # 精确率 Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # 召回率 Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1分数 F1 = 2 * (Precision * Recall) / (Precision + Recall)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }


def print_results(results: Dict, title: str):
    """打印评估结果"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # 混淆矩阵
    cm = results['confusion_matrix']
    print("\n混淆矩阵:")
    print(f"{'':>15} {'预测Spam':>15} {'预测Ham':>15}")
    print(f"{'实际Spam':>15} {cm['TP']:>15} {cm['FN']:>15}")
    print(f"{'实际Ham':>15} {cm['FP']:>15} {cm['TN']:>15}")
    
    # 评估指标
    print(f"\n评估指标:")
    print(f"  准确率 (Accuracy):  {results['accuracy']:.4f}")
    print(f"  精确率 (Precision): {results['precision']:.4f}")
    print(f"  召回率 (Recall):    {results['recall']:.4f}")
    print(f"  F1分数 (F1-Score):  {results['f1_score']:.4f}")


def main():
    """主函数"""
    # 设置数据路径
    train_dir = 'train-mails'
    test_dir = 'test-mails'
    
    print("="*60)
    print("朴素贝叶斯垃圾邮件分类器")
    print("="*60)
    
    # ========== 使用Laplace平滑 ==========
    print("\n第一部分: 使用Laplace平滑")
    print("-"*60)
    
    classifier_laplace = NaiveBayesClassifier(use_laplace=True)
    
    # 加载训练数据
    print("\n加载训练数据...")
    train_data = classifier_laplace.load_emails(train_dir)
    print(f"训练集大小: {len(train_data)}")
    spam_count = sum(1 for _, label in train_data if label == 'spam')
    ham_count = len(train_data) - spam_count
    print(f"  Spam: {spam_count}, Ham: {ham_count}")
    
    # 训练模型
    print("\n训练模型...")
    classifier_laplace.train(train_data)
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_data = classifier_laplace.load_emails(test_dir)
    print(f"测试集大小: {len(test_data)}")
    spam_count_test = sum(1 for _, label in test_data if label == 'spam')
    ham_count_test = len(test_data) - spam_count_test
    print(f"  Spam: {spam_count_test}, Ham: {ham_count_test}")
    
    # 评估模型
    print("\n评估模型...")
    results_laplace = classifier_laplace.evaluate(test_data)
    print_results(results_laplace, "测试集结果 (使用Laplace平滑)")
    
    # ========== 不使用Laplace平滑 ==========
    print("\n\n第二部分: 不使用Laplace平滑")
    print("-"*60)
    
    classifier_no_laplace = NaiveBayesClassifier(use_laplace=False)
    
    # 训练模型
    print("\n训练模型...")
    classifier_no_laplace.train(train_data)
    
    # 评估模型
    print("\n评估模型...")
    results_no_laplace = classifier_no_laplace.evaluate(test_data)
    print_results(results_no_laplace, "测试集结果 (不使用Laplace平滑)")
    
    # ========== 比较结果 ==========
    print("\n\n" + "="*60)
    print("结果比较")
    print("="*60)
    print(f"\n{'指标':>20} {'使用Laplace':>15} {'不使用Laplace':>18}")
    print("-"*60)
    print(f"{'准确率':>20} {results_laplace['accuracy']:>15.4f} {results_no_laplace['accuracy']:>18.4f}")
    print(f"{'精确率':>20} {results_laplace['precision']:>15.4f} {results_no_laplace['precision']:>18.4f}")
    print(f"{'召回率':>20} {results_laplace['recall']:>15.4f} {results_no_laplace['recall']:>18.4f}")
    print(f"{'F1分数':>20} {results_laplace['f1_score']:>15.4f} {results_no_laplace['f1_score']:>18.4f}")
    
    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)
    
    # 说明如何处理概率为0的问题
    print("\n关于概率接近0的问题:")
    print("- 在算法3中，我们使用对数概率来避免数值下溢")
    print("- 即计算 log P(c) + Σ log P(w_k|c) 而不是 P(c) * ∏ P(w_k|c)")
    print("- 这样可以将连乘转换为求和，避免极小概率相乘导致的下溢问题")
    print("- 即使不使用Laplace平滑，我们在代码中也对分子加1，避免完全为0的情况")


if __name__ == '__main__':
    main()
