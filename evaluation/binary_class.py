from sklearn import metrics
import numpy as np


class Metrics():
    def __init__(self, y_pred, y_true, y_score, plot_roc=False) -> None:
        self.y_pred = y_pred
        self.y_true = y_true
        self.y_score = y_score
        self.plot_roc = plot_roc
        self.checkData()
        
    def checkData(self) -> None:
        assert len(set(self.y_pred)) <= 2, f"Not a binary classification task. The predicted label is more than 2 types: {set(self.y_pred)}"
        assert len(set(self.y_true)) <= 2, f"Not a binary classification task. The actual label is more than 2 types: {set(self.y_true)}"
        assert np.max(self.y_score) <= 1, f"The y_score is greater than 1"
        assert np.min(self.y_score) >= 0, f"The y_score is less than 0"           
    
    def confusion_matrix(self) -> None:
        self.tn, self.fp, self.fn, self.tp = metrics.confusion_matrix(y_true=self.y_true, y_pred=self.y_pred).ravel()
        
    def accuracy(self) -> None:
        self.acc = metrics.accuracy_score(y_true=self.y_true, y_pred=self.y_pred, normalize=True)
    
    def balanced_accuracy(self) -> None:
        """ 
        The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. 
        It is defined as the average of recall obtained on each class.
        """
        self.b_acc = metrics.balanced_accuracy_score(y_true=self.y_true, y_pred=self.y_pred)
        
    def precision(self) -> None:
        self.precision = metrics.precision_score(y_true=self.y_true, y_pred=self.y_pred)

    def recall(self) -> None:
        "Recall / TPR / Sensitivity"
        self.recall_score = metrics.recall_score(y_true=self.y_true, y_pred=self.y_pred)
    
    def f1(self) -> None:
        """
        F1 = 2 * (precision * recall) / (precision + recall)
        """
        self.f1_score = metrics.f1_score(y_true=self.y_true, y_pred=self.y_pred)
    
    def auc(self) -> None:
        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(y_true=self.y_true, y_score=self.y_score)
        self.auc_score = metrics.roc_auc_score(y_true=self.y_true, y_score=self.y_score)


# TODO: 数据输出形式（JSON？）；绘制ROC曲线（要美化的）；绘制PR曲线（要美化的）
    
        
        