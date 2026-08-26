# graph_classifier.py

import numpy as np
import wisardpkg as wp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


class GraphClassifier:

    def __init__(
        self,
        classifier_name="rf",
        n_splits=10,
        random_state=42
    ):

        self.classifier_name = classifier_name
        self.n_splits = n_splits
        self.random_state = random_state

    def _build_classifier(self):

        if self.classifier_name == "rf":

            return RandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state,
                n_jobs=-1
            )

        elif self.classifier_name == "svm":

            return SVC(
                kernel="rbf"
            )

        elif self.classifier_name == "knn":

            return KNeighborsClassifier(
                n_neighbors=5
            )

        elif self.classifier_name == "mlp":

            return MLPClassifier(
                hidden_layer_sizes=(128,),
                max_iter=500,
                random_state=self.random_state
            )

        else:

            raise ValueError(
                f"Classificador desconhecido: "
                f"{self.classifier_name}"
            )

    def evaluate(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        confusion_sum = None

        fold = 1

        for train_idx, test_idx in cv.split(X, y):

            model = self._build_classifier()

            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            model.fit(
                X_train,
                y_train
            )

            y_pred = model.predict(
                X_test
            )

            acc = accuracy_score(
                y_test,
                y_pred
            )

            prec = precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            rec = recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            cm = confusion_matrix(
                y_test,
                y_pred
            )

            if confusion_sum is None:

                confusion_sum = cm

            else:

                confusion_sum += cm

            accuracy_scores.append(acc)
            precision_scores.append(prec)
            recall_scores.append(rec)
            f1_scores.append(f1)

            print(
                f"Fold {fold:02d} "
                f"| ACC={acc:.4f}"
            )

            fold += 1

        return {

            "accuracy_mean":
                np.mean(
                    accuracy_scores
                ),

            "accuracy_std":
                np.std(
                    accuracy_scores
                ),

            "precision_mean":
                np.mean(
                    precision_scores
                ),

            "recall_mean":
                np.mean(
                    recall_scores
                ),

            "f1_mean":
                np.mean(
                    f1_scores
                ),

            "confusion_matrix":
                confusion_sum,

            "all_scores":
                accuracy_scores
        }


def benchmark_classifiers(X, y):

    classifiers = [

        "rf",
        "svm",
        "knn",
        "mlp"
    ]

    results = {}

    for clf_name in classifiers:

        print(
            f"\n===== {clf_name.upper()} ====="
        )

        clf = GraphClassifier(
            classifier_name=clf_name
        )

        result = clf.evaluate(
            X,
            y
        )

        results[clf_name] = result

        print(
            f"Acurácia Média: "
            f"{result['accuracy_mean']:.4f}"
        )

        print(
            f"Desvio Padrão: "
            f"{result['accuracy_std']:.4f}"
        )

        print(
            f"F1 Médio: "
            f"{result['f1_mean']:.4f}"
        )

    return results

class WisardWrapper:

    def __init__(
        self,
        address_size=8,
        bleaching=True
    ):

        self.address_size = address_size
        self.bleaching = bleaching

        self.model = None

    def fit(self, X, y):

        self.model = wp.Wisard(
            self.address_size,
            bleachingActivated=self.bleaching
        )

        X = X.astype(int).astype(str)

        X = [
            "".join(row)
            for row in X
        ]

        y = [
            str(label)
            for label in y
        ]

        self.model.train(X, y)

    def predict(self, X):

        X = X.astype(int).astype(str)

        X = [
            "".join(row)
            for row in X
        ]

        pred = self.model.classify(X)

        return np.array(pred)
