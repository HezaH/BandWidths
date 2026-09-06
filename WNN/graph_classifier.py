# graph_classifier.py

import numpy as np

try:
    import tin_man_py
except ModuleNotFoundError:
    tin_man_py = None

try:
    import wisardpkg as wp
except ModuleNotFoundError:
    wp = None

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

        elif self.classifier_name in ("wisard", "wisard_tin_man", "tin_man_wisard"):

            return TinManWisardClassifier(
                address_size=8,
                bleaching=True,
                confidence_threshold=0.1,
                ignore_zero=False,
                parallel=False,
                n_splits=self.n_splits,
                random_state=self.random_state
            )

        else:

            raise ValueError(
                f"Classificador desconhecido: "
                f"{self.classifier_name}"
            )

    def evaluate(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y).astype(int)

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

            # Garante que a saída do classificador
            # tenha o mesmo tipo dos rótulos reais.
            y_pred = np.asarray(y_pred).astype(int)

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
        "mlp",
        "wisard"
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


class TinManWisardClassifier:

    def __init__(
        self,
        address_size=8,
        bleaching=True,
        confidence_threshold=0.1,
        ignore_zero=False,
        parallel=False,
        n_splits=10,
        random_state=42
    ):
        if tin_man_py is None:
            raise ImportError(
                "O módulo tin_man_py não foi encontrado. "
                "Instale o pacote do WiSARD do tin_man antes de usar o classificador.")

        self.address_size = address_size
        self.bleaching = bleaching
        self.confidence_threshold = confidence_threshold
        self.ignore_zero = ignore_zero
        self.parallel = parallel
        self.n_splits = n_splits
        self.random_state = random_state
        self.model = None

    def _build_model(self, input_size):
        return tin_man_py.Wisard(
            input_size=input_size,
            address_size=self.address_size,
            confidence_threshold=self.confidence_threshold,
            bleaching_enabled=self.bleaching,
            ignore_zero=self.ignore_zero,
            parallel=self.parallel,
        )

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.model = self._build_model(X.shape[1])

        for sample, label in zip(X, y):
            self.model.train(
                [int(v) for v in sample],
                str(label)
            )

        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("O modelo WiSARD do tin_man ainda não foi treinado.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []

        for sample in X:
            result = self.model.classify(
                [int(v) for v in sample]
            )

            if result is None:
                predictions.append("unknown")
            else:
                predictions.append(result[0])

        return np.asarray(predictions)

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
            model = self._build_model(X.shape[1])

            for sample, label in zip(X[train_idx], y[train_idx]):
                model.train(
                    [int(v) for v in sample],
                    str(label)
                )

            y_pred = []
            for sample in X[test_idx]:
                result = model.classify(
                    [int(v) for v in sample]
                )
                y_pred.append(result[0] if result is not None else "unknown")

            y_pred = np.asarray(y_pred)
            y_test = y[test_idx]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            if confusion_sum is None:
                confusion_sum = cm
            else:
                confusion_sum += cm

            accuracy_scores.append(acc)
            precision_scores.append(prec)
            recall_scores.append(rec)
            f1_scores.append(f1)

            print(f"Fold {fold:02d} | ACC={acc:.4f}")
            fold += 1

        return {
            "accuracy_mean": np.mean(accuracy_scores),
            "accuracy_std": np.std(accuracy_scores),
            "precision_mean": np.mean(precision_scores),
            "recall_mean": np.mean(recall_scores),
            "f1_mean": np.mean(f1_scores),
            "confusion_matrix": confusion_sum,
            "all_scores": accuracy_scores,
        }


class WisardWrapper:

    def __init__(
        self,
        address_size=8,
        bleaching=True,
        backend="wisardpkg"
    ):

        self.address_size = address_size
        self.bleaching = bleaching
        self.backend = backend
        self.model = None

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        if self.backend == "tin_man":
            if tin_man_py is None:
                raise ImportError("tin_man_py não está instalado.")

            self.model = tin_man_py.Wisard(
                input_size=X.shape[1],
                address_size=self.address_size,
                confidence_threshold=0.1,
                bleaching_enabled=self.bleaching,
                ignore_zero=False,
                parallel=False,
            )

            for sample, label in zip(X, y):
                self.model.train(
                    [int(v) for v in sample],
                    str(label)
                )

            return self

        if wp is None:
            raise ImportError("wisardpkg não está instalado.")

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
        return self

    def predict(self, X):

        X = np.asarray(X)

        if self.backend == "tin_man":
            if self.model is None:
                raise ValueError("O modelo WiSARD do tin_man ainda não foi treinado.")

            predictions = []
            for sample in X:
                result = self.model.classify(
                    [int(v) for v in sample]
                )
                predictions.append(result[0] if result is not None else "unknown")
            return np.asarray(predictions)

        X = X.astype(int).astype(str)

        X = [
            "".join(row)
            for row in X
        ]

        pred = self.model.classify(X)

        return np.array(pred)
