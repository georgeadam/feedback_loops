import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model
import sklearn.naive_bayes as naive_bayes
import sklearn.svm as svm


def lr(num_features=2, class_weight=None):
    model = linear_model.SGDClassifier(max_iter=10000, tol=1e-3, warm_start=True, loss="log", class_weight=class_weight)

    return model


def linear_svm(num_features=2, class_weight=None):
    model = linear_model.SGDClassifier(max_iter=10000, tol=1e-3, warm_start=True, loss="hinge", class_weight=class_weight)

    return model


def rbf_svm(num_features=2, class_weight=None):
    model = svm.SVC(probability=True, gamma="auto")

    return model


def random_forest(num_features=2, class_weight=None):
    model = ensemble.RandomForestClassifier(class_weight=class_weight, n_jobs=4)

    return model


def adaboost(num_features=2, class_weight=None):
    model = ensemble.AdaBoostClassifier()

    return model