import sklearn.linear_model as linear_model
import sklearn.naive_bayes as naive_bayes
import sklearn.svm as svm


def lr(class_weight=None):
    model = linear_model.SGDClassifier(max_iter=10000, tol=1e-3, warm_start=True, loss="log", class_weight=class_weight)

    return model


def linear_svm(class_weight=None):
    model = linear_model.SGDClassifier(max_iter=10000, tol=1e-3, warm_start=True, loss="hinge", class_weight=class_weight)

    return model


def rbf_svm():
    model = svm.SVC(gamma="auto")

    return model