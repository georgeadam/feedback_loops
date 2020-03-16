import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model
import sklearn.naive_bayes as naive_bayes
import sklearn.svm as svm


def lr(num_features=2, class_weight=None):
    model = linear_model.LogisticRegression(max_iter=10000, tol=1e-3, warm_start=True, class_weight=class_weight,
                                            penalty="none")
    model.evaluate = evaluate

    return model


def linear_svm(num_features=2, class_weight=None):
    model = linear_model.SGDClassifier(max_iter=10000, tol=1e-3, warm_start=True, loss="hinge", class_weight=class_weight)
    model.evaluate = evaluate

    return model


def rbf_svm(num_features=2, class_weight=None):
    model = svm.SVC(probability=True, gamma="auto")
    model.evaluate = evaluate

    return model


def random_forest(num_features=2, class_weight=None):
    model = ensemble.RandomForestClassifier(class_weight=class_weight, n_jobs=4)
    model.evaluate = evaluate

    return model


def adaboost(num_features=2, class_weight=None):
    model = ensemble.AdaBoostClassifier()
    model.evaluate = evaluate

    return model


def xgboost(num_features=2, class_weight=None):
    model = ensemble.GradientBoostingClassifier()
    model.evaluate = evaluate

    return model


def evaluate(x, y):
    return 0.0