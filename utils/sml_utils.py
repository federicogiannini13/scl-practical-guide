import sklearn.metrics as skm

def test_cl(cl_table, models, X_test, y_test):
    for m in models:
        for metric in ["accuracy", "kappa"]:
            cl_table[m][metric].append([])
        for task, (X_test_task, y_test_task) in enumerate(zip(X_test, y_test)):
            pred = []
            for x_cl in X_test_task:
                x_cl = {f"feat{i}": x_cl[i] for i in range(len(x_cl))}
                if type(models[m]) == list:
                    task = min(task, len(models[m])-1)
                    y_hat = models[m][task].predict_one(x_cl)
                else:
                    y_hat = models[m].predict_one(x_cl)
                y_hat = 0 if y_hat is None else y_hat
                pred.append(y_hat)
            cl_table[m]["accuracy"][-1].append(skm.accuracy_score(y_test_task, pred))
            cl_table[m]["kappa"][-1].append(skm.cohen_kappa_score(y_test_task, pred))
    return cl_table