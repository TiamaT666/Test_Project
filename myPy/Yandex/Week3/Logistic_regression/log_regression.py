import pandas as pd
from math import exp
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score
from copy import copy

# 1. Загрузите данные из файла data-logistic.csv. Это двумерная выборка, целевая переменная на которой
# принимает значения -1 или 1.

logistic_data = pd.read_csv('data-logistic.csv', header=None)

logistic_classes = logistic_data[0]
logistic_observations = logistic_data.ix[:,1:].copy()

print(logistic_observations[0:6])
print(logistic_classes[0:6])

# 2. Убедитесь, что выше выписаны правильные формулы для градиентного спуска.
# Обратите внимание, что мы используем полноценный градиентный спуск, а не его стохастический вариант!

# 3. Реализуйте градиентный спуск для обычной и L2-регуляризованной (с коэффициентом регуляризации 10)
# логистической регрессии. Используйте длину шага k=0.1. В качестве начального приближения используйте
# вектор (0, 0).

def gradient_descent(step = 0.1, initial_approach = [0, 0], eps = 0.00001, max_iter = 10000):
    weights_old = copy(initial_approach)
    weights_new = [_+2*abs(eps) for _ in weights_old] # Для того, чтобы пошла хотя бы первая итерация, нужно, чтобы weights_new превосходили weights_old хотя бы на 2*eps
    # Веса проинициализированы, теперь можно запускать итерации
    iter_num = 0
    while(distance.euclidean(weights_old, weights_new) > eps and iter_num <= max_iter):
        print(distance.euclidean(weights_old, weights_new))
        print(iter_num)
        iter_num = iter_num + 1
    print(weights_old)
    print(weights_new)
    print(distance.euclidean(weights_old, weights_new))
    return weights_new

def gradient_descent_regularized_L2(k = 0.1, initial_approach = [0, 0], eps = 0.00001, max_iter = 10000, regularization_coef = 10):
    return 0



# 4. Запустите градиентный спуск и доведите до сходимости (евклидово расстояние между векторами весов
# на соседних итерациях должно быть не больше 1e-5). Рекомендуется ограничить сверху число итераций
# десятью тысячами.

estimates_non_regularized = gradient_descent()
print(estimates_non_regularized)

# 5. Какое значение принимает AUC-ROC на обучении без регуляризации и при ее использовании?
# Эти величины будут ответом на задание. В качестве ответа приведите два числа через пробел. О
# братите внимание, что на вход функции roc_auc_score нужно подавать оценки вероятностей,
# подсчитанные обученным алгоритмом. Для этого воспользуйтесь сигмоидной функцией:
# a(x) = 1 / (1 + exp(-w1 x1 - w2 x2)).

auc_roc_no_regularization = 0.0
auc_roc_regularization = 0.01

resulting_string = " ".join([repr(round(auc_roc_no_regularization, 3)), repr(round(auc_roc_regularization, 3))])

print(resulting_string)

file_answer = open("answers.txt", "w")
file_answer.write(resulting_string)
file_answer.close()


# 6. Попробуйте поменять длину шага. Будет ли сходиться алгоритм, если делать более длинные шаги?
# Как меняется число итераций при уменьшении длины шага?


# 7. Попробуйте менять начальное приближение. Влияет ли оно на что-нибудь?