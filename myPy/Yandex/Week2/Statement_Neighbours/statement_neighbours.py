import pandas as pd
import sklearn

from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# 1. Загрузите выборку Wine по адресу https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data (файл также приложен к этому заданию)
# 2. Извлеките из данных признаки и классы. Класс записан в первом столбце (три варианта), признаки — в столбцах со второго по последний.
# Более подробно о сути признаков можно прочитать по адресу https://archive.ics.uci.edu/ml/datasets/Wine (см. также файл wine.names, приложенный к заданию)

df = pd.read_csv('wine.data', header=None)
y = df[0]
X = df.loc[:, 1:]

# 3. Оценку качества необходимо провести методом кросс-валидации по 5 блокам (5-fold).
# Создайте генератор разбиений, который перемешивает выборку перед формированием блоков (shuffle=True).
# Для воспроизводимости результата, создавайте генератор KFold с фиксированным параметром random_state=42.
# В качестве меры качества используйте долю верных ответов (accuracy).

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. Найдите точность классификации на кросс-валидации для метода k ближайших соседей (sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50.
# При каком k получилось оптимальное качество? Чему оно равно (число в интервале от 0 до 1)?
# Данные результаты и будут ответами на вопросы 1 и 2.

def test_accuracy(kf, X, y):
    scores = list()
    k_range = range(1, 51)
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        scores.append(cross_val_score(model, X, y, cv=kf, scoring='accuracy'))

    return pd.DataFrame(scores, k_range).mean(axis=1).sort_values(ascending=False)


accuracy = test_accuracy(kf, X, y)
top_accuracy = accuracy.head(1)
print(1, top_accuracy.index[0])
print(2, top_accuracy.values[0])

# 5. Произведите масштабирование признаков с помощью функции sklearn.preprocessing.scale.
# Снова найдите оптимальное k на кросс-валидации.

X = sklearn.preprocessing.scale(X)
accuracy = test_accuracy(kf, X, y)

# 6. Какое значение k получилось оптимальным после приведения признаков к одному масштабу?
# Приведите ответы на вопросы 3 и 4. Помогло ли масштабирование признаков?

top_accuracy = accuracy.head(1)
print(3, top_accuracy.index[0])
print(4, top_accuracy.values[0])