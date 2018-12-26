# 1. Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
# 2. Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare), возраст пассажира (Age) и его пол (Sex).
# 3. Обратите внимание, что признак Sex имеет строковые значения.
# 4. Выделите целевую переменную — она записана в столбце Survived.
# 5. В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст. Такие записи при чтении их в pandas принимают значение nan. Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.
# 6. Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию (речь идет о параметрах конструктора DecisionTreeСlassifier).
# 7. Вычислите важности признаков и найдите два признака с наибольшей важностью. Их названия будут ответами для данной задачи (в качестве ответа укажите названия признаков через запятую или пробел, порядок не важен).

import pandas as pd
import numpy as np
from sklearn import tree

data_auth = pd.read_csv('titanic.csv', index_col='PassengerId')

data = data_auth.copy()

# Impute Age NaN values
data["Age"] = data["Age"] = data["Age"].fillna(data["Age"].median())

# Convert male and female to integer form
data["Sex"][data["Sex"] == "male"] = 0
data["Sex"][data["Sex"] == "female"] = 1

# Create the target and features numpy arrays: target, features
target = data["Survived"].values
features = data[["Pclass", "Sex", "Age", "Fare"]].values

# Fit decision tree: my_tree
my_tree = tree.DecisionTreeClassifier(random_state=241)
my_tree = my_tree.fit(features, target)

print('Pclass = ', my_tree.feature_importances_[0:1])
print('Sex = ', my_tree.feature_importances_[1:2])
print('Age = ', my_tree.feature_importances_[2:3])
print('Fare = ', my_tree.feature_importances_[3:4])
print(my_tree.score(features, target))