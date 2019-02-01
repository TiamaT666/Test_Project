import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

def save_clean_data(cleaner, X_train, y_train, X_test, name='simple'):
    path = './data/clean/' + name
    if not os.path.exists(path):
        os.makedirs(path)

    y_train.to_csv(path + '/y_train.csv')
    cleaner(X_train).to_csv(path + '/X_train.csv')
    cleaner(X_test).to_csv(path + '/X_test.csv')

def get_clean_data(cleaner_name='simple'):
	path = './data/clean/' + cleaner_name
	X_train = pd.read_csv(path + '/X_train.csv', index_col='match_id')
	y_train = pd.read_csv(path + '/y_train.csv', index_col='match_id')
	X_test = pd.read_csv(path + '/X_test.csv', index_col='match_id')
	return X_train, y_train['radiant_win'], X_test

def kaggle_save(name, model, X_test):
	y_test = model.predict_proba(X_test)[:, 1]
	result = pd.DataFrame({'radiant_win': y_test}, index=X_test.index)
	result.index.name = 'match_id'
	result.to_csv('./data/kaggle/{}.csv'.format(name))


# Загружаем данные
train = pd.read_csv('./data/features.csv', index_col='match_id')
test = pd.read_csv('./data/features_test.csv', index_col='match_id')

# Удаляем признаки, связанные с итогами матча
train.drop(['duration',
         'tower_status_radiant',
         'tower_status_dire',
         'barracks_status_radiant',
         'barracks_status_dire'
        ], axis=1, inplace=True)

# И разделяем датасет на признаки и целевую переменную
X = train
y = train['radiant_win'].to_frame()
del train['radiant_win']

# Заменяем пропуски на 0
def clean(X):
    return X.fillna(0)

# В данных присутствует 11 категориальных признаков, удаляем их
def clean_category(X):
    X = clean(X)
    del X['lobby_type']
    for n in range(1, 6):
        del X['r{}_hero'.format(n)]
        del X['d{}_hero'.format(n)]

    return X

heroes = pd.read_csv('./data/dictionaries/heroes.csv')
print('Всего героев в игре:', len(heroes))

# Формируем "мешок слов" по героям
def hero_bag(X):
    X_pick = np.zeros((X.shape[0], len(heroes)))
    for i, match_id in enumerate(X.index):
        for p in range(5):
            X_pick[i, X.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, X.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

    return pd.DataFrame(X_pick, index=X.index)

# Сохраняем очищенные данные
save_clean_data(clean, X, y, test)
save_clean_data(clean_category, X, y, test, name='clean_category')
save_clean_data(hero_bag, X, y, test, name='hero_bag')

# Подход 1: градиентный бустинг "в лоб"
X, y, X_kaggle = get_clean_data()
kf = KFold(y.size, shuffle=True, random_state=42)

# Подбираем оптимальное число деревьев
scores = []
nums = [10, 20, 30, 50, 100, 250]
for n in nums:
    print('#', str(n))
    model = GradientBoostingClassifier(n_estimators=n, random_state=42)
    start_time = datetime.datetime.now()
    model_scores = cross_val_score(model, X, y, cv=kf, n_jobs=-1)
    print('Time elapsed:', datetime.datetime.now() - start_time)
    print(model_scores)
    scores.append(np.mean(model_scores))

# Рисуем график
plt.plot(nums, scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.show()

