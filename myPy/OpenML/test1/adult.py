import pandas as pd

data = pd.read_csv('adult.data.csv')

#1. Сколько женщин (признак sex) представлено в этом наборе данных?
print(data['sex'].value_counts())

#2. Каков средний возраст (признак age) женщин?
print(data[data['sex'] == 'Female']['age'].mean())

#3. Какова доля граждан Германии (признак native-country) в процентах?
print(data[data['native-country'] == 'Germany']['native-country'].value_counts()/data.shape[0])

#4. Каковы средние и среднеквадратичные отклонения возраста тех, кто получает более 50K в год
# (признак salary)?
print(data[data['salary'] == '>50K']['age'].mean())
print(data[data['salary'] == '>50K']['age'].std())

#5. Каковы средние значения и среднеквадратичные отклонения возраста тех, кто получает менее 50K в год
# (признак salary)?
print(data[data['salary'] == '<=50K']['age'].mean())
print(data[data['salary'] == '<=50K']['age'].std())

#6. Правда ли, что люди, которые получают больше 50k, имеют как минимум высшее образование?
# (признак education равен Bachelors, Prof-school, Assoc-acdm, Assoc-voc, Masters или Doctorate)
print(data[data['salary'] == '>50K']['education'].unique())

#7. Каков максимальный возраст мужчин расы Amer-Indian-Eskimo?
print(data[data['race'] == 'Amer-Indian-Eskimo']['age'].max())

#8. Среди кого больше доля зарабатывающих много (>50K): среди женатых или холостых мужчин
# (признак marital-status для женатых равен Married-civ-spouse, Married-spouse-absent или Married-AF-spouse)?
print(data[data['salary'] == '>50K']['marital-status'].value_counts(normalize=True))

#9. Какое максимальное число часов человек работает в неделю (признак hours-per-week)?
# Сколько людей работают такое количество часов и каков среди них процент зарабатывающих много?
print(data['hours-per-week'].max())
print(len(data[data['hours-per-week'] == data['hours-per-week'].max()]))
print(data[data['hours-per-week'] == data['hours-per-week'].max()].salary.value_counts(normalize=True)['>50K'])

#10. Посчитайте среднее время работы (hours-per-week) зарабатывающих мало и много (salary)
# для каждой страны (native-country).
print(data.groupby(['native-country', 'salary']).agg({'hours-per-week': 'mean'}))

