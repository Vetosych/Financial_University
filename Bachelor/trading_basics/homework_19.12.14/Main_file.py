# Задание 1 - замена слов

print("Введите предложение")
a = str(input())
ans = 1
while ans != 0:
    print("Введите символ, который требуется заменить")
    old = str(input())
    print("Введите замену")
    new = str(input())
    a = a.replace(old,new)
    print("Хотите заменить другие символы?")
    print("1 - Да")
    print("2 - Нет")
    ans = int(input())
print(a)

# Задание 2 - Датасет с ирисами

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
names = ['sepal length','sepal width','petal length','petal   width','class']
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',names = names)
iris.head()
sns_plot = sns.distplot(iris['sepal width'])
fig = sns_plot.get_figure()
#распределение длин чашелистников
x = np.array(iris['sepal length'])
y = np.array(iris['petal length'])
x = x.reshape((-1, 1))
#задаем переменные, между которыми исследуем взаимосвязь
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
#строим модель
r_sq = model.score(x, y)
#рассчитываем коэффициент детерминации (>50% -> модель значимая, взаимосвязь высокая)
print('coefficient of determination:', r_sq)
#предскажем значением Petal Length по значению Sepal Length
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

# Задание 3 - Файл FR-2019

# 1 задача FR_2019

newarr=[]
for i in arr:
    if arr[i]%2== 0:
        newarr.append(-1)
    elif arr[i]%2 !=0:
        newarr.append(i)
print(newarr)

#2 задача FR_2019

import numpy as np
import random
import scipy.stats as sps
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
a = np.array([2, 6, 1, 9, 10, 3, 27, 8])
n=[int(i) for i in a]
newarr=[]
for i in n:
    if i>5 and i<10:
        newarr.append(i)
    continue
print (newarr)

#3 задача FR_2019

dfs = df.groupby(by = ['director_name', 'color']).aggregate({'imdb_score' : 'mean'})
dfs.head()

#4 задача FR_2019

df['facebook_likes'] = df.director_facebook_likes + df.actor_3_facebook_likes + df.actor_1_facebook_likes + df.actor_2_facebook_likes + df.movie_facebook_likes
dfp = df[df.country == 'UK'].groupby(by = ['director_name', 'color', 'country']).aggregate({'facebook_likes' : 'mean'})
dfp.head()