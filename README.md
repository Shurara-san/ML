# ML-FinalTask

##### ML: sklearn, pandas, numpy API: flask

### Источник данных: 
https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

### Задача:
Определить качество вина (по шкале от 0 до 10) по имеющимся признакам. Многоклассовая классификация.

#### Используемые признаки (11):

1 - fixed acidity (Decimal, 4.6 - 15.9)

2 - volatile acidity (Decimal, 0.12 - 1.58)

3 - citric acid (Decimal, 0 - 1)

4 - residual sugar (Decimal, 0.9 - 15.5)

5 - chlorides (Decimal, 0.01 - 0.61)

6 - free sulfur dioxide (Decimal, 1 - 72)

7 - total sulfur dioxide (Decimal, 6 -289)

8 - density (Decimal, 0.99 - 1)

9 - pH (Decimal, 2.74 - 4.01)

10 - sulphates (Decimal, 0.33 - 2)

11 - alcohol (Decimal, 8.4 - 14.9)

### Модель:

CatBoostClassifier (с подбором оптимальных параметров с помощью GridSearchCV).

### Клонируем репозиторий и создаем образ:

git clone https://github.com/Shurara-san/ML.git
cd ML
docker build -t ml_bs/final_task .

### Запускаем контейнер:
docker run -d -p 8181:8181 -v C:\Users\ShuRaRa_SaN\HW\FinalTask\Final_Model:/models ml_bs/final_task
12367292e2837a97204e9c1affd66e8dfa10eb469a09b371f00d916f49fae790
