
import joblib
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('housedata.csv')
df.dropna(inplace=True)
df.head()

X = df.drop(['median_house_value', 'ocean_proximity'], axis=1)
y = df['median_house_value']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

md = LinearRegression()
md.fit(X_train, y_train)

score = md.score(X_train, y_train)
print(score * 100)
score1 = md.score(X_test, y_test)
print(score1 * 100)

y_pred = md.predict(X_test)
plt.scatter(y_test, y_pred)

joblib.dump(md, "model.joblib")