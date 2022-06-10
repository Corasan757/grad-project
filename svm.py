import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('F.csv')
#df = df.drop('Unnamed: 3', axis=1)
#print(df)

x = df.drop("label", axis = 1 )
y = df["label"]

from sklearn.model_selection import train_test_split
x_Train, x_Test, y_Train, y_Test = train_test_split(x, y, test_size = 0.25, random_state = 1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_Train = sc_X.fit_transform(x_Train)
x_Test = sc_X.transform(x_Test)
#print(x_Train)



from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_Train, y_Train)
y_Pred = classifier.predict(x_Test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_Test, y_Pred)
print(cm)
from matplotlib.colors import ListedColormap
X_Set, Y_Set = x_Train, y_Train
X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
 plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Support Vector Machine (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
X_Set, Y_Set = x_Test, y_Test
X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
 plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
             c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Support Vector Machine (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()