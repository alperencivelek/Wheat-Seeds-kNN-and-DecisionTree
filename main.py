from sklearn import tree, neighbors
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn .model_selection import cross_val_score

# Veri Okuma
#3 adet buğday tohumu çeşidini temsil eden veriseti
dataset = np.loadtxt("seeds_dataset.txt")
x = dataset[:, 0:7]
y = dataset[:, 7]

k=5 #karşılaştıracağımız en yakın komşu sayısı 5 olsun

x_egitim, x_deneme, y_egitim, y_deneme = train_test_split(x, y, test_size=.5)

print("Decision Tree Başarı Oranı k=10:")

print(cross_val_score(tree.DecisionTreeClassifier(),x_egitim,y_egitim,cv=10))

print("KNN başarı oranı k=10:")

print(cross_val_score(neighbors.KNeighborsClassifier(n_neighbors=k),x_egitim,y_egitim,cv=10))

print("Decision Tree Confusion Matrice")
siniflandirici = tree.DecisionTreeClassifier()
y_tahmin = siniflandirici.fit(x_egitim, y_egitim).predict(x_deneme)

# Compute confusion matrix
cm = confusion_matrix(y_deneme, y_tahmin)

print(cm)

plt.matshow(cm)
plt.title('Decision Tree')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("KNN Confusion Matrice")
siniflandirici = neighbors.KNeighborsClassifier(n_neighbors=k)
y_tahmin = siniflandirici.fit(x_egitim, y_egitim).predict(x_deneme)

# Compute confusion matrix
cm = confusion_matrix(y_deneme, y_tahmin)

print(cm)

plt.matshow(cm)
plt.title('KNN')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()



