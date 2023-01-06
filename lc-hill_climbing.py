# Import libraries for building and evaluating a KNN model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy
import random
import copy
from sklearn import svm
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("survey lung cancer.csv")
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['LUNG_CANCER']=encoder.fit_transform(data['LUNG_CANCER'])
data['GENDER']=encoder.fit_transform(data['GENDER'])


# import seaborn as sns
# import matplotlib.pyplot as plt


Y = data['LUNG_CANCER']
data = data.drop('LUNG_CANCER',axis=1)
X = data

def MatrixCreate(rows, cols):
    # matrix = [[0 for y in range(cols)] for x in range(rows)]
    matrix = numpy.zeros(shape=(rows, cols))
    return matrix
    
def MatrixRandomize(v):
    random_m = [[random.random() for y in range(len(v[x]))] for x in range(len(v))]
    return random_m

#MatrixPerturb fonksiyonu, verilen bir matrisin (p) bir kopyasını oluşturur (c). 
#Daha sonra, matrisin her satırı ve her sütunu için bir döngü oluşturur.
# Eğer verilen bir probabilite değeri (prob) rastgele üretilen bir sayıdan daha büyükse,
# o sütünün değeri rastgele bir sayı ile değiştirilir. Bu işlemler tamamlandıktan sonra, oluşturulan matris (c) döndürülür.

def MatrixPerturb(p, prob):
    c = copy.deepcopy(p)
    for x in range(len(c)):
        for y in range(len(c[x])):
            if prob > random.random():
                c[x][y] = random.random()
    return c


# Fitness fonksiyonu  modeller için en iyi değeri buluyor
def Fitness(matrix):
    # Use the input matrix as features to predict a target variable
   # Replace this with your target values
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
    
    # Train a KNN model on the training data
    #model = KNeighborsClassifier(n_neighbors=11)
    model=svm.SVC(kernel='linear')# Linear Kernel
    #model=LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)
    
    # Return the accuracy as the fitness value
    return accuracy

# Define the Hill Climbing function using the modified Fitness function
def HillClimbing():
    # Başlangıç random çözümleri
    current_solution = MatrixCreate(1, 15)
    current_solution = MatrixRandomize(current_solution)
    
    # Set the initial fitness value
    current_fitness = Fitness(current_solution)
    
    #maximum iterations
    max_iterations = 1000
    
    # adım sayısı değişiklik için
    step_size = 0.5
    
    for i in range(max_iterations):
        #Mevcut çözümde küçük bir değişiklik yapma
        new_solution = MatrixPerturb(current_solution, step_size)
        
        # yeni fitness hesapla
        new_fitness = Fitness(new_solution)
        
        # yeni fitness iyiyse kaydet
        if new_fitness > current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness
            
    # sonuçları ve uygunluk-fitness değerlerini döndür
    return current_solution, current_fitness

# Hill Climbing çalıştur
best_solution, best_fitness = HillClimbing()

# Değerleri yazdır
print(f'Best Solution: {best_solution}')
print(f'Best Value: {best_fitness}')