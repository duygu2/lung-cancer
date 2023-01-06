import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
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

#Eğitim ve test verileri bölünür
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2,random_state=42)

#modelin performans değerini hesaplar ve onu geri döndürür
def objective_function(solution):
    if(sum(solution) == 0):
        return 0
    #model = KNeighborsClassifier(n_neighbors=3)
    model=svm.SVC(kernel='linear')# Linear Kernel
    #model=LogisticRegression(solver='liblinear')
    model.fit(X_train.loc[:,solution],Y_train)
    score = model.score(X_test.loc[:,solution], Y_test)
    print(score)
    return score

#çözüm komşuları bulunması hedeflenir.
#özellikler sırayla değiştirilir etkileri gösterilir
#önceki değişkenle kontrol eder iyileşti mi iyileşmedi mi
def neighborhood_function(solution, obj_val):
    neighbors = []
    for i in range(len(solution)):
        temp_sol = solution.copy()
        temp_sol[i] = ~temp_sol[i]
        if ( objective_function(temp_sol) < obj_val ):
            neighbors.append(temp_sol)
    if len(neighbors) == 0:
        return None
    rand_ind = np.random.randint(0, len(neighbors))
    return neighbors[rand_ind]
    

#Sıcaklık değeri başlangıçta yüksek olur ve soğutma adımıyla düşürülür. 
initial_temp = 1
cooling_coef = 0.8
target_temp = 1e-8
max_iterations = 1000

#15 elemanlı rastgele bir dizi atıp oluşturuyoruz
#hedeflenen doğruluk değerini döndürür
solution = np.random.rand(15)>0.5
obj_val = objective_function(solution)

best_solution = solution.copy()
best_val = obj_val

#aday çözüm değeri max iterasyon sayısı kadar dönerek hesaplanır
#komşu çözüm hesaplanır aralarından en iyisi seçilir
#seçilen en iyi çözüm değeri bir önceki en iyi çözümle karşılaştırılır 
convergence = []
for i in range(max_iterations):
    candidate_solution = neighborhood_function(solution, obj_val)
    if candidate_solution is None:
        break
    cand_val = objective_function(candidate_solution)
    
    if(cand_val < obj_val ):
        obj_val, solution = cand_val, candidate_solution.copy()
        if(cand_val < best_val):
            best_val, best_solution = cand_val, candidate_solution.copy()
    else:
        acceptance_prob = np.exp(-(cand_val - obj_val)/initial_temp)
        if acceptance_prob > np.random.random():
            obj_val, solution = cand_val, candidate_solution.copy()
    convergence.append(best_val)
    initial_temp *= 1 - cooling_coef
    if initial_temp < target_temp:
        break
#sıcaklık değeri hedeflenen sıcaklık değeri düşene kadar devam eder 
print("Best objective function value:", best_val)
print("Convergence:", convergence)
#Boş dönerse belirlenen maks iterasyon sayısına ulaşmadan sonlandığı anlamına gelir