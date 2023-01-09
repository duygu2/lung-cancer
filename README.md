# Akciğer Kanser Tahmini

### Kullanılan makine öğrenmesi modelleri:
     SVM
     Lojistik Regresyon
     KNN

### Kullanılan optimizasyon yöntemleri:
     Hill Climbing
     Simulated Annealing 
     
SVM (Support Vector Machines) yüksek boyutlu veriler için çok yönlü karar sınırı (decision boundary) 
oluşturarak sınıflandırma yapabilen bir makine öğrenimi algoritmasıdır. SVM, veri noktalarının etrafında
en uygun karar sınırını oluşturmak için matematiksel optimizasyon yöntemlerini kullanır. Bu sayede, SVM,
veri noktalarını en iyi şekilde ayırabilir ve yeni verilere göre sınıflandırma yapabilir.

KNN (K-Nearest Neighbors) yöntemi, verilen bir sınıflandırma problemine göre, bir veri noktasının
en yakın "k" komşusunun sınıfına göre o veri noktasını sınıflandırmak için kullanılan
bir makine öğrenimi algoritmasıdır. 

Lojistik regresyon, bir sınıflandırma modelidir ve bir çıktı değişkeninin binary (ikili)
olup olmadığını tahmin etmeye yöneliktir. Bu model, veri setindeki özellikleri (bağımsız değişkenler) 
kullanarak bir sınıflandırma eşiği oluşturur ve bu eşik değerine göre tahminler yapar. 
Lojistik regresyon modeli, veri setinin çoğunlukla ikili olduğu durumlarda en yaygın olarak kullanılır, ancak birden fazla sınıf için de kullanılabilir.

Hill Climbing (Tepe Tırmanışı) Algoritması
Tepe tırmanma algoritması, dağın zirvesini veya problemin en iyi çözümünü bulmak için
sürekli artan yükseklik/değer yönünde hareket eden yerel bir arama algoritmasıdır. 
Hiçbir komşunun daha yüksek bir değere sahip olmadığı bir tepe değerine ulaştığında sona erer.

Simulated Annealing (Benzetilmiş Tavlama) Algoritması
Benzetimli tavlama (Simulated Annealing) yöntemi, ayrık ve daha az ölçüdeki sürekli 
optimizasyon problemlerini ele almak için kullanılan popüler bir metasezgisel yerel arama yöntemidir.
Benzetimli tavlamanın temel özelliği, küresel bir optimum bulma umudu ile tepe tırmanma hareketlerine
(amaç fonksiyon değerini kötüleştiren hareketlere) izin vererek yerel optimumdan kaçma aracı sağlamasıdır. 

***Akciğer Kanseri veri setindeki modellerin ve optimizasyonların sonuç tablosu:***

![image](https://user-images.githubusercontent.com/56012686/211342624-c157c33f-ac00-4ce2-b26b-f4f7adf132e4.png)

