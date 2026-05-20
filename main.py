# Pandas kitabxanasını import edirik (dataset ilə işləmək üçün)
import pandas as pd

# Support Vector Machine modeli
from sklearn.svm import SVC

# Dataseti train və test olaraq bölmək üçün
from sklearn.model_selection import train_test_split


# CSV faylını oxuyuruq
df = pd.read_csv("Cancer_Data.csv")


# Lazımsız "id" sütununu silirik
df.drop("id", axis=1, inplace=True)

# Boş və ya lazımsız olan "Unnamed: 32" sütununu silirik
df.drop("Unnamed: 32", axis=1, inplace=True)


# Diagnosis sütununu mətn formatından rəqəmə çeviririk
# B = Benign (xeyirxassəli) -> 0
# M = Malignant (bədxassəli) -> 1
df["diagnosis"] = df["diagnosis"].map({
    "B": 0,
    "M": 1
})


# Bütün sütunların korelyasiya matrisini hesablayırıq
a = df.corr()

# Diagnosis ilə ən çox əlaqəli sütunları çap edirik
print(a["diagnosis"].sort_values(ascending=True))


# Modelə az təsir edən bəzi sütunları silirik
df.drop(
    columns=[
        "fractal_dimension_se",
        "texture_se",
        "symmetry_se",
        "concavity_se",
        "compactness_se",
        "fractal_dimension_worst",
        "symmetry_mean",
        "smoothness_mean"
    ],
    inplace=True
)


# X -> giriş məlumatları (feature)
x = df.drop("diagnosis", axis=1)

# Y -> çıxış nəticəsi (target)
y = df["diagnosis"]


# Dataseti train və test olaraq bölürük
# test_size=0.2 -> məlumatın 20%-i test üçün ayrılır
# random_state=42 -> nəticələrin həmişə eyni olması üçün
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# SVM modelini yaradırıq
sv = SVC()


# Modeli train məlumatları ilə öyrədirik
model = sv.fit(x_train, y_train)