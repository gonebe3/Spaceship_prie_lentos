import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from duomenu_paruosimas import pasalinti_nereikalingas

def ikelti_duomenis(failo_kelias):
    try:
        return pd.read_csv(failo_kelias, index_col=0)
    except FileNotFoundError:
        print(f"Klaida, failas nerastas")
        raise FileNotFoundError ("Klaida, failas nerastas")
    
df = ikelti_duomenis("data/train.csv")
df_test = ikelti_duomenis("data/test.csv")

# print(df.head())

# df.info()
# print(df.describe())
df = pasalinti_nereikalingas(df)
df_test = pasalinti_nereikalingas(df_test)
# print(df.describe(include="object"))

# print(df.head())

# for column in df.select_dtypes(include='object').columns:
#     sns.countplot(df,x=column)
#     plt.show()

# for column in df.select_dtypes(include='number').columns:
#     sns.histplot(df,x=column, kde=True)
#     plt.show()

df_skaitinis = df.select_dtypes(include=[np.number, bool])
koreliacija = df_skaitinis.corr()

# plt.figure(figsize=(8, 6))
# sns.heatmap(koreliacija, annot=True, cmap='viridis', fmt='.2f', cbar=True)
# plt.title("Koreliaciju zemelapis")
# plt.show()
columns_with_outliers = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# sns.histplot(df,x='Spa', kde=True)
# plt.show()


# Q1 = df['Spa'].quantile(0.25)
# Q3 = df['Spa'].quantile(0.75)
# IQR = Q3 - Q1

# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
 
# outliers = df[(df['Spa'] < lower_bound) | (df['Spa'] > upper_bound)]
for column in columns_with_outliers:
    lower = df[column].quantile(0.01)   # 1st percentile
    upper = df[column].quantile(0.95)   # 99th percentile
    
    # Clip values
    df[column] = df[column].clip(lower, upper)
# df['Spa_clipped'] = df['Spa'].clip(lower_bound, upper_bound)

for column in columns_with_outliers:
    lower = df_test[column].quantile(0.01)   # 1st percentile
    upper = df_test[column].quantile(0.95)   # 99th percentile
    
    # Clip values
    df_test[column] = df_test[column].clip(lower, upper)
# df['Spa_clipped'] = df['Spa'].clip(lower_bound, u


# sns.histplot(df,x='Spa', kde=True)
# plt.show()
df = df.dropna()
# print(df.duplicated().sum())
df = df.drop_duplicates()

for col in df_test.select_dtypes(exclude=['number']).columns:
    mode_val = df_test[col].mode()
    df_test[col].fillna(mode_val[0], inplace=True)

for col in df_test.select_dtypes(include=['number']).columns:
    mode_val = df_test[col].median()
    df_test[col].fillna(mode_val, inplace=True)
# print(outliers)

print(df.isnull().sum())

print(df.shape)

df = pd.get_dummies(df, columns=['Destination','HomePlanet'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['Destination','HomePlanet'], drop_first=True)
# print(df.head())

X = df.drop('Transported', axis=1)
y = df['Transported']

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

scaler = StandardScaler()

x_scaled = scaler.fit_transform(X)
# print(x_scaled)
x_test_scaled = scaler.transform(df_test)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

galimas_k = list(range(2,3))
visi_acc = []
for k in galimas_k:
    modelis_su_k = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    dabartinis_acc = cross_val_score(estimator=modelis_su_k, X= x_scaled, y=y, cv=5, scoring='accuracy')
    visi_acc.append(dabartinis_acc.mean())
    
best_k = galimas_k[np.argmax(visi_acc)]
print(f'Best K: {best_k}')

# sns.lineplot(x=galimas_k, y=visi_acc)
# plt.show()

from sklearn.svm import SVC
# modelis = KNeighborsClassifier(n_neighbors=best_k)

modelis = SVC()
modelis.fit(x_scaled, y)

spejimai = modelis.predict(x_test_scaled)

submission = pd.DataFrame({
    'PassengerId': df_test.index,
    'Transported': spejimai
})

submission_path = 'data/submission.csv'

submission.to_csv(submission_path, index=False)