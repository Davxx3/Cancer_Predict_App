

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



import plotly.express as px

# import the csv files using pandas
cancer_df = pd.read_csv('cervical_cancer.csv')
print(cancer_df.columns)

# (int) Age
# (int) Number of sexual partners
# (int) First sexual intercourse (age)
# (int) Num of pregnancies
# (bool) Smokes
# (bool) Smokes (years)
# (bool) Smokes (packs/year)
# (bool) Hormonal Contraceptives
# (int) Hormonal Contraceptives (years)
# (bool) IUD ("IUD" stands for "intrauterine device" and used for birth control
# (int) IUD (years)
# (bool) STDs (Sexually transmitted disease)
# (int) STDs (number)
# (bool) STDs:condylomatosis
# (bool) STDs:cervical condylomatosis
# (bool) STDs:vaginal condylomatosis
# (bool) STDs:vulvo-perineal condylomatosis
# (bool) STDs:syphilis
# (bool) STDs:pelvic inflammatory disease
# (bool) STDs:genital herpes
# (bool) STDs:molluscum contagiosum
# (bool) STDs:AIDS
# (bool) STDs:HIV
# (bool) STDs:Hepatitis B
# (bool) STDs:HPV
# (int) STDs: Number of diagnosis
# (int) STDs: Time since first diagnosis
# (int) STDs: Time since last diagnosis
# (bool) Dx:Cancer
# (bool) Dx:CIN
# (bool) Dx:HPV
# (bool) Dx
# (bool) Hinselmann: target variable - A colposcopy is a procedure in which doctors examine the cervix.
# (bool) Schiller: target variable - Schiller's Iodine test is used for cervical cancer diagnosis
# (bool) Cytology: target variable - Cytology is the exam of a single cell type used for cancer screening.
# (bool) Biopsy: target variable - Biopsy is performed by removing a piece of tissue and examine it under microscope,
# Biopsy is the main way doctors diagnose most types of cancer.



# Let's replace '?' with NaN
cancer_df = cancer_df.replace('?', np.nan)
cancer_df

# Plot heatmap

# Get data frame info
cancer_df.info()

# Since STDs: Time since first diagnosis  and STDs: Time since last diagnosis have more than 80% missing values
# we can drop them
cancer_df = cancer_df.drop(columns = ['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'])
cancer_df

# Since most of the column types are object, we are not able to get the statistics of the dataframe.
# Convert them to numeric type

cancer_df = cancer_df.apply(pd.to_numeric)
cancer_df.info()

# Get the statistics of the dataframe
cancer_df.describe()





# Replace null values with mean
cancer_df = cancer_df.fillna(cancer_df.mean())
cancer_df

# Nan heatmap
sns.heatmap(cancer_df.isnull(), yticklabels = False)


cancer_df['Age'].min()

cancer_df['Age'].max()

cancer_df[cancer_df['Age'] == 84]



corr_matrix = cancer_df.corr()
corr_matrix

# Get the correlation matrix

# Plot the correlation matrix
plt.figure(figsize = (30, 30))
sns.heatmap(corr_matrix, annot = True)
plt.show()




cancer_df.hist(bins = 10, figsize = (30, 30), color = 'b')

cancer_df

target_df = cancer_df['Biopsy']
input_df = cancer_df.drop(columns = ['Biopsy', 'Hinselmann', 'Schiller', 'Citology'])



target_df.shape



X = np.array(input_df).astype('float32')
y = np.array(target_df).astype('float32')

# reshaping the array from (421570,) to (421570, 1)
# y = y.reshape(-1,1)
y.shape

# scaling the data before feeding the model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

X



# spliting the data in to test and train sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)




import numpy as np
import xgboost as xgb

# Trening modelu
model = xgb.XGBClassifier(learning_rate=0.1, max_depth=50, n_estimators=100)
model.fit(X_train, y_train)

# Predykcje prawdopodobieństw
y_proba = model.predict_proba(X_test)

# Ustawienie nowego progu klasyfikacji
threshold = 0.1
y_pred = (y_proba[:, 1] >= threshold).astype(int)

# y_pred teraz zawiera przewidywane klasy przy nowym progu

result_train = model.score(X_train, y_train)
result_train

# predict the score of the trained model using the testing dataset
result_test = model.score(X_test, y_test)
result_test

# make predictions on the test data
y_predict = model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, y_predict))

cm = confusion_matrix(y_predict, y_test)
sns.heatmap(cm, annot = True)




import numpy as np

# Przykładowe dane wejściowe
custom_data = [
    [
        28,  # 'Age' (wiek)
        1,   # 'Number of sexual partners' (liczba partnerów seksualnych)
        18,  # 'First sexual intercourse' (wiek przy pierwszym stosunku)
        0,   # 'Num of pregnancies' (liczba ciąż)
        0,   # 'Smokes' (czy pali papierosy: 0 = nie)
        0,   # 'Smokes (years)' (liczba lat palenia: 0)
        0,   # 'Smokes (packs/year)' (liczba paczek papierosów na rok: 0)
        0,   # 'Hormonal Contraceptives' (czy stosowała środki antykoncepcyjne hormonalne)
        0,   # 'Hormonal Contraceptives (years)' (liczba lat stosowania antykoncepcji hormonalnej)
        0,   # 'IUD' (czy stosowała wkładkę wewnątrzmaciczną)
        0,   # 'IUD (years)' (liczba lat stosowania wkładki wewnątrzmacicznej)
        0,   # 'STDs' (czy miała choroby przenoszone drogą płciową)
        0,   # 'STDs (number)' (liczba chorób przenoszonych drogą płciową)
        0,   # 'STDs:condylomatosis' (czy miała kondylomatozę)
        0,   # 'STDs:vaginal condylomatosis' (czy miała kłykciny pochwy)
        0,   # 'STDs:vulvo-perineal condylomatosis' (czy miała kłykciny w okolicy narządów płciowych)
        0,   # 'STDs:syphilis' (czy miała kiłę)
        0,   # 'STDs:pelvic inflammatory disease' (czy miała zapalenie miednicy)
        0,   # 'STDs:genital herpes' (czy miała opryszczkę narządów płciowych)
        0,   # 'STDs:molluscum contagiosum' (czy miała mięczaka zakaźnego)
        0,   # 'STDs:HIV' (czy miała HIV)
        0,   # 'STDs:Hepatitis B' (czy miała wirusowe zapalenie wątroby typu B)
        0,   # 'STDs:HPV' (czy miała wirusa HPV)
        0,   # 'STDs: Number of diagnosis' (liczba diagnoz STDs)
        0,   # 'STDs: Time since first diagnosis' (czas od pierwszej diagnozy STDs)
        0,   # 'STDs: Time since last diagnosis' (czas od ostatniej diagnozy STDs)
        0,   # 'Dx:Cancer' (czy miała raka)
        0,   # 'Dx:CIN' (czy miała zmiany przedrakowe)
        0,   # 'Dx:HPV' (czy miała HPV)
        0,   # 'Dx' (czy miała inne choroby)

    ]
]


# Skalowanie danych wejściowych
custom_data_scaled = scaler.transform(custom_data)

# Predykcja prawdopodobieństw
prediction_proba = model.predict_proba(custom_data_scaled)

# Ustawienie nowego progu decyzyjnego
threshold = 0.1
prediction = (prediction_proba[:, 1] >= threshold).astype(int)

# Wyświetlenie wyników
print("Predicted class based on threshold 0.1:", prediction[0])
print("Prediction probabilities:", prediction_proba[0])

# Interpretacja wyniku
if prediction[0] == 0:
    print("\nInterpretacja:")
    print("- Model przewiduje, że osoba NIE wymaga biopsji.")
    print("- Ryzyko raka szyjki macicy jest niskie.")
    print("- Prawdopodobieństwo przypisania do klasy 0 (brak konieczności biopsji): {:.2f}%.".format(prediction_proba[0][0] * 100))
    print("- Prawdopodobieństwo przypisania do klasy 1 (konieczność biopsji): {:.2f}%.".format(prediction_proba[0][1] * 100))
    print("- Zalecenie: Kontynuować rutynowe badania kontrolne.")
else:
    print("\nInterpretacja:")
    print("- Model przewiduje, że osoba MOŻE wymagać biopsji.")
    print("- Ryzyko raka szyjki macicy jest podwyższone.")
    print("- Prawdopodobieństwo przypisania do klasy 0 (brak konieczności biopsji): {:.2f}%.".format(prediction_proba[0][0] * 100))
    print("- Prawdopodobieństwo przypisania do klasy 1 (konieczność biopsji): {:.2f}%.".format(prediction_proba[0][1] * 100))
    print("- Zalecenie: Skonsultować wyniki z lekarzem specjalistą i rozważyć biopsję.")

import numpy as np

# Przykładowe dane wejściowe
custom_data = [
    [
        45,   # 'Age' (wiek) - starsza osoba, wyższe ryzyko
        5,    # 'Number of sexual partners' (liczba partnerów seksualnych) - większa liczba partnerów, wyższe ryzyko STDs
        18,   # 'First sexual intercourse' (wiek przy pierwszym stosunku) - wczesny wiek przy pierwszym stosunku
        3,    # 'Num of pregnancies' (liczba ciąż) - 3 ciąże, typowe dla wielu kobiet
        1,    # 'Smokes' (czy pali papierosy: 1 = tak) - palenie zwiększa ryzyko
        15,   # 'Smokes (years)' (liczba lat palenia) - długotrwałe palenie
        10,   # 'Smokes (packs/year)' (liczba paczek papierosów na rok)
        1,    # 'Hormonal Contraceptives' (czy stosowała środki antykoncepcyjne hormonalne: 1 = tak)
        10,   # 'Hormonal Contraceptives (years)' (liczba lat stosowania antykoncepcji hormonalnej)
        1,    # 'IUD' (czy stosowała wkładkę wewnątrzmaciczną: 1 = tak)
        5,    # 'IUD (years)' (liczba lat stosowania wkładki)
        1,    # 'STDs' (czy miała choroby przenoszone drogą płciową: 1 = tak)
        2,    # 'STDs (number)' (liczba chorób przenoszonych drogą płciową)
        1,    # 'STDs:condylomatosis' (czy miała kondylomatozę: 1 = tak)
        0,    # 'STDs:vaginal condylomatosis' (czy miała kłykciny pochwy: 0 = nie)
        0,    # 'STDs:vulvo-perineal condylomatosis' (czy miała kłykciny w okolicy narządów płciowych: 0 = nie)
        1,    # 'STDs:syphilis' (czy miała kiłę: 1 = tak)
        1,    # 'STDs:pelvic inflammatory disease' (czy miała zapalenie miednicy: 1 = tak)
        1,    # 'STDs:genital herpes' (czy miała opryszczkę narządów płciowych: 1 = tak)
        1,    # 'STDs:molluscum contagiosum' (czy miała mięczaka zakaźnego: 1 = tak)
        1,    # 'STDs:HIV' (czy miała HIV: 1 = tak)
        0,    # 'STDs:Hepatitis B' (czy miała wirusowe zapalenie wątroby typu B: 0 = nie)
        1,    # 'STDs:HPV' (czy miała wirusa HPV: 1 = tak)
        3,    # 'STDs: Number of diagnosis' (liczba diagnoz STDs)
        1,    # 'STDs: Time since first diagnosis' (czas od pierwszej diagnozy STDs)
        0,    # 'STDs: Time since last diagnosis' (czas od ostatniej diagnozy STDs)
        1,    # 'Dx:Cancer' (czy miała raka: 1 = tak)
        0,    # 'Dx:CIN' (czy miała zmiany przedrakowe: 0 = nie)
        1,    # 'Dx:HPV' (czy miała HPV: 1 = tak)
        0,    # 'Dx' (czy miała inne choroby: 0 = nie)

    ]
]



# Skalowanie danych wejściowych
custom_data_scaled = scaler.transform(custom_data)

# Predykcja prawdopodobieństw
prediction_proba = model.predict_proba(custom_data_scaled)

# Ustawienie nowego progu decyzyjnego
threshold = 0.1
prediction = (prediction_proba[:, 1] >= threshold).astype(int)

# Wyświetlenie wyników
print("Predicted class based on threshold 0.1:", prediction[0])
print("Prediction probabilities:", prediction_proba[0])

# Interpretacja wyniku
if prediction[0] == 0:
    print("\nInterpretacja:")
    print("- Model przewiduje, że osoba NIE wymaga biopsji.")
    print("- Ryzyko raka szyjki macicy jest niskie.")
    print("- Prawdopodobieństwo przypisania do klasy 0 (brak konieczności biopsji): {:.2f}%.".format(prediction_proba[0][0] * 100))
    print("- Prawdopodobieństwo przypisania do klasy 1 (konieczność biopsji): {:.2f}%.".format(prediction_proba[0][1] * 100))
    print("- Zalecenie: Kontynuować rutynowe badania kontrolne.")
else:
    print("\nInterpretacja:")
    print("- Model przewiduje, że osoba MOŻE wymagać biopsji.")
    print("- Ryzyko raka szyjki macicy jest podwyższone.")
    print("- Prawdopodobieństwo przypisania do klasy 0 (brak konieczności biopsji): {:.2f}%.".format(prediction_proba[0][0] * 100))
    print("- Prawdopodobieństwo przypisania do klasy 1 (konieczność biopsji): {:.2f}%.".format(prediction_proba[0][1] * 100))
    print("- Zalecenie: Skonsultować wyniki z lekarzem specjalistą i rozważyć biopsję.")

import numpy as np

# Funkcja do zbierania danych od użytkownika
def get_user_input():
    print("Wprowadź dane pacjenta:")
    data = [
        int(input("Age (wiek): ")),
        int(input("Number of sexual partners (liczba partnerów seksualnych): ")),
        int(input("First sexual intercourse (wiek przy pierwszym stosunku): ")),
        int(input("Num of pregnancies (liczba ciąż): ")),
        int(input("Smokes (czy pali papierosy, 1 = tak, 0 = nie): ")),
        int(input("Smokes (years) (liczba lat palenia): ")),
        int(input("Smokes (packs/year) (liczba paczek papierosów na rok): ")),
        int(input("Hormonal Contraceptives (czy stosowała środki antykoncepcyjne hormonalne, 1 = tak, 0 = nie): ")),
        int(input("Hormonal Contraceptives (years) (liczba lat stosowania antykoncepcji hormonalnej): ")),
        int(input("IUD (czy stosowała wkładkę wewnątrzmaciczną, 1 = tak, 0 = nie): ")),
        int(input("IUD (years) (liczba lat stosowania wkładki wewnątrzmacicznej): ")),
        int(input("STDs (czy miała choroby przenoszone drogą płciową, 1 = tak, 0 = nie): ")),
        int(input("STDs (number) (liczba chorób przenoszonych drogą płciową): ")),
        int(input("STDs:condylomatosis (czy miała kondylomatozę, 1 = tak, 0 = nie): ")),
        int(input("STDs:vaginal condylomatosis (czy miała kłykciny pochwy, 1 = tak, 0 = nie): ")),
        int(input("STDs:vulvo-perineal condylomatosis (czy miała kłykciny w okolicy narządów płciowych, 1 = tak, 0 = nie): ")),
        int(input("STDs:syphilis (czy miała kiłę, 1 = tak, 0 = nie): ")),
        int(input("STDs:pelvic inflammatory disease (czy miała zapalenie miednicy, 1 = tak, 0 = nie): ")),
        int(input("STDs:genital herpes (czy miała opryszczkę narządów płciowych, 1 = tak, 0 = nie): ")),
        int(input("STDs:molluscum contagiosum (czy miała mięczaka zakaźnego, 1 = tak, 0 = nie): ")),
        int(input("STDs:HIV (czy miała HIV, 1 = tak, 0 = nie): ")),
        int(input("STDs:Hepatitis B (czy miała wirusowe zapalenie wątroby typu B, 1 = tak, 0 = nie): ")),
        int(input("STDs:HPV (czy miała wirusa HPV, 1 = tak, 0 = nie): ")),
        int(input("STDs: Number of diagnosis (liczba diagnoz STDs): ")),
        int(input("STDs: Time since first diagnosis (czas od pierwszej diagnozy STDs): ")),
        int(input("STDs: Time since last diagnosis (czas od ostatniej diagnozy STDs): ")),
        int(input("Dx:Cancer (czy miała raka, 1 = tak, 0 = nie): ")),
        int(input("Dx:CIN (czy miała zmiany przedrakowe, 1 = tak, 0 = nie): ")),
        int(input("Dx:HPV (czy miała HPV, 1 = tak, 0 = nie): ")),
        int(input("Dx (czy miała inne choroby, 1 = tak, 0 = nie): "))
    ]
    return np.array([data])

# Pobranie danych od użytkownika
custom_data = get_user_input()

# Skalowanie danych wejściowych
custom_data_scaled = scaler.transform(custom_data)

# Ustawienie nowego progu decyzyjnego
threshold = 0.1

# Predykcja prawdopodobieństw
prediction_proba = model.predict_proba(custom_data_scaled)

# Przypisanie klasy na podstawie progu decyzyjnego
prediction = (prediction_proba[:, 1] >= threshold).astype(int)

# Wyświetlenie wyników
print("\nPredicted class based on threshold {:.2f}: {}".format(threshold, prediction[0]))
print("Prediction probabilities:", prediction_proba[0])

# Interpretacja wyniku
if prediction[0] == 0:
    print("\nInterpretacja:")
    print("- Model przewiduje, że osoba NIE wymaga biopsji.")
    print("- Ryzyko raka szyjki macicy jest niskie.")
    print("- Prawdopodobieństwo przypisania do klasy 0 (brak konieczności biopsji): {:.2f}%.".format(prediction_proba[0][0] * 100))
    print("- Prawdopodobieństwo przypisania do klasy 1 (konieczność biopsji): {:.2f}%.".format(prediction_proba[0][1] * 100))
    print("- Zalecenie: Kontynuować rutynowe badania kontrolne.")
else:
    print("\nInterpretacja:")
    print("- Model przewiduje, że osoba MOŻE wymagać biopsji.")
    print("- Ryzyko raka szyjki macicy jest podwyższone.")
    print("- Prawdopodobieństwo przypisania do klasy 0 (brak konieczności biopsji): {:.2f}%.".format(prediction_proba[0][0] * 100))
    print("- Prawdopodobieństwo przypisania do klasy 1 (konieczność biopsji): {:.2f}%.".format(prediction_proba[0][1] * 100))
    print("- Zalecenie: Skonsultować wyniki z lekarzem specjalistą i rozważyć biopsję.")









