from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import traceback

app = Flask(__name__)

# Load and prepare the model
def load_model():
    # Load the dataset
    cancer_df = pd.read_csv('cervical_cancer.csv')
    print("Original columns:", cancer_df.columns.tolist())
    
    # Replace '?' with NaN
    cancer_df = cancer_df.replace('?', np.nan)
    
    # Drop columns with high missing values
    cancer_df = cancer_df.drop(columns=['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'])
    print("Columns after dropping:", cancer_df.columns.tolist())
    
    # Convert to numeric
    cancer_df = cancer_df.apply(pd.to_numeric)
    
    # Fill missing values with mean
    cancer_df = cancer_df.fillna(cancer_df.mean())
    
    # Prepare target and input data
    target_df = cancer_df['Biopsy']
    input_df = cancer_df.drop(columns=['Biopsy', 'Hinselmann', 'Schiller', 'Citology'])
    print("Final input columns:", input_df.columns.tolist())
    print("Number of input columns:", len(input_df.columns))
    
    # Convert to numpy arrays
    X = np.array(input_df).astype('float32')
    y = np.array(target_df).astype('float32')
    
    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data - identycznie jak w cervical_cancer_prediction_using_xg_boost_algorithm_4-1.py
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    # Train the model
    model = xgb.XGBClassifier(learning_rate=0.1, max_depth=50, n_estimators=100)
    model.fit(X_train, y_train)
    
    return model, scaler, input_df.columns.tolist()

# Load model and scaler
model, scaler, feature_columns = load_model()
print("Feature columns used in model:", feature_columns)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Otrzymane dane:", data)  # Dodajemy logowanie otrzymanych danych

        # Sprawdź, czy wszystkie wymagane pola są obecne
        required_fields = [
            'age', 'sexual_partners', 'first_sex', 'pregnancies',
            'smokes', 'smokes_years', 'smokes_packs_year',
            'hormonal_contraceptives', 'hormonal_contraceptives_years',
            'iud', 'iud_years', 'stds', 'stds_number',
            'stds_condylomatosis', 'stds_cervical_condylomatosis',
            'stds_vaginal_condylomatosis', 'stds_vulvo_perineal_condylomatosis',
            'stds_syphilis', 'stds_pelvic_inflammatory_disease',
            'stds_genital_herpes', 'stds_molluscum_contagiosum',
            'stds_aids', 'stds_hiv', 'stds_hepatitis_b',
            'stds_hpv', 'stds_diagnosis_number',
            'dx_cancer', 'dx_cin', 'dx_hpv', 'dx'
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f"Brakujące pola: {', '.join(missing_fields)}"
            })

        # Tworzenie słownika z danymi
        features = {
            'Age': float(data['age']),
            'Number of sexual partners': float(data['sexual_partners']),
            'First sexual intercourse': float(data['first_sex']),
            'Num of pregnancies': float(data['pregnancies']),
            'Smokes': float(data['smokes']),
            'Smokes (years)': float(data['smokes_years']),
            'Smokes (packs/year)': float(data['smokes_packs_year']),
            'Hormonal Contraceptives': float(data['hormonal_contraceptives']),
            'Hormonal Contraceptives (years)': float(data['hormonal_contraceptives_years']),
            'IUD': float(data['iud']),
            'IUD (years)': float(data['iud_years']),
            'STDs': float(data['stds']),
            'STDs (number)': float(data['stds_number']),
            'STDs:condylomatosis': float(data['stds_condylomatosis']),
            'STDs:cervical condylomatosis': float(data['stds_cervical_condylomatosis']),
            'STDs:vaginal condylomatosis': float(data['stds_vaginal_condylomatosis']),
            'STDs:vulvo-perineal condylomatosis': float(data['stds_vulvo_perineal_condylomatosis']),
            'STDs:syphilis': float(data['stds_syphilis']),
            'STDs:pelvic inflammatory disease': float(data['stds_pelvic_inflammatory_disease']),
            'STDs:genital herpes': float(data['stds_genital_herpes']),
            'STDs:molluscum contagiosum': float(data['stds_molluscum_contagiosum']),
            'STDs:AIDS': float(data['stds_aids']),
            'STDs:HIV': float(data['stds_hiv']),
            'STDs:Hepatitis B': float(data['stds_hepatitis_b']),
            'STDs:HPV': float(data['stds_hpv']),
            'STDs: Number of diagnosis': float(data['stds_diagnosis_number']),
            'Dx:Cancer': float(data['dx_cancer']),
            'Dx:CIN': float(data['dx_cin']),
            'Dx:HPV': float(data['dx_hpv']),
            'Dx': float(data['dx'])
        }

        print("Przygotowane dane:", features)  # Dodajemy logowanie przygotowanych danych

        # Konwersja na DataFrame
        df = pd.DataFrame([features])
        print("DataFrame:", df)  # Dodajemy logowanie DataFrame

        # Skalowanie danych przed przekazaniem do modelu
        df_array = np.array(df).astype('float32')
        df_scaled = scaler.transform(df_array)
        print("Dane po skalowaniu:", df_scaled)  # Dodajemy logowanie skalowanych danych

        # Wykonanie predykcji
        probability = model.predict_proba(df_scaled)[0][1]
        # Użyj progu 0.1 do klasyfikacji, tak jak w oryginalnym skrypcie
        threshold = 0.1
        prediction = 1 if probability >= threshold else 0
        
        print(f"Prawdopodobieństwo: {probability}, Próg: {threshold}, Przewidywana klasa: {prediction}")

        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability_1': float(probability)
        })

    except Exception as e:
        print("Wystąpił błąd:", str(e))  # Dodajemy logowanie błędów
        return jsonify({
            'success': False,
            'error': f"Wystąpił błąd podczas przetwarzania danych: {str(e)}"
        })

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Nie można uruchomić aplikacji: {str(e)}")
        print("Szczegóły błędu:")
        print(traceback.format_exc()) 