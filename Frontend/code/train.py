import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import joblib
import gzip
import os
import json
import warnings
warnings.filterwarnings('ignore')

class DiseaseModelTrainer:
    """
    Comprehensive trainer for all disease prediction models
    """

    def __init__(self):
        self.models_dir = 'models'
        self.data_dir = 'data'
        self.model_dir = 'model'

        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def train_symptom_based_disease_model(self):
        """Train the main symptom-based disease prediction model (XGBoost)"""
        print("Training Symptom-based Disease Prediction Model...")

        # Import the dataset
        dataset_df = pd.read_csv(f'{self.data_dir}/dataset.csv')

        # Preprocess
        dataset_df = dataset_df.apply(lambda col: col.str.strip())

        test = pd.get_dummies(dataset_df.filter(regex='Symptom'), prefix='', prefix_sep='')
        test = test.groupby(test.columns, axis=1).agg(np.max)
        clean_df = pd.merge(test, dataset_df['Disease'], left_index=True, right_index=True)

        clean_df.to_csv(f'{self.data_dir}/clean_dataset.tsv', sep='\t', index=False)

        # Preprocessing
        X_data = clean_df.iloc[:,:-1]
        y_data = clean_df.iloc[:,-1]

        # Convert y to categorical values
        y_data = y_data.astype('category')

        # Convert y categories to numbers with encoder
        le = preprocessing.LabelEncoder()
        le.fit(y_data)

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

        # Convert labels to numbers
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)

        # Init classifier
        model = xgb.XGBClassifier(random_state=42)

        # Fit
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Test accuracy
        accuracy = accuracy_score(y_test, preds)
        print(f"Symptom-based Disease Model Accuracy: {accuracy:.4f}")

        # Export model
        joblib.dump(model, gzip.open(f'{self.model_dir}/model_binary.dat.gz', "wb"))
        model.save_model(f"{self.model_dir}/xgboost_model.json")

        return accuracy

    def create_synthetic_diabetes_data(self, n_samples=1000):
        """Create synthetic diabetes dataset for training"""
        np.random.seed(42)

        # Generate synthetic features
        pregnancies = np.random.randint(0, 18, n_samples)
        glucose = np.random.normal(120, 30, n_samples)
        blood_pressure = np.random.normal(70, 20, n_samples)
        skin_thickness = np.random.normal(20, 15, n_samples)
        insulin = np.random.normal(80, 100, n_samples)
        bmi = np.random.normal(32, 8, n_samples)
        diabetes_pedigree = np.random.uniform(0.08, 2.5, n_samples)
        age = np.random.randint(21, 81, n_samples)

        # Create target based on realistic conditions
        diabetes_risk = (
            (glucose > 140) * 0.3 +
            (bmi > 30) * 0.2 +
            (age > 45) * 0.2 +
            (diabetes_pedigree > 0.5) * 0.15 +
            (pregnancies > 5) * 0.1 +
            (blood_pressure > 90) * 0.05
        )

        # Add some randomness
        diabetes_risk += np.random.normal(0, 0.1, n_samples)
        outcome = (diabetes_risk > 0.5).astype(int)

        # Create DataFrame
        data = pd.DataFrame({
            'Pregnancies': pregnancies,
            'Glucose': np.clip(glucose, 0, 200),
            'BloodPressure': np.clip(blood_pressure, 0, 122),
            'SkinThickness': np.clip(skin_thickness, 0, 100),
            'Insulin': np.clip(insulin, 0, 846),
            'BMI': np.clip(bmi, 0, 67),
            'DiabetesPedigreeFunction': diabetes_pedigree,
            'Age': age,
            'Outcome': outcome
        })

        return data

    def calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate comprehensive metrics for model evaluation"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        print(f"{model_name} Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        return metrics

    def train_diabetes_model(self):
        """Train diabetes prediction model"""
        print("Training Diabetes Prediction Model...")

        # Create synthetic data
        data = self.create_synthetic_diabetes_data()

        X = data.drop('Outcome', axis=1)
        y = data['Outcome']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        metrics = self.calculate_metrics(y_test, preds, "Diabetes Model")

        # Save model and metrics
        joblib.dump(model, f'{self.models_dir}/diabetes_model.sav')
        joblib.dump(metrics, f'{self.models_dir}/diabetes_metrics.pkl')

        return metrics

    def create_synthetic_heart_data(self, n_samples=1000):
        """Create synthetic heart disease dataset"""
        np.random.seed(42)

        age = np.random.randint(29, 78, n_samples)
        sex = np.random.randint(0, 2, n_samples)
        cp = np.random.randint(0, 4, n_samples)
        trestbps = np.random.normal(130, 20, n_samples)
        chol = np.random.normal(245, 50, n_samples)
        fbs = np.random.randint(0, 2, n_samples)
        restecg = np.random.randint(0, 3, n_samples)
        thalach = np.random.normal(150, 25, n_samples)
        exang = np.random.randint(0, 2, n_samples)
        oldpeak = np.random.uniform(0, 6.2, n_samples)
        slope = np.random.randint(0, 3, n_samples)
        ca = np.random.randint(0, 4, n_samples)
        thal = np.random.randint(0, 3, n_samples)

        # Create target based on risk factors
        heart_risk = (
            (age > 55) * 0.2 +
            (sex == 1) * 0.15 +  # Male higher risk
            (cp == 0) * 0.2 +    # Typical angina
            (trestbps > 140) * 0.15 +
            (chol > 240) * 0.1 +
            (thalach < 120) * 0.1 +
            (exang == 1) * 0.1
        )

        heart_risk += np.random.normal(0, 0.1, n_samples)
        target = (heart_risk > 0.4).astype(int)

        data = pd.DataFrame({
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': np.clip(trestbps, 94, 200),
            'chol': np.clip(chol, 126, 564),
            'fbs': fbs,
            'restecg': restecg,
            'thalach': np.clip(thalach, 71, 202),
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal,
            'target': target
        })

        return data

    def train_heart_model(self):
        """Train heart disease prediction model"""
        print("Training Heart Disease Prediction Model...")

        data = self.create_synthetic_heart_data()

        X = data.drop('target', axis=1)
        y = data['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = self.calculate_metrics(y_test, preds, "Heart Disease Model")

        joblib.dump(model, f'{self.models_dir}/heart_disease_model.sav')
        joblib.dump(metrics, f'{self.models_dir}/heart_metrics.pkl')
        return metrics

    def create_synthetic_parkinsons_data(self, n_samples=1000):
        """Create synthetic Parkinson's disease dataset"""
        np.random.seed(42)

        # Generate features based on typical Parkinson's voice measurements
        mdvp_fo = np.random.normal(154, 40, n_samples)
        mdvp_fhi = np.random.normal(197, 90, n_samples)
        mdvp_flo = np.random.normal(116, 40, n_samples)
        mdvp_jitter = np.random.uniform(0.00168, 0.03316, n_samples)
        mdvp_jitter_abs = np.random.uniform(0.000007, 0.000260, n_samples)
        mdvp_rap = np.random.uniform(0.00068, 0.02144, n_samples)
        mdvp_ppq = np.random.uniform(0.00092, 0.01958, n_samples)
        jitter_ddp = np.random.uniform(0.00204, 0.06433, n_samples)
        mdvp_shimmer = np.random.uniform(0.00954, 0.11908, n_samples)
        mdvp_shimmer_db = np.random.uniform(0.085, 1.302, n_samples)
        shimmer_apq3 = np.random.uniform(0.00455, 0.05647, n_samples)
        shimmer_apq5 = np.random.uniform(0.00757, 0.07946, n_samples)
        mdvp_apq = np.random.uniform(0.00719, 0.13778, n_samples)
        shimmer_dda = np.random.uniform(0.01364, 0.16942, n_samples)
        nhr = np.random.uniform(0.00065, 0.31482, n_samples)
        hnr = np.random.uniform(8.441, 33.047, n_samples)
        rpde = np.random.uniform(0.256570, 0.685151, n_samples)
        dfa = np.random.uniform(0.574282, 0.825288, n_samples)
        spread1 = np.random.uniform(-7.964984, -2.434031, n_samples)
        spread2 = np.random.uniform(0.006274, 0.450493, n_samples)
        d2 = np.random.uniform(1.423287, 3.671155, n_samples)
        ppe = np.random.uniform(0.044539, 0.527367, n_samples)

        # Create target based on voice abnormalities
        parkinsons_risk = (
            (mdvp_jitter > 0.01) * 0.2 +
            (mdvp_shimmer > 0.05) * 0.2 +
            (nhr > 0.1) * 0.15 +
            (hnr < 15) * 0.15 +
            (rpde > 0.5) * 0.15 +
            (dfa > 0.7) * 0.15
        )

        parkinsons_risk += np.random.normal(0, 0.1, n_samples)
        status = (parkinsons_risk > 0.5).astype(int)

        data = pd.DataFrame({
            'MDVP:Fo(Hz)': mdvp_fo,
            'MDVP:Fhi(Hz)': mdvp_fhi,
            'MDVP:Flo(Hz)': mdvp_flo,
            'MDVP:Jitter(%)': mdvp_jitter,
            'MDVP:Jitter(Abs)': mdvp_jitter_abs,
            'MDVP:RAP': mdvp_rap,
            'MDVP:PPQ': mdvp_ppq,
            'Jitter:DDP': jitter_ddp,
            'MDVP:Shimmer': mdvp_shimmer,
            'MDVP:Shimmer(dB)': mdvp_shimmer_db,
            'Shimmer:APQ3': shimmer_apq3,
            'Shimmer:APQ5': shimmer_apq5,
            'MDVP:APQ': mdvp_apq,
            'Shimmer:DDA': shimmer_dda,
            'NHR': nhr,
            'HNR': hnr,
            'RPDE': rpde,
            'DFA': dfa,
            'spread1': spread1,
            'spread2': spread2,
            'D2': d2,
            'PPE': ppe,
            'status': status
        })

        return data

    def train_parkinsons_model(self):
        """Train Parkinson's disease prediction model"""
        print("Training Parkinson's Disease Prediction Model...")

        data = self.create_synthetic_parkinsons_data()

        X = data.drop('status', axis=1)
        y = data['status']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = SVC(kernel='rbf', random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        print(f"Parkinson's Disease Model Accuracy: {accuracy:.4f}")

        joblib.dump(model, f'{self.models_dir}/parkinsons_model.sav')
        return accuracy

    def create_synthetic_liver_data(self, n_samples=1000):
        """Create synthetic liver disease dataset"""
        np.random.seed(42)

        age = np.random.randint(4, 90, n_samples)
        gender = np.random.randint(0, 2, n_samples)  # 0: Male, 1: Female
        total_bilirubin = np.random.uniform(0.4, 75.0, n_samples)
        direct_bilirubin = np.random.uniform(0.1, 19.7, n_samples)
        alkaline_phosphotase = np.random.randint(63, 2110, n_samples)
        alamine_aminotransferase = np.random.randint(10, 2000, n_samples)
        aspartate_aminotransferase = np.random.randint(10, 4929, n_samples)
        total_protiens = np.random.uniform(2.7, 9.6, n_samples)
        albumin = np.random.uniform(0.9, 5.5, n_samples)
        albumin_and_globulin_ratio = np.random.uniform(0.3, 2.8, n_samples)

        # Create target based on liver function indicators
        liver_risk = (
            (total_bilirubin > 1.2) * 0.2 +
            (alkaline_phosphotase > 147) * 0.15 +
            (alamine_aminotransferase > 56) * 0.2 +
            (aspartate_aminotransferase > 40) * 0.2 +
            (albumin < 3.5) * 0.15 +
            (albumin_and_globulin_ratio < 1.0) * 0.1
        )

        liver_risk += np.random.normal(0, 0.1, n_samples)
        dataset = (liver_risk > 0.4).astype(int)

        data = pd.DataFrame({
            'Age': age,
            'Gender': gender,
            'Total_Bilirubin': total_bilirubin,
            'Direct_Bilirubin': direct_bilirubin,
            'Alkaline_Phosphotase': alkaline_phosphotase,
            'Alamine_Aminotransferase': alamine_aminotransferase,
            'Aspartate_Aminotransferase': aspartate_aminotransferase,
            'Total_Protiens': total_protiens,
            'Albumin': albumin,
            'Albumin_and_Globulin_Ratio': albumin_and_globulin_ratio,
            'Dataset': dataset
        })

        return data

    def train_liver_model(self):
        """Train liver disease prediction model"""
        print("Training Liver Disease Prediction Model...")

        data = self.create_synthetic_liver_data()

        X = data.drop('Dataset', axis=1)
        y = data['Dataset']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        print(f"Liver Disease Model Accuracy: {accuracy:.4f}")

        joblib.dump(model, f'{self.models_dir}/liver_model.sav')
        return accuracy



    def create_synthetic_hepatitis_data(self, n_samples=1000):
        """Create synthetic hepatitis dataset"""
        np.random.seed(42)

        age = np.random.randint(7, 78, n_samples)
        sex = np.random.randint(1, 3, n_samples)  # 1: Male, 2: Female
        alb = np.random.uniform(14.9, 82.2, n_samples)
        alp = np.random.uniform(11.3, 416.6, n_samples)
        alt = np.random.uniform(4.0, 325.3, n_samples)
        ast = np.random.uniform(10.6, 324.0, n_samples)
        bil = np.random.uniform(0.8, 254.0, n_samples)
        che = np.random.uniform(1.42, 16.41, n_samples)
        chol = np.random.uniform(1.43, 9.67, n_samples)
        crea = np.random.uniform(8.0, 1079.1, n_samples)
        ggt = np.random.uniform(4.5, 650.9, n_samples)
        prot = np.random.uniform(44.8, 90.0, n_samples)

        # Create target based on liver function indicators
        hepatitis_risk = (
            (alt > 56) * 0.2 +
            (ast > 40) * 0.2 +
            (bil > 1.2) * 0.15 +
            (ggt > 61) * 0.15 +
            (alp > 147) * 0.15 +
            (age > 50) * 0.15
        )

        hepatitis_risk += np.random.normal(0, 0.1, n_samples)
        category = (hepatitis_risk > 0.5).astype(int)

        data = pd.DataFrame({
            'Age': age,
            'Sex': sex,
            'ALB': alb,
            'ALP': alp,
            'ALT': alt,
            'AST': ast,
            'BIL': bil,
            'CHE': che,
            'CHOL': chol,
            'CREA': crea,
            'GGT': ggt,
            'PROT': prot,
            'Category': category
        })

        return data

    def train_hepatitis_model(self):
        """Train hepatitis prediction model"""
        print("Training Hepatitis Prediction Model...")

        data = self.create_synthetic_hepatitis_data()

        X = data.drop('Category', axis=1)
        y = data['Category']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        print(f"Hepatitis Model Accuracy: {accuracy:.4f}")

        joblib.dump(model, f'{self.models_dir}/hepititisc_model.sav')
        return accuracy

    def create_synthetic_chronic_kidney_data(self, n_samples=1000):
        """Create synthetic chronic kidney disease dataset with balanced classes"""
        np.random.seed(42)

        # Create balanced dataset - 70% healthy, 30% with kidney disease
        n_healthy = int(n_samples * 0.7)
        n_diseased = n_samples - n_healthy

        # Initialize arrays
        age = np.zeros(n_samples)
        bp = np.zeros(n_samples)
        sg = np.zeros(n_samples)
        al = np.zeros(n_samples)
        su = np.zeros(n_samples)
        rbc = np.zeros(n_samples)
        pc = np.zeros(n_samples)
        pcc = np.zeros(n_samples)
        ba = np.zeros(n_samples)
        bgr = np.zeros(n_samples)
        bu = np.zeros(n_samples)
        sc = np.zeros(n_samples)
        sod = np.zeros(n_samples)
        pot = np.zeros(n_samples)
        hemo = np.zeros(n_samples)
        pcv = np.zeros(n_samples)
        wc = np.zeros(n_samples)
        rc = np.zeros(n_samples)
        htn = np.zeros(n_samples)
        dm = np.zeros(n_samples)
        cad = np.zeros(n_samples)
        appet = np.zeros(n_samples)
        pe = np.zeros(n_samples)
        ane = np.zeros(n_samples)

        # Generate healthy samples (first n_healthy samples)
        age[:n_healthy] = np.random.randint(20, 60, n_healthy)  # Younger age
        bp[:n_healthy] = np.random.randint(90, 140, n_healthy)  # Normal BP
        sg[:n_healthy] = np.random.uniform(1.010, 1.025, n_healthy)  # Normal specific gravity
        al[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.9, 0.1])  # Mostly no albumin
        su[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.95, 0.05])  # Mostly no sugar
        rbc[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.1, 0.9])  # Normal RBC
        pc[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.9, 0.1])  # Normal pus cells
        pcc[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.95, 0.05])  # No pus cell clumps
        ba[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.9, 0.1])  # No bacteria
        bgr[:n_healthy] = np.random.randint(70, 120, n_healthy)  # Normal glucose
        bu[:n_healthy] = np.random.randint(10, 25, n_healthy)  # Normal blood urea
        sc[:n_healthy] = np.random.uniform(0.5, 1.2, n_healthy)  # Normal serum creatinine
        sod[:n_healthy] = np.random.randint(135, 145, n_healthy)  # Normal sodium
        pot[:n_healthy] = np.random.uniform(3.5, 5.0, n_healthy)  # Normal potassium
        hemo[:n_healthy] = np.random.uniform(12.0, 16.0, n_healthy)  # Normal hemoglobin
        pcv[:n_healthy] = np.random.randint(35, 50, n_healthy)  # Normal PCV
        wc[:n_healthy] = np.random.randint(4000, 11000, n_healthy)  # Normal WBC
        rc[:n_healthy] = np.random.uniform(4.0, 6.0, n_healthy)  # Normal RBC count
        htn[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.8, 0.2])  # Less hypertension
        dm[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.9, 0.1])  # Less diabetes
        cad[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.95, 0.05])  # Less CAD
        appet[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.1, 0.9])  # Good appetite
        pe[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.95, 0.05])  # No pedal edema
        ane[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.9, 0.1])  # No anemia

        # Generate diseased samples (last n_diseased samples)
        age[n_healthy:] = np.random.randint(45, 85, n_diseased)  # Older age
        bp[n_healthy:] = np.random.randint(140, 200, n_diseased)  # High BP
        sg[n_healthy:] = np.random.uniform(1.005, 1.015, n_diseased)  # Low specific gravity
        al[n_healthy:] = np.random.choice([0, 1, 2, 3, 4], n_diseased, p=[0.2, 0.3, 0.3, 0.15, 0.05])  # More albumin
        su[n_healthy:] = np.random.choice([0, 1, 2, 3, 4], n_diseased, p=[0.4, 0.3, 0.2, 0.08, 0.02])  # More sugar
        rbc[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.6, 0.4])  # Abnormal RBC
        pc[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.3, 0.7])  # Abnormal pus cells
        pcc[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.5, 0.5])  # Pus cell clumps
        ba[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.4, 0.6])  # Bacteria present
        bgr[n_healthy:] = np.random.randint(80, 300, n_diseased)  # Higher glucose
        bu[n_healthy:] = np.random.randint(25, 150, n_diseased)  # High blood urea
        sc[n_healthy:] = np.random.uniform(1.5, 15.0, n_diseased)  # High serum creatinine
        sod[n_healthy:] = np.random.randint(120, 150, n_diseased)  # Variable sodium
        pot[n_healthy:] = np.random.uniform(3.0, 7.0, n_diseased)  # Variable potassium
        hemo[n_healthy:] = np.random.uniform(6.0, 12.0, n_diseased)  # Low hemoglobin
        pcv[n_healthy:] = np.random.randint(15, 40, n_diseased)  # Low PCV
        wc[n_healthy:] = np.random.randint(3000, 15000, n_diseased)  # Variable WBC
        rc[n_healthy:] = np.random.uniform(2.5, 5.0, n_diseased)  # Low RBC count
        htn[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.3, 0.7])  # More hypertension
        dm[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.5, 0.5])  # More diabetes
        cad[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.7, 0.3])  # More CAD
        appet[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.6, 0.4])  # Poor appetite
        pe[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.4, 0.6])  # Pedal edema
        ane[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.3, 0.7])  # Anemia

        # Create classification labels
        classification = np.zeros(n_samples)
        classification[n_healthy:] = 1  # Diseased samples

        # Shuffle the data to mix healthy and diseased samples
        indices = np.random.permutation(n_samples)

        data = pd.DataFrame({
            'age': age[indices], 'bp': bp[indices], 'sg': sg[indices], 'al': al[indices], 'su': su[indices],
            'rbc': rbc[indices], 'pc': pc[indices], 'pcc': pcc[indices], 'ba': ba[indices], 'bgr': bgr[indices],
            'bu': bu[indices], 'sc': sc[indices], 'sod': sod[indices], 'pot': pot[indices], 'hemo': hemo[indices],
            'pcv': pcv[indices], 'wc': wc[indices], 'rc': rc[indices], 'htn': htn[indices], 'dm': dm[indices],
            'cad': cad[indices], 'appet': appet[indices], 'pe': pe[indices], 'ane': ane[indices],
            'classification': classification[indices]
        })

        return data

    def train_chronic_kidney_model(self):
        """Train chronic kidney disease prediction model"""
        print("Training Chronic Kidney Disease Prediction Model...")

        data = self.create_synthetic_chronic_kidney_data()
        print(f"Dataset created with {len(data)} samples")
        print(f"Class distribution: {data['classification'].value_counts().to_dict()}")

        X = data.drop('classification', axis=1)
        y = data['classification']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Use RandomForest with balanced class weights
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X_train, y_train)

        # Make predictions
        preds = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)

        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='weighted')
        recall = recall_score(y_test, preds, average='weighted')
        f1 = f1_score(y_test, preds, average='weighted')

        print(f"Chronic Kidney Disease Model Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")

        # Save comprehensive metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'model_type': 'Random Forest',
            'features': list(X.columns),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }

        with open(f'{self.models_dir}/chronic_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save the trained model
        joblib.dump(model, f'{self.models_dir}/chronic_model.sav')
        print("Chronic Kidney Disease model saved successfully!")

        return accuracy



    def train_all_models(self):
        """Train all disease prediction models"""
        print("\n===== TRAINING ALL DISEASE PREDICTION MODELS =====\n")

        # Train the main symptom-based disease model
        symptom_accuracy = self.train_symptom_based_disease_model()

        # Train individual disease models
        diabetes_accuracy = self.train_diabetes_model()
        heart_accuracy = self.train_heart_model()
        parkinsons_accuracy = self.train_parkinsons_model()
        liver_accuracy = self.train_liver_model()
        hepatitis_accuracy = self.train_hepatitis_model()
        chronic_kidney_accuracy = self.train_chronic_kidney_model()

        # Print summary of model accuracies
        print("\n===== MODEL TRAINING SUMMARY =====")
        print(f"Symptom-based Disease Model: {symptom_accuracy:.4f}")
        print(f"Diabetes Model: {diabetes_accuracy:.4f}")
        print(f"Heart Disease Model: {heart_accuracy:.4f}")
        print(f"Parkinson's Disease Model: {parkinsons_accuracy:.4f}")
        print(f"Liver Disease Model: {liver_accuracy:.4f}")
        print(f"Hepatitis Model: {hepatitis_accuracy:.4f}")
        print(f"Chronic Kidney Disease Model: {chronic_kidney_accuracy:.4f}")
        print("\nAll models have been trained and saved successfully!")


# Execute the training process if this script is run directly
if __name__ == "__main__":
    trainer = DiseaseModelTrainer()
    trainer.train_all_models()
