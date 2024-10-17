import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

@st.cache_data
def load_data():
    data = pd.read_csv('dataSet.csv')
    data.columns = data.columns.str.strip()  
    return data

data = load_data()
if "train_model" not in st.session_state:
    st.session_state.train_model = False

features=['season','age','childish','accident','surgery','fevers','alcohol','smoking','sitting']
X = data[features]
y = data['output']

pd.set_option('future.no_silent_downcasting', True)
y = y.replace({'n': 0, 'o': 1})
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.title("Predicción de fertilidad en hombres utilizando Super Vector Machine")
st.write("""
La aplicación utiliza SVM para predecir si un hombre es fértil dependiendo de su estilo de vida (cirugías, tiempo que emplea sentado, si fuma, edad
    y en qué época del año se hizo la prueba)
""")

def entrenar_y_mostrar_modelo(best_svm):
    y_pred = best_svm.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")

    y_score = best_svm.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    st.write(f"AUC: {roc_auc:.4f}")
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', label='No Skill')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('ROC Curve - Decision Tree')
    ax.legend(loc="lower right")
    st.pyplot(fig)

svm = SVC(class_weight='balanced')
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4]
}
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_svm = grid_search.best_estimator_
entrenar_y_mostrar_modelo(best_svm)
