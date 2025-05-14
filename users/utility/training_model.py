import numpy as np
import pandas as pd
from django.shortcuts import redirect, render
from django.conf import settings
import os
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from users.models import QuestionnaireResponse

# Load and preprocess dataset
def load_and_preprocess_data():
    # Load the dataset
    data_path = os.path.join(settings.MEDIA_ROOT, 'data', 'sleep_health_and_lifestyle_dataset.csv')
    df = pd.read_csv(data_path)

    # Fill missing values in 'Sleep Disorder' column with 'None'
    df['Sleep Disorder'].fillna('None', inplace=True)

    # Encode categorical features using LabelEncoder
    gender_encoder = LabelEncoder()
    df['Gender'] = gender_encoder.fit_transform(df['Gender'])

    bmi_encoder = LabelEncoder()
    df['BMI Category'] = bmi_encoder.fit_transform(df['BMI Category'])

    bp_encoder = LabelEncoder()
    df['Blood Pressure'] = bp_encoder.fit_transform(df['Blood Pressure'])

    disorder_encoder = LabelEncoder()
    df['Sleep Disorder'] = disorder_encoder.fit_transform(df['Sleep Disorder'])

    # Features and target variable
    x = df[['Gender', 'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
            'Stress Level', 'BMI Category', 'Blood Pressure', 'Heart Rate', 'Daily Steps']]
    y = df['Sleep Disorder']

    # Normalize the features for better predictions
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Split the data into training and testing sets
    return train_test_split(x, y, test_size=0.2, random_state=39), scaler

def load_data():
    data_path = os.path.join(settings.MEDIA_ROOT, 'data', 'sleep_health_and_lifestyle_dataset.csv')
    df = pd.read_csv(data_path)

    le=LabelEncoder()
    for col in ['Gender',"BMI Category",'Sleep Disorder',"Blood Pressure"]:
        df[col]=le.fit_transform(df[col])


    x=df[['Gender', 'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
            'Stress Level', 'BMI Category', 'Blood Pressure', 'Heart Rate', 'Daily Steps']]
    y=df['Sleep Disorder']

    return train_test_split(x,y ,test_size=0.2,random_state=39)

# Function to train a new model
def train_model(x_train, y_train):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    return model


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


   

def ANNMODEL():

    x_train, x_test, y_train, y_test=load_data()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    model=Sequential([
        Dense(64, activation='relu',input_shape=(x_train.shape[1],)),
        Dense(32,activation='relu'),
        Dense(16,activation='relu'),
        Dense(1,activation='sigmoid')
        ])
    
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    model.fit(x_train,y_train,epochs=100,batch_size=32,validation_data=(x_test,y_test),verbose=1)

    ANN_pred_prob=model.predict(x_test)
    Ann_prob=(ANN_pred_prob > 0.5 ).astype(int)

    ANN_accurecy=accuracy_score(y_test,Ann_prob)

    print(ANN_accurecy)
    return ANN_accurecy
# Function to train and evaluate multiple models
def Model_Evaluation_View(request):
    (x_train, x_test, y_train, y_test), scaler = load_and_preprocess_data()

    models = {
        'SVC': SVC(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'KNeighbors': KNeighborsClassifier()
    }

    accuracies = {}
    reports = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracies[name] = accuracy_score(y_test, predictions)
        reports[name] = classification_report(y_test, predictions)
        plot_confusion_matrix(y_test, predictions, name)

    # Find the best model
    best_model_name = max(accuracies, key=accuracies.get)
    best_model_accuracy = accuracies[best_model_name]

    ANN_accurecy=ANNMODEL()
    print(ANN_accurecy)


    context = {
        'accuracy_svc': accuracies['SVC'],
        'accuracy_rf': accuracies['RandomForest'],
        'accuracy_knn': accuracies['KNeighbors'],
        'accuracy_dt': accuracies['DecisionTree'],
        
        'ANN_ACCURECY': ANN_accurecy,
        'rf_report': reports['RandomForest'],
        'svc_report': reports['SVC'],
        'knn_report': reports['KNeighbors'],
        'dt_report': reports['DecisionTree'],
       
    }

    return render(request, 'analysis/model_evaluation.html', context)

# Function to plot and save confusion matrix
def plot_confusion_matrix(y_test, predictions, model_name):
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(os.path.join(settings.MEDIA_ROOT, f'{model_name}.png'))
    plt.close()
from django.contrib import messages
# Prediction function
def ModelPrediction(request):
    if request.method == 'POST':
        # Encode Gender (Manually Apply Same Encoding as Training)
        gender_map = {'Male': 1, 'Female': 0}
        gender = gender_map.get(request.POST['gender'], -1)

        # Encode BMI Category (Manually Apply Same Encoding)
        bmi_map = {'Normal': 0, 'Overweight': 1,'Obese':2}
        bmi_category = bmi_map.get(request.POST['bmi'], -1)

        # Ensure valid encoding
        if gender == -1 or bmi_category == -1:
            return render(request, 'analysis/predictionpage.html', {'error': "Invalid input for Gender or BMI Category"})
        
        
        id=request.POST['id']
        # Get remaining numerical input values
        age = float(request.POST['age'])
        sleep_duration = float(request.POST['sleep'])
        quality_of_sleep = float(request.POST['quality'])
        physical_activity_level = float(request.POST['activity'])
        stress_level = float(request.POST['stress'])
        blood_pressure = float(request.POST['bloodpressure'])
        heart_rate = float(request.POST['heartrate'])
        daily_steps = float(request.POST['dailysteps'])

        try:
            user_questionnaire = QuestionnaireResponse.objects.get(id=id)
            errors = []

            if user_questionnaire.stress_level == "always" and int(stress_level) < 8:
                errors.append("Your questionnaire indicates extreme stress, but you entered a low stress level.")

            if user_questionnaire.wake_freshness == "no" and int(sleep_duration) > 8:
                errors.append("Your questionnaire suggests you wake up tired, but your sleep duration is too high.")

            if user_questionnaire.exercise_frequency in ["rarely", "1-2"] and int(sleep_duration) > 9:
                errors.append("You reported low exercise levels, which might not align with high sleep duration.")

            if user_questionnaire.caffeine_alcohol == "yes" and int(sleep_duration) > 7:
                errors.append("Your caffeine or alcohol intake may impact your sleep duration.")

            if errors:
                for error in errors:
                    messages.error(request, error)
                return redirect('predictionpage')

            # Load dataset and preprocess
            (x_train, x_test, y_train, y_test), scaler = load_and_preprocess_data()

            # Train a new model
            model = train_model(x_train, y_train)

            # Normalize input
            input_data = np.array([[gender, age, sleep_duration, quality_of_sleep, physical_activity_level, 
                                    stress_level, bmi_category, blood_pressure, heart_rate, daily_steps]])
            input_data = scaler.transform(input_data)  # Normalize the input

            # Predict using the newly trained model
            pred = model.predict(input_data)
            prob = model.predict_proba(input_data)

            # Map prediction to readable label
            pred_map = {1: 'No Sleep Disorder', 2: 'Sleep Apnea', 0: 'Insomnia'}
            prediction = pred_map.get(pred[0], 'None')

            # Get probability of the predicted class
            confidence = round(np.max(prob) * 100, 2)

            return render(request, 'analysis/predictionpage.html', {'pred': prediction, 'confidence': confidence})

        except:
            messages.error(request,'please fill your daily response form Daily responseform')
            return render(request, 'analysis/predictionpage.html')

    try:
        id=int(request.GET['id'])
        print(id)
        user=QuestionnaireResponse.objects.get(id=id)
        print(user.wake_frequency)
        if user:
            return render(request, 'analysis/predictionpage.html')
        else:
            messages.error(request,'Please fill the form first')
            return render(request, 'analysis/questionnaire_form.html')
    except:
        messages.error(request,'Please fill the form first')
        return render(request, 'analysis/questionnaire_form.html')



