import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
def diabetes_prediction():
    st.title("Diabetes Prediction")
    st.markdown(
    """
    <div>
        <img src="https://www.expresshealthcaremd.com/wp-content/uploads/2022/08/diabetes-inn-desktop.jpg" style="width:100%;height:100%;object-fit:cover">
    </div>
    """,
    unsafe_allow_html=True
    )
    st.sidebar.header("Enter Patient Information")
    gender = st.sidebar.number_input("Mention gender: Male-1, Female-0",min_value=0,max_value=1,step=1)
    age = st.sidebar.number_input("Age",min_value=5,max_value=100,step=1)
    hypertension = st.sidebar.number_input("Do you have Hypertension?",min_value=0,max_value=1,step=1)
    heart_disease = st.sidebar.number_input("Do you have Heart disease?",min_value=0,max_value=1,step=1)
    bmi = st.sidebar.number_input("BMI")
    HbA1c_level = st.sidebar.number_input("HbA1c level")
    blood_glucose = st.sidebar.number_input("Present Glucose level")
    if st.button("Predict Diabetes"):
        # Load the datasets
        df = pd.read_csv('dataset.csv')
        # Split the data into features and target
        x = df.drop(['diabetes', 'gender'], axis=1).values
        y = df['diabetes'].values
        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=3)
        # Create the Random Forest Classifier
        rf_classifier = RandomForestClassifier(random_state=0)
        # Fit the classifier to the training data
        rf_classifier.fit(x_train, y_train)
        new_data = [[age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose]]
        new_data_predictions = rf_classifier.predict(new_data)
        # Display the predictions
        if new_data_predictions == 1:
            st.write("Oops you are tested Positive!!!!")
        else:
            st.write("Well done!! you are tested Negative!!!!")  
def  madhumeha_prediction():
    st.title('Madhumeha Prediction')
    st.markdown(
    """
    <div>
        <img src="https://www.doctorsfinder.in/wp-content/uploads/2023/04/Ayurvedic-Doctor.webp">
    </div>
    """,
    unsafe_allow_html=True
    )
    st.sidebar.header("Enter your details")
    Excessivesleep=st.sidebar.number_input("Do you have excessive sleep?",min_value=0,max_value=1,step=1)
    drowsiness=st.sidebar.number_input("Do you have drowsiness?",min_value=0,max_value=1,step=1)
    drymouth=st.sidebar.number_input("dry mouth?",min_value=0,max_value=1,step=1)
    burninghandslegs=st.sidebar.number_input("burning hands and legs?",min_value=0,max_value=1,step=1)
    frequenturination=st.sidebar.number_input("frequent urination?",min_value=0,max_value=1,step=1)
    increasedhunger=st.sidebar.number_input("Increased hunger?",min_value=0,max_value=1,step=1)
    laziness=st.sidebar.number_input("laziness?",min_value=0,max_value=1,step=1)
    if st.button("Predict Madhumeha"):
        df= pd.read_csv('dataset1.csv')
        x = df.drop(['diabetes'], axis=1).values  # Independent variables
        y = df['diabetes'].values  # Dependent features
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        # Create the Random Forest Classifier
        rf_classifier = RandomForestClassifier(random_state=0)
        # Fit the classifier to the training data
        rf_classifier.fit(x_train, y_train)
        new_data = [[Excessivesleep,drowsiness,drymouth,burninghandslegs,frequenturination,increasedhunger,laziness]]
        new_data_predictions = rf_classifier.predict(new_data)
        # Display the predictions
        if new_data_predictions == 1:
            st.write("Oops you are tested Positive!!!!")
        else:
            st.write("Well done!! you are tested Negative!!!!")  
def ayurvedic_food_recommendation():
    st.title("Ayurvedic Food Recommendation")
    st.markdown(
    """
    <div>
        <img src="https://butterflyayurveda.com/cdn/shop/articles/ayurvedic_treatment_for_diabetes.jpg?v=1669207818" 
        style="width:400px;height:200px;">
    </div>
    """,
    unsafe_allow_html=True)
    st.sidebar.header("Enter your information")
    age = st.sidebar.number_input("Enter your Age:", min_value=19, step=1)
    if(age<20):
        st.sidebar.warning("Please enter your age")
        return
    elif(age>=101):
        st.sidebar.warning("Please enter your age")
        return
    # Vegetarian preference (binary selection)
    veg = st.sidebar.text_input("Are you vegetarian(yes/no)?")
    # Weight should be a positive number
    weight = st.sidebar.number_input("Enter your weight:", min_value=44, step=1)
    # Height should be a positive number
    if(weight<45):
        st.sidebar.warning("Please enter your weight")
        return
    elif(weight>=121):
        st.sidebar.warning("Please enter your weight")
        return     
    height = st.sidebar.number_input("Enter your Height:", min_value=119, step=1)
    if(height<120):
        st.sidebar.warning("Please enter your height")
        return
    elif(height>=230):
        st.sidebar.warning("Please enter your height")
        return
    nutrition_data = pd.read_csv('nutrition.csv')
    modified_data = pd.read_csv('Modified.csv')
    # Performing  KMeans clustering on nutrition data
    kmeans_nutrition = KMeans(n_clusters=5)
    kmeans_nutrition.fit(nutrition_data)
    centroids = kmeans_nutrition.cluster_centers_
    if(age>=20 and weight>=45 and height>=120 and age<=100 and weight<=120 and height<=229):
        if st.button('Get Recommendations!'):
            # Calculate BMI and other calculations...
            bmi = weight / ((height / 100) ** 2)
            st.write(f"Calculated BMI: {bmi}")
            # Filter food items related to the predicted cluster label
            selected_features = modified_data[['Calories', 'Fats', 'Proteins', 'Iron', 'Calcium', 'Sodium', 'Pottasium', 'Carbohydrates', 'Fibre', 'VitaminD', 'Sugars']]
            kmeans_modified = KMeans(n_clusters=5)
            kmeans_modified.fit(selected_features)
            distances = []
            for centroid in centroids:
                distance = np.linalg.norm(selected_features - centroid, axis=1)
                distances.append(distance)
            closest_clusters = np.argmin(distances, axis=0)

            # Assign class label based on closest clusters
            modified_data['Class_Label'] = closest_clusters

            # Use Random Forest for classification
            rf_model = RandomForestClassifier(n_estimators=100)
            rf_model.fit(selected_features, kmeans_modified.labels_)

            # Predict cluster label based on the user's BMI
            predicted_label = rf_model.predict([[bmi, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
            related_foods = modified_data[modified_data['Class_Label'] == predicted_label[0]]['Food']

            # Display recommended foods based on BMI prediction
            Food_itemsdata=modified_data['Food']
            st.header("Breakfast Foods")
            bf=[]
            breakfast_foods = modified_data[modified_data['Breakfast'] == 1]['Food']
            for food in breakfast_foods:
                if food in related_foods.tolist():
                    bf.append(food)
            if len(bf)==0:
                total_food_items = len(Food_itemsdata)
                random_indices = np.random.choice(total_food_items, size=6, replace=False)
                for index in random_indices:
                    st.write(Food_itemsdata[index])
            elif len(bf) >= 6:
                selected_random_foods = random.sample(bf, k=6)
                for food1 in selected_random_foods:
                    st.write(food1)
            else:
                st.write("Recommended Foods:")
                for food1 in bf:
                    st.write(food1)
            st.header("Lunch Foods")
            lf=[]
            lunch_foods = modified_data[modified_data['Lunch'] == 1]['Food']
            for food in lunch_foods:
                if food in related_foods.tolist():
                    lf.append(food)
            if len(lf)==0:
                total_food_items = len(Food_itemsdata)
                random_indices = np.random.choice(total_food_items, size=6, replace=False)
                for index in random_indices:
                    st.write(Food_itemsdata[index])
            elif len(lf) >= 6:
                selected_random_foods = random.sample(lf, k=6)
                for food1 in selected_random_foods:
                    st.write(food1)
            else:
                st.write("Recommended Foods:")
                for food1 in lf:
                    st.write(food1)
            st.header("Dinner Foods")
            df=[]
            dinner_foods = modified_data[modified_data['Dinner'] == 1]['Food']
            for food in dinner_foods:
                if food in related_foods.tolist():
                    df.append(food)
            if len(df)==0:
                total_food_items = len(Food_itemsdata)
                random_indices = np.random.choice(total_food_items, size=6, replace=False)
                for index in random_indices:
                    st.write(Food_itemsdata[index])
            elif len(df) >= 6:
                selected_random_foods = random.sample(df, k=6)
                for food1 in selected_random_foods:
                    st.write(food1)
            else:
                st.write("Recommended Foods:")
                for food1 in df:
                    st.write(food1)
app_choice = st.sidebar.radio("Select App", ("Diabetes Prediction","Madhumeha prediction", "Ayurvedic Food Recommendation"))

if app_choice == "Diabetes Prediction":
    diabetes_prediction()
elif app_choice == "Ayurvedic Food Recommendation":
    ayurvedic_food_recommendation()
elif app_choice=="Madhumeha prediction":
    madhumeha_prediction()
