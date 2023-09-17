import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np

st.write("Created by : Abhishek Bhardwaj")

def main():
    st.title("Medical cost prediction")

    df=pd.read_csv("medical_cost.csv")
    sc=MinMaxScaler()
    sc.fit(df[['age','bmi']])

    gender=[x.upper() for x in df['sex'].unique()]
    smoker=[x.upper() for x in df['smoker'].unique()]
    region=[x.upper() for x in df['region'].unique()]
    age = st.number_input('Enter your Age.',step=1)
    bmi = st.number_input('Enter your BMI.')
    children = st.number_input('How many children you have?',step=1)
    gen=st.selectbox(label="Enter your Gender.",options=gender)
    smo=st.selectbox(label="Do you smoke?",options=smoker)
    reg=st.selectbox(label="Enter your region.",options=region)

    gender_d={"FEMALE":0,
            "MALE":1}
    smoker_d={"YES":1,
            "NO":0}
    region_d={"NORTHWEST":0,
            "NORTHEAST":1,
            "SOUTHEAST":2,
            "SOUTHWEST":3}

    da={"age":age,
        "sex":gender_d[gen],
    "bmi":bmi,
    "children":children,
    "smoker":smoker_d[smo],
    "region":region_d[reg]}


    df_temp=pd.DataFrame(da,index=[0])
    df_temp[['age','bmi']]=sc.transform(df_temp[['age','bmi']])

    i=st.button("Predict", type="primary")
    if i:
        model= pickle.load(open("gb_model", 'rb'))
        y_value=model.predict(np.array(df_temp.iloc[0].values).reshape(1,6))
        st.write(f"Your medical cost for the entered information is approximately {round(y_value[0],2)} dollars.")

if __name__=='__main__':
    main()


