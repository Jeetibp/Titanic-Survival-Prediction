{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "6b83e59d-8554-4aaf-9487-cbadb6fe7ddb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import pandas as pd\n",
    "import numpy as np\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7820f7d3-4576-42cf-abb0-dbbf0d68d767",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the saved model and scaler\n",
    "with open('model.pkl', 'rb') as f:\n",
    "    model = pickle.load(f)\n",
    "\n",
    "with open('scaler.pkl', 'rb') as f:\n",
    "    scaler = pickle.load(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "c1d69421-9034-4b4a-9aef-b02a7781faf7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-03-25 12:12:53.595 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Jeet\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Set up the Streamlit app\n",
    "st.title('Titanic Survival Prediction')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "3a62f2cc-f7b9-4f98-83a2-e57d3799b4e1",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-03-25 12:13:13.420 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Create input fields for user data\n",
    "sex = st.selectbox('Sex', ['Male', 'Female'])\n",
    "pclass = st.selectbox('Passenger Class', [1, 2, 3])\n",
    "fare = st.number_input('Fare', min_value=0.0, max_value=500.0, value=32.2)\n",
    "embarked = st.selectbox('Port of Embarkation', ['Queenstown', 'Southampton', 'Other'])\n",
    "family_size = st.number_input('Family Size', min_value=0, max_value=10, value=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "4cfeef10-7dce-48bb-bd28-5fecb6f153a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a dataframe from user inputs\n",
    "input_data = pd.DataFrame({\n",
    "    'Sex': [1 if sex == 'Male' else 0],\n",
    "    'Pclass': [pclass],\n",
    "    'Fare': [fare],\n",
    "    'Embarked_Q': [1 if embarked == 'Queenstown' else 0],\n",
    "    'Embarked_S': [1 if embarked == 'Southampton' else 0],\n",
    "    'FamilySize': [family_size]\n",
    "})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "1b122c3b-ad7d-4164-b4d0-12199fca42c0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Scale the input data\n",
    "input_scaled = scaler.transform(input_data)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "794a3519-d55e-4af5-860c-1b99232a83c6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DeltaGenerator(_root_container=1, _parent=DeltaGenerator())"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Make prediction\n",
    "if st.button('Predict'):\n",
    "    prediction = model.predict(input_scaled)\n",
    "    probability = model.predict_proba(input_scaled)[0][1]\n",
    "    \n",
    "    st.subheader('Prediction Result')\n",
    "    if prediction[0] == 1:\n",
    "        st.write('The passenger would likely survive.')\n",
    "    else:\n",
    "        st.write('The passenger would likely not survive.')\n",
    "    \n",
    "    st.write(f'Survival probability: {probability:.2f}')\n",
    "\n",
    "st.sidebar.header('About')\n",
    "st.sidebar.info('This app predicts the survival of Titanic passengers based on input features.')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "56d2358f-bf3a-4b71-bac3-507450345a72",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
