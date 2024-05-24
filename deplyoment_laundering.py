import streamlit as st
import pandas as pd
import sklearn
import pickle
import altair as alt

# Load the saved model from file
with open('rf_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Title for the app with money laundering icon
st.title('💸 Money Laundering Prediction')

# Input features in the middle of the app
st.header('Input Features')
sender_account = st.text_input('Sender Account')
receiver_account = st.text_input('Receiver Account')
amount = st.text_input('Amount')
payment_currency = st.text_input('Payment Currency')
received_currency = st.text_input('Received Currency')
sender_bank_location = st.text_input('Sender Bank Location')
receiver_bank_location = st.text_input('Receiver Bank Location')
payment_type = st.text_input('Payment Type')

# Create feature dataframe
features = pd.DataFrame({
    'Sender_account': [sender_account],
    'Receiver_account': [receiver_account],
    'Amount': [amount],
    'Payment_currency': [payment_currency],
    'Received_currency': [received_currency],
    'Sender_bank_location': [sender_bank_location],
    'Receiver_bank_location': [receiver_bank_location],
    'Payment_type': [payment_type]
})

# Predict button
if st.button('Predict'):
    # Prediction
    prediction = loaded_model.predict(features)

    # Map prediction to "is laundering" or "not laundering"
    prediction_label = "<h1 style='color:red;'>Predict: Is laundering</h1>" if prediction[0] == 1 else "<h1 style='color:green;'>Predict: Not laundering</h1>"

    # Display prediction
    st.markdown(prediction_label, unsafe_allow_html=True)

# Title for the CSV file upload and prediction
st.title('CSV File Money Laundering Prediction')

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)

    # Predict on the uploaded data
    predictions = loaded_model.predict(df)

    # Add predictions to the DataFrame
    df['Prediction'] = ['Is laundering' if pred == 1 else 'Not laundering' for pred in predictions]

    # Display the results in a table
    st.write(df)

    # Allow the user to download the results as CSV
    st.download_button(
        label="Download CSV",
        data=df.to_csv().encode('utf-8'),
        file_name='predictions.csv',
        mime='text/csv'
    )

    # Plot the results
    chart_data = df['Prediction'].value_counts().reset_index()
    chart_data.columns = ['Prediction', 'Count']
    
    bar_chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Prediction', axis=alt.Axis(labelAngle=0)),  # Set label angle to 0 for horizontal labels
        y='Count',
        color='Prediction'
    ).properties(
        title='Prediction Distribution'
    )

    text = bar_chart.mark_text(
        align='center',
        baseline='middle',
        dy=-10  # Nudge text above bars
    ).encode(
        text='Count:Q'
    )
    
    st.altair_chart(bar_chart + text, use_container_width=True)