import streamlit as st
import requests

st.set_page_config(page_title = 'AI-Powered Food Delivery Intelligence',page_icon = '🍔',layout = 'wide')

st.markdown("""
<style>
.stApp {
    background-color: #FFAE61;
}
</style>
""", unsafe_allow_html = True)


st.title('🍔 AI Powered Food Delivery Intelligence')

st.write('Choose a prediction')

st.divider()

col1, col2, col3 = st.columns(3)

with col1:

    st.header('📈')
    st.subheader('Demand Prediction')

    st.write('Predict the expected number of food orders')

    if st.button('Select', key = 'demand'):

        st.session_state.page = 'demand'

with col2:

    st.header('🛵')
    st.subheader('Delivery Time Prediction')

    st.write('Predict the estimated delivery time')

    if st.button('Select', key = 'delivery'):

        st.session_state.page = 'delivery'


with col3:

    st.header('💬')
    st.subheader('Review Sentiment Analysis')

    st.write('Analyze customer reviews')

    if st.button('Select', key = 'sentiment'):

        st.session_state.page = 'sentiment'



if st.session_state.get('page') == 'demand':

    st.divider()

    st.header('📈 Demand Prediction')

    order_timestamp = st.text_input('Enter Order Date and Time', placeholder = '2025-10-05 15:54')
    is_weekend = st.selectbox('Is Weekend?',['No', 'Yes'])
    weather = st.selectbox('Weather',['Clear', 'Cloudy', 'Rain', 'Storm'])
    traffic_level = st.selectbox('Traffic Level',['Low', 'Medium', 'High'])

    if st.button('Predict Demand'):

        data = {'order_timestamp': order_timestamp,
        'is_weekend': 1 if is_weekend == 'Yes' else 0,
        'weather': weather,
        'traffic_level': traffic_level}

        response = requests.post('https://food-delivery-backend-ijvg.onrender.com/predict/demand',json = data)

        if response.status_code == 200:

            result = response.json()

            if 'error' in result:
                st.error(result['error'])
            else:
                st.success(f'Predicted Orders: {result["predicted_order_count"]}')

        else:
            st.error('Prediction failed')


if st.session_state.get('page') == 'delivery':

    st.divider()

    st.header('🛵 Delivery Time Prediction')

    distance = st.number_input('Distance (km)', min_value = 0.0 , value = 5.0)

    preparation_time = st.number_input('Restaurant Preparation Time (minutes)',min_value = 0,value = 20)

    estimated_time = st.number_input('Estimated Delivery Time (minutes)',min_value = 0,value = 30)

    experience = st.number_input('Delivery Partner Experience (months)',min_value = 0,value = 12)

    partner_rating = st.number_input('Delivery Partner Rating',min_value = 0.0,max_value = 5.0,value = 4.5)

    is_weekend = st.selectbox('Weekend?',['No', 'Yes'])

    traffic = st.selectbox('Traffic Level', ['Low', 'Medium', 'High'])

    weather = st.selectbox('Weather',['Clear', 'Cloudy', 'Rain', 'Storm'])

    if st.button('Predict Delivery Time'):

        data = {'distance_km': distance,
                'restaurant_preparation_time_minutes': preparation_time,
                'estimated_delivery_time_minutes': estimated_time,
                'delivery_partner_experience_months': experience,
                'delivery_partner_rating': partner_rating,
                'is_weekend': 1 if is_weekend == 'Yes' else 0,
                'traffic_level': traffic,
                'weather': weather}

        response = requests.post('https://food-delivery-backend-ijvg.onrender.com/predict/delivery',json = data)

        if response.status_code == 200:

            result = response.json()

            st.success(f"Predicted Delivery Time: {result['predicted_delivery_time_minutes']} minutes")

        else:
            st.error('Prediction failed')


if st.session_state.get('page') == 'sentiment':

    st.divider()

    st.header('💬 Review Sentiment Analysis')

    review = st.text_area('Enter Customer Review',placeholder = 'Enter the customer review here...')

    if st.button('Analyze Sentiment'):

        data = {'review': review}

        response = requests.post('https://food-delivery-backend-ijvg.onrender.com/predict/sentiment',json = data)

        if response.status_code == 200:

            result = response.json()

            st.success(f"Sentiment: {result['sentiment']}")

            st.write(f'Confidence: {result["confidence"] * 100:.2f}%')

        else:
            st.error('Prediction failed')
