import streamlit as st
#st.title("My first streamlit app")
#st.header('Artificial Intelligence')
#st.subheader('ML , DL')

#st.write("AICW program has been started in our college and I am enjoying learning ")
user_input = st.text_input("Enter your name")
st.text(f"Hello, {user_input}!")

st.title("Streamlit text input example")
# single line text
user_name = st.text_input("Enter your name ")
st.write("your name is: ", user_name)
user_bio =st.text_area('enter your biodata')
st.write('your bio is:', user_bio)
# number input

user_age = st.number_input('enter your age', min_value=18, max_value=100, value=25)

st.write('your age is', user_age)

appt_date = st.date_input('select appointment date')

st.write('appointment date:', appt_date)

#time input
appt_time = st.time_input('Select appointment time')
st.write('Appointment time:',appt_time)

# combining inputs
if st.button('Submit'):
    st.write('Name:',user_name)
    st.write('Bio:',user_bio)
    st.write('Age:',user_age)
    st.write('Appointment Date:',appt_date)
    st.write('Appoint Time:',appt_time)
    