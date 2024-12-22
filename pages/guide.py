import streamlit as st

st.title("How to Use Model Builder")

st.write("""
### Installation
Clone the repository and install the dependencies:
""")
st.code('git clone https://github.com/Badhrinadhgvs/Model_builder.git\ncd Model_builder\npip install -r requirements.txt', language='bash')

st.write("""
### Running the App
Run the Streamlit app locally with:
""")
st.code('streamlit run app.py', language='bash')

st.write("""
### Using the Model
The model is saved in Pickle format (i.e., .pkl or Serialized). To use it, you need to deserialize it. In Python, we use the `pickle` module. Follow these steps:
""")
st.write("""
- Import pickle:
""")
st.code('import pickle', language='python')
st.write("""
- Load the model:
""")
st.code("load_model = pickle.load(open('Model.pkl', 'rb'))", language='python')
st.write("""
- Make predictions with the model:
""")
st.code('load_model.predict([[input]])', language='python')