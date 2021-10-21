import numpy as np
import pickle
import streamlit as st 
from datetime import datetime


pkl_filename = "taller_dp.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

labels = ['Sano','enfermo']       

def prediccion(xin):
    print(xin)
    yout=model.predict(xin)
    print(yout)
    mensaje = ''
    for y_out in yout:
        mensaje =mensaje + 'El paciente corresponde a la clase {}\n'.format(labels[y_out])  
        
    return mensaje
   

def main():
    # Título
    html_temp = """
    <h1 style="color:##600a0a;text-align:center;">SISTEMA DE DIAGNOSTICO QUE PERMITE ESTABLECER SI EL PACIENTE POSEE UNA ENFERMEDAD CARDIACA  </h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    # Lecctura de datos
    Datos = st.text_input(style="color:##600a0a Ingrese los valores : "")

    
    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción :"): 
        x_in = list(np.float_((Datos.title().split('\t'))))
        x_in = np.asarray(x_in).reshape(1,8)
        print(x_in.shape)

        predictS = prediccion(x_in)
        st.success(predictS)

if __name__=='__main__':
    main()