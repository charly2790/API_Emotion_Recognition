import os
import cv2
import dlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request
from Functions import *
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static'


@app.route("/")
def upload_file():
    return render_template("index.html")


@app.route("/upload", methods=['POST'])
def uploader():
    texto = ""

    if request.method == 'POST':
        # Inicialización de variables
        vFile = None
        vFileName = None
        vImagen = None
        vImgGray = None
        vRostros = None
        vRostro = None
        vLandmarks = None
        vTheta = None  # DataFrame Theta
        ck_redim = 0
        vError = ""

        # Variables p/ redimensionado de la imágen
        vDimRef = 371
        vLimInf = vDimRef - 5
        vLimSup = vDimRef + 5
        # ****************************************

        # Recuperación del archivo de imagen***************
        try:
            vFile = request.files['archivo']
        except:
            vFile = None

        if vFile is None:
            vError = "Error 01 - Error al recuperar el archivo de imágen"
            return render_template('Prediction.html', vError=vError)
        # ************************************************

        # Detector de caras frontales********************
        vDetector = None
        try:
            vDetector = dlib.get_frontal_face_detector()
        except:
            vDetector = None

        if vDetector is None:
            vError = "Error 02 - Error al cargar el detector de rostros"
            return render_template('Prediction.html', vError=vError)
        # ************************************************

        # Detector de landmarks********************
        vShapePredictor = None
        try:
            vShapePredictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        except:
            vShapePredictor = None

        if vShapePredictor is None:
            vError = "Error 03 - Error al cargar el detector de landmarks"
            return render_template('Prediction.html', vError=vError)
        # ******************************************

        # Carga de Matriz de parámetros**************
        vNomFileTheta = 'mat_parametros_RaFD_CK_1616.csv'
        vTheta = None

        try:
            dfTheta = pd.read_csv(vNomFileTheta, delimiter=';', header=None, dtype=np.float64, decimal=',',
                                  float_precision='high')
        except:
            dfTheta = None

        if dfTheta is None:
            vError = "Error 04 - Error al cargar la matriz de parámetros"
            return render_template('Prediction.html', vError=vError)
        else:
            vTheta = np.asarray(dfTheta)
            # vError = "Dimensiones: " + str(vTheta.shape[1]) + " x " + str(vTheta.shape[0])
            # return render_template('Prediction.html', vError=vError)
        # ******************************************

        vFilename = secure_filename(vFile.filename)

        # Validación de archivo de entrada**********
        if validarFormato(vFilename) is False:
            vError = "Error 02 - Formato de archivo erroneo"
            return render_template('Prediction.html', vError=vError)
        # ******************************************

        # Guardo la imágen en el directorio static
        vFile.save(os.path.join(app.config['UPLOAD_FOLDER'], vFilename))

        vPath = Path(os.path.join(app.config['UPLOAD_FOLDER']))

        vPath = vPath / vFilename

        # Lectura de la imágen con OpenCV***************
        try:
            vImagen = cv2.imread(str(vPath))
        except:
            vImagen = None

        if vImagen is None:
            vError = "Error 05 - Error al cargar la imágen"
            return render_templ2ate('Prediction.html', vError=vError, nom_imagen=vFilename)
        # **********************************************

        # Si se logro leer la imágen la convierto a escala de grises
        vImgGray = cv2.cvtColor(vImagen, cv2.COLOR_BGR2GRAY)

        # Detección de rostros en la imágen en escala de grises
        vRostros = vDetector(vImgGray)

        # Si se detectan varios rostros se considera solo uno para la predicción
        if len(vRostros) > 0:
            vRostro = vRostros[0]

        if vRostro is None:
            vError = "Error 06 - No se detectaron rostros en la imágen cargada"
            return render_template('Prediction.html', vError=vError, nom_imagen=vFilename)

        # Redimensionado de la imágen*********************
        if ck_redim == 1:
            vLado = 0
            vx1 = vRostro.left()
            vx2 = vRostro.right()

            vLado = vx2 - vx1

            if vLado < vLimInf or vLado > vLimSup:

                vPorc = (vDimRef / vLado)

                vNuevoAncho = int(vImgGray.shape[0] * vPorc)
                vNuevoAlto = int(vImgGray.shape[1] * vPorc)

                vDim = (vNuevoAlto, vNuevoAlto)

                vNuevaImg = None

                try:
                    vNuevaImg = cv2.resize(vImgGray, vDim, interpolation=cv2.INTER_AREA)
                except:
                    vNuevaImg = None

                if vNuevaImg is not None:
                    vCara = None
                    vImgGray = vNuevaImg
                    vCaras = vDetector(vImgGray)

                    if len(vCaras) > 0:
                        vCara = vCaras[0]

                    if vCara is not None:
                        vRostro = vCara
        # **********************************************

        vLandmarks = vShapePredictor(vImgGray, vRostro)
        # Matriz de características "X"
        vFeatures = np.zeros((1, 136))
        j = 0
        for i in range(0, 68):
            x = vLandmarks.part(i).x
            y = vLandmarks.part(i).y

            vFeatures[0, j] = int(x)
            j = j + 1
            vFeatures[0, j] = int(y)
            j = j + 1

        #Calculo de la función hipótesis*********************************
        try:
            h = hipotesisRL(vFeatures,vTheta)
        except:
            h = None

        if h is None:
            vError = "Error 07 - Error al calcular la función hipotesis"
            return render_template('Prediction.html', vError=vError)
        #****************************************************************

        #'h' tiene la salida de los 8 clasificadores

        #Formato del resultado*******************************************
        vResultado = None

        try:
            vResultado = darFormato(h,vFilename)
        except:
            vResultado = None

        if vResultado is None:
            vError = "Error 08 - Error al dar formato a la salida de los clasificadores"
            return render_template('Prediction.html', vError=vError)
        #****************************************************************
                
        #Limpio el servidor
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], vFilename))

        return render_template('Prediction.html', vResultado=str(vResultado))

    else:
        vError = "Error 00 - Error en la recepción de los parámetros"
        return render_template('Prediction.html', vError=vError)

if __name__ == '__main__':
    app.run(debug=True, port=8000)



