import os
import cv2
import dlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

FORMATOS_PERMITIDOS = set(['png','jpg','JPG','PNG','bmp'])

def validarFormato(nomArchivo):
    extArchivo = nomArchivo.rsplit('.',1)[1]
    return '.' in nomArchivo and extArchivo in FORMATOS_PERMITIDOS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static'

@app.route("/")
def upload_file():
	return render_template("index.html")

@app.route("/upload", methods = ['POST'])
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
        vTheta = None #DataFrame Theta
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
        #************************************************
        
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

        #Carga de Matriz de parámetros**************
        vNomFileTheta = 'mat_parametros_RaFD_CK_1616.csv'
        vTheta = None

        try:
            dfTheta = pd.read_csv(vNomFileTheta, delimiter=';', header=None, dtype=np.float64, decimal=',',float_precision='high')
        except:
            dfTheta = None

        if dfTheta is None:
            vError = "Error 04 - Error al cargar la matriz de parámetros"
            return render_template('Prediction.html', vError=vError)
        else:
            vTheta = np.asarray(dfTheta)
            #vError = "Dimensiones: " + str(vTheta.shape[1]) + " x " + str(vTheta.shape[0])
            #return render_template('Prediction.html', vError=vError)
        #******************************************

        vFilename = secure_filename(vFile.filename)

        #Validación de archivo de entrada**********
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

                vDim = (vNuevoAlto,vNuevoAlto)

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
        #**********************************************

        vLandmarks = vShapePredictor(vImgGray,vRostro)
        #Matriz de características "X"
        vFeatures = np.zeros((1,136))
        j = 0
        for i in range(0, 68):
            x = vLandmarks.part(i).x
            y = vLandmarks.part(i).y

            vFeatures[0,j] = int(x)
            j = j + 1
            vFeatures[0,j] = int(y)
            j = j + 1

            cv2.circle(vImgGray, center = (x, y), radius = 5, color=(0, 255, 0), thickness=-1)

        #Cantidad de imágenes
        m = vFeatures.shape[0]

        #Número de emociones
        vNumLabels = vTheta.shape[0]

        #Traspuesta de la matriz theta
        vTranspTheta = np.transpose(vTheta)

        h = np.zeros((vNumLabels,m),dtype = np.float64)
        z = np.zeros((vNumLabels, m), dtype=np.float64)

        #se agrega una columna de 1s al comienzo de la matriz de características (vFeatures)
        vFeatures = np.concatenate((np.ones((m,1)),vFeatures),axis=1)

        #calculo el polinomio 'z'
        z = np.dot(vFeatures,vTranspTheta)

        #aplico la función sigmoide
        h = 1/(1+np.exp(-z))

        h = np.transpose(h)

        vCantCols = h.shape[1]
        dicCol = {}

        for i in range(0,vCantCols):

            vEmociónPredicha = ""
            vCol = h[:,i]

            #calculo el máximo
            vMax = np.max(vCol)

            vPosNumMax = int(np.where(vCol == vMax)[0][0])

            if vPosNumMax == 0:
                vEmociónPredicha = 'Enojo'
            elif vPosNumMax == 1:
                vEmociónPredicha = 'Desprecio'
            elif vPosNumMax == 2:
                vEmociónPredicha = 'Asco'
            elif vPosNumMax == 3:
                vEmociónPredicha = 'Miedo'
            elif vPosNumMax == 4:
                vEmociónPredicha = 'Felicidad'
            elif vPosNumMax == 5:
                vEmociónPredicha = 'Neutral'
            elif vPosNumMax == 6:
                vEmociónPredicha = 'Tristeza'
            elif vPosNumMax == 7:
                vEmociónPredicha = 'Sorpresa'

            dicCol = {'Enojo': vCol[0], 'Desprecio': vCol[1], 'Asco': vCol[2], 'Miedo': vCol[3],
                           'Felicidad': vCol[4], 'Neutral': vCol[5], 'Tristeza': vCol[6], 'Sorpresa': vCol[7]}

            dicResultado = {'Archivo':vFilename, 'salida_clasificadores':dicCol,'emocion_predicha':vEmociónPredicha}

        vResult = json.dumps(dicResultado)

        #return render_template('Prediction.html', vResultado = str(vResult), nom_imagen=vFilename)
        return render_template('Prediction.html', vResultado=str(vResult))
                
    else:
        vError = "Error 00 - Error en la recepción de los parámetros"
        return render_template('Prediction.html', vError=vError)

if __name__ =='__main__':
	app.run(debug = True,port = 8000)



