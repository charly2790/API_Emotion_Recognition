import os
import cv2
import dlib
from pathlib import Path
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static'

@app.route("/")
def upload_file():
	return render_template("index.html")

@app.route("/upload", methods = ['POST'])
def uploader():

	texto = ""
	
	if request.method == 'POST':		
		
		#Inicialización de variables
		vFile = None
        vFileName = None
		vImagen = None
		vImgGray = None
		vRostros = None
		vRostro = None
        vLandmarks = None
		vError = ""
        #Variables p/ redimensionado de la imágen
        vDimRef = 371
        vLimInf = vDimRef - 5
        vLimSup = vDimRef + 5
        #****************************************

        #Recuperación del archivo de imagen***************
        try:
            vFile = request.files['archivo']
        except:
            vFile = None

        if vFile is None:
            vError = "Error 01 - Error al recuperar el archivo de imágen"
            return render_template('Prediction.html', vError=vError, nom_imagen=filename)
        # ************************************************

		#Detector de caras frontales********************
		vDetector = None
		try:
			vDetector = dlib.get_frontal_face_detector()
		except:
			vDetector = None

        if vDetector is None:
            vError = "Error 02 - Error al cargar el detector de rostros"
            return render_template('Prediction.html', vError=vError, nom_imagen=filename)
		#************************************************

        # Detector de landmarks********************
        vShapePredictor = None
        try:
            vShapePredictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        except:
            vShapePredictor = None

        if vShapePredictor is None:
            vError = "Error 03 - Error al cargar el detector de landmarks"
			return render_template('Prediction.html', vError = vError , nom_imagen=filename)
        #************************************************

		vFilename = secure_filename(vfile.filename)

		#Guardo la imágen en el directorio static
		vfile.save(os.path.join(app.config['UPLOAD_FOLDER'],vFilename))

		vPath = Path(os.path.join(app.config['UPLOAD_FOLDER']))

		vPath = vPath / filename

        #Lectura de la imágen con OpenCV***************
		try:
			vImagen = cv2.imread(str(vPath))
		except:
			vImagen = None

        if vImagen is None:
            vError = "Error 04 - Error al cargar la imágen"
            return render_templ2ate('Prediction.html', vError=vError, nom_imagen=vFilename)
        #**********************************************

        #Si se logro leer la imágen la convierto a escala de grises
        vImgGray = cv2.cvtColor(vImagen,cv2.COLOR_BGR2GRAY)

        #Detección de rostros en la imágen en escala de grises
        vRostros = vDetector(vImgGray)

        #Si se detectan varios rostros se considera solo uno para la predicción
        if len(vRostros) > 0:
            vRostro = vRostros[0]

        if vRostro is None:
            vError = "Error 05 - No se detectaron rostros en la imágen cargada"
            return render_template('Prediction.html', vError=vError, nom_imagen=vFilename)

        #Redimensionado de la imágen*********************
        vLado = 0
        vx1 = vRostro.left()
        vx2 = vRostro.right()

        vLado = vx2 - vx1

        if vLado < vLimInf or vLado > vLimSup:

            vPorc = (vDimRef / vLado)

            vNuevoAncho = int(vImgGray.shape[0] * vPorc)
            vNuevoAlto  = int(vImgGray.shape[1] * vPorc)

            vDim = (vNuevoAncho,vNuevoAlto)

            vNuevaImg = None

            try:
                vNuevaImg = cv2.resize(vImgGray,vDim,interpolation=cv2.INTER_AREA)
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

        vLandmarks = vShapePredictor(vImgGray,vCara)

        for i in range (0,68):
            x = vLandmarks.part(i).x
            y = vLandmarks.part(i).y

            cv2.circle(vImgGray,(x,y),radius=5,color(0,255,0),thickness=-1)

        vImgLndmks = "img_with_landmarks.png"

        vPathImgWithLmrks = Path(os.path.join(app.config['UPLOAD_FOLDER']))

        vPathImgWithLmrks = vPathImgWithLmrks / vImgLndmks

        cv2.imwrite(str(vPathImgWithRect), vImgGray)

        return render_template('Prediction.html', ruta = vImgLndmks, nom_imagen=vImgLndmks)

        #Prueba detección de rostros***************************************************

        #x1 = vRostro.left()
        #y1 = vRostro.top()
        #x2 = vRostro.right()
        #y2 = vRostro.bottom()

        #cv2.rectangle(vImagen,(x1,y1),(x2,y2),color=(0,255,0),thickness=4)

        #vImgRect = "img_with_rectangle.png"

        #vPathImgWithRect = Path(os.path.join(app.config['UPLOAD_FOLDER']))

        #vPathImgWithRect = vPathImgWithRect / vImgRect

        #cv2.imwrite(str(vPathImgWithRect), vImagen)

        #Fin - Prueba detección de rostros**********************************************

        #vNomImgGs = "img_grayscale.png"

        #vPathGrayScale = Path(os.path.join(app.config['UPLOAD_FOLDER']))

        #vPathGrayScale = vPathGrayScale / vNomImgGs

        #Guardo la imágen en escala de grises
        #cv2.imwrite(str(vPathGrayScale),vImgGray)

        #return render_template('Prediction.html', ruta = vImgRect, nom_imagen=vImgRect)

			
		#else:
			#vError = "Error 02 - Error al cargar la imágen"
			#return render_template('Prediction.html', vError=vError, nom_imagen=filename)

			#return render_template('Prediction.html', ruta = "No se leyo la imágen",nom_imagen = filename)
		#else:
			#return render_template('Prediction.html', ruta = "imágen leida",nom_imagen = filename)

		#return render_template('Prediction.html', nom_imagen=filename)

if __name__ =='__main__':
	app.run(debug = True,port = 8000)



