import os
import cv2
import dlib
import json
import numpy as np
import pandas as pd

FORMATOS_PERMITIDOS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

def validarFormato(nomArchivo):
    extArchivo = nomArchivo.rsplit('.', 1)[1]
    return '.' in nomArchivo and extArchivo in FORMATOS_PERMITIDOS

def hipotesisRL(mat_x,mat_param):
    #Entradas
    #mat_x: matriz de características faciales extraidas del rostro a analizar
    #mat_param: matriz theta.

    #Dimensiones matriz theta: 8x136
    #¿por que 8? cantidad de emociones consideradas
    #¿por que 136? cantidad de parámetros ajustados de acuerdo a entrenamiento del polinomio
    #Dimensiones matriz x: 1 x 136
    #¿Por que 1? porque será siempre una imágen de entrada
    #¿Por que 136? se corresponde con la cantidad de landmarks faciales multiplicados por 2 (68 x 2)

    # Cantidad de imágenes
    m = mat_x.shape[0]

    # Número de emociones
    vNumLabels = mat_param.shape[0]

    # Traspuesta de la matriz theta
    vTranspTheta = np.transpose(mat_param)

    #Variable donde se almacenarán los resultados
    #Se declaran 2 matrices de 8 x 1. Es decir la salida de los 8 clasificadores para 1 sola imágen.
    h = np.zeros((vNumLabels, m), dtype=np.float64)
    z = np.zeros((vNumLabels, m), dtype=np.float64)

    #se agrega una columna de 1s al comienzo de la matriz de características (mat_x)
    mat_x = np.concatenate((np.ones((m, 1)), mat_x), axis=1)

    #calculo el polinomio 'z'. Producto entre las matrices x y theta.
    z = np.dot(mat_x,vTranspTheta)

    h = 1 / (1 + np.exp(-z))

    h = np.transpose(h)

    return h

def darFormato(mat_result,nom_archivo):

    #Retorna una cadena en formato JSON con el nombre del archivo sobre el cual se realizó la clasificación
    #y los valores calculados por los clasificadores identificando además el máximo y a que emoción se
    #encuentra asociado.

    vCantCols = mat_result.shape[1]
    dicCol = {}
    dicResultado = {}
    vJSONResultado = None

    for i in range(0, vCantCols):

        vEmociónPredicha = ""
        vCol = mat_result[:, i]

        # calculo el máximo
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

        dicResultado = {'Archivo': nom_archivo, 'salida_clasificadores': dicCol, 'emocion_predicha': vEmociónPredicha}

    vJSONResultado = json.dumps(dicResultado)

    return vJSONResultado
