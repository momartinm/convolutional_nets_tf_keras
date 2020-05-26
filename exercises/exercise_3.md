## Taller de construcción de redes de neuronas convolucionales 
### Machine Learning, Tensor Flow, Keras, Redes de neuronas

## Ejercicio 3 - Descarga y análisis de la información 

El objetivo de este ejercicio es descargar la información y analizarla con el objetivo de entender el formato de los datos de entrenamiento, el número de clases y dejar la información preparada para el proceso de entrenamiento. 

**Paso 1: Instalando paquetes en Jupyter**

Los notebooks son entidades independientes que permiten la utilización de cualquier tipo de páquete python y para ellos nos ofrece la posibilidad de instalar paquete mediante la utilización de la sistema de instalación de paquetes pip. Para la instalación de los diferentes paquetes que utilizaremos para la realización de nuestro paquetes tenemos que ejecutar el siguiente comando:

```
!sudo apt update
!sudo apt -y install graphviz
!pip install pandas scikit-learn numpy seaborn matplotlib numpy requests pydot
```

Como podemos observar, es necesario incluir el caracter __!__ antes del comando de instalación. A continuación hay que seleccionar el fragmento y pulsar la tecla play para ejecutar el código contenido en el fragmento. Siendo el resultado de la ejecución de esta linea, el siguiente:

```
Requirement already satisfied: pandas in /opt/conda/lib/python3.7/site-packages (1.0.3)
Requirement already satisfied: scikit-learn in /opt/conda/lib/python3.7/site-packages (0.22.2.post1)
Requirement already satisfied: numpy in /opt/conda/lib/python3.7/site-packages (1.18.1)
Requirement already satisfied: seaborn in /opt/conda/lib/python3.7/site-packages (0.10.1)
Requirement already satisfied: matplotlib in /opt/conda/lib/python3.7/site-packages (3.2.1)
Requirement already satisfied: tensorflow==1.15 in /opt/conda/lib/python3.7/site-packages (1.15.0)
Requirement already satisfied: requests in /opt/conda/lib/python3.7/site-packages (2.23.0)
Requirement already satisfied: pytz>=2017.2 in /opt/conda/lib/python3.7/site-packages (from pandas) (2020.1)
Requirement already satisfied: python-dateutil>=2.6.1 in /opt/conda/lib/python3.7/site-packages (from pandas) (2.8.1)
Requirement already satisfied: scipy>=0.17.0 in /opt/conda/lib/python3.7/site-packages (from scikit-learn) (1.4.1)
Requirement already satisfied: joblib>=0.11 in /opt/conda/lib/python3.7/site-packages (from scikit-learn) (0.14.1)
Requirement already satisfied: cycler>=0.10 in /opt/conda/lib/python3.7/site-packages (from matplotlib) (0.10.0)
Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/lib/python3.7/site-packages (from matplotlib) (2.4.7)
Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/lib/python3.7/site-packages (from matplotlib) (1.2.0)
Requirement already satisfied: gast==0.2.2 in /opt/conda/lib/python3.7/site-packages (from tensorflow==1.15) (0.2.2)
Requirement already satisfied: termcolor>=1.1.0 in /opt/conda/lib/python3.7/site-packages (from tensorflow==1.15) (1.1.0)
Requirement already satisfied: six>=1.10.0 in /opt/conda/lib/python3.7/site-packages (from tensorflow==1.15) (1.14.0)
Requirement already satisfied: keras-applications>=1.0.8 in /opt/conda/lib/python3.7/site-packages (from tensorflow==1.15) (1.0.8)
Requirement already satisfied: absl-py>=0.7.0 in /opt/conda/lib/python3.7/site-packages (from tensorflow==1.15) (0.9.0)
Requirement already satisfied: astor>=0.6.0 in /opt/conda/lib/python3.7/site-packages (from tensorflow==1.15) (0.8.1)
Requirement already satisfied: protobuf>=3.6.1 in /opt/conda/lib/python3.7/site-packages (from tensorflow==1.15) (3.11.4)
Requirement already satisfied: tensorboard<1.16.0,>=1.15.0 in /opt/conda/lib/python3.7/site-packages (from tensorflow==1.15) (1.15.0)
Requirement already satisfied: keras-preprocessing>=1.0.5 in /opt/conda/lib/python3.7/site-packages (from tensorflow==1.15) (1.1.0)
Requirement already satisfied: grpcio>=1.8.6 in /opt/conda/lib/python3.7/site-packages (from tensorflow==1.15) (1.28.1)
Requirement already satisfied: wrapt>=1.11.1 in /opt/conda/lib/python3.7/site-packages (from tensorflow==1.15) (1.12.1)
Requirement already satisfied: google-pasta>=0.1.6 in /opt/conda/lib/python3.7/site-packages (from tensorflow==1.15) (0.2.0)
Requirement already satisfied: wheel>=0.26 in /opt/conda/lib/python3.7/site-packages (from tensorflow==1.15) (0.34.2)
Requirement already satisfied: opt-einsum>=2.3.2 in /opt/conda/lib/python3.7/site-packages (from tensorflow==1.15) (3.2.1)
Requirement already satisfied: tensorflow-estimator==1.15.1 in /opt/conda/lib/python3.7/site-packages (from tensorflow==1.15) (1.15.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/conda/lib/python3.7/site-packages (from requests) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/lib/python3.7/site-packages (from requests) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.7/site-packages (from requests) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/conda/lib/python3.7/site-packages (from requests) (2.9)
Requirement already satisfied: h5py in /opt/conda/lib/python3.7/site-packages (from keras-applications>=1.0.8->tensorflow==1.15) (2.10.0)
Requirement already satisfied: setuptools in /opt/conda/lib/python3.7/site-packages (from protobuf>=3.6.1->tensorflow==1.15) (46.1.3.post20200325)
Requirement already satisfied: werkzeug>=0.11.15 in /opt/conda/lib/python3.7/site-packages (from tensorboard<1.16.0,>=1.15.0->tensorflow==1.15) (1.0.1)
Requirement already satisfied: markdown>=2.6.8 in /opt/conda/lib/python3.7/site-packages (from tensorboard<1.16.0,>=1.15.0->tensorflow==1.15) (3.2.1)
```

En este caso no se ha realizado la instalación de ningún paquete debido a que todos ya estaban instalados en el servidor Jupyter. 

**Paso 2: Importando librerías**

Una vez que se ha realizado la instalación de los diferentes paquetes python, es necesario importar aquellas clases y métodos necesarios para la realización del ejercicio.

```
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import os.path
import requests 
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

```

Para el desarrollo de los diferentes ejercicios vamos a necesitar un conjunto de liberías que servirán para lo siguiente:

* numpy: Nos ofrece funciones para la manipulación de arrays y conjunto de datos. 
* matplotlib: Nos ofrece funciones para la visualización de datos. 
* tensorflow: Nos ofrece funciones para la construacción de procesos de entrenamiento. 
* os: Nos ofrece funciones para la manipulación de recursos del sistema operativo. 
* os: Nos ofrece funciones para la manipulación del sistema de ficheros del sistema operativo.
* requests: Nos ofrece funciones para la descarga de archivos de localizaciones remotas.
* math: Nos ofrece funciones para la realización de operaciones matemáticas complejas (no elementales).

Además, vamos a incluir algunas variables de entorno:

* TF_CPP_MIN_LOG_LEVEL para definir el nivel de visualización de los mensajes del framework TensorFlow. Existen 4 niveles de visualización incrementales: (I) 0 para mostrar todos los mejores; (II) 1 para filtrar los mensajes de tipo INFO; (III) 2 para filtrar los mensajes de tipo WARN; (3) 3 para filtrar los mensajes de tipo ERROR. 
* CUDA_VISIBLE_DEVICES para indicar que los dispositivos de tipo CUDA no sean visibles. 

**Paso 3: Descarga de datos**

A continuación vamos a realizar la carga de datos. En primer lugar, tenemos que definir las urls donde se encuentran los datasets para el proceso de entrenamiento y test. Para este ejercicio vamos a utilizar los datos contenidos en el dataset de [Zalando sobre ropa](https://github.com/zalandoresearch/fashion-mnist). 

```
urls = ['http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
         'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
         'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
         'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz']
```

Una vez que hayamos definidos las localizaciones de nuestro ficheros de ejemplos (entrenamiento y test) podemos crear un directorio para almacenarlos y descargalos. Para ellos incluiremos el siguiente código. 

```
data_path = "data"

try:
    os.mkdir(data_path)
except OSError:
    print ("El directorio %s no hay podido ser creado" % (data_path))
else:
    print ("El directorio %s ha sido creado correctamente" % (data_path))

```
Una vez hayamos creado el directorio __data__ podemos proceder a descargar todos los elementos disponibles en el array __urls__ mediante el siguiente fragmento de código. 

```
for url in urls:
    save_path = os.path.join("data", url.split('/')[-1])
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
        print ("Fichero %s ha sido descargado correctamente" % (save_path))
```


**Paso 4: Definición de clases y datos**

A continuación tenemos que cargar los datos en las estructuras de datos básicas para comenzar a trabajar con ellos. Por lo que necesitaremos dos conjuntos de datos:

* Ejemplos (entrenamiento/validacion/test): Conjunto de ejemplos de información (imágenes) para los procesos de entrenamiento, validación y test.
* Labels (clases): Conjunto de clases asignadas a cada una de las imágenes de los diferentes conjuntos. Cada conjunto tendrá un conjunto de labels del mismo tamaño. 

Para poder cargar los datos en formato minist tenemos que utilizar las funcionalidades de importación propuesta por el equipo de TensorFlow (version 2016). Para ello debemos cargar el código de las funciones de cargar mediante la inclusión de un archivo local que denominaremos __input_data.py__. El código fuente de este archivo, se puede descargar en la siguiente [url](./resources/exercise_4/input_data.py). Una vez incluido este archivo podemos realizar la carga de datos. Para ellos utilizaremos la función __read_data_sets__ que nos permite cargar dataset desde una url, utilizando las siguiente opciones:

- Nombre del dataset
- source_url: Se corresponde con la url donde estará almacenada la información. 
- one_hot:  Realiza una transformación sobre las variables categorizadas a una codificación binaria. Es decir si tenemos n valores para una variables categorica se crearan n features binarias (0,1) de forma que sólo una de ellas tendrá el valor 1 correspondiendose con uno de los valores de la variable categorica. En este ejercicio, se utiliza para convertar la caraterística label (Clase de salida) en una coficiación binaria. 

```
full_data = input_data.read_data_sets('data', one_hot=True)

# Condificación one hot para un ejemplo de tipo Dress
# 3 => [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# El valor tres se convierte en una array de probabilidades donde la posición que se corresponde con la etiqueta (3) tiene una probabilidad de 1 y el resto tienen una probabilidad de cero.


LABELS = {
 0: 'Camiseta/top',
 1: 'Pantalones',
 2: 'Sudadera',
 3: 'Vestido',
 4: 'Abrigo/Gabardina',
 5: 'Sandalias/Zapato',
 6: 'Camisa',
 7: 'Zapatillas',
 8: 'Bolso/Bolsa',
 9: 'Botas',
}
```

**Paso 5: Análisis de datos**

Una vez que hemos cargado los datos, tenemos que analizar los datos para entender su estructura, formato, si tenemos disponibles los suficientes conjuntos de entrenamiento, etc. Para ellos vamos a analizar algunas caracteristicas de los datos. Primero comprobaremos el tamaño (shape) de los conjuntos de datos que vamos a utilizar:

```
print("Conjunto de entrenamiento (Imágenes) shape: {shape}".format(shape=full_data.train.images.shape))
print("Conjunto de entrenamiento (Classes) shape: {shape}".format(shape=full_data.train.labels.shape))
print("Conjunto de test (Imágenes) shape: {shape}".format(shape=full_data.test.images.shape))
print("Conjunto de test (Clases) shape: {shape}".format(shape=full_data.test.labels.shape))
```

Cómo podemos observar tenemos 55000 ejemplos (imágenes) de entrenamiento donde cada uno de ellas está formada por un array de 784 pixeles y pueden pertener a 10 clases diferentes. 

Lo primero a analizar, es comocer si el número de clases y la estructura de los ejemplos es similar tanto en el conjunto de test como entrenamiento. En este caso ambos conjuntos están bien formados, tenemos un conjunto global formado por 65000 ejemplos donde un 18% se corresponde con el conjunto de test y todas la imágenes tienen el mismo tamaño (784 pixeles). 

Lo segundo a comprobar es el formato de la imagen, tenemos que definir cual es la estructura de la imagen. Para comprobarlo, bastaría con calcular la raíz cuadrada de 784 que se corresponde con 28. Esto significa que las imágenes tiene un tamaño de 28x28 pixeles. __Es obligatorio que todas las imágenes tenga el mismo tamaño, sino no podremos construir nuestra red de neuronas__. 

```
print(full_data.train.images[0].shape)
print(math.sqrt(full_data.train.images[0].shape[0]))
image_size = int(math.sqrt(full_data.train.images[0].shape[0]))
print(image_size)

print(full_data.train.labels[0])
```

Lo tercero es comprobar el volumen de ejemplos de cada clase en el conjunto de entrenamiento y en el conjunto de test. Normalmente nosotros tendremos que crear estos conjuntos, por lo que es muy útil comprobar si los conjuntos están balanceados, con el objetivo de balancearlos en caso de que no ocurra. 

```
train_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
test_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for label in full_data.train.labels:
    train_labels = np.add(train_labels, label)
    
print(train_labels)

for label in full_data.test.labels:
    test_labels = np.add(test_labels, label)

print(test_labels)
```

**Paso 6: Visualización de los datos**

Por último vamos a crear una función para visualizar las imágenes con las que estamos trabajando con el objetivo de ver el tipo de imágenes que estamos utilizando. La función se denominará __plot_image__ y nos permitirá visualizar imágenes con su etiqueta y utilizará 5 parámetros de entrada:

- plt: Es la figura sobre la que se insertará la imagen. 
- data: Se corresponde con la imagen que queremos visualizar. 
- label: Se corresponde con la etiqueta asignada a la imagen. Es un vector de n valores. 
- size: El tamaño de la imagen. Es una tupla con dos valores. 
- location: Es la localización de la imagen en la figura. Se corresponde con un secuencia de tres número enteros. 

```
def plot_image(plt, data, label, size, location):
    plt.subplot(location)
    img = np.reshape(data, size)
    label = np.argmax(label)
    plt.imshow(img)
    plt.title("(Label: " + str(LABELS[label]) + ")")
```

Una vez que hemos generado la función para visualizar la estructura de los ejemplos y las etiquetas (labels) podemos utilizarla para mostrar algunos de nuestros ejemplos mediante el siguiente fragmento de código:

```
plt.figure(figsize=[18,18])

plot_image(plt, 
           full_data.train.images[4], 
           full_data.train.labels[4,:], 
           (image_size, image_size),
           121)

plot_image(plt, 
           full_data.test.images[95], 
           full_data.test.labels[95,:], 
           (image_size, image_size),
           122)
```

**Congratulations Ninja!**

Has aprendido como preparar los datos para el proceso de aprendizaje, como definir las clases del modelo y como distribuir los conjunto de entrenamiento y test. Has conseguido aprender:

1. Como instalar paquetes en un notebook. 
2. Como descargar archivo mediante python request.
3. Como definir las clases o etiquetas (label) para la construcción de un modelo de aprendizaje. 
4. Como realizar un análisis básico sobre los datos de entrenamamiento y test.
5. Como visualizar imágenes mediante matplotlib. 

<img src="../img/ejercicio_3_congrats.png" alt="Congrats ejercicio 3" width="800"/>


