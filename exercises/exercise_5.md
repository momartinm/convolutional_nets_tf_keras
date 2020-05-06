## Taller de construcción de redes de neuronas convolucionales 
### Machine Learning, Tensor Flow, Keras, Redes de neuronas

## Ejercicio 5 - Desarrollo de una red convolucional con TensorFlow

El objetivo de este ejercicio es construir nuestra red de neuronas convolucional mediante la utilización de tensorflow. Para su realización vamos a reutilizar parte del código que hemos desarrollado en el (ejercicio 3)[./ejercicio_3.md]. 


**Paso 1. Instalación de liberías y despliegue de TensorBoard**

En este primer paso hay que incluir los paquetes que deben ser instalados con el objetivo de utilizar keras y TensorFlow Board. Para ello es necesario incluir el siguiente código al inicio del cuaderno. 

```
!pip install pandas scikit-learn numpy seaborn matplotlib numpy tensorflow==1.15 requests
```

Este comando permite cargar la extensión de TensorFlow Board. 

```
%load_ext tensorboard
```

**Paso 3. Definición de conjuntos de entrenamiento y test para el proceso de entrenamiento**

El primer paso consiste en dividir la información entre los conjuntos de entrenamiento y test. Para ello crearemos 4 variables denominadas train y test e identificadas con __X__ para los ejemplos e __y__ para las clases o etiquetas (labels). 

```
train_X = full_data.train.images
test_X = full_data.test.images

train_y = full_data.train.labels
test_y = full_data.test.labels

n_input = image_size * image_size
n_output = len(LABELS)
weights = list()
biases = list()
```

Para la construcción de la red de neuronas vamos a definir una serie de variables para almacenar la información de las diferentes capas:

- n_input: Se corresponde con el número de neuronas de entrada.
- n_output: Se corresponde con el número de neuronas de salida. Este valor se corresponderá con el número de labels o etiquetas. 
- weights: Se corresponde con un conjunto de pesos, cada uno aplicado a una capa. 
- biases: Se corresponde con un conjunto de bias, cada uno aplicado a una de las capas de las red. Existe una correspondencias 1 a 1 entre los elementos de la lista weights y la lista biases. 


**Paso 4. Construcción de capa convolución (Función)**

A continuación vamos a construir tres funciones para construir nuestras capas convolucionales (convolución + bias + function) y nuestras capas fully connected. Para ello crearemos una función denominada __conv2d__ que utilizará 4 parámetros de entrada:

- x: Se corresponde con la capa anterior. En el caso de la primera capa, este valor se corresponde con la imagen de entrada. Es importante tener en cuenta que el shape de este valor debe ser igual que el shape de los pesos (weights) de la capa. 
- W: Se corresponde con los pesos que se asignarán a la capa convolucional. 
- b: Se corresponde con el bias que se aplicará sobre las capa convolucional. 
- strides: Se corresponde con el salto de la ventana deslizante para cada una de las dimensiones de la entrada de la capa. Su valor puedes ser 1, 2 o 4 y por defecto es siempre 1. En este caso el valor de la primera capa debe ser 1 debido a que estamos trabajando con imágenes en blanco y negro que sólo tiene un canal y por tanto una dimensión. Si se da un único valor, se replica en la dimensión Height y Weight.

```
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 
```

Esta función construye la estructura de una red de neuronas basada en una capa convolucional que utiliza los pesos, un bias (sesgo) y por última la utilización de una función de activación. En este caso hemos decidido utilizar un función [relu](https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning). 

Con respecto a la función __conv2d__ de tensorflow que aplicar el filtro convoluciona, es importande describir el objeto de la opción __padding__. Se selecciona el valor  __SAME__ para la opción padding con el objetivo de garantizar que los pixeles de los bordes la imagen no se omitan durante el proceso de convolución. De forma que se añadiran pixeles con ceros alrededor de la imagen con el objetivo que se pueda aplicar el filtro de la convolución sobre todos los pixeles. 

**Paso 5. Construcción de capa pool (Función)**

La segunda función que vamos a definir es al función denominada __maxpool2d__ que tendrá dos parámetros de entrada:

- x: Se corresponde con la entrada y es el resultado de la la función __conv2d__. 
- k: Se corresponde con el tamaño del kernel aplicado a la capa de pooling (k x k). Su valor por defecto es 2. 

```
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
```

Esta función realiza una operación de tipo max pool con un kernell de 2x2. Al igual que en el filtro convolucional, se realiza un padding con el fin de incluir todos los pixeles de la imagen. 

**Paso 6. Construcción de capa fully connected (Función)**

La tercera función que vamos a definir es la función denominado __fully_connected__ que tendrá 3 paramétros de entrada:

- x: Se corresponde con la capa anterior. En el caso de la primera capa, este valor se corresponde con la imagen de entrada. Es importante tener en cuenta que el shape de este valor debe ser igual que el shape de los pesos (weights) de la capa. 
- W: Se corresponde con los pesos que se asignarán a la capa convolucional. 
- b: Se corresponde con el bias que se aplicará sobre la capa convolucional.

```
def fully_connected(x, w, b):
    x = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
    x = tf.add(tf.matmul(x, w), b)
    return tf.nn.relu(x)
```

Debido a que estamos realizando un proceso de clasificación debemos aplicar una función de fully connected que conecta cada neurona en la capa anterior (última capa convolucional) a cada neurona en la capa siguiente. Este proceso se puede considerar como como la aplicación de perceptrón multicapa (MLP) tradicional. La matriz de aplanada atraviesa una capa completamente conectada para clasificar las imágenes.

**Paso 7. Construcción de red de neuronas (Función)**

Una vez que hemos definidos nuestras funciones auxiliares para la creación de la capas de nuestra red vamos construir una función para la generación de redes convolucionales genéricas. Para ellos crearemos la función __generate_network__ que tendrá 6 parámetros de entrada:

- x: Se corresponde con la estructura de la imagen de entrada. 
- weights: Se corresponde con la lista de pesos de todas la capas de la red. El orden de los pesos en la lista se corresponde con el orden de las capas de las red.
- biases: Se corresponde con la lista de bias de todas la capas de la red. El orden de los bias en la lista se corresponde con el orden de las capas de las red y por tanto existe una correspondecia 1 a 1 con la lista de pesos. 
- layers: Es un array que indica el tipo de las capas. Siendo su valor 1 para las capas de tipo convolucional y 2 para la capas de tipo Fully connected layer. Normalmente las red de neuronas convolucional sólo tiene una capa de fully connected al final pero es posible añadir más. 
- drop_out: Es un valor boolean que indica si se quiere aplicar drop out para la red. Su valor por defecto es False. 
- dropout: Es el valor de drop our que se quiere aplicar sobre la red. 


```
def generate_network(x, weights, biases, layers, drop_out=False, dropout=None):
    
    # Tensor input 4-D: [Batch Size, Height, Width, Channel]
    previous_layer = tf.reshape(x, shape=[-1, 28, 28, 1])
    new_layer = None
    
    for i in range(len(layers)-1):
        if layers[i] == 1: #Convolutional layer
            new_layer = conv2d(previous_layer, weights[i], biases[i])
            new_layer = maxpool2d(new_layer, k=2)
        
        if layers[i] == 2: #Fully connected layer
            new_layer = fully_connected(previous_layer, weights[i], biases[i])
        
        previous_layer = new_layer
        
    if drop_out:
        previous_layer = tf.nn.dropout(previous_layer, dropout)
    
    return tf.add(tf.matmul(previous_layer, weights[len(layers)-1]), biases[len(layers)-1])
```

Esta función construye una red de neuronas formada por diferentes capas convoluciones y fully connected añadiendo al final el coeficiente de dropout y la última capa que generar el resultado de las n neuronas de salida que se corresponde con las diferentes clases en las que queremos clasificar nuestras imagenes. Cada una de las neuronas devolverá una valor numérico siendo la clases con mayor valor la que indica la clasificación. 


**Paso 8. Inicialización del grafo (TensorFLow)**

```
tf.reset_default_graph()
```

**Paso 9. Inicialización de placeHolders**

Una vez que hemos definido la función de generación, podemos construir nuestra red de neuronas y definir las variables necesarias para el proceso de aprendiaje.  En este caso utilizaremos sólo dos variables de tensorflow:

- placeholder: - placeholder: Son las variables de entrada (inputs) y salida (output) del algoritmo. Se generan mediante el método __tf.placeholder__ y se utilizan para definir el grafo de tensorflow sin la necesidad de haberles asignado un valor inicial. 
- variable: Son las variables que se modificarán durante el proceso de entrenamiento. Se generan mediante el método __tf.variable__ y se utilizan para definir variables dinámicas que tienen valor desde el inicio. En caso de que no se asigne valor, estas variables son inicializadas de manera aleatoria. 

```
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])
```

**Paso 10. Inicialización de variables**

Una vez definadas la variables de entrada y salida con su formato (shape) podemos construir nuestra red de neuronas que estará compuesta de tres 4 capas: 

- Capa convolucional 1: Capa convolucional que aplica un filtro convolucional de 3 x 3, pooling de 2 x 2 con una función de activación ReLU con entrada de 28 neuronas y salida de 32 neuronas. 
- Capa convolucional 2: Capa convolucional que aplica un filtro convolucional de 3 x 3, pooling de 2 x 2 con una función de activación ReLU con entrada de 32 neuronas y salida de 64 neuronas. 
- Capa fully connected: Capa fully connected con una función de activación ReLU con entrada de 64 y salida 512 neuronas. 
- Capa salida: Capa de salida con entrada de 512 neuronas y salida de 10 neuronas (labels). 

```
#Capa convolucional 32 filtro 3x3
weights.append(tf.get_variable('W0', shape=(3,3,1,32)))
biases.append(tf.get_variable('B0', shape=(32)))

#Capa convolucional 64 filtro 3x3
weights.append(tf.get_variable('W1', shape=(3,3,32,64)))
biases.append(tf.get_variable('B1', shape=(64)))

#Capa Densa 1024 unidades
weights.append(tf.get_variable('D1', shape=(7*7*64, 1024)))
biases.append(tf.get_variable('BD1', shape=(1024)))

#Capa salida 10 clases
weights.append(tf.get_variable('OUT', shape=(1024,n_output)))
biases.append(tf.get_variable('BOUT', shape=(n_output)))
```

<img src="./img/neurons_1.png" alt="Estructura de la red de neuronas" width="800"/>

**Paso 11. Generación de la red**

```
net = generate_network(x, weights, biases, [1, 1, 2, 3])
```

**Paso 12. Definición de función de loss**


Para la realización del proceso de aprendizaje es necesario definir algunos elementos básicos. En primer lugar vamos a definir la función de activación de la salida de la red de neuronas. Debido a que estamos construyendo un modelo de clasificación multi-clase utilizaremos un función de activación de tipo __softmax__ sobre las neuronas de salida de forma que obtengamos un valor probabilstico para cada uno de los labels. Esta función será combinada con un  [cross-entropy](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html) para calcular la función de loss. Vamos a utilizar esta función debido dos propiedades esenciales que se esperan para una función de coste: (1) el resultado es siempre positivo; y (2) el resultado tiende a cero según mejora la salida deseada (y) para todas las entradas del conjunto de entrenamiento (X). 


```
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=net, 
        labels=y))
```

**Paso 13. Definición de función de optimización**

A continuación tenemos que definir la función de obtimización que utilizaremos para minimizar el valor de función de coste. Para este ejecicio vamos a utilizar el algoritmo de [Adam](https://arxiv.org/abs/1412.6980https://arxiv.org/abs/1412.6980) con el fin de minimizar el coste del error mediante la función __tf.train.AdamOptimizer__. 

```
learning_rate = 0.001

optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(cost)
```

**Paso 14. Definición de función de entrenamiento**

Además de las dos operaciones básicas de un proceso de aprendizaje, vamos a añadir dos funciones más a nuestro grafo de operaciones más para realizar el proceso de test. Estas operaciones serán __correct_prediction__ y __accuracy__ que nos permitirán evaluar el modelo después de cada iteración de entrenamiento utilizando todos los elementos del conjunto de entrenamiento. 


```
correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

**Paso 15. Definición de bucle de entrenamiento (Función)**

Una vez que se han definido todas las variables y funciones necesarias para el proceso de aprendizaje, podemos construir el bucle en tensorflow. Para ello primero deberemos inicializar la variables mediante el método __tf.global_variables_initializer__. Este método inicializará todas las variables definidas previamente cuando se ejecute dentro de la sesion. A continuación será necesario crear una sesión en tensorflow para poder ejecutar todas nuestras funciones, siendo la primera la que inicializa la variables mediante la ejecución del método __session.run(init)__ de la sesión previamente creada. A continuación definiremos un conjunto de variables que almacenarán la información de cada una de nuestras iteraciones con el fin de poder visualizar la evolución del proceso. 

```

log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def train(training_iters, learning_rate, batch_size = 128):

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init) 

        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []

        writer = tf.summary.FileWriter(log_dir, sess.graph)

```

Una vez inicializadas las variables podemos comenzar el bucle de aprendizaje que se ejecutará tantas veces como iteraciones de entrenamiento. Cada iteraciones (epoch) se divirá en un conjunto de micro iteraciones utiliando barch de imágenes. Para cada uno de esto barch se aplicará la función de optimización y luego se obtendrá el coste (loss) y la exactitud (accuracy) del modelo. 

```
        for epoch in range(training_iters):

            for batch in range(len(train_X)//batch_size):

                batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
                batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    

                opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y})
	    
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss)]), 
                               epoch)
            
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=acc)]), 
                                epoch)

```
Estos valores se mostrarán cada 10 iteraciones con el fin de visualizar la evolución de nuestro proceso de entrenamiento. 

```
            if (epoch + 1) % 10 == 0:
                print("Iteración " + str(epoch+1) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Exactitud= " + \
                      "{:.5f}".format(acc))
	
		save_path = saver.save(sess, "./models/model" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".ckpt")
                
            test_acc, valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
            train_loss.append(loss)
            test_loss.append(valid_loss)
            train_accuracy.append(acc)
            test_accuracy.append(test_acc)

        print("Aprendizaje finalizado")

```

Una vez finalizado el proceso de entrenamiento, calcularemos los resultados finales con respeto al conjunto de test. 

```

        test_acc, valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Accuracy test:","{:.5f}".format(test_acc))

        summary_writer.close()
        
        return [train_loss, test_loss, train_accuracy, test_accuracy]
```

**Paso 16. Visualización de resultados (Función)**

Una vez realizado el proceso de entrenamiento vamos a utilizar la información recolectada por el proceso de entrenamiento con el fin de visualizar su evolución. Para ellos vamos a crear una función que denominaremos __print_results__ y utilizará 7 parámetros de entrada:

- train: Es el conjunto de los valores de coste (loss) del conjunto de entrenamiento en cada iteración. 
- test: Es el conjunto de los valores de coste (loss) del conjunto de test en cada iteración. 
- labels: Es un array bidimensional donde se incluye el nombre que se le dará a cada una de las curvas.
- Legend: Es un string que almacena el nombre que se le dará a la gráfica. 

```
def print_results(train, test, labels, legend):
    
    plt.plot(range(len(train)), train, 'b', label=labels[0])
    plt.plot(range(len(test)), test, 'r', label=labels[1])
    plt.title('Entrenamiento y test')
    plt.xlabel('Iteraciones',fontsize=16)
    plt.ylabel(legend,fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()

```

**Paso 17. Ejecución del proceso de entrenamiento**

Una vez construidas nuestras funciones podemos ejecutar nuestro proceso de aprendizaje de la siguiente manera, ejecutando el proceso de aprendizaje durante 100 iteraciones con una tasa de aprendizaje del 0.001 y un tamaño de batch de 128 imágenes. 

```
!rm -rf ./logs/
results = train(10, 0.001, 128)
print_results(results[0], results[1], ['Loss entrenamiento', 'Loss test'], 'Loss')
print_results(results[2], results[3], ['Acurracy entrenamiento', 'Acurracy test'], 'Acurracy')
```

Siendo el resultado obtenido tras ejecutar el codígo el siguiente:

```
Iteración 10, Loss= 0.091082, Exactitud= 0.94531
Aprendizaje finalizado
Exactitud test: 0.89510
```

<img src="../img/regresion_conv_neurons_1.png" alt="Resultado de aprendizaje tras 10 iteraciones" width="800"/>


**Paso 18. Ejecución del proceso de entrenamiento**

Es posible visualizar la información mediante TensorFlow Board con el objetivo de poder obtener toda la información sobre el proceso de aprendizaje. Para ello es necesario incluir el siguiente comando y ejercutar el fragmento del cuarderno. TensorBoard utilizar los ficheros de logs que se han generado en el fichero que indiquemos como valor del parámetro __logdir__, que en este caso se corresponde con la carpeta logs que hemos utilizado para almacenzar los logs generados en el proceso de entrenamiento del paso 13. 

```
%tensorboard --logdir logs
```


