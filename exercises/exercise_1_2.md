## Taller de construcción de redes de neuronas convolucionales 
### Machine Learning, Tensor Flow, Keras, Redes de neuronas

## Ejercicio 2 - Despliegue del entorno de trabajo 

El objetivo de este ejercicio es desplegar el entorno de trabajo mediante la utilización de un servidor Jupyter Notebook. Para ello, vamos a construir un contenedor mediante Docker que contenga un servidor Jupyter Notebooks fin de poder desarrollar diferentes tipos de algoritmos de manera sencilla. Docker nos permite desplegar de forma sencilla contenedores utilizando imágenes previamente creadas, para aprender como reutilizar estas imágenes vamos a desplegar un servidor Jupyter Notebook. Existe diferentes maneras de construir nuestro contenedor Jupyter Notebook: 

* Mediante la utilización el despligue de una imagen
* Mediante la generación de un fichero de despliegue (docker-compose.yml)

**Paso 1: Descargando la imagen**

En primer lugar vamos a descarga la imagen que queremos instalar, para comprobar que imágenes tenemos disponibles podemos ir acceder al listado de imágenes disponibles en dockerhub. Para este ejercicio vamos a utilizar una imagen espécifica para Data Scientisth, que puede descargarse en el siguiente [enlace](https://hub.docker.com/r/jupyter/datascience-notebook/) . 

```
$ docker pull jupyter/datascience-notebook:latest
```

A continuación comprobaremos si la imagen se ha descargado correctamente y está disponible en nuestro repositorio local de imágenes, mediante el siguiente comando:

```
$ docker images 
```

Obteniendo la siguiente salida que nos indica que hemos descargado la imagen mongo en su versión (tag) 3.4.21-xenial hace 6 semanas. 

```
REPOSITORY                     TAG                 IMAGE ID            CREATED             SIZE
jupyter/datascience-notebook   latest              9e64f3a158ed        2 weeks ago         4.91GB
```

**Paso 2: Desplegandando la imagen**

Una vez que hemos descargado la imagen podemos deplegarla para levantas nuestro servidor MongoDB, mediante el siguiente comando:

```
$ docker run --name=jupyter_server -p 8888:8888 -p 6006:6006 -e JUPYTER_ENABLE_LAB=yes -e GRANT_SUDO=yes --user root jupyter/datascience-notebook:latest -d
```

**Paso 3: Desplegandando la imagen mediante compose**

La otra alternativa a la creación de nuestro contenedor por linea de comando, es crear nuestro contenedor mediante un fichero de despliegue, para ello tenemos que crear nuestro fichero docker-compose.yml. Este fichero de despligue nos permite definir la configuración de nuestro contener y incluir algunas opciones como:

* Ports: Nos permite indicar los puertos que vamos a abrir para acceder a los servicios del contenedor. En este caso vamos ha abrir dos puertos: (i) 8888 para acceder al servidor de jupyter notebook; y (ii) 6006 para acceder al interfaz gráfico de TensorBoard.
* Volumes: Nos permite mapear ciertos directorios del contenedor en nuestra máquina host (tu ordenador) con el objetivo de conservar los archivos cuando dejemos de ejecutar el contenedor.
* Networks: Nos permite asociar a nuestro contenedor una dirección ip a un red que nos permitirá conectarlo con otros contenedores en el que caso de que queramos incluir otros servicios en el futuro. Es necesario definir la configuración de la red en fichero de despliegue. 

__IMPORTANTE: Si estamos trabajando en un sistema operativo Windows debemos eliminar las dos líneas referente a los volúmenes debido a que en algunas versiones de windows este tipo de configuración no funciona correctamente.__

```
version: '3.4'
services:
  
  jupyter:
    restart: always
    image: jupyter/datascience-notebook:latest
    container_name: jupyter_server 
    ports:
      - "8888:8888"
      - "6006:6006"
    volumes:
      - ./notebooks:/home/jovyan/work
    user: root
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - GRANT_SUDO=yes
    networks:
      workshop_ml:
        ipv4_address: 172.24.1.3

networks:
  workshop_ml:
    driver: bridge
    driver_opts:
      com.docker.network.enable_ipv6: "true"
    ipam:
      driver: default
      config:
        - subnet: 172.24.1.0/24
```

Una vez construido nuestro fichero de despliegue podemos lanzar nuestro fichero de despliegue mediante el siguiente comando:

```
$ docker-compose -f docker_compose.yml up --build -d
```

**Paso 5: Configurando los permisos de escritura**

Tras la correcta ejecución del comando de despliegue, se habrá creado una nueva carpeta en el directorio sobre el cual lo hemos ejecutado denominada __notebooks__. Para comprobar la existencia de este directorio podemos utilizar el siguiente comando:

```
ls -la 
```

de forma que obtendremos el siguiente resultado:

```
total 12
drwxrwxr-x 6 momartin momartin  4096 may  3 01:34 .
drwxrwxr-x 4 momartin momartin  4096 mar 30 06:07 ..
drwxr-xr-x 2 root     root      4096 may  3 01:34 notebooks
```

Como podemos observar, este directorio pertenece al usuario __root__ por lo que no se nos permitirá el almacenaminamiento de archivo dentro de el. Para modificar su propietario debemos utilizar el siguiente comando:


```
sudo chown momartin:momartin notebooks
```

**Paso 5: Accediendo a nuestro Jupyter Notebook server**

Una vez que hemos desplegado correctamente nuestro servidor Jupyter Notebook podremos acceder a el mediante la siguiente url:

```
http://localhost:8888/
```

Al acceder a la página web nos solicitará un token de acceso por motivos de seguridad. Para obtener el token de acceso es necesario acceder a los logs del contenedor que hemos desplegado mediante el siguiente comando:


```
$ docker logs jupyter_server
```

La salida de este comando nos mostará todo el log del contenedor donde podremos encontrar el token de acceso para acceder a nuestro servidor Jupyter Notebooks. 

```
Executing the command: jupyter lab
[I 06:21:03.420 LabApp] Writing notebook server cookie secret to /home/jovyan/.local/share/jupyter/runtime/notebook_cookie_secret
[I 06:21:04.777 LabApp] JupyterLab extension loaded from /opt/conda/lib/python3.7/site-packages/jupyterlab
[I 06:21:04.778 LabApp] JupyterLab application directory is /opt/conda/share/jupyter/lab
[I 06:21:05.881 LabApp] Serving notebooks from local directory: /home/jovyan
[I 06:21:05.881 LabApp] The Jupyter Notebook is running at:
[I 06:21:05.881 LabApp] http://5778cabb64d7:8888/?token=1e6d710c051275c055ab068fe46b7ef9f5b8a6eb24519bc0
[I 06:21:05.881 LabApp]  or http://127.0.0.1:8888/?token=1e6d710c051275c055ab068fe46b7ef9f5b8a6eb24519bc0
[I 06:21:05.881 LabApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 06:21:05.886 LabApp] 
    
    To access the notebook, open this file in a browser:
        file:///home/jovyan/.local/share/jupyter/runtime/nbserver-6-open.html
    Or copy and paste one of these URLs:
        http://5778cabb64d7:8888/?token=1e6d710c051275c055ab068fe46b7ef9f5b8a6eb24519bc0
     or http://127.0.0.1:8888/?token=1e6d710c051275c055ab068fe46b7ef9f5b8a6eb24519bc0
```

**Paso 6: Creando nuestro notebook mediante python**

Una vez que hemos desplegado nuestro servidor Jupyter Notebook y hemos accedido a la consola de inicio. Podremos trabajar con su interfaz gráfico:

<img src="../img/Jupyter_1.png" alt="Interfaz inicial Jupyter Notebook" width="800"/>

Como podemos observar en la imagen, el posible crear diferentes tipos de elementos (notebooks, scripts, etc). Para la realización de este ejercicio vamos a utilizar un notebook de tipo Python 3. De forma que una vez que pulsemos sobre el botón de python 3 crearemos un nuevo notebook, como se muestra en la siguiente imagen:

<img src="../img/Jupyter_2.png" alt="Interfaz inicial Jupyter Notebook" width="800"/>

Este interfaz nos permite crear incluir fragmentos de código python y analizar su resultado una vez que ha sido ejecutado el fragmento de código, cada uno de estos fragmento es representado mediante un id entre corchetes, siendo su significado el siguiente:

- [id]: Identifica un fragmento de código con el id. 
- [*]: Identificado un fragmento de códico en ejecución. 
