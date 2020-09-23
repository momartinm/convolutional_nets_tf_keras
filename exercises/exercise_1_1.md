## Taller de construcción de redes de neuronas convolucionales 
### Machine Learning, Tensor Flow, Keras, Redes de neuronas

## Ejercicio 1 - Instalación de docker

El objetivo de este ejercicio es instalar el entorno principal de trabajo para el desarrollo del taller. Docker es un proyecto de software que permite crear, construir y deplegar aplicaciones software de forma rápica y sencilla mediante virtualización basada en contenedores. Docker empaqueta el software en unidades estandarizadas llamadas contenedores que incluyen todo lo necesario para que el software se ejecute, incluyendo bibliotecas, herramientas de sistema, código y tiempo de ejecución. Con Docker, se puede implementar y ajustar la escala de aplicaciones rápidamente en cualquier entorno con la certeza de saber que su código se ejecutará. Ya que el contenedor es autocontenido desde el punto de vista del SSOO y las aplicaciones que se ejecutan dentro. 

### Ciclo de vida de un contenedor

El ciclo de vida de un contenedor es el conjunto de estados por los que puede pasar un contenedor a lo largo de su vida útil. 

<img src="./img/contenedores-2.svg" alt="La metáfora del contenedor" width="800"/>

* Definición (DockerFile): Es el estado en el cual se están definiendo los elementos básicos del contenedor que se corresponden con la configuración del propio contenedor (SSOO), las diferentes libreras que serán instaladas, el código fuente que será desplegado y la forma en la que se ejecutará el código fuente. 
* Imagen (Image): Es un imagen construida, es decir, todos los componentes han sido instalados y desplegados para su ejecución. 
* Parada (Stopped): Es un contenedor en que encuentra en estado de reposo, es decir la imagen está congelada.  
* Ejecución (Running): Es un contenedor en ejecución, es decir la imagen está en ejecución.   

### Comando básicos de docker

versión
```
$ docker --version
```

descarga de imágenes 
```
$ docker pull nombre[:tag|@digest]
```

Despliegue de un contenedor (básico)
```
$ docker run --name=id_contenedor imagen
```

Despliegue de un contenedor en segundo plano (básico)
```
$ docker run --name=id_contenedor -d imagen
```

Listar/visualizar todas la imágenes finales
```
$ docker images
```

Listar/visualizar todas la imágenes (incluyenod las imágenes ocultas)
```
$ docker images -a 
```

Eliminar una imagen
```
$ docker images rm <id_imagen>
```

Acceder al contenedor 
```
$ docker exec -it <id_contenedor> <programa>
```

Listar/visualizar todos los contenedores en ejecución
```
$ docker container ps
```

Listar/visualizar todos los contenedores
```
$ docker container ps -a
```

Parar un contenedor
```
$ docker container stop <id_contenedor>
```

Arrancar de un contenedor (Debe encontrarse en el estado de parada (Stop))
```
$ docker container start <id_contenedor>
```

Borrar de un contenedor (Debe encontrarse en el estado de parada (Stop))
```
$ docker container rm <id_contenedor>
```

Borrar de un contenedor
```
$ docker container rm -f <id_contenedor>
```

visualizar información del contenedor
```
$ docker inspect <id_contenedor>
```

Crear un volumen 
```
$ docker volume create --name <nombre_volumen>
```

eliminar un volumen 
```
$ docker volume rm <nombre_volumen>
```

### Instalación

Para poder utilizar docker es necesario instalarlo previamente. En algunas ocasiones este se instala de forma conjunta con algunas aplicaciones por lo que es posible que se encuentre en tu sistema operativo. Para poder comprobar si docker se encuentra en tu sistema debes abrir un terminal o interfaz de comando y ejecutar el siguiente comando:

```
$ docker --version
```

y deberás obtener un resultado similar al siguiente (Este resultado se obtiene al ejecutar el comando sobre el sistema operativo Ubuntu 18.10): 

```
Docker version 19.03.5, build 633a0ea838
```

#### Instalación en Linux (Ubuntu)
Mac
Para realizar la instalación sobre Ubuntu es necesario seguir los siguientes pasos:

**Paso 1: Desintalación de versiones anteriores (opcional)**

Se recomienda instalar la última versión estable de docker, por lo que si tu versión es inferior a la 19.03 o tu versión se corresponde con docker.io o docker-engine deberías eliminar la versión que tienes instalada e instalar una nueva versión. Para ello deberás utilizar el siguiente comando:

```
$ sudo apt-get remove docker docker-engine docker.io containerd runc
```

**Paso 2: Instalación de docker mediante apt-get**

A continuación, podemos realizar una instalación limpia de docker. Para ello vamos a utilizar la versión del repositorio de ubuntu, aunque si lo deseas puede utilizar el repositorio oficial de docker siguiente las instrucciones de su [página web](https://docs.docker.com/engine/install/ubuntu/). 


1. Instalación de paquetes básicos para permitir la utilización de paquetes apt sobre HTTPS.

```
$ sudo apt-get update

$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
```

2. Añadimos las claves GPG oficiales de docker

```
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

3. Añadimos el último repositorio stable. 

```
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```

4. Realizar la instalación de Docker mediante los siguiente comandos:

```
$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io 
```

5. Si todo ha ido bien tras la instalación podrás utilizar el comando para obtener la versión de docker. Estamos listos para poder empezar con el taller. 

```
$ docker --version
```

#### Instalación en Mac

Para realizar la instalación sobre Mac es necesario seguir los siguientes pasos:

**Paso 1: Descarga de Docker Desktop**

En primer lugar es necesario descarga la versión disponibles para Mac en el siguiente [enlace](https://hub.docker.com/editions/community/docker-ce-desktop-mac/).

**Paso 2: Instalación de Docker Desktop**

1. Hacer doble click sobre el archivo Docker.dmg para comenzar con el proceso de instalación. 

2. A continuación, tienes que mover el icono de docker al directorio de aplicaciones como se muestra en la siguiente figura.

<img src="../img/docker-mac-1.png" alt="Instalación de docker en Mac 1" width="800"/>

3. Hacer doble click en el icono de docker que está en tu directorio de aplicaciones para iniciar la ejecución, cómo se muestra en la siguiente figura. Para la ejecución de docker se te solicitará autorización mediante usuario y contraseña. Es necesario tener privilegios para la instalación de ciertos componentes. 

<img src="../img/docker-mac-2.png" alt="Instalación de docker en Mac 2" width="800"/>

4. Si la aplicación ha podido ejecutarse correctamente deberá aparecer el icono de docker en la barra superior derecha de tu escritorio. Este icono indica que Docker Desktop está ejecutandose y puedes utilizarlo en tu terminal. 

<img src="../img/docker-mac-3.png" alt="Instalación de docker en Mac 3" width="160"/>

5. Si todo ha ido bien tras la instalación podrás utilizar el comando para obtener la versión de docker como se indica a continuación. 

```
$ docker --version
```

6. Si todo ha ido bien, ya estamos listos para poder empezar con el taller. 

#### Instalación en Windows

Para realizar la instalación sobre Windows es necesario seguir los siguientes pasos:

**Paso 1: Descarga de Docker Desktop**

En primer lugar es necesario descarga la versión disponibles para Windows en el siguiente [enlace](https://hub.docker.com/editions/community/docker-ce-desktop-windows/). Debes elegir la versión que se adapte a tu sistema operativo. 

**Paso 2: Instalación de Docker Desktop**

1. Hacer doble click sobre el archivo Installer.exe para comenzar con el proceso de instalación. 

2. Seguir las instrucciones del instalador y aceptar las condiciones de uso. Para la instalación de docker se te solicitará autorización mediante usuario y contraseña. Es necesario tener privilegios para la instalación de ciertos componentes. 

3. Pulsar el boton de finalizar (Finish) para completar la instalación. 

4. Ejecutar Docker Desktop application como se muestra en la siguiente figura.

<img src="../img/docker-win-1.png" alt="Instalación de docker en Windows 1" width="320"/>

5. Si la aplicación ha podido ejecutarse correctamente deberá aparecer el icono de docker (Ballena) en la barra inferior derecha de tu escritorio. Este icono indica que Docker Desktop está ejecutandose y puedes utilizarlo en tu terminal. 

<img src="../img/docker-win-2.png" alt="Instalación de docker en Windows 2" width="160"/>


5. Si todo ha ido bien tras la instalación podrás utilizar el comando para obtener la versión de docker como se indica a continuación. 

```
$ docker --version
```

6. Si todo ha ido bien, ya estamos listos para poder empezar con el taller. 

### Recursos

- [Página oficial | Docker](https://docs.docker.com/)
- [Página oficial | Referencias](https://docs.docker.com/reference/)

