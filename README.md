# Laboratorio: Computación de Alto Rendimiento con CUDA

* **Estudiante:** Juan Carlos Barajas Quintero 
* **Curso:** Arquitectura de Computadores - Unidad 11
* **Institución:** Universidad Francisco de Paula Santander
  
Este repositorio contiene las implementaciones en CUDA C/C++ para el benchmark de suma de vectores (`vectorAdd`) y multiplicación de matrices (`matMul`), evaluando el impacto del procesamiento paralelo y la optimización mediante memoria compartida (*Shared Memory*).

---

##  Descripción del entorno

El benchmark y las pruebas de rendimiento fueron ejecutados bajo la siguiente configuración de hardware y software:

* **Modelo de GPU:** NVIDIA GeForce RTX 2050 (4GB VRAM / 2048 Núcleos CUDA)
* **Versión de CUDA:** CUDA Toolkit 13.2 (Driver API / Runtime API compatible)
* **Sistema Operativo (OS):** Windows 11 Home (Arquitectura x64)

---

##  Tabla de resultados: `vectorAdd`

A continuación se detallan los tiempos obtenidos al procesar la suma de vectores para diferentes cargas de trabajo ($N$). Los tiempos están expresados en milisegundos (ms).

| Tamaño ($N$) | Tiempo CPU (ms) | GPU Kernel (ms) | GPU Total con `cudaMemcpy` (ms) | Errores vs CPU |
| :--- | :--- | :--- | :--- | :--- |
| **1M** ($1,048,576$) | ~1.20 ms | ~0.05 ms | ~1.10 ms | 0 |
| **4M** ($4,194,304$) | ~4.75 ms | ~0.18 ms | ~3.90 ms | 0 |
| **16M** ($16,777,216$) | 19.00 ms | 0.70 ms | 15.10 ms | 0 |

---

##  Tabla de resultados: `matMul`

Comparativa de rendimiento en la multiplicación de matrices utilizando el enfoque directo (*Naïve*) frente a la optimización por bloques (*Tiling*) con Memoria Compartida.

| Tamaño de Matriz ($N \times N$) | GPU Kernel Naïve (ms) | GPU Kernel Tiling (ms) | Speedup Obtenido | Errores vs CPU |
| :--- | :--- | :--- | :--- | :--- |
| **512 × 512** | 0.70 ms | 0.25 ms | 2.80x | 0 |
| **1024 × 1024** | 4.00 ms | 1.20 ms | 3.33x | 0 |

---

##  Análisis y Conclusiones

### 1. ¿Por qué el GPU Kernel es más rápido que la CPU para un $N$ grande?
La CPU está diseñada bajo una arquitectura de procesamiento secuencial optimizada con grandes memorias caché para ejecutar hilos complejos de forma consecutiva. Cuando $N$ crece masivamente (como en $N = 16\text{M}$), un bucle secuencial satura la capacidad de procesamiento por ciclo de la CPU. 

Por el contrario, la GPU posee una arquitectura masivamente paralela orientada al rendimiento (*throughput*), compuesta en este caso por **2048 núcleos CUDA**. Al fragmentar el problema en miles de bloques y millones de hilos concurrentes que ejecutan la misma instrucción de forma simultánea (modelo **SIMT**), la GPU distribuye la carga matemática de manera uniforme. Esto reduce el tiempo de cómputo puro a fracciones de milisegundo, haciendo evidente su superioridad en tareas de cálculo intensivo e independiente.

### 2. ¿Por qué el tiempo total de la GPU (con `memcpy`) puede ser mayor que el de la CPU para un $N$ pequeño?
Para tamaños de datos reducidos (como $N = 1\text{M}$ o inferiores), el tiempo secuencial de la CPU suele ser extremadamente bajo debido a que los datos entran por completo en sus cachés internas de alta velocidad (L1/L2/L3). 

En la GPU, antes de poder realizar el cálculo en los núcleos, el sistema operativo debe transferir los vectores desde la memoria RAM del sistema (Host) hacia la VRAM de la tarjeta dedicada (Device) a través del bus **PCIe**, y repetir el proceso a la inversa para recuperar los resultados (`cudaMemcpy`). El ancho de banda del bus PCIe introduce un retardo constante (*overhead* de latencia de transferencia). Cuando el problema es pequeño, **el costo de mover los datos de un componente a otro es mayor que el tiempo que le toma a la CPU resolver el problema por sí misma**, superando el beneficio de la paralelización.

### 3. Eficiencia de Tiling frente a Naïve en la GPU
En el benchmark `matMul`, el algoritmo *Naïve* sufre un severo cuello de botella debido a que cada hilo debe leer continuamente datos directamente de la memoria global de la GPU (VRAM), la cual posee una latencia alta. 

Al implementar *Tiling*, el problema se segmenta en subbloques y los hilos colaboran para cargar una porción de las matrices en la **Memoria Compartida (*Shared Memory*)** del chip, sincronizándose mediante `__syncthreads()`. Dado que la Memoria Compartida reside físicamente dentro del multiprocesador de flujo (SM) y opera a una velocidad cercana a los registros, el acceso repetitivo a los datos se vuelve órdenes de magnitud más rápido. Esto explica el **Speedup de hasta 3.33x** observado al escalar la matriz a $1024 \times 1024$, demostrando que la optimización de memoria es tan crucial en HPC como la paralelización aritmética.
