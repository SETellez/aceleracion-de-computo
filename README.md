# Aceleracion de computo

Laboratorio inicial de CUDA C++: ejecucion de un kernel **Hello World** y suma de vectores en la GPU. Este repositorio sera la base para los siguientes ejercicios de aceleracion de computo.

## Estado de la entrega

| Parte | Estado | Evidencia |
| --- | --- | --- |
| Hello World en CUDA | Completo en codigo fuente | `hello_cuda.cu` |
| Suma de vectores en CUDA | Completo en codigo fuente y con salida guardada | `vector_add.cu`, `suma-vectorescuda.ipynb` |
| Ejecucion en Kaggle | Parcialmente documentada | Notebooks descargados desde Kaggle |
| Guia del laboratorio | Incluida | [`CUDA Hello World — Kaggle Lab Guide.pdf`](CUDA%20Hello%20World%20%E2%80%94%20Kaggle%20Lab%20Guide.pdf) |

Los notebooks se conservan como trazabilidad de la practica realizada en linea. La version mantenible y portable de los ejercicios esta en los archivos `.cu`.

> **Pendiente para cerrar la evidencia al 100 %:** ejecutar tambien `hello_cuda.cu` en una sesion GPU de Kaggle y guardar la salida en `helloworldcuda.ipynb`. El notebook actual de Hello World conserva la plantilla de Kaggle, pero no contiene aun la ejecucion del kernel.

## Estructura

```text
.
├── hello_cuda.cu
├── vector_add.cu
├── helloworldcuda.ipynb
├── suma-vectorescuda.ipynb
├── CUDA Hello World — Kaggle Lab Guide.pdf
└── README.md
```

## 1. Hello World

El programa consulta la GPU activa y lanza un kernel con un bloque y un hilo:

```cpp
helloKernel<<<1, 1>>>();
```

Compilacion y ejecucion:

```bash
nvcc -O2 hello_cuda.cu -o hello_cuda
./hello_cuda
```

Salida esperada, con el nombre de la GPU variando segun el entorno:

```text
GPU activa: NVIDIA ...
Hello World desde CUDA | bloque 0 | hilo 0
Ejecucion finalizada correctamente.
```

## 2. Suma de vectores en la GPU

El kernel calcula un elemento por hilo. La posicion global se obtiene con:

```text
index = blockIdx.x * blockDim.x + threadIdx.x
```

Cada hilo ejecuta `C[index] = A[index] + B[index]` si su indice esta dentro del arreglo. Para `1,048,576` elementos se usan bloques de `256` hilos:

```text
blocks = (N + threadsPerBlock - 1) / threadsPerBlock
```

Compilacion y ejecucion:

```bash
nvcc -O3 vector_add.cu -o vector_add
./vector_add
```

El programa muestra el primer y ultimo elemento y valida todos los resultados contra una suma calculada en el host. Tambien verifica los errores de la API CUDA, la configuracion del kernel y las transferencias de memoria.

## Flujo de memoria

```mermaid
flowchart LR
    A[Host: A y B] -->|cudaMemcpy HostToDevice| B[Device: A y B]
    B --> C[Kernel vectorAdd]
    C --> D[Device: C]
    D -->|cudaMemcpy DeviceToHost| E[Host: C verificado]
```

El ejercicio usa `cudaMalloc` para reservar memoria en la GPU, `cudaMemcpy` para transferir los datos y `cudaFree` para liberar los recursos del device.

## Ejecucion en Kaggle

En una notebook con acelerador GPU, la secuencia recomendada es:

```python
!nvidia-smi
!nvcc --version
!nvcc -O2 hello_cuda.cu -o hello_cuda
!./hello_cuda
!nvcc -O3 vector_add.cu -o vector_add
!./vector_add
```

Los notebooks incluidos funcionan como bitacora de la plataforma. Para reproducirlos, activa el acelerador **NVIDIA GPU** en Kaggle y ejecuta las celdas en orden.

## Requisitos

- GPU NVIDIA con soporte CUDA para la ejecucion.
- CUDA Toolkit y `nvcc`.
- Compilador C++ compatible con la version instalada del toolkit.
- En Kaggle: una notebook con acelerador GPU habilitado.

## Proximos ejercicios

- Medicion de tiempo CPU vs. GPU con eventos CUDA.
- Comparacion de distintos tamanos de bloque.
- Manejo de errores y sincronizacion avanzada.
- Suma de matrices y operaciones con memoria compartida.