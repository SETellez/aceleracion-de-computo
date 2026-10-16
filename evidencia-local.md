# Evidencia de ejecucion local

**Fecha:** 2026-10-16  
**GPU:** NVIDIA GeForce GTX 1650  
**Driver:** 610.62  
**Memoria GPU:** 4096 MiB  
**CUDA Toolkit:** 13.4 (`nvcc V13.4.59`)  
**Compilador host:** MSVC 19.44.35229 x64

## Hello World

Comando:

```powershell
nvcc -O2 hello_cuda.cu -o hello_cuda
.\hello_cuda
```

Salida:

```text
GPU activa: NVIDIA GeForce GTX 1650
Hello World desde CUDA | bloque 0 | hilo 0
Ejecucion finalizada correctamente.
```

## Suma de vectores

Comando:

```powershell
nvcc -O3 vector_add.cu -o vector_add
.\vector_add
```

Salida:

```text
Elementos: 1048576 | bloques: 4096 | hilos por bloque: 256
c[0] = 1.000000 | esperado = 1.000000
c[1048575] = 0.172421 | esperado = 0.172421
Verificacion: OK (0 errores)
```

## Resultado

Los dos programas compilaron y ejecutaron correctamente en la GPU local. La suma de vectores validó los `1,048,576` resultados contra el cálculo esperado en el host.