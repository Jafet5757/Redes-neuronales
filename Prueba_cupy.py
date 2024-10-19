import cupy as cp
import numpy as np
import time

# Tamaño de la matriz grande
n = 15000

# Generamos dos matrices grandes aleatorias con NumPy
a_cpu = np.random.rand(n, n).astype(np.float32)
b_cpu = np.random.rand(n, n).astype(np.float32)

# =============================
# Multiplicación en la CPU
# =============================
start_cpu = time.time()

# Multiplicación de matrices en la CPU
c_cpu = np.dot(a_cpu, b_cpu)

end_cpu = time.time()
print(f"Tiempo de multiplicación en la CPU: {end_cpu - start_cpu:.4f} segundos")


# =============================
# Multiplicación en la GPU
# =============================

# Transferimos las matrices a la GPU utilizando CuPy
a_gpu = cp.asarray(a_cpu)
b_gpu = cp.asarray(b_cpu)

# Medimos el tiempo de la multiplicación en la GPU
start_gpu = time.time()

# Multiplicación de matrices en la GPU
c_gpu = cp.dot(a_gpu, b_gpu)

# Sincronizamos para asegurarnos de que todas las operaciones en la GPU han terminado
cp.cuda.Device(0).synchronize()

end_gpu = time.time()
print(f"Tiempo de multiplicación en la GPU: {end_gpu - start_gpu:.4f} segundos")

# c_cpu_from_gpu = cp.asnumpy(c_gpu)