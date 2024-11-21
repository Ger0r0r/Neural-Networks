import numpy as np
from numba import njit
import time
import psutil
import os

##################################################
##################################################
##################################################

def get_cpu_time(): # Функция для получения времени использования CPU текущим процессом
	process = psutil.Process(os.getpid()) # Получаем информацию о процессе
	return process.cpu_times().user + process.cpu_times().system # Возвращаем сумму пользовательского и системного времени CPU

def manual_matrix_multiply(A, B):
	n = A.shape[0]
	C = np.zeros((n, n))

	for i in range(n):
		for j in range(n):
			for k in range(n):
				C[i, j] += A[i, k] * B[k, j]

	return C

@njit
def numba_matrix_multiply(A, B, C, n):

	for i in range(n):
		for j in range(n):
			for k in range(n):
				C[i, j] += A[i, k] * B[k, j]

	return C

def numpy_dot_multiply(A, B):
	return np.dot(A, B)

##################################################
##################################################
##################################################

file = open('data/jit.txt', 'a')

current_time = 0
m_size = 32
jit_time = np.array([])
jit_size = np.array([])
while (current_time < 10):
	matrix_A = np.random.randint(0, 10, size=(m_size, m_size))
	matrix_B = np.random.randint(0, 10, size=(m_size, m_size))
	matrix_C = np.zeros((m_size,m_size))

	start_real_time = get_cpu_time()
	numba_matrix_multiply(matrix_A, matrix_B, matrix_C, m_size)
	end_real_time = get_cpu_time()
	current_time = end_real_time - start_real_time
	print(current_time)
	jit_time = np.append(jit_time, current_time)
	jit_size = np.append(jit_size, m_size)
	file.write(str(m_size)+","+str(current_time)+"\n")
	m_size = m_size + 32

file.close()

##################################################
##################################################
##################################################

file = open('data/manualy.txt', 'a')

current_time = 0
m_size = 8
manualy_time = np.array([])
manualy_size = np.array([])
while (current_time < 10):
	matrix_A = np.random.randint(0, 10, size=(m_size, m_size))
	matrix_B = np.random.randint(0, 10, size=(m_size, m_size))
	matrix_C = np.zeros((m_size,m_size))

	start_real_time = get_cpu_time()
	matrix_C = manual_matrix_multiply(matrix_A, matrix_B)
	end_real_time = get_cpu_time()
	current_time = end_real_time - start_real_time
	print(current_time)
	manualy_time = np.append(manualy_time, current_time)
	manualy_size = np.append(manualy_size, m_size)
	file.write(str(m_size)+","+str(current_time)+"\n")
	m_size = m_size + 8

file.close()

##################################################
##################################################
##################################################

file = open('data/dot.txt', 'a')

current_time = 0
m_size = 64
dot_time = np.array([])
dot_size = np.array([])
while (current_time < 10):
	matrix_A = np.random.randint(0, 10, size=(m_size, m_size))
	matrix_B = np.random.randint(0, 10, size=(m_size, m_size))
	matrix_C = np.zeros((m_size,m_size))

	start_real_time = get_cpu_time()
	matrix_C = numpy_dot_multiply(matrix_A, matrix_B)
	end_real_time = get_cpu_time()
	current_time = end_real_time - start_real_time
	print(current_time)
	dot_time = np.append(dot_time, current_time)
	dot_size = np.append(dot_size, m_size)
	file.write(str(m_size)+","+str(current_time)+"\n")
	m_size = m_size + 64

file.close()