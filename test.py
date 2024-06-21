import time
import multiprocessing


def soma(x, y):
  time.sleep(1)
  return x + y

def subtracao(x, y):
  time.sleep(1)
  return x - y

inicio = time.time()
multiprocessing.set_start_method('fork')
pool = multiprocessing.Pool(processes=2)
result = pool.apply_async(soma, (1, 2))
result2 = pool.apply_async(subtracao, (1, 2))


print(result.get())
print(result2.get())
fim = time.time()
print("Tempo de execução:", fim - inicio)

