import math
from concurrent.futures import ThreadPoolExecutor
import numba as nb
import numpy as np
import time

@nb.jit(nopython=True, nogil=False)
def kernel1(result, x, y):
    sum_num = 0
    for i in range(x,y):
        sum_num += i            
        #print(sum_num)
    result = sum_num

    return result

@nb.jit(nopython=True, nogil=True)
def kernel2(result, x, y):
    sum_num = 0
    for i in range(x,y):
        sum_num += i
        #print(sum_num)
    result = sum_num

    return result

def make_single_task(kernel):
    def func(length, *args):
        #result = np.empty(length, dtype=np.float32)
        result = kernel(length, *args)
        #print(result)
        return result
    return func

def make_multi_task(kernel, n_thread):
    def func(length, *args):

        all_len = args[1]
        single = round(all_len/4)
        count_num = 0
        with ThreadPoolExecutor(max_workers=n_thread) as executor:
            executor.submit(kernel, length, 1, single)
            executor.submit(kernel, length, single, single*2)
            executor.submit(kernel, length, single*2, single*3)
            executor.submit(kernel, length, single*3, all_len)
           
        executor.shutdown(wait=True)

    return func

nb_func1 = make_single_task(kernel1)
nb_func2 = make_multi_task(kernel1, 4)
nb_func3 = make_single_task(kernel2)
nb_func4 = make_multi_task(kernel2, 4)

result = np.array(0)

no_gil = 0 
multi_no_gil = 0
yes_gil = 0
multi_yes_gil = 0
step = 10 

for i in range(1, step):

    start_time = time.time()
    nb_func1(result, 1, 100000000)
    #print('no gil: {} sec'.format(time.time()-start_time))
    no_gil += (time.time() - start_time)


    start_time = time.time()
    nb_func2(result, 1, 100000000)
    #print('muti no gil: {} sec'.format(time.time()-start_time))
    multi_no_gil += (time.time() - start_time)

    time.sleep(0.5)


    start_time = time.time()
    nb_func3(result, 1, 100000000)
    #print('Have gil: {} sec'.format(time.time()-start_time))
    yes_gil += (time.time() - start_time)


    start_time = time.time()
    nb_func4(result, 1, 100000000)
    #print('mutli gil: {} sec'.format(time.time()-start_time))
    multi_yes_gil += (time.time() - start_time)

print("no_gil : \n",no_gil/step)
print("multi_no_gil : \n",multi_no_gil/step)
print("yes_gil : \n",yes_gil/step)
print("multi_yes_gil :\n",multi_yes_gil/step)


