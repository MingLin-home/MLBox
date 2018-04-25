import multiprocessing
import os
import psutil
import time


def restric_computation_resource(compute_device_str_list):
    gpu_id_list = []
    cpu_id_list = []
    mapped_compute_device_str_list = []
    
    for device_str in compute_device_str_list:
        device_type = device_str[0:5] # '/cpu:0' -> '/cpu:
        device_id = device_str[5:]
        if device_type == '/cpu:':
            cpu_id_list.append(device_id)
        else:
            gpu_id_list.append(device_id)
        pass
    pass # end if

    if len(gpu_id_list) > 0 and len(cpu_id_list) > 0:
        raise RuntimeError('Cannot restrict computation on both GPU and CPU!')
    
    if len(gpu_id_list) == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        CUDA_VISIBLE_DEVICES = ','.join(gpu_id_list)
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
        for i in range(len(gpu_id_list)):
            mapped_compute_device_str_list.append('/gpu:%d' % i )
        pass # end for i
    pass # end if
    
    if len(cpu_id_list) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        the_process = psutil.Process()
        the_process.cpu_affinity([int(i) for i in cpu_id_list])
        mapped_compute_device_str_list = compute_device_str_list
    pass # end if
    
    return mapped_compute_device_str_list
pass # end def

def decode_job_queu_and_run(para):
    par_do_func, task_parameters = para
    # print('debug: in decode_job_queu_and_run!')
    the_free_device_str = decode_job_queu_and_run.queue.get()
    # print('debug: get device %s' % the_free_device_str)
    par_do_func(the_free_device_str, task_parameters)
    decode_job_queu_and_run.queue.put(the_free_device_str)
   
def init_decode_job_queu_and_run(the_queue):
    decode_job_queu_and_run.queue = the_queue

def parallel_GPU_CPU_do(num_gpus=None, num_cpus=None, par_do_func=None, task_list=None,
                        gpu_offset=0, cpu_offset=0, per_task_num_gpus=None, per_task_num_cpus=None,
                        compute_device_str_list=None,):
    """
    
    :param par_todo_func: must accept par_do_func(device_str='/gpu:0', task_list[i])
    :param task_list: list of tasks.
    """

    num_jobs = 0
    free_device_queue = multiprocessing.Queue()
    
    if num_gpus is not None and num_gpus > 0 and per_task_num_gpus is not None and per_task_num_gpus > 0:
        for i in range(int(num_gpus/per_task_num_gpus)):
            if per_task_num_gpus == 1:
                free_device_queue.put(['/gpu:%d' % (gpu_offset + i),])
            else:
                free_device_str_list = []
                for j in range(per_task_num_gpus):
                    free_device_str_list.append('/gpu:%d' %(gpu_offset + i*per_task_num_gpus+j))
                pass # end for j
                free_device_queue.put(free_device_str_list)
            pass # end if per_task_num_gpu
        pass  # end for
        num_jobs += int(num_gpus/per_task_num_gpus)
    pass # end if num_gpus is not None:
    
    if num_cpus is not None and num_cpus >0 and per_task_num_cpus is not None and per_task_num_cpus > 0:
        for i in range(int(num_cpus/per_task_num_cpus)):
            if per_task_num_cpus == 1:
                free_device_queue.put(['/cpu:%d' % (cpu_offset + i ),])
            else:
                free_device_str_list = []
                for j in range(per_task_num_cpus):
                    free_device_str_list.append('/cpu:%d' % (cpu_offset + i*per_task_num_cpus+j))
                pass # end for j
                free_device_queue.put(free_device_str_list)
            pass # end if per_task_num_gpu
        pass  # end for
        num_jobs += int(num_cpus/per_task_num_cpus)
    pass # end if num_gpus is not None:
    
    if compute_device_str_list is not None:
        for device_str in compute_device_str_list:
            free_device_queue.put(device_str)
        pass # end for device_str in compute_device_str_list:
        num_jobs += len(compute_device_str_list)
    pass # end if compute_device_str_list is not None:
    
    
    worker_pool = multiprocessing.Pool(num_jobs, init_decode_job_queu_and_run,[free_device_queue,])
    results = worker_pool.map_async(decode_job_queu_and_run,[(par_do_func,task_parameters) for task_parameters in task_list],
                                    chunksize=1)
    for i in range(10):
        if results.ready():
            break
        time.sleep(1)
    pass # end for i

    print('len(task_list)=%d' % len(task_list))
    start_timer = time.time()
    total_num_jobs = results._number_left
    print('total number of job chunks=%d' % total_num_jobs, flush=True)
    while not results.ready():
        num_left_jobs = results._number_left
        elasped_time = time.time() - start_timer
        num_done_jobs = total_num_jobs - num_left_jobs
        if total_num_jobs> num_left_jobs:
            remaining_time = elasped_time / (total_num_jobs - num_left_jobs) * num_left_jobs / 3600
            print('Done job chunks %d/%d, elasped time=%g, remaining time=%g (hours)' % (num_done_jobs, total_num_jobs, elasped_time, remaining_time), flush=True)
        pass
        time.sleep(60)
    real_result = results.get()
    
    worker_pool.close()
    worker_pool.join()
    return real_result
pass # end def


if __name__ == '__main__':
    # multiprocessing.freeze_support()
    pass