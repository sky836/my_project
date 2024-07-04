import torch
import torch.distributed as dist
import os

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'

# torch.distributed.init_process_group(backend, init_method=None, world_size=-1, rank=-1, store=None)
dist.init_process_group('nccl', init_method='env://')
rank = dist.get_rank()   # 每个进程在执行同一份代码得到对应的rank
local_rank = os.environ['LOCAL_RANK']  # 每个进程在执行同一份代码得到对应的local_rank
master_addr = os.environ['MASTER_ADDR']
master_port = os.environ['MASTER_PORT']
print(f"rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank}")
torch.cuda.set_device(rank)
tensor = torch.tensor([1, 2, 3, 4]).cuda()
print(tensor)

