# Synchronization Methods in Distributed Training

Synchronization ensures that model parameters are updated consistently across all workers. There are two main approaches: synchronous and asynchronous training.

## Synchronous Training

### All-Reduce Communication

**Concept:**
- All workers compute gradients on their data.
- Gradients are averaged (all-reduce) and all workers update their models with the same parameters.

**Ring All-Reduce:**
- Gradients are passed in a ring among workers, reducing communication overhead.

**Math:**
$$
T_{ring} = 2 \times (K-1) \times \frac{\text{data\_size}}{K \times \text{bandwidth}}
$$

**Python Example (PyTorch):**
```python
import torch.distributed as dist
# Assume gradients are already computed
for param in model.parameters():
    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
    param.grad.data /= dist.get_world_size()
```

### Synchronization Overhead

**Communication Time:**
$$
T_{sync} = \frac{\text{model\_parameters} \times 4 \text{ bytes}}{\text{network\_bandwidth}}
$$

**Computation-Communication Overlap:**
$$
T_{effective} = \max(T_{comp}, T_{comm})
$$

## Asynchronous Training

### Parameter Server Architecture

**Concept:**
- Workers send gradients to a central parameter server.
- The server updates parameters as gradients arrive (no global sync).

**Math:**
$$
\theta_{t+1} = \theta_t - \alpha \sum_{k \in S_t} g_k
$$

**Python Example (Pseudo-code):**
```python
# Worker
while True:
    gradients = compute_gradients()
    send_to_server(gradients)
    params = get_params_from_server()
    update_local_model(params)

# Parameter Server
while True:
    gradients = receive_from_workers()
    update_params(gradients)
    send_params_to_workers()
```

### Hogwild! Algorithm

**Concept:**
- Lock-free, asynchronous updates to shared parameters.
- Works well for sparse models.

**Python Example:**
```python
# Shared tensor in multiprocessing
import torch.multiprocessing as mp
shared_tensor = torch.zeros(10).share_memory_()

def worker(shared_tensor):
    for _ in range(100):
        shared_tensor += torch.randn(10)

processes = [mp.Process(target=worker, args=(shared_tensor,)) for _ in range(4)]
for p in processes: p.start()
for p in processes: p.join()
```

---

Synchronous methods ensure consistency but can be slower due to waiting for all workers. Asynchronous methods are faster but may introduce staleness and inconsistency. The choice depends on the application and infrastructure. 