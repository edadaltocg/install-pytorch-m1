from itertools import product
import torch.utils.benchmark as benchmark
import timeit
import torch

# num_threads = torch.get_num_threads()


def batched_dot_mul_sum(a, b):
    '''Computes batched dot by multiplying and summing'''
    return a.mul(b).sum(-1)


def batched_dot_bmm(a, b):
    '''Computes batched dot by reducing to bmm'''
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)


# Input for benchmarking
# x = torch.randn(10000, 64)

# Ensure that both functions compute the same output
# assert batched_dot_mul_sum(x, x).allclose(batched_dot_bmm(x, x))

# print(f'Benchmarking on {num_threads} threads')

# t0 = benchmark.Timer(
#     stmt='batched_dot_mul_sum(x, x)',
#     setup='from __main__ import batched_dot_mul_sum',
#     globals={'x': x},
#     num_threads=num_threads,
#     label='Multithreaded batch dot',
#     sub_label='Implemented using mul and sum')

# t1 = benchmark.Timer(
#     stmt='batched_dot_bmm(x, x)',
#     setup='from __main__ import batched_dot_bmm',
#     globals={'x': x},
#     num_threads=num_threads,
#     label='Multithreaded batch dot',
#     sub_label='Implemented using bmm')

# print(t0.timeit(100))
# print(t1.timeit(100))


# Compare takes a list of measurements which we'll save in results.
results = []

sizes = [1, 64, 1024, 10000]
for b, n in product(sizes, sizes):
    # label and sub_label are the rows
    # description is the column
    label = 'Batched dot'
    sub_label = f'[{b}, {n}]'
    x = torch.ones((b, n))
    for num_threads in [32]:
        results.append(benchmark.Timer(
            stmt='batched_dot_mul_sum(x, x)',
            setup='from __main__ import batched_dot_mul_sum',
            globals={'x': x},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='mul/sum',
        ).blocked_autorange(min_run_time=1))
        results.append(benchmark.Timer(
            stmt='batched_dot_bmm(x, x)',
            setup='from __main__ import batched_dot_bmm',
            globals={'x': x},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='bmm',
        ).blocked_autorange(min_run_time=1))

compare = benchmark.Compare(results)

compare.trim_significant_figures()
compare.colorize()
compare.print()
