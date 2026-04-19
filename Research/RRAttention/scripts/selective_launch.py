"""
Selective launch script.

Usage: python script/selective_launch.py <port> <ranks> <ranks> <ranks> ...
"""
import os
import sys


def parse_ranks(ranks_strs):
    """
    parse_ranks
    """
    # NOTE: You can return ranks directly here to change script/train_gpu.sh
    # and script/kill_process.sh together

    # Example 1: Use contiguous nodes [8, 16)
    # return range(8, 16)

    # Example 2: Use non-contiguous nodes [4, 8) + {10} + [30, 32), i.e., [4, 5, 6, 7, 10, 30, 31]
    # return list(range(4, 8)) + [10] + list(range(30, 32))

    # Example 3:
    # Just Python code, return any nodes you want!
    return range(0, 1)
    if not ranks_strs:
        return None

    ranks = []
    for r in ranks_strs:
        r = eval(r)
        if isinstance(r, int):
            ranks.append(r)
        else:
            ranks.extend(r)
    return ranks


def main(port, ranks):
    """
    main
    """
    trainer_instances = os.getenv("TRAINER_INSTANCES")
    if not trainer_instances:
        print(f"--master_addr 127.0.0.1 --master_port {port} --node_rank 0 --nnodes 1 --nproc_per_node 1")
        return

    ips = [ip.strip() for ip in trainer_instances.split(",") if ip.strip()]
    if ranks is None:
        ranks = list(range(len(ips)))
    ranks = sorted(list(set(ranks)))
    my_rank = int(os.getenv("POD_INDEX", "0"))
    if my_rank not in ranks:
        return

    rank = ranks.index(my_rank)
    nranks = len(ranks)

    master = ips[ranks[0]]
    print(f"--master_addr {master} --master_port {port} --node_rank {rank} --nnodes {nranks} --nproc_per_node 8")


if __name__ == "__main__":
    main(int(sys.argv[1]), parse_ranks(sys.argv[2:]))
