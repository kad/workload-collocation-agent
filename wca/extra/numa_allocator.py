import logging
from pprint import pprint
from typing import List

from dataclasses import dataclass

from wca.allocators import Allocator, TasksAllocations, AllocationType
from wca.detectors import TasksMeasurements, TasksResources, TasksLabels, Anomaly
from wca.metrics import Metric, MetricName
from wca.platforms import Platform, encode_listformat, decode_listformat

log = logging.getLogger(__name__)


@dataclass
class NUMAAllocator(Allocator):

    def allocate(
            self,
            platform: Platform,
            tasks_measurements: TasksMeasurements,
            tasks_resources: TasksResources,
            tasks_labels: TasksLabels,
            tasks_allocations: TasksAllocations,
    ) -> (TasksAllocations, List[Anomaly], List[Metric]):
        log.info('NUMA allocator random policy here....')
        log.debug('NUMA allocator input data:')

        print('Measurements:')
        pprint(tasks_measurements)
        print('Resources:')
        pprint(tasks_resources)
        print('Labels:')
        pprint(tasks_labels)
        print('Allocations (current):')
        pprint(tasks_allocations)
        print("Platform")
        pprint(platform)

        # # Example stupid policy
        # cpu1 = random.randint(0, platform.cpus-1)
        # cpu2 = random.randint(cpu1, platform.cpus-1)
        # log.debug('random cpus: %s-%s', cpu1, cpu2)
        # memory_migrate = random.randint(0, 1)
        # log.debug('random memory_migrate: %s-%s', cpu1, cpu2)
        # allocations = {
        #     'task1': {
        #         AllocationType.CPUSET_CPUS: '%s-%s' % (cpu1, cpu2),
        #         AllocationType.CPUSET_MEMS: '%s-%s' % (cpu1, cpu2),
        #         AllocationType.CPUSET_MEM_MIGRATE: memory_migrate,
        #         # Other options:
        #         # 'cpu_quota': 0.5,
        #         # 'cpu_shares': 20,
        #         # only when rdt is enabled!
        #         # 'rdt': RDTAllocation(
        #         #     name = 'be',
        #         #     l3 = '0:10,1:110',
        #         #     mb = '0:100,1:20',
        #         # )
        #     }
        # }
        # # You can put any metrics here for debugging purposes.
        # extra_metrics = [Metric('some_debug', value=1)]

        print("Policy:")

        allocations = {}

        # Total host memory
        total_memory = _platform_total_memory(platform)
        print("Total memory: %d\n" % total_memory)

        # Collect tasks sizes and NUMA node usages
        tasks_memory = []
        for task in tasks_labels:
            tasks_memory.append(
                (task,
                 _get_task_memory_limit(tasks_measurements[task], total_memory),
                 _get_numa_node_preferences(tasks_measurements[task], platform)))
        tasks_memory = sorted(tasks_memory, reverse=True, key=lambda x: x[1])
        # pprint(tasks_memory)

        # Current state of the system
        balanced_memory = {x: [] for x in platform.node_memory_used}

        balance_task = None
        balance_task_node = None

        # First, get current state of the system
        for task, memory, preferences in tasks_memory:
            current_node = _get_current_node(decode_listformat(tasks_allocations[task][AllocationType.CPUSET_CPUS]), platform.node_cpus)
            log.debug("Task: %s Memory: %d Preferences: %s, Current node: %d" % (task, memory, preferences, current_node))
            if current_node >= 0:
                log.debug("task already placed, recording state")
                balanced_memory[current_node].append((task, memory))


        log.debug("Current state of the system: %s" % balanced_memory)

        log.debug("Starting re-balancing")
        for task, memory, preferences in tasks_memory:
            log.debug("Task: %s Memory: %d Preferences: %s" % (task, memory, preferences))
            current_node = _get_current_node(decode_listformat(tasks_allocations[task][AllocationType.CPUSET_CPUS]), platform.node_cpus)
            #log.debug("Task current node: %d", current_node)
            if current_node >= 0:
                log.debug("task already placed on the node %d, taking next" % current_node)
                #balanced_memory[current_node].append((task, memory))
                continue

            if memory == 0:
                # Handle missing data for "ghost" tasks e.g. cgroup without processes when using StaticNode
                log.warning('skip allocation for %r task - not enough data - maybe there are no processes there!', task)
                continue

            most_used_node = _get_most_used_node(preferences)
            # print("Most used node: %d" % most_used_node)
            # memory based score:
            best_memory_node = _get_best_memory_node(memory, balanced_memory)
            # print("Best memory node: %d" % best_memory_node)
            most_free_memory_node = _get_most_free_memory_node(memory, platform.node_memory_free)
            # print("Best free memory node: %d" % most_free_memory_node)

            log.debug("Task %s: Most used node: %d, Best free node: %d, Best memory node: %d" %
                      (task, most_used_node, most_free_memory_node, best_memory_node))

            # Give a chance for AutoNUMA to re-balance memory
            if preferences[most_used_node] < 0.66:
                log.debug("not most of the memory balanced, continue")
                continue

            if most_used_node == best_memory_node or most_used_node == most_free_memory_node:
            #if most_used_node == most_free_memory_node or most_used_node == best_memory_node:
                balance_task = task
                balance_task_node = most_used_node
                break

            # if most_used_node != most_free_memory_node:
            #     continue

            # pref_nodes = {}
            # for node in preferences:
            #     print(preferences[node])
            #     print(( memory / (sum([k[1] for k in balanced_memory[node]])+memory))/2)
            #     # pref_nodes[node] = max(preferences[node],
            #     #     ( memory / (sum([k[1] for k in balanced_memory[node]])+memory))/2)
            #     pref_nodes[node] = preferences[node]
            #     # pref_nodes[node] = ( memory / (sum([k[1] for k in balanced_memory[node]])+memory))
            # pprint(pref_nodes)
            # best_node = sorted(pref_nodes.items(), reverse=True, key=lambda x: x[1])[0][0]
            # # pprint(best_node)
            # balanced_memory[best_node].append((task, memory))

        pprint(balanced_memory)
        print(balance_task, balance_task_node)

        if balance_task is not None and balance_task_node is not None:
            allocations[balance_task] = {
                AllocationType.CPUSET_MEM_MIGRATE: 1,
                AllocationType.CPUSET_CPUS: encode_listformat(platform.node_cpus[balance_task_node]),
                AllocationType.CPUSET_MEMS: encode_listformat({balance_task_node}),
            }

        # for node in balanced_memory:
        #     for task, _ in balanced_memory[node]:
        #         if decode_listformat(tasks_allocations[task]['cpu_set']) == platform.node_cpus[node]:
        #             continue
        #         allocations[task] = {
        #             AllocationType.CPUSET_MEM_MIGRATE: 1,
        #             AllocationType.CPUSET: platform.node_cpus[node],
        #         }

        pprint(allocations)

        # for task in tasks_labels:
        #     allocations[task] = {
        #         AllocationType.CPUSET_MEM_MIGRATE: memory_migrate,
        #     }

        return allocations, [], []
        # return allocations, [], extra_metrics


def _platform_total_memory(platform):
    return sum(platform.node_memory_free.values()) + sum(platform.node_memory_used.values())


def _get_task_memory_limit(task, total):
    "Returns detected maximum memory for the task"
    limits_order = [
        MetricName.MEM_LIMIT_PER_TASK,
        MetricName.MEM_SOFT_LIMIT_PER_TASK,
        MetricName.MEM_MAX_USAGE_PER_TASK,
        MetricName.MEM_USAGE_PER_TASK, ]
    for limit in limits_order:
        if limit not in task:
            continue
        if task[limit] > total:
            continue
        return task[limit]
    return 0


def _get_numa_node_preferences(task, platform):
    prefix = "memory_numa_stat"
    used = {}
    for node in platform.node_memory_free:
        metric = "%s_%d" % (prefix, node)
        if metric in task:
            used[node] = task[metric]
        else:
            used[node] = 0
    ret = {}
    for node in used:
        ret[node] = used[node] / max(1, sum(used.values()))
    return ret


def _get_most_used_node(preferences):
    return sorted(preferences.items(), reverse=True, key=lambda x: x[1])[0][0]


def _get_current_node(cpus, nodes):
    for node in nodes:
        if nodes[node] == cpus:
            return node
    return -1


def _get_best_memory_node(memory, balanced_memory):
    d = {}
    for node in balanced_memory:
        d[node] = memory / (sum([k[1] for k in balanced_memory[node]]) + memory)
    # pprint(d)
    return sorted(d.items(), reverse=True, key=lambda x: x[1])[0][0]


def _get_most_free_memory_node(memory, node_memory_free):
    d = {}
    for node in node_memory_free:
        d[node] = memory / node_memory_free[node]
    # pprint(d)
    return sorted(d.items(), key=lambda x: x[1])[0][0]
