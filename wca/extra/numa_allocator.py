import logging
from typing import List, Dict
from pprint import pprint
import math

from dataclasses import dataclass

from wca.allocators import Allocator, TasksAllocations, AllocationType
from wca.detectors import TasksMeasurements, TasksResources, TasksLabels, Anomaly
from wca.metrics import Metric, MetricName
from wca.platforms import Platform, encode_listformat, decode_listformat

log = logging.getLogger(__name__)


@dataclass
class NUMAAllocator(Allocator):

    # intrusive set of options
    # parse15
    # preferences_threshold: float = 0.66
    preferences_threshold: float = 0.0  # always migrate
    #memory_migrate: bool = False
    memory_migrate: bool = True

    # use candidate
    candidate = False

    def allocate(
            self,
            platform: Platform,
            tasks_measurements: TasksMeasurements,
            tasks_resources: TasksResources,
            tasks_labels: TasksLabels,
            tasks_allocations: TasksAllocations,
    ) -> (TasksAllocations, List[Anomaly], List[Metric]):
        log.info('NUMA allocator policy reconcile loop')
        log.debug('NUMA allocator input data:')
        #
        # print('Measurements:')
        # pprint(tasks_measurements)
        # print('Resources:')
        # pprint(tasks_resources)
        # print('Labels:')
        # pprint(tasks_labels)
        # print('Allocations (current):')
        # pprint(tasks_allocations)
        # print("Platform")
        # pprint(platform)

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

        # print("Policy:")

        allocations = {}

        # Total host memory
        total_memory = _platform_total_memory(platform)
        # log.debug("Total memory: %d\n" % total_memory)
        extra_metrics = []

        # Collect tasks sizes and NUMA node usages
        tasks_memory = []
        for task in tasks_labels:
            tasks_memory.append(
                (task,
                 _get_task_memory_limit(tasks_measurements[task], total_memory),
                 _get_numa_node_preferences(tasks_measurements[task], platform)))
        tasks_memory = sorted(tasks_memory, reverse=True, key=lambda x: x[1])
        # FOR DEBUGGING PURPOSES just sort by name (it should work the same assuming
        # memory is equal for all tasks)
        # tasks_memory = sorted(tasks_memory, reverse=False, key=lambda x: x[0])  # by name
        # pprint(tasks_memory)

        # Current state of the system
        balanced_memory = {x: [] for x in platform.measurements[MetricName.MEM_NUMA_USED]}

        balance_task = None
        balance_task_node = None
        balance_task_candidate = None
        balance_task_node_candidate = None

        # First, get current state of the system
        for task, memory, preferences in tasks_memory:
            current_node = _get_current_node(
                decode_listformat(tasks_allocations[task][AllocationType.CPUSET_CPUS]),
                platform.node_cpus)
            log.debug("Task: %s Memory: %d Preferences: %s, Current node: %d" % (
                task, memory, preferences, current_node))
            if current_node >= 0:
                # log.debug("task already placed, recording state")
                balanced_memory[current_node].append((task, memory))

        log.debug("Current state of the system: %s" % balanced_memory)

        for node, tasks_with_memory in balanced_memory.items():
            extra_metrics.extend([
                Metric('numa__balanced_memory_tasks', value=len(tasks_with_memory),
                       labels=dict(numa_node=str(node))),
                Metric('numa__balanced_memory_size', value=sum([m for t, m in tasks_with_memory]),
                       labels=dict(numa_node=str(node)))
            ])

        log.debug("Starting re-balancing")

        for task, memory, preferences in tasks_memory:
            log.debug("Task: %s Memory: %d Preferences: %s" % (task, memory, preferences))
            current_node = _get_current_node(
                decode_listformat(tasks_allocations[task][AllocationType.CPUSET_CPUS]),
                platform.node_cpus)
            most_used_node = _get_most_used_node(preferences)
            most_used_nodes = _get_most_used_node_v2(preferences)
            best_memory_node = _get_best_memory_node(memory, balanced_memory)
            best_memory_nodes = _get_best_memory_node_v3(memory, balanced_memory)
            most_free_memory_node = \
                _get_most_free_memory_node(memory,
                                           platform.measurements[MetricName.MEM_NUMA_FREE])
            most_free_memory_nodes = \
                _get_most_free_memory_node_v3(memory,
                                           platform.measurements[MetricName.MEM_NUMA_FREE])
            extra_metrics.extend([
                Metric('numa__task_current_node', value=current_node,
                       labels=tasks_labels[task]),
                Metric('numa__task_most_used_node', value=most_used_node,
                       labels=tasks_labels[task]),
                Metric('numa__task_best_memory_node', value=best_memory_node,
                       labels=tasks_labels[task]),
                Metric('numa__task_best_memory_node_preference', value=preferences[most_used_node],
                       labels=tasks_labels[task]),
                Metric('numa__task_most_free_memory_mode', value=most_free_memory_node,
                       labels=tasks_labels[task])
            ])

            if current_node >= 0:
                log.debug("   task already placed on the node %d, taking next" % current_node)
                continue

            if memory == 0:
                # Handle missing data for "ghost" tasks
                # e.g. cgroup without processes when using StaticNode
                log.warning(
                    'skip allocation for %r task - not enough data - '
                    'maybe there are no processes there!',
                    task)
                continue

            # log.debug("Task %s: Most used node: %d, Best free node: %d, Best memory node: %d" %
            #           (task, most_used_node, most_free_memory_node, best_memory_node))
            log.debug("Task %s: Most used node: %d (%s), Best free node: %s, Best memory node: %s" %
                      (task, most_used_node, most_used_nodes, most_free_memory_nodes, best_memory_nodes))

            # if not yet task found for balancing
            if balance_task is None and balance_task_node is None:

                # Give a chance for AutoNUMA to re-balance memory
                if preferences[most_used_node] < self.preferences_threshold:
                    log.debug("   THRESHOLD: not most of the memory balanced, continue")
                    continue

                # if most_used_node == best_memory_node or most_used_node == most_free_memory_node:
                if len(most_used_nodes.intersection(best_memory_nodes)) == 1:
                    log.debug("   OK: found task for best memory node")
                    balance_task = task
                    balance_task_node = list(most_used_nodes.intersection(best_memory_nodes))[0]

                elif len(most_used_nodes.intersection(most_free_memory_nodes)) == 1:
                    log.debug("   OK: found task for most free memory node")
                    balance_task = task
                    balance_task_node = list(most_used_nodes.intersection(most_free_memory_nodes))[0]

                elif most_used_node in best_memory_nodes or most_used_node in most_free_memory_nodes:
                    log.debug("   OK: minimized migrations case")
                    balance_task = task
                    balance_task_node = most_used_node
                    # break # commented to give a chance to generate other metrics

                elif len(best_memory_nodes.intersection(most_free_memory_nodes)) == 1:
                    log.debug("   OK: task not local, but both best available has only one alternative")
                    balance_task = task
                    balance_task_node = list(best_memory_nodes.intersection(most_free_memory_nodes))[0]

                #elif self.candidate and balance_task_candidate is None \
                #        and balance_task_node_candidate is None:
                #    log.debug("   CANDIT: not perfect match, but remember as candidate, continue")
                #    balance_task_candidate = task
                #    balance_task_node_candidate = best_memory_node
                #    # balance_task_node_candidate = most_free_memory_node

                else:
                    log.debug("   IGNORE: no good decisions can be made now for this task, continue")
                # break # commented to give a chance to generate other metrics

                # Validate if we have enough memory to migrate to desired node:
                if balance_task is not None and balance_task_node is not None:
                    if memory >= platform.measurements[MetricName.MEM_NUMA_FREE].get(balance_task_node, 0):
                        log.debug(" We can't migrate task '%s' to node '%d', because not enough memory on the target. Looking for another candidate" % 
                                    (balence_task, balance_task_node))
                        balance_task = None
                        balance_task_node = None


        if balance_task is None and balance_task_node is None:
            if balance_task_candidate is not None and balance_task_node_candidate is not None:
                log.warn("Cannot find by most_used, use candidate from 'best node' rule!")
                balance_task = balance_task_candidate
                balance_task_node = balance_task_node_candidate

        if balance_task is not None and balance_task_node is not None:
            log.debug("   Assign task %s to node %s." % (balance_task, balance_task_node))
            allocations[balance_task] = {
                AllocationType.CPUSET_CPUS: encode_listformat(
                    platform.node_cpus[balance_task_node]),
                AllocationType.CPUSET_MEMS: encode_listformat({balance_task_node}),
            }
            # Instant memory migrate.
            if self.memory_migrate:
                log.debug("Assign task %s to node %s with memory migrate" %
                          (balance_task, balance_task_node))
                allocations[balance_task][AllocationType.CPUSET_MEM_MIGRATE] = 1

        return allocations, [], extra_metrics


def _platform_total_memory(platform):
    return sum(platform.measurements[MetricName.MEM_NUMA_FREE].values()) + \
           sum(platform.measurements[MetricName.MEM_NUMA_USED].values())


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
        log.debug('task limit %s %s %s %s %s %s', task, 'by', limit, 'is', task[limit], 'bytes')
        return task[limit]
    return 0


def _get_numa_node_preferences(task_measurements, platform: Platform) -> Dict[int, float]:
    ret = {node_id: 0 for node_id in range(0, platform.numa_nodes)}
    if MetricName.MEM_NUMA_STAT_PER_TASK in task_measurements:
        metrics_val_sum = sum(task_measurements[MetricName.MEM_NUMA_STAT_PER_TASK].values())
        for node_id, metric_val in task_measurements[MetricName.MEM_NUMA_STAT_PER_TASK].items():
            ret[int(node_id)] = metric_val / max(1, metrics_val_sum)
    else:
        log.warning('{} metric not available'.format(MetricName.MEM_NUMA_STAT_PER_TASK))
    return ret


def _get_most_used_node(preferences):
    return sorted(preferences.items(), reverse=True, key=lambda x: x[1])[0][0]


def _get_current_node(cpus, nodes):
    for node in nodes:
        if nodes[node] == cpus:
            return node
    return -1


def _get_best_memory_node(memory, balanced_memory):
    """for equal task memory, choose node with less allocated memory by WCA"""
    d = {}
    for node in balanced_memory:
        d[node] = memory / (sum([k[1] for k in balanced_memory[node]]) + memory)
    best = sorted(d.items(), reverse=True, key=lambda x: x[1])
    # print('best:')
    # pprint(best)
    return best[0][0]


def _get_most_free_memory_node(memory, node_memory_free):
    d = {}
    for node in node_memory_free:
        d[node] = memory / node_memory_free[node]
    # pprint(d)
    free_nodes = sorted(d.items(), key=lambda x: x[1])
    # print('free:')
    # pprint(free_nodes)
    return free_nodes[0][0]

def _get_best_memory_node_v2(memory, balanced_memory):
    """for equal task memory, choose node with less allocated memory by WCA"""
    d = {}
    for node in balanced_memory:
        d[node] = round(math.log1p((memory / (sum([k[1] for k in balanced_memory[node]]) + memory))*100),0)
    best = sorted(d.items(), reverse=True, key=lambda x: x[1])
    z = best[0][1]
    best_nodes = [x[0] for x in best if x[1] == z]
    return best_nodes

def _get_most_free_memory_node_v2(memory, node_memory_free):
    d = {}
    for node in node_memory_free:
        d[node] = round(math.log1p((memory / node_memory_free[node]) * 100),0)
    free_nodes = sorted(d.items(), key=lambda x: x[1])
    z = free_nodes[0][1]
    best_free_nodes = [x[0] for x in free_nodes if x[1] == z]
    return best_free_nodes

def _get_best_memory_node_v3(memory, balanced_memory):
    """for equal task memory, choose node with less allocated memory by WCA"""
    d = {}
    for node in balanced_memory:
        d[node] = round(math.log10((sum([k[1] for k in balanced_memory[node]]) + memory)),1)
    best = sorted(d.items(), key=lambda x: x[1])
    z = best[0][1]
    best_nodes = {x[0] for x in best if x[1] == z}
    return best_nodes

def _get_most_free_memory_node_v3(memory, node_memory_free):
    d = {}
    for node in node_memory_free:
        if memory >= node_memory_free[node]:
            # if we can't fit into free memory, don't consider that node at all
            continue
        d[node] = round(math.log10(node_memory_free[node] - memory),1)
    free_nodes = sorted(d.items(), reverse=True, key=lambda x: x[1])
    best_free_nodes = set()
    if len(free_nodes) > 0:
        z = free_nodes[0][1]
        best_free_nodes = {x[0] for x in free_nodes if x[1] == z}
    return best_free_nodes

def _get_most_used_node_v2(preferences):
    d = {}
    for node in preferences:
        d[node] = round(math.log1p(preferences[node]*1000))
    nodes = sorted(d.items(), reverse=True, key=lambda x: x[1])
    z = nodes[0][1]
    best_nodes = {x[0] for x in nodes if x[1] == z}
    return best_nodes
