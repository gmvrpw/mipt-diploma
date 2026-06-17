from collections import deque

import structlog
from structlog.stdlib import BoundLogger

from .pipeline import Pipeline
from .tasks.input import PipelineInput, LinkInput

log: BoundLogger = structlog.get_logger(__name__)


def topsort(pipeline: Pipeline):
    log.info("Ordering tasks...")

    tasks = pipeline.tasks
    in_degree = {tid: 0 for tid in tasks}
    adj: dict[str, list[str]] = {tid: [] for tid in tasks}

    for task_id, task in tasks.items():
        try:
            seen_deps: set[str] = set()

            for inp in task.inputs:
                match inp:
                    case PipelineInput(input_id=input_name):
                        if input_name not in pipeline.inputs:
                            raise ValueError(
                                f"Pipeline input '{input_name}' not found")

                    case LinkInput(task_id=dep_id, output_id=output_name):
                        if dep_id not in tasks:
                            raise ValueError(
                                f"Task '{dep_id}' referenced in '{task_id}' not found")

                        if output_name not in tasks[dep_id].outputs:
                            raise ValueError(
                                f"Output '{output_name}' not found in task '{dep_id}'")

                        if dep_id not in seen_deps:
                            seen_deps.add(dep_id)
                            adj[dep_id].append(task_id)
                            in_degree[task_id] += 1
        except ValueError as e:
            raise ValueError(
                f"Invalid reference detected for task {task_id}: {e.args[0]}")

    queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
    result = []

    while queue:
        tid = queue.popleft()
        result.append(tasks[tid])
        for next_tid in adj[tid]:
            in_degree[next_tid] -= 1
            if in_degree[next_tid] == 0:
                queue.append(next_tid)

    if len(result) != len(tasks):
        raise ValueError("Cycle detected in task dependencies")

    log.info("Tasks ordered.")

    return result
