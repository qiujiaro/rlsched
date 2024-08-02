"""Train-Ticket-calibrated microservice scheduling environment.

This module is a readable stand-in for KuTARA, which was not publicly
available. It models the cluster as a stochastic fluid queue:

    arrivals -> route to nodes -> process up to capacity -> keep the backlog

Paper mapping:
  Eq. (1) state: queue, utilisation, dependency pressure, latency change
  Eq. (2) action: routing, replica scaling, CPU-quota adjustment
  Eq. (3) reward: latency, imbalance, cost, completed throughput
  Eq. (5) disturbance: traffic surge, node degradation, CPU jitter
  Eq. (7) constraints: node capacity, queue capacity, latency SLA
"""

import copy
import json
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces


DEFAULT_PROFILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "trainticket_profile.json"
)

# Actions store indexes into these arrays, not the values themselves.
QUOTA_LEVELS = np.array([0.25, 0.5, 1.0, 2.0])  # CPU cores per replica
ROUTE_LEVELS = np.array([0.5, 1.0, 2.0])  # relative node routing weight
ARRIVAL_PROPAGATION_STEPS = 8
LATENCY_PROPAGATION_STEPS = 6


def sample_disturbance(rng, eps, n_nodes):
    """Draw one disturbance from the admissible set Omega(eps)."""
    disturbance = {
        "surge": 1.0 + 2.0 * eps * rng.random(),
        "degrade": np.ones(n_nodes),
        "jitter": 0.2 * eps,
    }
    if eps > 0:
        affected_node = rng.integers(n_nodes)
        disturbance["degrade"][affected_node] = 1.0 - 0.6 * eps * rng.random()
    return disturbance


# Kept for compatibility with callers that import this name.
NOMINAL_XI = None


class MicroserviceSchedEnv(gym.Env):
    """A stochastic fluid-queue simulator for collaborative scheduling."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        profile_path=DEFAULT_PROFILE,
        n_nodes=6,
        node_cores=8.0,
        node_mem_gb=16.0,
        base_rps=60.0,
        max_steps=100,
        r_max=8,
        q_max=2000.0,
        sla_ms=None,
        weights=(1.0, 0.5, 0.2, 1.0),
        core_price=0.05,
        fault_schedule=None,
    ):
        super().__init__()
        with open(profile_path) as file:
            profile = json.load(file)

        self.M = n_nodes
        self.node_cores = node_cores
        self.node_mem = node_mem_gb
        self.r_max = r_max
        self.q_max = q_max
        self.max_steps = max_steps
        self.w = weights
        self.core_price = core_price
        self.fault_schedule = fault_schedule or []

        self._build_service_model(profile, base_rps)
        self.sla_ms = sla_ms if sla_ms is not None else self._default_sla_ms()

        # State = [queue, service util, node CPU util, node memory util,
        #          dependency pressure, latency fluctuation, time].
        observation_size = 4 * self.S + 2 * self.M + 1
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(observation_size,), dtype=np.float32
        )

        # Action = [routing per node, replica delta per service,
        #           CPU-quota delta per service].
        self.action_space = spaces.MultiDiscrete(
            [len(ROUTE_LEVELS)] * self.M + [3] * self.S + [3] * self.S
        )
        self.xi = None

    # ------------------------------------------------------------------
    # Static model construction
    # ------------------------------------------------------------------
    def _build_service_model(self, profile, base_rps):
        """Build the service graph and nominal workload from calibration data."""
        active = [
            index
            for index, latency in enumerate(profile["latency"])
            if latency["n"] > 0
        ]
        old_to_new = {old: new for new, old in enumerate(active)}
        self.names = [profile["services"][index] for index in active]
        self.S = len(active)

        # R[caller, callee] = downstream calls per request served by caller.
        self.R = np.zeros((self.S, self.S))
        for caller, callee, _static, _dynamic, ratio in profile["edges"]:
            if caller in old_to_new and callee in old_to_new:
                self.R[old_to_new[caller], old_to_new[callee]] = min(ratio, 3.0)

        entry_mix = np.array([profile["entry_mix"][index] for index in active])
        entry_mix += 1e-6
        self.mix = entry_mix / entry_mix.sum()

        inclusive_ms = np.array(
            [profile["latency"][index]["p50_ms"] for index in active]
        )
        expected_downstream_ms = self.R @ inclusive_ms
        exclusive_ms = np.maximum(
            inclusive_ms - expected_downstream_ms, 0.2 * inclusive_ms
        )
        self.svc_ms = np.clip(exclusive_ms, 2.0, 2000.0)

        # Approximate CPU core-seconds per request (most span time is I/O wait).
        self.cpu_per_req = 0.1 * self.svc_ms / 1000.0

        # Propagate external traffic through the call graph using a finite series.
        self.lam0 = np.zeros(self.S)
        traffic_at_depth = self.mix * base_rps
        for _ in range(ARRIVAL_PROPAGATION_STEPS):
            self.lam0 += traffic_at_depth
            traffic_at_depth = self.R.T @ traffic_at_depth

    def _default_sla_ms(self):
        """Return 1.5x the unloaded, mix-weighted end-to-end latency."""
        end_to_end_ms = self.svc_ms.copy()
        for _ in range(LATENCY_PROPAGATION_STEPS):
            end_to_end_ms = self.svc_ms + self.R @ end_to_end_ms
        return 1.5 * float(self.mix @ end_to_end_ms)

    # ------------------------------------------------------------------
    # Gymnasium lifecycle
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Create empty queues and a balanced initial deployment."""
        super().reset(seed=seed)
        self.t = 0
        self.q = np.zeros(self.S)

        nominal_cpu_demand = self.lam0 * self.cpu_per_req
        self.quota = np.full(self.S, 1, dtype=int)  # level 1 = 0.5 core
        self.rep = np.clip(
            np.ceil(nominal_cpu_demand / (QUOTA_LEVELS[1] * 0.6)),
            1,
            self.r_max,
        ).astype(int)

        self.place = np.zeros((self.S, self.M), dtype=int)
        for service in range(self.S):
            for _ in range(self.rep[service]):
                self._place_one(service)

        self.route = np.ones(self.M, dtype=int)  # level 1 = weight 1.0
        self.lat_ema = self.svc_ms.copy()
        self.W = self.svc_ms.copy()
        self.util_s = np.zeros(self.S)
        self.xi = None
        self.stats = {
            "arrived": 0.0,
            "completed": 0.0,
            "dropped": 0.0,
            "rejected": 0,
            "sched_ops": 0,
        }

        empty_nodes = np.zeros(self.M)
        empty_services = np.zeros(self.S)
        return self._obs(empty_nodes, empty_nodes, empty_services), {}

    def step(self, action):
        """Apply one joint action and advance the fluid queue by one step."""
        surge, node_health, cpu_jitter = self._active_disturbance()
        changes, rejected = self._apply_action(action)
        arrivals = self._sample_arrivals(surge)

        allocated_cores, allocated_memory, capacity = self._service_capacity(
            node_health, cpu_jitter
        )
        routing_share = self._routing_shares()
        completed_by_node, completed, dropped, total_capacity = self._process_requests(
            arrivals, routing_share, capacity
        )

        used_cores = (completed_by_node * self.cpu_per_req[:, None]).sum(axis=0)
        node_cpu_util = used_cores / self.node_cores
        node_mem_util = allocated_memory / self.node_mem
        latency_ms, latency_fluctuation = self._latency(total_capacity)

        reward, reward_terms = self._reward(
            arrivals,
            completed,
            node_cpu_util,
            allocated_cores,
            changes,
            latency_ms,
        )
        constraints = self._constraints(allocated_cores, latency_ms)

        self._update_stats(arrivals, completed, dropped, changes, rejected)
        self.t += 1
        truncated = self.t >= self.max_steps

        info = self._info(
            arrivals=arrivals,
            completed=completed,
            dropped=dropped,
            rejected=rejected,
            changes=changes,
            allocated_cores=allocated_cores,
            allocated_memory=allocated_memory,
            used_cores=used_cores,
            latency_ms=latency_ms,
            constraints=constraints,
            allocation_fraction=reward_terms["allocation_fraction"],
        )
        observation = self._obs(
            node_cpu_util, node_mem_util, latency_fluctuation
        )
        return observation, float(reward), False, truncated, info

    # ------------------------------------------------------------------
    # Disturbances and arrivals
    # ------------------------------------------------------------------
    def set_disturbance(self, disturbance):
        """Set the disturbance used by the next transition."""
        self.xi = disturbance

    def _active_disturbance(self):
        """Return (traffic surge, node health, CPU jitter) for this step."""
        disturbance = self.xi
        for fault in self.fault_schedule:
            if fault["t0"] <= self.t < fault["t1"]:
                disturbance = fault["xi"]

        if disturbance is None:
            return 1.0, np.ones(self.M), 0.0
        return (
            disturbance["surge"],
            disturbance["degrade"],
            disturbance["jitter"],
        )

    def _sample_arrivals(self, surge):
        """Sample one step of diurnal traffic with optional surge."""
        phase = self.t + self.np_random.random() * 3
        traffic_profile = 1.0 + 0.3 * np.sin(2 * np.pi * phase / 50.0)
        rates = self.lam0 * traffic_profile * surge
        return self.np_random.poisson(rates).astype(float)

    # ------------------------------------------------------------------
    # Replica placement and actions
    # ------------------------------------------------------------------
    def _node_alloc(self):
        """Return allocated CPU cores and memory for every node."""
        quota_per_service = QUOTA_LEVELS[self.quota]
        cores = (self.place * quota_per_service[:, None]).sum(axis=0)
        memory_per_replica = 0.5 + 0.25 * quota_per_service
        memory = (self.place * memory_per_replica[:, None]).sum(axis=0)
        return cores, memory

    def _place_one(self, service):
        """Place one replica on the feasible node with most free CPU."""
        allocated_cores, allocated_memory = self._node_alloc()
        required_cores = QUOTA_LEVELS[self.quota[service]]
        required_memory = 0.5 + 0.25 * required_cores
        free_cores = self.node_cores - allocated_cores
        feasible = np.where(
            (free_cores >= required_cores)
            & (self.node_mem - allocated_memory >= required_memory)
        )[0]
        if len(feasible) == 0:
            return False

        target_node = feasible[np.argmax(free_cores[feasible])]
        self.place[service, target_node] += 1
        return True

    def _remove_one(self, service):
        """Remove one replica from the most CPU-allocated hosting node."""
        allocated_cores, _ = self._node_alloc()
        hosting_nodes = np.where(self.place[service] > 0)[0]
        target_node = hosting_nodes[np.argmax(allocated_cores[hosting_nodes])]
        self.place[service, target_node] -= 1

    def _decode_action(self, action):
        """Split the joint action into routing, replica, and quota decisions."""
        action = np.asarray(action, dtype=int)
        route_end = self.M
        scale_end = self.M + self.S
        return (
            action[:route_end],
            action[route_end:scale_end],
            action[scale_end:],
        )

    def _apply_action(self, action):
        """Apply routing, scaling, and CPU-quota decisions."""
        route_action, scale_action, quota_action = self._decode_action(action)
        self.route = route_action.copy()

        changes = 0
        rejected = 0
        for service in range(self.S):
            replica_delta = scale_action[service] - 1  # 0/1/2 -> -1/0/+1
            if replica_delta > 0 and self.rep[service] < self.r_max:
                if self._place_one(service):
                    self.rep[service] += 1
                    changes += 1
                else:
                    rejected += 1
            elif replica_delta < 0 and self.rep[service] > 1:
                self._remove_one(service)
                self.rep[service] -= 1
                changes += 1

            quota_delta = quota_action[service] - 1  # 0/1/2 -> -1/0/+1
            next_quota = int(
                np.clip(
                    self.quota[service] + quota_delta,
                    0,
                    len(QUOTA_LEVELS) - 1,
                )
            )
            if next_quota == self.quota[service]:
                continue

            previous_quota = self.quota[service]
            self.quota[service] = next_quota
            allocated_cores, allocated_memory = self._node_alloc()
            exceeds_capacity = np.any(
                allocated_cores > self.node_cores + 1e-9
            ) or np.any(allocated_memory > self.node_mem + 1e-9)
            if exceeds_capacity:
                self.quota[service] = previous_quota
                rejected += 1
            else:
                changes += 0.5

        return changes, rejected

    # ------------------------------------------------------------------
    # Fluid queue transition
    # ------------------------------------------------------------------
    def _service_capacity(self, node_health, cpu_jitter):
        """Return allocations and request capacity for each service/node."""
        allocated_cores, allocated_memory = self._node_alloc()
        random_efficiency = np.clip(
            1.0 + cpu_jitter * self.np_random.standard_normal(self.M), 0.3, 1.5
        )
        node_efficiency = node_health * random_efficiency
        replica_cpu = self.place * QUOTA_LEVELS[self.quota][:, None]
        capacity = (
            replica_cpu * node_efficiency[None, :] / self.cpu_per_req[:, None]
        )
        return allocated_cores, allocated_memory, capacity

    def _routing_shares(self):
        """Convert node routing choices into per-service routing fractions."""
        weighted_replicas = ROUTE_LEVELS[self.route][None, :] * self.place
        total_weight = weighted_replicas.sum(axis=1, keepdims=True)
        return weighted_replicas / np.maximum(total_weight, 1e-9)

    def _process_requests(self, arrivals, routing_share, capacity):
        """Route fluid load, process up to capacity, and retain the backlog."""
        pending = self.q + arrivals
        load_by_node = pending[:, None] * routing_share
        completed_by_node = np.minimum(load_by_node, capacity)
        completed = completed_by_node.sum(axis=1)

        next_queue = pending - completed
        dropped = np.maximum(next_queue - self.q_max, 0.0)
        self.q = next_queue - dropped

        total_capacity = capacity.sum(axis=1)
        self.util_s = np.clip(
            (arrivals + 1e-9) / np.maximum(total_capacity, 1e-9), 0, 5
        )
        return completed_by_node, completed, dropped, total_capacity

    # ------------------------------------------------------------------
    # Latency, reward, constraints, and reporting
    # ------------------------------------------------------------------
    def _latency(self, total_capacity):
        """Estimate service and end-to-end latency in milliseconds."""
        stable_util = np.clip(self.util_s, 0, 0.95)
        utilisation_delay = self.svc_ms / (1 - stable_util)
        queue_delay = 1000.0 * np.minimum(
            self.q / np.maximum(total_capacity, 1e-6), 30.0
        )
        self.W = utilisation_delay + queue_delay

        end_to_end_ms = self.W.copy()
        for _ in range(LATENCY_PROPAGATION_STEPS):
            end_to_end_ms = self.W + self.R @ end_to_end_ms
        end_to_end_ms = np.minimum(end_to_end_ms, 60_000.0)
        aggregate_latency_ms = float(self.mix @ end_to_end_ms)

        fluctuation = np.abs(self.W - self.lat_ema) / np.maximum(
            self.lat_ema, 1e-6
        )
        self.lat_ema = 0.8 * self.lat_ema + 0.2 * self.W
        return aggregate_latency_ms, fluctuation

    def _reward(
        self,
        arrivals,
        completed,
        node_cpu_util,
        allocated_cores,
        changes,
        latency_ms,
    ):
        """Compute Eq. (3) and expose named terms for reporting."""
        throughput_ratio = completed.sum() / max(arrivals.sum(), 1.0)
        imbalance = float(
            np.std(node_cpu_util) + np.std(np.clip(self.util_s, 0, 2))
        )
        allocation_fraction = allocated_cores.sum() / (
            self.node_cores * self.M
        )
        control_cost = changes / self.S + allocation_fraction
        normalized_latency = latency_ms / self.sla_ms

        latency_weight, imbalance_weight, cost_weight, throughput_weight = self.w
        reward = (
            -latency_weight * min(normalized_latency, 10.0)
            - imbalance_weight * imbalance
            - cost_weight * control_cost
            + throughput_weight * throughput_ratio
        )
        return reward, {
            "allocation_fraction": allocation_fraction,
            "normalized_latency": normalized_latency,
        }

    def _constraints(self, allocated_cores, latency_ms):
        """Return residuals; positive values mean a constraint is violated."""
        return np.array(
            [
                allocated_cores.max() / self.node_cores - 1.0,
                self.q.max() / self.q_max - 1.0,
                latency_ms / self.sla_ms - 1.0,
            ]
        )

    def _update_stats(self, arrivals, completed, dropped, changes, rejected):
        self.stats["arrived"] += arrivals.sum()
        self.stats["completed"] += completed.sum()
        self.stats["dropped"] += dropped.sum()
        self.stats["rejected"] += rejected
        self.stats["sched_ops"] += changes + rejected

    def _info(
        self,
        *,
        arrivals,
        completed,
        dropped,
        rejected,
        changes,
        allocated_cores,
        allocated_memory,
        used_cores,
        latency_ms,
        constraints,
        allocation_fraction,
    ):
        """Build the stable public metrics consumed by the benchmark."""
        return {
            "constraints": constraints,
            "lat_ms": latency_ms,
            "thr_rps": float(completed.sum()),
            "arr_rps": float(arrivals.sum()),
            "ru": float(used_cores.sum() / max(allocated_cores.sum(), 1e-9)),
            "cpu_alloc": float(allocation_fraction),
            "mem_alloc": float(
                allocated_memory.sum() / (self.node_mem * self.M)
            ),
            "dropped": float(dropped.sum()),
            "rejected": rejected,
            "changes": changes,
            "cost": float(
                allocated_cores.sum() * self.core_price + 0.01 * changes
            ),
            "sla_ok": float(latency_ms <= self.sla_ms),
        }

    def _obs(self, node_cpu_util, node_mem_util, latency_fluctuation):
        """Construct Eq. (1)'s normalized global state vector."""
        normalized_queue = np.log1p(self.q) / np.log1p(self.q_max)
        bounded_service_util = np.clip(self.util_s, 0, 5)
        dependency_pressure = (
            self.R @ np.clip(self.util_s, 0, 2)
            + self.R.T @ np.clip(self.util_s, 0, 2)
        )
        bounded_fluctuation = np.clip(latency_fluctuation, 0, 5)
        normalized_time = [self.t / self.max_steps]

        return np.concatenate(
            [
                normalized_queue,
                bounded_service_util,
                node_cpu_util,
                node_mem_util,
                dependency_pressure,
                bounded_fluctuation,
                normalized_time,
            ]
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Robust wrapper support
    # ------------------------------------------------------------------
    def clone(self):
        """Return an independent simulator copy for disturbance evaluation."""
        return copy.deepcopy(self)


gym.register(
    id="TrainTicketSched-v0",
    entry_point="simulator.env:MicroserviceSchedEnv",
)
