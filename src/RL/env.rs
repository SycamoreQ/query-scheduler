use crate::structures::_graph::*;
use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use database::low::DatabaseManager;
use rand::{distributions::Distribution, Rng};
use rand_distr::Poisson;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

pub struct EdgeMLEnv {
    pub clusters: Vec<EdgeCluster>,
    pub global_task_queue: VecDeque<MLTask>,
    pub current_time: DateTime<Utc>,
    pub timestep: usize,
    pub completed_tasks: VecDeque<CompletedTask>,
    pub config: EnvironmentConfig,

    rng: rand::rngs::ThreadRng,
    pub db: Option<Arc<DatabaseManager>>,

    // Robustness features
    max_completed_history: usize,
    failed_tasks: VecDeque<FailedTask>,
    max_failed_history: usize,
    cluster_health: Vec<ClusterHealth>,
}


#[derive(Clone, Debug)]
pub struct EdgeCluster {
    pub id: usize,
    pub cpus: Vec<CPU>,
    pub gpus: Vec<GPU>,
    pub total_memory_mb: usize,
    pub bandwidth_mbps: f64,
    pub pending_tasks: VecDeque<MLTask>,
    pub running_tasks: Vec<RunningTask>,
}

impl EdgeCluster {
    pub fn new(id: usize, config: &EnvironmentConfig) -> Self {
        let cpus = (0..config.cpu_cores_per_cluster)
            .map(|i| CPU::new(i))
            .collect();

        let gpus = (0..config.gpus_per_cluster)
            .map(|i| GPU::new(i))
            .collect();

        Self {
            id,
            cpus,
            gpus,
            total_memory_mb: config.memory_mb_per_cluster,
            bandwidth_mbps: 1000.0,
            pending_tasks: VecDeque::new(),
            running_tasks: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        for cpu in &mut self.cpus {
            cpu.core_available = cpu.core_total;
            cpu.memory_available_mb = cpu.memory_total_mb;
        }
        for gpu in &mut self.gpus {
            gpu.core_available = gpu.core_total;
            gpu.memory_available_mb = gpu.memory_total_mb;
        }
        self.pending_tasks.clear();
        self.running_tasks.clear();
    }

    /// Get total available CPU cores across all CPUs
    pub fn total_cpu_available(&self) -> usize {
        self.cpus.iter().map(|c| c.core_available).sum()
    }

    /// Get total available memory across all CPUs
    pub fn total_memory_available(&self) -> usize {
        self.cpus.iter().map(|c| c.memory_available_mb).sum()
    }

    /// Get cluster resource snapshot
    pub fn get_resources(&self) -> ClusterResources {
        ClusterResources {
            cpu_available: self.total_cpu_available(),
            gpu_available: self.gpus.iter().map(|g| g.core_available).collect(),
            memory_available: self.total_memory_available(),
        }
    }

    /// Convert to features for graph
    pub fn to_features(&self) -> ClusterFeatures {
        ClusterFeatures {
            cluster_id: self.id,
            cpu_available: self.total_cpu_available() as f32,
            gpu_available: self.gpus
                .iter()
                .flat_map(|g| vec![g.core_available as f32, g.memory_available_mb as f32])
                .collect(),
            memory_available: self.total_memory_available() as f32,
            queue_length: self.pending_tasks.len() as f32,
        }
    }
}


#[derive(Debug, Clone)]
pub struct CPU {
    pub id: usize,
    pub core_available: usize,
    pub core_total: usize,
    pub memory_available_mb: usize,
    pub memory_total_mb: usize,
}

impl CPU {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            core_available: 100,
            core_total: 100,
            memory_available_mb: 16000,
            memory_total_mb: 16000,
        }
    }

    pub fn can_allocate(&self, cores: usize, memory_mb: usize) -> bool {
        self.core_available >= cores && self.memory_available_mb >= memory_mb
    }
}

#[derive(Debug, Clone)]
pub struct GPU {
    pub id: usize,
    pub core_available: usize,
    pub core_total: usize,
    pub memory_available_mb: usize,
    pub memory_total_mb: usize,
}

impl GPU {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            core_available: 100,
            core_total: 100,
            memory_available_mb: 16000,
            memory_total_mb: 16000,
        }
    }

    pub fn can_allocate(&self, cores: usize, memory_mb: usize) -> bool {
        self.core_available >= cores && self.memory_available_mb >= memory_mb
    }
}


#[derive(Clone, Debug)]
pub struct MLTask {
    pub id: String,
    pub task_type: MLTaskType,
    pub min_cpu: usize,
    pub min_gpu_core: usize,
    pub min_gpu_memory: usize,
    pub min_memory_mb: usize,
    pub arrival_time: DateTime<Utc>,
    pub eligible_clusters: Vec<usize>,
    pub duration_estimate: Duration,
    pub resource_sensitivity: ResourceSensitivity,
}

impl MLTask {
    pub fn to_features(&self) -> TaskFeatures {
        TaskFeatures {
            task_id: self.id.clone(),
            task_type: self.task_type.to_one_hot(),
            cpu_required: self.min_cpu as f32,
            gpu_core_required: self.min_gpu_core as f32,
            gpu_mem_required: self.min_gpu_memory as f32,
            memory_required: self.min_memory_mb as f32,
            elapsed_time: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MLTaskType {
    ImageClassificationTraining,
    ObjectDetectionInference,
    NLPTransformer,
    ReinforcementLearning,
    LinearRegression,
    LogisticRegression,
    GraphAttention,
    GraphNeuralNetwork,
    RetrievalAugmentedGeneration,
    SupportVectorMachine,
}

impl MLTaskType {
    pub fn to_one_hot(&self) -> Vec<f32> {
        let mut vec = vec![0.0; 10];
        let idx = match self {
            Self::ImageClassificationTraining => 0,
            Self::ObjectDetectionInference => 1,
            Self::NLPTransformer => 2,
            Self::ReinforcementLearning => 3,
            Self::LinearRegression => 4,
            Self::LogisticRegression => 5,
            Self::GraphAttention => 6,
            Self::GraphNeuralNetwork => 7,
            Self::RetrievalAugmentedGeneration => 8,
            Self::SupportVectorMachine => 9,
        };
        vec[idx] = 1.0;
        vec
    }
}

#[derive(Clone, Debug)]
pub struct ResourceSensitivity {
    pub cpu_scaling: f32,
    pub gpu_scaling: f32,
    pub memory_scaling: f32,
}

#[derive(Clone, Debug)]
pub struct RunningTask {
    pub task: MLTask,
    pub allocated_cpu: usize,
    pub allocated_gpu_id: Option<usize>,
    pub allocated_gpu_cores: usize,
    pub allocated_memory_mb: usize,
    pub start_time: DateTime<Utc>,
    pub progress: f64, // 0.0 to 1.0
}

#[derive(Clone, Debug)]
pub struct CompletedTask {
    pub task: MLTask,
    pub task_id: String,
    pub cluster_id: usize,
    pub arrival_time: DateTime<Utc>,
    pub start_time: DateTime<Utc>,
    pub completion_time: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct ClusterHealth {
    pub cluster_id: usize,
    pub is_healthy: bool,
    pub consecutive_failures: usize,
    pub last_failure: Option<DateTime<Utc>>,
    pub total_failures: usize,
    pub recovery_time: Option<DateTime<Utc>>,
}

impl ClusterHealth {
    pub fn new(cluster_id: usize) -> Self {
        Self {
            cluster_id,
            is_healthy: true,
            consecutive_failures: 0,
            last_failure: None,
            total_failures: 0,
            recovery_time: None,
        }
    }

    pub fn mark_failure(&mut self, timestamp: DateTime<Utc>) {
        self.consecutive_failures += 1;
        self.total_failures += 1;
        self.last_failure = Some(timestamp);

        if self.consecutive_failures > 5 {
            self.is_healthy = false;
            error!("Cluster {} marked unhealthy after {} consecutive failures",
                self.cluster_id, self.consecutive_failures);
        }
    }

    pub fn mark_success(&mut self) {
        if self.consecutive_failures > 0 {
            debug!("Cluster {} recovered after {} failures",
                self.cluster_id, self.consecutive_failures);
        }
        self.consecutive_failures = 0;

        if !self.is_healthy {
            self.is_healthy = true;
            self.recovery_time = Some(Utc::now());
            info!("Cluster {} marked healthy again", self.cluster_id);
        }
    }
}

#[derive(Debug, Clone)]
pub struct FailedTask {
    pub task: MLTask,
    pub cluster_id: usize,
    pub failure_reason: String,
    pub timestamp: DateTime<Utc>,
    pub attempted_allocation: AgentAction,
}


pub struct SchedulingContext {
    pub graph: GraphTensors,
    pub task_lookup: TaskLookup,
    pub cluster_resources: ClusterResources,
}

pub struct TaskLookup {
    pub tasks: HashMap<i64, TaskInfo>,
}

impl TaskLookup {
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
        }
    }

    pub fn insert(&mut self, node_idx: i64, info: TaskInfo) {
        self.tasks.insert(node_idx, info);
    }

    pub fn get(&self, node_idx: &i64) -> Option<&TaskInfo> {
        self.tasks.get(node_idx)
    }
}

pub struct TaskInfo {
    pub task_id: String,
    pub min_cpu: usize,
    pub min_gpu_core: usize,
    pub min_gpu_memory: usize,
    pub min_memory_mb: usize,
    pub task_type: MLTaskType,
}

#[derive(Debug, Clone)]
pub struct ClusterResources {
    pub cpu_available: usize,
    pub gpu_available: Vec<usize>,
    pub memory_available: usize,
}

impl Default for ClusterResources {
    fn default() -> Self {
        Self {
            cpu_available: 0,
            gpu_available: Vec::new(),
            memory_available: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AgentAction {
    pub selected_task: Option<String>,
    pub allocated_cpu: usize,
    pub allocated_gpu_id: Option<usize>,
    pub allocated_gpu_cores: usize,
    pub allocated_memory_mb: usize,
}

impl AgentAction {
    pub fn no_op() -> Self {
        Self {
            selected_task: None,
            allocated_cpu: 0,
            allocated_gpu_id: None,
            allocated_gpu_cores: 0,
            allocated_memory_mb: 0,
        }
    }
}


pub struct StepResult {
    pub observation: GraphTensors,
    pub rewards: Vec<f64>,
    pub done: bool,
    pub info: HashMap<String, f64>,
}


#[derive(Clone, Debug)]
pub struct EnvironmentConfig {
    pub num_clusters: usize,
    pub cpu_cores_per_cluster: usize,
    pub gpus_per_cluster: usize,
    pub memory_mb_per_cluster: usize,
    pub task_arrival_rate: f64,
    pub timestep_duration: Duration,
    pub max_pending_set_size: usize,
    pub episode_length: usize,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            num_clusters: 3,
            cpu_cores_per_cluster: 16,
            gpus_per_cluster: 8,
            memory_mb_per_cluster: 32768,
            task_arrival_rate: 2.0,
            timestep_duration: Duration::seconds(10),
            max_pending_set_size: 10,
            episode_length: 1000,
        }
    }
}

impl EdgeMLEnv {
    pub fn new(config: EnvironmentConfig, db: Arc<DatabaseManager>) -> Self {
        let clusters = (0..config.num_clusters)
            .map(|id| EdgeCluster::new(id, &config))
            .collect();

        let cluster_health = (0..config.num_clusters)
            .map(ClusterHealth::new)
            .collect();

        Self {
            clusters,
            global_task_queue: VecDeque::new(),
            current_time: Utc::now(),
            timestep: 0,
            completed_tasks: VecDeque::new(),
            config,
            rng: rand::thread_rng(),
            db: Some(db),
            max_completed_history: 1000,
            failed_tasks: VecDeque::new(),
            max_failed_history: 500,
            cluster_health,
        }
    }

    /// Reset environment for new episode
    pub fn reset(&mut self) -> GraphTensors {
        for cluster in &mut self.clusters {
            cluster.reset();
        }

        self.global_task_queue.clear();
        self.completed_tasks.clear();
        self.failed_tasks.clear();
        self.current_time = Utc::now();
        self.timestep = 0;

        // Reset cluster health
        for health in &mut self.cluster_health {
            health.is_healthy = true;
            health.consecutive_failures = 0;
        }

        // Generate initial batch of tasks
        self.generate_tasks(self.config.timestep_duration * 6);
        self.distribute_new_tasks();

        self.build_observation()
    }

    /// Build observation graph for all clusters
    fn build_observation(&self) -> GraphTensors {
        let mut graph = HANGraph::new();

        // Add cluster nodes
        let cluster_indices: Vec<_> = self
            .clusters
            .iter()
            .map(|c| graph.add_cluster(c.to_features()))
            .collect();

        for (cluster_idx, cluster) in self.clusters.iter().enumerate() {
            // Add pending tasks
            for (i, task) in cluster.pending_tasks.iter().enumerate() {
                if i >= self.config.max_pending_set_size {
                    break;
                }

                let task_idx = graph.add_pending_task(task.to_features());
                graph.connect_task_to_cluster(task_idx, cluster_indices[cluster_idx]);
            }

            // Add running tasks
            for running in &cluster.running_tasks {
                let elapsed = (self.current_time - running.start_time).num_seconds() as f32;
                let mut features = running.task.to_features();
                features.elapsed_time = elapsed;

                let task_idx = graph.add_running_task(features);
                graph.connect_task_to_cluster(task_idx, cluster_indices[cluster_idx]);
            }
        }

        graph.add_shortcut();
        graph.to_tensors()
    }

    /// Build scheduling context for a specific cluster (for MCTS)
    pub fn build_scheduling_context(&self, cluster_idx: usize) -> SchedulingContext {
        let mut graph = HANGraph::new();
        let mut task_lookup = TaskLookup::new();
        let cluster = &self.clusters[cluster_idx];

        let cluster_node_idx = graph.add_cluster(cluster.to_features()) as i64;

        let cluster_resources = cluster.get_resources();

        for (i, task) in cluster.pending_tasks.iter().enumerate() {
            if i >= self.config.max_pending_set_size {
                break;
            }

            let node_idx = graph.add_pending_task(task.to_features()) as i64;

            task_lookup.insert(
                node_idx,
                TaskInfo {
                    task_id: task.id.clone(),
                    min_cpu: task.min_cpu,
                    min_gpu_core: task.min_gpu_core,
                    min_gpu_memory: task.min_gpu_memory,
                    min_memory_mb: task.min_memory_mb,
                    task_type: task.task_type,
                },
            );

            graph.connect_task_to_cluster(node_idx as usize, cluster_node_idx as usize);
        }

        SchedulingContext {
            graph: graph.to_tensors(),
            task_lookup,
            cluster_resources,
        }
    }

    /// Execute one environment step
    pub async fn step(&mut self, actions: Vec<AgentAction>) -> Result<StepResult> {
        // 1. Apply agent actions (schedule tasks)
        let scheduled = self.apply_actions(actions).await?;
        debug!("Scheduled {} tasks this step", scheduled.len());

        // 2. Execute running tasks (make progress)
        self.execute_tasks();

        // 3. Generate new arriving tasks
        self.generate_tasks(self.config.timestep_duration);
        self.distribute_new_tasks();

        // 4. Calculate rewards
        let rewards = self.calculate_rewards();

        // 5. Build next observation
        let observation = self.build_observation();

        // 6. Check if done
        let done = self.is_done();

        self.current_time += self.config.timestep_duration;
        self.timestep += 1;

        Ok(StepResult {
            observation,
            rewards,
            done,
            info: self.get_info(),
        })
    }

    /// Apply actions with validation and rollback
    pub async fn apply_actions(&mut self, actions: Vec<AgentAction>) -> Result<Vec<String>> {
        let mut scheduled_tasks = Vec::new();

        for (cluster_id, action) in actions.iter().enumerate() {
            if cluster_id >= self.clusters.len() {
                continue;
            }

            // Check cluster health
            if !self.cluster_health[cluster_id].is_healthy {
                warn!("Cluster {} is unhealthy, skipping", cluster_id);
                continue;
            }

            if let Some(ref task_id) = action.selected_task {
                match self.try_schedule_task(cluster_id, task_id, action).await {
                    Ok(()) => {
                        scheduled_tasks.push(task_id.clone());
                        self.cluster_health[cluster_id].mark_success();
                    }
                    Err(e) => {
                        error!("Failed to schedule task {} on cluster {}: {}",
                            task_id, cluster_id, e);
                        self.cluster_health[cluster_id].mark_failure(self.current_time);
                    }
                }
            }
        }

        Ok(scheduled_tasks)
    }

    /// Try to schedule a task on a cluster
    async fn try_schedule_task(
        &mut self,
        cluster_id: usize,
        task_id: &str,
        action: &AgentAction,
    ) -> Result<()> {
        let cluster = &mut self.clusters[cluster_id];

        // Validate resources BEFORE allocation
        if !self.validate_resources(cluster_id, action) {
            return Err(anyhow::anyhow!("Insufficient resources for allocation"));
        }

        // Find and remove task from pending queue
        let task_pos = cluster
            .pending_tasks
            .iter()
            .position(|t| &t.id == task_id)
            .context("Task not found in pending queue")?;

        let task = cluster.pending_tasks.remove(task_pos).unwrap();

        // Try to allocate resources
        match self.allocate_resources(cluster_id, action, &task) {
            Ok(running_task) => {
                cluster.running_tasks.push(running_task);

                // Log to database if available
                if let Some(db) = &self.db {
                    // Async logging - don't block on failure
                    let task_clone = task.clone();
                    let db_clone = Arc::clone(db);
                    let cluster_id_copy = cluster_id;
                    tokio::spawn(async move {
                        if let Err(e) = log_task_start(&db_clone, &task_clone, cluster_id_copy).await {
                            warn!("Failed to log task start: {}", e);
                        }
                    });
                }

                Ok(())
            }
            Err(e) => {
                // Rollback: put task back at front
                cluster.pending_tasks.push_front(task.clone());

                // Track failure
                self.failed_tasks.push_back(FailedTask {
                    task,
                    cluster_id,
                    failure_reason: e.to_string(),
                    timestamp: self.current_time,
                    attempted_allocation: action.clone(),
                });

                // Keep bounded history
                if self.failed_tasks.len() > self.max_failed_history {
                    self.failed_tasks.pop_front();
                }

                Err(e)
            }
        }
    }

    /// Validate that resources are available
    fn validate_resources(&self, cluster_id: usize, action: &AgentAction) -> bool {
        let cluster = &self.clusters[cluster_id];

        // Check CPU
        if action.allocated_cpu > cluster.total_cpu_available() {
            debug!("Insufficient CPU: need {}, have {}",
                action.allocated_cpu, cluster.total_cpu_available());
            return false;
        }

        // Check GPU
        if let Some(gpu_id) = action.allocated_gpu_id {
            if gpu_id >= cluster.gpus.len() {
                debug!("Invalid GPU ID: {}", gpu_id);
                return false;
            }
            if action.allocated_gpu_cores > cluster.gpus[gpu_id].core_available {
                debug!("Insufficient GPU cores on GPU {}: need {}, have {}",
                    gpu_id, action.allocated_gpu_cores, cluster.gpus[gpu_id].core_available);
                return false;
            }
        }

        // Check Memory
        if action.allocated_memory_mb > cluster.total_memory_available() {
            debug!("Insufficient memory: need {} MB, have {} MB",
                action.allocated_memory_mb, cluster.total_memory_available());
            return false;
        }

        true
    }

    /// Allocate resources across CPUs/GPUs
    fn allocate_resources(
        &mut self,
        cluster_id: usize,
        action: &AgentAction,
        task: &MLTask,
    ) -> Result<RunningTask> {
        let cluster = &mut self.clusters[cluster_id];

        // Allocate CPU cores across available CPUs
        let mut cpu_allocated = 0;
        for cpu in &mut cluster.cpus {
            if cpu_allocated >= action.allocated_cpu {
                break;
            }
            let can_allocate = cpu.core_available.min(action.allocated_cpu - cpu_allocated);
            cpu.core_available = cpu.core_available.saturating_sub(can_allocate);
            cpu_allocated += can_allocate;
        }

        if cpu_allocated < action.allocated_cpu {
            // Rollback CPU allocation
            self.free_cpu_resources(cluster_id, cpu_allocated);
            return Err(anyhow::anyhow!("Could not allocate enough CPU cores"));
        }

        // Allocate GPU if requested
        if let Some(gpu_id) = action.allocated_gpu_id {
            if gpu_id < cluster.gpus.len() {
                let gpu = &mut cluster.gpus[gpu_id];

                if !gpu.can_allocate(action.allocated_gpu_cores, task.min_gpu_memory) {
                    // Rollback CPU allocation
                    self.free_cpu_resources(cluster_id, cpu_allocated);
                    return Err(anyhow::anyhow!("GPU {} cannot allocate requested resources", gpu_id));
                }

                gpu.core_available = gpu.core_available.saturating_sub(action.allocated_gpu_cores);
                gpu.memory_available_mb = gpu.memory_available_mb.saturating_sub(task.min_gpu_memory);
            }
        }

        // Allocate Memory across CPUs
        let mut mem_allocated = 0;
        for cpu in &mut cluster.cpus {
            if mem_allocated >= action.allocated_memory_mb {
                break;
            }
            let can_allocate = cpu.memory_available_mb.min(action.allocated_memory_mb - mem_allocated);
            cpu.memory_available_mb = cpu.memory_available_mb.saturating_sub(can_allocate);
            mem_allocated += can_allocate;
        }

        if mem_allocated < action.allocated_memory_mb {
            // Rollback all allocations
            self.free_cpu_resources(cluster_id, cpu_allocated);
            if let Some(gpu_id) = action.allocated_gpu_id {
                self.free_gpu_resources(cluster_id, gpu_id, action.allocated_gpu_cores, task.min_gpu_memory);
            }
            return Err(anyhow::anyhow!("Could not allocate enough memory"));
        }

        Ok(RunningTask {
            task: task.clone(),
            allocated_cpu: cpu_allocated,
            allocated_gpu_id: action.allocated_gpu_id,
            allocated_gpu_cores: action.allocated_gpu_cores,
            allocated_memory_mb: mem_allocated,
            start_time: self.current_time,
            progress: 0.0,
        })
    }

    /// Free CPU resources (for rollback or completion)
    fn free_cpu_resources(&mut self, cluster_id: usize, cores: usize) {
        let cluster = &mut self.clusters[cluster_id];
        let mut cores_to_free = cores;

        for cpu in &mut cluster.cpus {
            if cores_to_free == 0 {
                break;
            }
            let free_amount = cores_to_free.min(cpu.core_total - cpu.core_available);
            cpu.core_available += free_amount;
            cores_to_free -= free_amount;
        }
    }

    /// Free GPU resources (for rollback or completion)
    fn free_gpu_resources(&mut self, cluster_id: usize, gpu_id: usize, cores: usize, memory_mb: usize) {
        let cluster = &mut self.clusters[cluster_id];
        if gpu_id < cluster.gpus.len() {
            let gpu = &mut cluster.gpus[gpu_id];
            gpu.core_available = (gpu.core_available + cores).min(gpu.core_total);
            gpu.memory_available_mb = (gpu.memory_available_mb + memory_mb).min(gpu.memory_total_mb);
        }
    }

    /// Free memory resources (for rollback or completion)
    fn free_memory_resources(&mut self, cluster_id: usize, memory_mb: usize) {
        let cluster = &mut self.clusters[cluster_id];
        let mut mem_to_free = memory_mb;

        for cpu in &mut cluster.cpus {
            if mem_to_free == 0 {
                break;
            }
            let free_amount = mem_to_free.min(cpu.memory_total_mb - cpu.memory_available_mb);
            cpu.memory_available_mb += free_amount;
            mem_to_free -= free_amount;
        }
    }

    /// Execute running tasks and complete finished ones
    fn execute_tasks(&mut self) {
        for cluster in &mut self.clusters {
            let mut completed_indices = Vec::new();

            for (idx, running) in cluster.running_tasks.iter_mut().enumerate() {
                let progress_delta = self.calculate_progress(running);
                running.progress += progress_delta;

                if running.progress >= 1.0 {
                    completed_indices.push(idx);
                }
            }

            // Remove completed tasks in reverse order
            for &idx in completed_indices.iter().rev() {
                let completed = cluster.running_tasks.remove(idx);

                // Free resources
                self.free_cpu_resources(cluster.id, completed.allocated_cpu);
                if let Some(gpu_id) = completed.allocated_gpu_id {
                    self.free_gpu_resources(
                        cluster.id,
                        gpu_id,
                        completed.allocated_gpu_cores,
                        completed.task.min_gpu_memory,
                    );
                }
                self.free_memory_resources(cluster.id, completed.allocated_memory_mb);

                // Add to completed history
                let completed_task = CompletedTask {
                    task: completed.task.clone(),
                    task_id: completed.task.id.clone(),
                    cluster_id: cluster.id,
                    arrival_time: completed.task.arrival_time,
                    start_time: completed.start_time,
                    completion_time: self.current_time,
                };

                self.completed_tasks.push_back(completed_task);

                // Keep bounded history
                if self.completed_tasks.len() > self.max_completed_history {
                    self.completed_tasks.pop_front();
                }

                // Async logging to database
                if let Some(db) = &self.db {
                    let task_clone = completed.task.clone();
                    let db_clone = Arc::clone(db);
                    let cluster_id = cluster.id;
                    let completion_time = self.current_time;
                    let start_time = completed.start_time;

                    tokio::spawn(async move {
                        let execution_time = (completion_time - start_time).num_seconds() as f32;
                        if let Err(e) = log_task_completion(
                            &db_clone,
                            &task_clone.id,
                            cluster_id,
                            completion_time,
                            execution_time,
                        ).await {
                            warn!("Failed to log task completion: {}", e);
                        }
                    });
                }
            }
        }
    }

    /// Calculate task progress based on resource allocation
    fn calculate_progress(&self, running: &RunningTask) -> f64 {
        let sens = &running.task.resource_sensitivity;

        // Base progress (if given minimum resources)
        let base_progress = 1.0 / running.task.duration_estimate.num_seconds() as f64
            * self.config.timestep_duration.num_seconds() as f64;

        // CPU scaling - more CPUs = faster execution
        let cpu_ratio = running.allocated_cpu as f32 / running.task.min_cpu.max(1) as f32;
        let cpu_factor = 1.0 + sens.cpu_scaling * (cpu_ratio - 1.0).max(0.0);

        // GPU scaling
        let gpu_factor = if running.allocated_gpu_cores > 0 {
            let gpu_ratio = running.allocated_gpu_cores as f32
                / running.task.min_gpu_core.max(1) as f32;
            1.0 + sens.gpu_scaling * (gpu_ratio - 1.0).max(0.0)
        } else {
            1.0
        };

        // Memory scaling (diminishing returns via sqrt)
        let mem_ratio = running.allocated_memory_mb as f32
            / running.task.min_memory_mb.max(1) as f32;
        let mem_factor = 1.0 + sens.memory_scaling * (mem_ratio.sqrt() - 1.0).max(0.0);

        base_progress * cpu_factor as f64 * gpu_factor as f64 * mem_factor as f64
    }

    /// Generate tasks according to Poisson arrival process
    fn generate_tasks(&mut self, time_delta: Duration) {
        let seconds = time_delta.num_seconds() as f64;
        let lambda = self.config.task_arrival_rate * seconds;

        let poisson = Poisson::new(lambda).unwrap();
        let num_tasks = poisson.sample(&mut self.rng) as usize;

        for _ in 0..num_tasks {
            let task = self.sample_task();
            self.global_task_queue.push_back(task);
        }
    }

    /// Sample a random ML task with realistic characteristics
    fn sample_task(&mut self) -> MLTask {
        let task_types = [
            MLTaskType::ImageClassificationTraining,
            MLTaskType::ObjectDetectionInference,
            MLTaskType::NLPTransformer,
            MLTaskType::ReinforcementLearning,
        ];

        let task_type = task_types[self.rng.gen_range(0..task_types.len())];

        // Task characteristics vary by type
        let (min_cpu, min_gpu_core, duration_base, sensitivity) = match task_type {
            MLTaskType::ImageClassificationTraining => (
                4,
                50,
                120,
                ResourceSensitivity {
                    cpu_scaling: 0.3,
                    gpu_scaling: 0.8,
                    memory_scaling: 0.2,
                },
            ),
            MLTaskType::ObjectDetectionInference => (
                2,
                25,
                30,
                ResourceSensitivity {
                    cpu_scaling: 0.2,
                    gpu_scaling: 0.6,
                    memory_scaling: 0.1,
                },
            ),
            MLTaskType::NLPTransformer => (
                8,
                80,
                180,
                ResourceSensitivity {
                    cpu_scaling: 0.4,
                    gpu_scaling: 0.9,
                    memory_scaling: 0.5,
                },
            ),
            MLTaskType::ReinforcementLearning => (
                4,
                40,
                200,
                ResourceSensitivity {
                    cpu_scaling: 0.7,
                    gpu_scaling: 0.7,
                    memory_scaling: 0.3,
                },
            ),
            _ => (
                2,
                20,
                60,
                ResourceSensitivity {
                    cpu_scaling: 0.5,
                    gpu_scaling: 0.5,
                    memory_scaling: 0.3,
                },
            ),
        };

        MLTask {
            id: uuid::Uuid::new_v4().to_string(),
            task_type,
            min_cpu,
            min_gpu_core,
            min_gpu_memory: 2000, // 2GB minimum
            min_memory_mb: 4096,
            arrival_time: self.current_time,
            eligible_clusters: (0..self.config.num_clusters).collect(),
            duration_estimate: Duration::seconds(duration_base),
            resource_sensitivity: sensitivity,
        }
    }

    /// Distribute tasks from global queue to cluster queues
    fn distribute_new_tasks(&mut self) {
        while let Some(task) = self.global_task_queue.pop_front() {
            // Send to all eligible clusters
            for &cluster_id in &task.eligible_clusters {
                if cluster_id < self.clusters.len()
                    && self.cluster_health[cluster_id].is_healthy {
                    self.clusters[cluster_id]
                        .pending_tasks
                        .push_back(task.clone());
                }
            }
        }
    }

    /// Calculate rewards (negative queue length encourages clearing queue)
    fn calculate_rewards(&self) -> Vec<f64> {
        self.clusters
            .iter()
            .map(|cluster| {
                let queue_penalty = -(cluster.pending_tasks.len() as f64);
                let running_penalty = -(cluster.running_tasks.len() as f64) * 0.5;
                let utilization_reward = self.calculate_utilization_reward(cluster);

                queue_penalty + running_penalty + utilization_reward
            })
            .collect()
    }

    /// Reward for good resource utilization
    fn calculate_utilization_reward(&self, cluster: &EdgeCluster) -> f64 {
        let cpu_util = 1.0 - (cluster.total_cpu_available() as f64
            / (cluster.cpus.len() * 100) as f64);

        let gpu_util: f64 = cluster.gpus.iter()
            .map(|g| 1.0 - (g.core_available as f64 / g.core_total as f64))
            .sum::<f64>() / cluster.gpus.len().max(1) as f64;

        // Reward 50-80% utilization, penalize over-utilization
        let target_util = 0.65;
        let cpu_reward = -((cpu_util - target_util).abs() * 5.0);
        let gpu_reward = -((gpu_util - target_util).abs() * 5.0);

        cpu_reward + gpu_reward
    }

    /// Check if episode is done
    fn is_done(&self) -> bool {
        self.timestep >= self.config.episode_length
    }

    /// Get info dictionary for logging
    fn get_info(&self) -> HashMap<String, f64> {
        let mut info = HashMap::new();

        info.insert("timestep".to_string(), self.timestep as f64);
        info.insert("completed_tasks".to_string(), self.completed_tasks.len() as f64);
        info.insert("failed_tasks".to_string(), self.failed_tasks.len() as f64);

        let total_queue: usize = self.clusters.iter()
            .map(|c| c.pending_tasks.len())
            .sum();
        info.insert("total_queue".to_string(), total_queue as f64);

        let total_running: usize = self.clusters.iter()
            .map(|c| c.running_tasks.len())
            .sum();
        info.insert("total_running".to_string(), total_running as f64);

        let healthy_clusters = self.cluster_health.iter()
            .filter(|h| h.is_healthy)
            .count();
        info.insert("healthy_clusters".to_string(), healthy_clusters as f64);

        // Average utilization
        let avg_cpu_util: f64 = self.clusters.iter()
            .map(|c| 1.0 - (c.total_cpu_available() as f64
                / (c.cpus.len() * 100) as f64))
            .sum::<f64>() / self.clusters.len() as f64;
        info.insert("avg_cpu_utilization".to_string(), avg_cpu_util);

        let avg_gpu_util: f64 = self.clusters.iter()
            .flat_map(|c| &c.gpus)
            .map(|g| 1.0 - (g.core_available as f64 / g.core_total as f64))
            .sum::<f64>() / (self.clusters.len() * self.config.gpus_per_cluster).max(1) as f64;
        info.insert("avg_gpu_utilization".to_string(), avg_gpu_util);

        info
    }

    /// Get cluster health status
    pub fn get_cluster_health(&self, cluster_id: usize) -> Option<&ClusterHealth> {
        self.cluster_health.get(cluster_id)
    }

    /// Get recent failed tasks
    pub fn get_recent_failures(&self, count: usize) -> Vec<&FailedTask> {
        self.failed_tasks.iter().rev().take(count).collect()
    }
}

async fn log_task_start(
    db: &DatabaseManager,
    task: &MLTask,
    cluster_id: usize,
) -> Result<()> {
    use database::low::JobRecord;

    let job = JobRecord {
        id: 0,
        job_id: task.id.clone(),
        task_type: format!("{:?}", task.task_type),
        cluster_id: cluster_id as i32,
        arrival_time: task.arrival_time,
        start_time: Some(Utc::now()),
        completion_time: None,
        status: "running".to_string(),
        cpu_allocated: task.min_cpu as i32,
        gpu_id: None,
        gpu_cores_allocated: task.min_gpu_core as i32,
        memory_mb_allocated: task.min_memory_mb as i64,
        priority: 1.0,
        queue_wait_time_sec: Some(
            (Utc::now() - task.arrival_time).num_seconds() as f32
        ),
        execution_time_sec: None,
    };

    db.record_job_start(&job).await?;
    Ok(())
}

async fn log_task_completion(
    db: &DatabaseManager,
    task_id: &str,
    cluster_id: usize,
    completion_time: DateTime<Utc>,
    execution_time_sec: f32,
) -> Result<()> {
    db.record_job_completion(
        task_id,
        cluster_id,
        completion_time,
        execution_time_sec,
    ).await?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_environment_creation() {
        let config = EnvironmentConfig::default();
        let db = Arc::new(DatabaseManager::new(
            "postgres://test:test@localhost/test",
            "valkey://localhost:6379"
        ).await.unwrap());

        let env = EdgeMLEnv::new(config.clone(), db);

        assert_eq!(env.clusters.len(), config.num_clusters);
        assert_eq!(env.timestep, 0);
        assert!(env.global_task_queue.is_empty());
    }

    #[tokio::test]
    async fn test_resource_validation() {
        let config = EnvironmentConfig::default();
        let db = Arc::new(DatabaseManager::new(
            "postgres://test:test@localhost/test",
            "valkey://localhost:6379"
        ).await.unwrap());

        let env = EdgeMLEnv::new(config, db);

        // Valid action
        let valid_action = AgentAction {
            selected_task: Some("test".to_string()),
            allocated_cpu: 4,
            allocated_gpu_id: Some(0),
            allocated_gpu_cores: 50,
            allocated_memory_mb: 4096,
        };
        assert!(env.validate_resources(0, &valid_action));

        // Invalid - too much CPU
        let invalid_action = AgentAction {
            selected_task: Some("test".to_string()),
            allocated_cpu: 10000,
            allocated_gpu_id: None,
            allocated_gpu_cores: 0,
            allocated_memory_mb: 1000,
        };
        assert!(!env.validate_resources(0, &invalid_action));
    }

    #[test]
    fn test_task_generation() {
        let config = EnvironmentConfig::default();
        let db = Arc::new(DatabaseManager::new(
            "postgres://test:test@localhost/test",
            "valkey://localhost:6379"
        ).await.unwrap());

        let mut env = EdgeMLEnv::new(config, db);

        env.generate_tasks(Duration::seconds(10));
        assert!(!env.global_task_queue.is_empty());
    }
}
