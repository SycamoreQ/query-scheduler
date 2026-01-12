use chrono::{DateTime, Duration, Utc};
use rand::{distributions::Distribution, Rng};
use rand_distr::Poisson;
use std::collections::VecDeque;

pub struct EdgeMLEnv {
    pub clusters: Vec<EdgeCluster>,
    pub global_task_queue: VecDeque<MLTask>,
    pub current_time: DateTime<Utc>,
    pub timestep: usize,
    pub completed_tasks: Vec<CompletedTask>,
    pub config: EnvironmentConfig,
    rng: rand::rngs::ThreadRng,
}

#[derive(Clone, Debug)]
pub struct EdgeCluster {
    pub id: usize,
    pub cpu_cores: ResourceState,
    pub gpus: Vec<GPU>,
    pub memory_mb: ResourceState,
    pub bandwidth_mbps: f64,
    pub pending_tasks: VecDeque<MLTask>,
    pub running_tasks: Vec<RunningTask>,
}

#[derive(Debug, Clone, Default)]
pub struct ResourceState {
    pub total: usize,
    pub available: usize,
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
            memory_available_mb: 16000, // 16GB
            memory_total_mb: 16000,
        }
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

#[derive(Debug, Clone, Copy)]
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

/// How much the task benefits from additional resources
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
    pub task_id: String,
    pub cluster_id: usize,
    pub arrival_time: DateTime<Utc>,
    pub start_time: DateTime<Utc>,
    pub completion_time: DateTime<Utc>,
}

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
    pub fn new(config: EnvironmentConfig) -> Self {
        let clusters = (0..config.num_clusters)
            .map(|id| EdgeCluster::new(id, &config))
            .collect();

        Self {
            clusters,
            global_task_queue: VecDeque::new(),
            current_time: Utc::now(),
            timestep: 0,
            completed_tasks: Vec::new(),
            config,
            rng: rand::thread_rng(),
        }
    }

    /// Reset environment for new episode
    pub fn reset(&mut self) -> HANGraph {
        for cluster in &mut self.clusters {
            cluster.reset();
        }

        self.global_task_queue.clear();
        self.completed_tasks.clear();
        self.current_time = Utc::now();
        self.timestep = 0;

        // Generate initial batch of tasks
        self.generate_tasks(self.config.timestep_duration * 6);
        self.distribute_new_tasks();

        self.build_observation()
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

    /// Sample a random ML task
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
                self.clusters[cluster_id]
                    .pending_tasks
                    .push_back(task.clone());
            }
        }
    }

    /// Execute one environment step
    pub fn step(&mut self, actions: Vec<AgentAction>) -> StepResult {
        // 1. Apply agent actions (schedule tasks)
        self.apply_actions(actions);

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

        StepResult {
            observation,
            rewards,
            done,
            info: self.get_info(),
        }
    }

    fn apply_actions(&mut self, actions: Vec<AgentAction>) {
        for (cluster_id, action) in actions.iter().enumerate() {
            if cluster_id >= self.clusters.len() {
                continue;
            }

            if let Some(ref task_id) = action.selected_task {
                let cluster = &mut self.clusters[cluster_id];

                // Find and remove task from pending queue
                if let Some(pos) = cluster.pending_tasks.iter().position(|t| &t.id == task_id) {
                    let task = cluster.pending_tasks.remove(pos).unwrap();

                    // Create running task with allocated resources
                    let running = RunningTask {
                        task,
                        allocated_cpu: action.allocated_cpu,
                        allocated_gpu_id: action.allocated_gpu_id,
                        allocated_gpu_cores: action.allocated_gpu_cores,
                        allocated_memory_mb: action.allocated_memory_mb,
                        start_time: self.current_time,
                        progress: 0.0,
                    };

                    // Allocate resources
                    cluster.cpu_cores.available -= action.allocated_cpu;
                    if let Some(gpu_id) = action.allocated_gpu_id {
                        if gpu_id < cluster.gpus.len() {
                            cluster.gpus[gpu_id].core_available -= action.allocated_gpu_cores;
                            cluster.gpus[gpu_id].memory_available_mb -= running.task.min_gpu_memory;
                        }
                    }
                    cluster.memory_mb.available -= action.allocated_memory_mb;

                    cluster.running_tasks.push(running);
                }
            }
        }
    }

    fn execute_tasks(&mut self) {
        for cluster in &mut self.clusters {
            let mut completed_indices = Vec::new();

            for (idx, running) in cluster.running_tasks.iter_mut().enumerate() {
                // Calculate progress based on allocated resources
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
                cluster.cpu_cores.available += completed.allocated_cpu;
                if let Some(gpu_id) = completed.allocated_gpu_id {
                    if gpu_id < cluster.gpus.len() {
                        cluster.gpus[gpu_id].core_available += completed.allocated_gpu_cores;
                        cluster.gpus[gpu_id].memory_available_mb += completed.task.min_gpu_memory;
                    }
                }
                cluster.memory_mb.available += completed.allocated_memory_mb;

                self.completed_tasks.push(CompletedTask {
                    task_id: completed.task.id,
                    cluster_id: cluster.id,
                    arrival_time: completed.task.arrival_time,
                    start_time: completed.start_time,
                    completion_time: self.current_time,
                });
            }
        }
    }

    /// Calculate task progress based on resource allocation
    fn calculate_progress(&self, running: &RunningTask) -> f64 {
        let sens = &running.task.resource_sensitivity;

        // Base progress (if given minimum resources)
        let base_progress = 1.0 / running.task.duration_estimate.num_seconds() as f64
            * self.config.timestep_duration.num_seconds() as f64;

        // CPU scaling
        let cpu_factor = 1.0
            + sens.cpu_scaling * (running.allocated_cpu as f32 / running.task.min_cpu as f32 - 1.0);

        // GPU scaling
        let gpu_factor = if running.allocated_gpu_cores > 0 {
            1.0 + sens.gpu_scaling
                * (running.allocated_gpu_cores as f32 / running.task.min_gpu_core as f32 - 1.0)
        } else {
            1.0
        };

        // Memory scaling (diminishing returns)
        let mem_factor = 1.0
            + sens.memory_scaling
                * ((running.allocated_memory_mb as f32 / running.task.min_memory_mb as f32).sqrt()
                    - 1.0);

        base_progress * cpu_factor as f64 * gpu_factor as f64 * mem_factor as f64
    }

    fn calculate_rewards(&self) -> Vec<f64> {
        // Reward = negative of current queue + running tasks
        self.clusters
            .iter()
            .map(|cluster| {
                let load = cluster.pending_tasks.len() + cluster.running_tasks.len();
                -(load as f64)
            })
            .collect()
    }

    fn build_observation(&self) -> HANGraph {
        use crate::utils::graph::*;

        let mut graph = HANGraph::new();

        // Add cluster nodes
        let cluster_indices: Vec<_> = self
            .clusters
            .iter()
            .map(|c| {
                graph.add_cluster(ClusterFeatures {
                    cluster_id: c.id,
                    cpu_available: c.cpu_cores.available as f32,
                    gpu_available: c
                        .gpus
                        .iter()
                        .flat_map(|g| vec![g.core_available as f32, g.memory_available_mb as f32])
                        .collect(),
                    memory_available: c.memory_mb.available as f32,
                    queue_length: c.pending_tasks.len() as f32,
                })
            })
            .collect();

        for (cluster_idx, cluster) in self.clusters.iter().enumerate() {
            for (i, task) in cluster.pending_tasks.iter().enumerate() {
                if i >= self.config.max_pending_set_size {
                    break;
                }

                let task_idx = graph.add_pending_task(task.to_features());
                graph.connect_task_to_cluster(task_idx, cluster_indices[cluster_idx]);
            }

            // Running tasks
            for running in &cluster.running_tasks {
                let elapsed = (self.current_time - running.start_time).num_seconds() as f32;
                let mut features = running.task.to_features();
                features.elapsed_time = elapsed;

                let task_idx = graph.add_running_task(features);
                graph.connect_task_to_cluster(task_idx, cluster_indices[cluster_idx]);
            }
        }

        graph.add_shortcut();

        graph
    }

    fn is_done(&self) -> bool {
        self.timestep >= self.config.episode_length
    }

    fn get_info(&self) -> std::collections::HashMap<String, f64> {
        use std::collections::HashMap;

        let mut info = HashMap::new();
        info.insert("timestep".to_string(), self.timestep as f64);
        info.insert(
            "completed_tasks".to_string(),
            self.completed_tasks.len() as f64,
        );
        info.insert(
            "total_queue".to_string(),
            self.clusters
                .iter()
                .map(|c| c.pending_tasks.len())
                .sum::<usize>() as f64,
        );
        info
    }
}

impl EdgeCluster {
    fn new(id: usize, config: &EnvironmentConfig) -> Self {
        let gpus = (0..config.gpus_per_cluster).map(GPU::new).collect();

        Self {
            id,
            cpu_cores: ResourceState {
                total: config.cpu_cores_per_cluster,
                available: config.cpu_cores_per_cluster,
            },
            gpus,
            memory_mb: ResourceState {
                total: config.memory_mb_per_cluster,
                available: config.memory_mb_per_cluster,
            },
            bandwidth_mbps: 1000.0,
            pending_tasks: VecDeque::new(),
            running_tasks: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.cpu_cores.available = self.cpu_cores.total;
        self.memory_mb.available = self.memory_mb.total;
        for gpu in &mut self.gpus {
            gpu.core_available = gpu.core_total;
            gpu.memory_available_mb = gpu.memory_total_mb;
        }
        self.pending_tasks.clear();
        self.running_tasks.clear();
    }
}

impl MLTask {
    fn to_features(&self) -> TaskFeatures {
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

impl MLTaskType {
    fn to_one_hot(&self) -> Vec<f32> {
        let mut vec = vec![0.0; 10];
        let idx = match self {
            MLTaskType::ImageClassificationTraining => 0,
            MLTaskType::ObjectDetectionInference => 1,
            MLTaskType::NLPTransformer => 2,
            MLTaskType::ReinforcementLearning => 3,
            MLTaskType::LinearRegression => 4,
            MLTaskType::LogisticRegression => 5,
            MLTaskType::GraphAttention => 6,
            MLTaskType::GraphNeuralNetwork => 7,
            MLTaskType::RetrievalAugmentedGeneration => 8,
            MLTaskType::SupportVectorMachine => 9,
        };
        vec[idx] = 1.0;
        vec
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

pub struct StepResult {
    pub observation: HANGraph,
    pub rewards: Vec<f64>,
    pub done: bool,
    pub info: std::collections::HashMap<String, f64>,
}
