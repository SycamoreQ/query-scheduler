use chrono::{DateTime , Duration , Utc};
use std::collections::{VecDeque, HashMap};
use structures::_graph::{HANGraph, ClusterFeatures, TaskFeatures};

pub struct EdgeMLEnv{ 
    pub clusters: Vec<EdgeCluster>,
    pub global_task_queue: VecDeque<MLTask>,
    pub current_time: DateTime<Utc>,
    pub timestep: usize,
    pub completed_tasks: Vec<CompletedTask>,
    pub config: EnvironmentConfig,
}

#[derive(Default , Clone , Debug)]
pub struct EdgeCluster { 
    pub id: usize , 
    pub cpu_cores: ResourceState, 
    pub gpus: Vec<GPU>,
    pub memory_mb: ResourceState,
    pub bandwidth_mbps: f64,
    pub pending_tasks: VecDeque<Task>,
    pub running_tasks: Vec<Task>,
}

#[derive(Debug , Clone , Default)]
pub struct ResourceState{ 
    pub total: usize,
    pub available: usize
}

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

pub struct RunningTask {
    pub task: MLTask,
    pub allocated_resources: AllocatedResources,
    pub start_time: DateTime<Utc>,
    pub progress: f64,  // 0.0 to 1.0
}

pub struct AllocatedResources {
    pub cpu_cores: usize,
    pub gpu_indices: Vec<usize>,
    pub gpu_cores: Vec<usize>,
    pub gpu_memory: Vec<usize>,
    pub memory_mb: usize,
}

pub struct CompletedTask {
    pub task_id: String,
    pub cluster_id: usize,
    pub arrival_time: DateTime<Utc>,
    pub start_time: DateTime<Utc>,
    pub completion_time: DateTime<Utc>,
}

impl EdgeMLEnvironment {
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
        }
    }
    
    pub fn generate_tasks(&mut self, time_delta: Duration) {
        let num_tasks = self.poisson_sample(self.config.task_arrival_rate, time_delta);
        
        for _ in 0..num_tasks {
            let task = self.sample_task();
            self.global_task_queue.push_back(task);
        }
    }
    
    pub fn step(&mut self , actions: Vec<AgentAction>) -> StepResult{
        //apply agent action
        self.apply_action(actions);
        
        //execute running tasks
        self.execute_tasks();
        
        //distribute new tasks to clusters
        self.distribute_tasks();
        
        //calculate reward
        let reward = self.calculate_rewards();
        
        //build HAN graph for next observation
        let observation = self.build_observation();
        
        //check if episode is done
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

    fn apply_action(&mut self , actions: Vec<AgentAction>) {
        for (cluster_id) 
    }
    
}



