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
        self.distribute_new_tasks();
        
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
        for (cluster_id , action) in actions.iter().enumerate(){
            if let Some(task_selection) = &actions.selected_task{
                let cluster = &clusters.remove_task_by_id(task_selection);
                let running_task = RunningTask{
                    task , 
                    allocated_resources : action.allocated_resources.clone(),
                    start_time: self.current_time,
                    progress: 0
                };
                
                cluster.allocate_resources = &action.resource_allocation; 
                cluster.running_tasks.push(running_task);
            } 
        } 
    }
    
    fn execute_task(&mut self) {
        for cluster in &mut self.clusters { 
            let mut completed_indices = Vec::new();
            
            for (idx , running_task) in self.running_tasks.iter_mut().enumerate(){
                let progress_delta = self.calculate_progress_delta(running_task);
                running_task.progress += progress_delta;
                
                if running_task >= 1.0{ 
                    completed_indices.push(idx);
                }
                
                for idx in self.completed_tasks.iter_mut().enumerate(){
                    let completed = self.completed_tasks.remove(*idx);
                    cluster.free_resources(&completed.allocated_resources); 
                    
                    self.completed.push(CompletedTask{
                        task_id: idx,
                        cluster_id: cluster.id,
                        arrival_time: completed.task.arrival_time,
                        start_time: completed.start_time, 
                        completion_time: self.current_time,
                    });
                }
            }
        }
    }
    
    fn distribute_new_task(&mut self){
        
    }
    
    fn calculate_rewards(&self) -> Vec<f64> {
        self.clusters.iter().map(|cluster| {
            let num_tasks = cluster.pending_queue.len() + cluster.running_tasks.len();
            -(num_tasks as f64)
        }).collect()
    }
    
    fn build_observation(&self) -> HANGraph {
        let mut graph = HANgraph::new();
        
        let cluster_indices: Vec<_> = self.cluster.iter().
            map(|c|{
                graph.add_cluster(ClusterFeatures{
                    cluster_id: c.id, 
                    cpu_available: c.cpu_available as f32, 
                    gpu_available: c.gpus.iter().
                        flat_map(|g| vec![g.core_available as f32, g.memory_available as f32]).collect(),
                    memory_available: c.memory_available_mb as f32,
                    queue_length: c.pending_queue.len() as f32,
                })
            }).collect();
        
        for (cluster_idx , cluster) in self.clusters.iter().enumerate(){
            for (i , task) in self.cluster.pending_queue.iter().enumerate(){
                if i>= self.config.max_pending_set_size{ 
                    break;
                }
                
                let task_idx = graph.add_pending_task(task.to_features());
                graph.connect_task_to_cluster(task_idx, cluster_indices[cluster_idx]);
            }
        }
        
        for (cluster_idx, cluster) in self.clusters.iter().enumerate() {
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
    
    pub fn reset(&mut self) -> HANGraph {
        for cluster in &mut self.clusters {
            cluster.reset();
        }
        
        self.global_task_queue.clear();
        self.completed_tasks.clear();
        self.current_time = Utc::now();
        self.timestep = 0;
        
        // Generate initial task batch
        self.generate_tasks(Duration::seconds(self.config.warmup_seconds));
        self.distribute_new_tasks();
        
        self.build_observation()
    }
}

pub struct AgentAction {
    pub selected_task: Option<String>,  // Task ID
    pub resource_allocation: AllocatedResources,
}

pub struct StepResult {
    pub observation: HANGraph,
    pub rewards: Vec<f64>,
    pub done: bool,
    pub info: HashMap<String, f64>,
}

pub struct EnvironmentConfig {
    pub num_clusters: usize,
    pub task_arrival_rate: f64,  // Tasks per second
    pub timestep_duration: Duration,
    pub max_pending_set_size: usize,
    pub episode_length: usize,
    pub warmup_seconds: i64,
}





