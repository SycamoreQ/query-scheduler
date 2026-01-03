use petgraph::{DiGraph , NodeIndex}; 
use tch::{nn, nn::Module, Device, Kind, Tensor};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum NodeType {
    Cluster,
    PendingTask,
    RunningTask,
    Shortcut,
}

#[derive(Debug, Clone)]
pub struct ClusterFeatures {
    pub cluster_id: usize,
    pub cpu_available: f32,
    pub gpu_available: Vec<f32>, 
    pub memory_available: f32,
    pub queue_length: f32,
}

#[derive(Debug, Clone)]
pub struct TaskFeatures {
    pub task_id: String,
    pub task_type: Vec<f32>,  
    pub cpu_required: f32,
    pub gpu_core_required: f32,
    pub gpu_mem_required: f32,
    pub memory_required: f32,
    pub elapsed_time: f32,  
}

#[derive(Debug, Clone)]
pub enum NodeFeatures {
    Cluster(ClusterFeatures),
    PendingTask(TaskFeatures),
    RunningTask(TaskFeatures),
    Shortcut,  
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeType {
    TaskToCluster,   
    ClusterToShortcut,
    ShortcutToCluster, 
}

pub struct HANGraph {
    pub graph: DiGraph<NodeFeatures, EdgeType>,
    pub node_type_map: HashMap<NodeIndex, NodeType>,
    pub cluster_indices: Vec<NodeIndex>,
    pub pending_indices: Vec<NodeIndex>,
    pub running_indices: Vec<NodeIndex>,
    pub shortcut_idx: Option<NodeIndex>,
}

impl HANGraph{ 
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_type_map: HashMap::new(),
            cluster_indices: Vec::new(),
            pending_indices: Vec::new(),
            running_indices: Vec::new(),
            shortcut_idx: None,
        }
    }
    
    pub fn add_cluster(&mut self , features: &ClusterFeatures) -> NodeIndex{
        let idx = self.graph.add_node(NodeFeatures::Cluster(features)); 
        self.node_type_map.insert(idx , NodeType::Cluster);
        self.cluster_indices.push(idx);
        idx
    }
    
    pub fn add_pending_task(&mut self, features: TaskFeatures) -> NodeIndex {
        let idx = self.graph.add_node(NodeFeatures::PendingTask(features));
        self.node_type_map.insert(idx, NodeType::PendingTask);
        self.pending_indices.push(idx);
        idx
    }

    pub fn add_running_task(&mut self, features: TaskFeatures) -> NodeIndex {
        let idx = self.graph.add_node(NodeFeatures::RunningTask(features));
        self.node_type_map.insert(idx, NodeType::RunningTask);
        self.running_indices.push(idx);
        idx
    }
    
    pub fn add_shortcut(&mut self) -> NodeIndex { 
        if self.shortcut_idx.is_none(){
            let idx = self.graph.add_node(NodeFeatures::Shortcut);
            self.node_type_map.insert(idx, NodeType::Shortcut);
            self.shortcut_idx = Some(idx);
            
            for &cluster_idx in &self.cluster_indices{ 
                self.graph.add_edge(cluster_idx , idx , EdgeType::ShortCuttoCluster);
                self.graph.add_edge(idx , cluster_idx , EdgeType::ClustertoShortCut)
            }
            idx
        }
        else{
            self.shortcut_idx.unwrap()
        }
    }
    
    pub fn connect_task_to_cluster(&mut self, task_idx: NodeIndex, cluster_idx: NodeIndex) {
        self.graph.add_edge(task_idx, cluster_idx, EdgeType::TaskToCluster);
    }
    
    pub fn to_tensor(&self , device: Device) -> GraphTensor{
        let num_nodes = self.graph.node_count();
        let cluster_dim = self.get_cluster_dim();
        let task_dim = self.get_task_dim();
        let max_dim = cluster_dim.max(task_dim);
        
        
        let mut features = vec![vec![0.0; max_dim]; num_nodes];
        let mut node_types = vec![0i64; num_nodes];
        
        for node_idx in self.graph.node_indices(){
            let idx = node_idx.index();
            let node_data = &self.graph[node_idx];
            let node_type = &self.node_type_map[&node_idx];
            
            node_types[idx] = match node_type {
                NodeType::Cluster => 0,
                NodeType::PendingTask => 1,
                NodeType::RunningTask => 2,
                NodeType::Shortcut => 3,
            };
            
            match node_data {
                NodeFeatures::Cluster(cf) => {
                    features[idx] = self.cluster_to_vec(cf, max_dim);
                }
                NodeFeatures::PendingTask(tf) => {
                    features[idx] = self.task_to_vec(tf, max_dim, false);
                }
                NodeFeatures::RunningTask(tf) => {
                    features[idx] = self.task_to_vec(tf, max_dim, true);
                }
                NodeFeatures::Shortcut => {
                    // Shortcut has zero features
                    features[idx] = vec![0.0; max_dim];
                }
            }
        }
        
        let mut edge_index = Vec::new();
        let mut edge_types = Vec::new();
        
        for edge in self.graph.edge_references() {
            edge_index.push(edge.source().index() as i64);
            edge_index.push(edge.target().index() as i64);
            
            let edge_type_id = match edge.weight() {
                EdgeType::TaskToCluster => 0,
                EdgeType::ClusterToShortcut => 1,
                EdgeType::ShortcutToCluster => 2,
            };
            edge_types.push(edge_type_id);
        }
        
        let feature_tensor = Tensor::of_slice2(&features)
            .to_kind(Kind::Float)
            .to_device(device);
        
        let node_type_tensor = Tensor::of_slice(&node_types)
            .to_device(device);
        
        let edge_index_tensor = if edge_index.is_empty() {
            Tensor::zeros(&[2, 0], (Kind::Int64, device))
        } else {
            Tensor::of_slice(&edge_index)
                .reshape(&[2, edge_index.len() as i64 / 2])
                .to_device(device)
        };
        
        let edge_type_tensor = Tensor::of_slice(&edge_types)
            .to_device(device);
        
        GraphTensors {
            node_features: feature_tensor,
            node_types: node_type_tensor,
            edge_index: edge_index_tensor,
            edge_types: edge_type_tensor,
            num_nodes: num_nodes as i64,
            cluster_indices: self.cluster_indices.iter().map(|i| i.index() as i64).collect(),
            pending_indices: self.pending_indices.iter().map(|i| i.index() as i64).collect(),
            running_indices: self.running_indices.iter().map(|i| i.index() as i64).collect(),
        }
    }
    
    fn get_cluster_dim(&self) -> usize{
        4 + 8 + 2
    }
    
    fn get_task_features(&self) -> usize{
        10 + 5
    }
    
    fn cluster_to_vec(&self, cf: &ClusterFeatures, target_dim: usize) -> Vec<f32> {
        let mut vec = Vec::new();
        vec.push(cf.cpu_available);
        
        // Flatten GPU features (pad to 8 GPUs)
        for i in 0..8 {
            if i < cf.gpu_available.len() / 2 {
                vec.push(cf.gpu_available[i * 2]);     // core
                vec.push(cf.gpu_available[i * 2 + 1]); // memory
            } else {
                vec.push(0.0);
                vec.push(0.0);
            }
        }
        
        vec.push(cf.memory_available);
        vec.push(cf.queue_length);
        
        // Pad to target dimension
        while vec.len() < target_dim {
            vec.push(0.0);
        }
        vec
    }
    
    fn task_to_vec(&self, tf: &TaskFeatures, target_dim: usize, is_running: bool) -> Vec<f32> {
        let mut vec = tf.task_type.clone();
        vec.push(tf.cpu_required);
        vec.push(tf.gpu_core_required);
        vec.push(tf.gpu_mem_required);
        vec.push(tf.memory_required);
        vec.push(if is_running { tf.elapsed_time } else { 0.0 });
        
        // Pad to target dimension
        while vec.len() < target_dim {
            vec.push(0.0);
        }
        vec
    }
}

pub struct GraphTensors {
    pub node_features: Tensor,
    pub node_types: Tensor,
    pub edge_index: Tensor,
    pub edge_types: Tensor,
    pub num_nodes: i64,
    pub cluster_indices: Vec<i64>,
    pub pending_indices: Vec<i64>,
    pub running_indices: Vec<i64>,
}
    
    
    
    
}