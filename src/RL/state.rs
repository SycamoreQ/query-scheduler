use serde::{Deserialize , Serialize};
use ndarray::{Array1, Array2};
use tch::{nn, Device, Tensor, Kind};

#[derive(Debug , Clone , Deserialize , Serialize)]
pub struct SchedulerState{ 
    pub query_complexity: f32 , 
    pub data_volume_score: f32 ,
    pub computation_score: f32 ,
    pub graph_traversal_score: f32 ,
    pub string_ops_score: f32,
    pub join_ops_score: f32,
    pub priority: f32,  
    pub estimated_memory_mb: f32,
    pub estimated_duration_sec: f32,
    pub estimated_core_usage: f32,
    pub gpu_states: Vec<GPUState>,
    pub queue_length: f32,
    pub avg_wait_time: f32,
    pub system_throughput: f32,
}

pub struct GPUState{ 
    pub gpu_id: String, 
    pub core_utilization: String ,
    pub memory_utilization: String ,
    pub core_available: f32,    
    pub memory_available_mb: f32, 
    pub active_jobs: f32,          
    pub avg_job_completion_time: f32,
    pub recent_throughput: f32, 
}

impl SchedulerState {
    pub fn to_tensor(&self, device: Device) -> Tensor {
        let mut features = vec![
            self.query_complexity,
            self.data_volume_score,
            self.computation_score,
            self.graph_traversal_score,
            self.string_ops_score,
            self.join_ops_score,
            self.priority,
            self.estimated_memory_mb / 10000.0, 
            self.estimated_duration_sec / 60.0,  
            self.estimated_core_usage / 100.0,
            self.queue_length / 100.0,
            self.avg_wait_time / 300.0,  
            self.system_throughput / 10.0,
        ];
        
        for gpu_state in &self.gpu_states {
            features.extend_from_slice(&[
                gpu_state.core_utilization,
                gpu_state.memory_utilization,
                gpu_state.core_available / 100.0,
                gpu_state.memory_available_mb / 10000.0,
                gpu_state.active_jobs / 10.0,
                gpu_state.avg_job_completion_time / 60.0,
                gpu_state.recent_throughput / 5.0,
            ]);
        }
        
        Tensor::of_slice(&features).to_device(device)
    }
    
    pub fn state_dim(num_gpus: usize) -> i64 {
        13 + (num_gpus * 7) as i64  // Query features + GPU features per GPU
    }
}