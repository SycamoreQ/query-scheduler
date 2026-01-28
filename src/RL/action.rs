use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use tch::{nn, Device, Kind, Tensor};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingAction {
    pub cpu_id: String,
    pub gpu_id: String,
    pub resource_allocation: ResourceAllocation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub core_percent: f32,
    pub memory: u64,
    pub priority_boost: f32,
}

impl SchedulingAction {
    pub fn from_tensor(action_tensor: &Tensor, num_gpus: usize) -> Self {
        let action_vec: Vec<f32> = action_tensor.try_into().unwrap();

        let gpu_id = action_vec[..num_gpus]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let core_percent = (action_vec[num_gpus].sigmoid() * 100.0).clamp(10.0, 100.0);
        let memory_mb = ((action_vec[num_gpus + 1].sigmoid() * 20000.0) as u64).max(512);
        let priority_boost = action_vec[num_gpus + 2].sigmoid();

        Self {
            gpu_id,
            resource_allocation: ResourceAllocation {
                core_percent,
                memory_mb,
                priority_boost,
            },
        }
    }

    pub fn to_tensor(&self, action: SchedulingAction, device: Device) -> Tensor {
        let mut features = vec![action.gpu_id, action.resource_allocation];

        Tensor::of_slice(&features).to_device(device)
    }
}
