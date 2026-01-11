use tch::{nn, nn::Module, Device, Kind, Tensor};
use crate::han::HANOutput;


pub struct TapFingerActor  { 
    pub task_selection : nn::Sequential,
    pub pointer_query: nn::Linear, 
    pub pointer_key: nn::Linear,
    pub cpu_allocator: nn::Sequential, 
    pub gpu_allocator: nn::Sequential, 
    pub memory_allocator: nn::Sequential, 
    pub hidden_dim: i64,
    pub resource_bins: i64,
}

impl TapFingerActor{ 
    pub fn new(vs: &nn::Path, hidden_dim: i64, num_resource_bins: i64) -> Self {
        let task_encoder = nn::seq()
            .add(nn::linear(vs / "task_enc_1", hidden_dim, hidden_dim, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "task_enc_2", hidden_dim, hidden_dim, Default::default()));
        
        let pointer_query = nn::linear(vs / "ptr_q", hidden_dim, hidden_dim, Default::default());
        let pointer_key = nn::linear(vs / "ptr_k", hidden_dim, hidden_dim, Default::default());
        
        let cpu_allocator = nn::seq()
            .add(nn::linear(vs / "cpu_1", hidden_dim, hidden_dim / 2, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "cpu_2", hidden_dim / 2, num_resource_bins, Default::default()));
        
        let gpu_allocator = nn::seq()
            .add(nn::linear(vs / "gpu_1", hidden_dim, hidden_dim / 2, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "gpu_2", hidden_dim / 2, num_resource_bins, Default::default()));
        
        let memory_allocator = nn::seq()
            .add(nn::linear(vs / "mem_1", hidden_dim, hidden_dim / 2, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "mem_2", hidden_dim / 2, num_resource_bins, Default::default()));
        
        Self {
            task_encoder,
            pointer_query,
            pointer_key,
            cpu_allocator,
            gpu_allocator,
            memory_allocator,
            hidden_dim,
            num_resource_bins,
        }
    }
    
    pub fn forward(&self , cluster_embedding: &Tensor , pending_embedding: &Tensor , action_mask: &ActionMask) -> ActorOutput{ 
        let num_pending = pending_embeddings.size()[0]; 
        
        let no_action_emb = Tensor::zeros(&[1 , self.hidden_dim] , pending_embedding.kind()); 
        let task_embedding = Tensor::cat(&[&no_action_emb , pending_embedding] , 0); 
        
        let encoded_task = self.task_encoder.forward(&task_embedding); 
        let query = self.pointer_query.forward(cluster_embedding); // [1, hidden]
        let keys = self.pointer_key.forward(&encoded_tasks); // [num_pending+1, hidden]
        
        let scores = query.matmul(&keys.transpose(0, 1)).squeeze_dim(0); // [num_pending+1]
        
        // Apply task selection mask (invalid tasks set to -inf)
        let masked_scores = scores + &action_mask.task_mask;
        let task_probs = masked_scores.softmax(0, Kind::Float);
        
        let task_action = task_probs.multinomial(1, true).squeeze();
        
        // If selected a real task (not no-action), allocate resources
        let task_idx = i64::from(&task_action);
        
        let resource_allocation = if task_idx > 0 {
            selected_task_emb = encoded_task.i(&task_idx); 
            let context = Torch::cat(&[cluster_embedding , &selected_task_emb.unsqueeze(0)] , 1); 
            
            let cpu_logits = self.cpu_allocator.forward(&context).squeeze_dim(0);
            let cpu_masked = cpu_logits + &action_mask.cpu_mask;
            let cpu_probs = cpu_masked.softmax(0, Kind::Float);
            let cpu_action = cpu_probs.multinomial(1, true);
            
            let gpu_logits = self.gpu_allocator.forward(&context).squeeze_dim(0);
            let gpu_masked = gpu_logits + &action_mask.gpu_mask;
            let gpu_probs = gpu_masked.softmax(0, Kind::Float);
            let gpu_action = gpu_probs.multinomial(1, true);
            
            let mem_logits = self.memory_allocator.forward(&context).squeeze_dim(0);
            let mem_masked = mem_logits + &action_mask.memory_mask;
            let mem_probs = mem_masked.softmax(0, Kind::Float);
            let mem_action = mem_probs.multinomial(1, true);
            
            Some(ResourceAction {
                cpu: cpu_action,
                gpu: gpu_action,
                memory: mem_action,
            })
        }
        else {
            None
        }; 
        
        ActorOutput {
            task_action,
            task_probs,
            resource_allocation,
        }
    }
}

pub struct ActionMask {
    pub task_mask: Tensor,      // [num_pending+1] - 0 for valid, -inf for invalid
    pub cpu_mask: Tensor,       // [num_resource_bins]
    pub gpu_mask: Tensor,
    pub memory_mask: Tensor,
}

pub struct ActorOutput {
    pub task_action: Tensor,
    pub task_probs: Tensor,
    pub resource_allocation: Option<ResourceAction>,
}

pub struct ResourceAction {
    pub cpu: Tensor,
    pub gpu: Tensor,
    pub memory: Tensor,
}

