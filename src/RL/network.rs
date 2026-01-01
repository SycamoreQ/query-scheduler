use serde::{Deserialize , Serialize};
use ndarray::{Array1, Array2};
use tch::{nn, Device, Tensor, Kind};
use tch::nn::{Module, OptimizerConfig};

pub struct ActorCriticNet { 
    actor: nn::Sequential,
    critic: nn::Sequential,
    device : Device,
}

impl ActorCriticNet { 
    pub fn new(vs: &nn::Path , state_dim: i64 , num_gpus: usize) -> Self{ 
        let device = vs.device();
        
        let shared_layers = nn::seq()
            .add(nn::linear(vs/"shared_1" , state_dim , 256 , default::Default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs/"shared_2" , 256 , 128 , default::Default()))
            .add_fn(|x| x.relu());
        
        let action_dim = num_gpus + 3; 
        let actor = nn::seq()
            .add(shared_layers.clone())
            .add(nn::linear(vs / "actor1", 128, 128, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "actor_out", 128, action_dim as i64, Default::default()));
        
        let critic = nn::seq()
            .add(shared_layers)
            .add(nn::linear(vs / "critic1", 128, 64, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "critic_out", 64, 1, Default::default()));
        
        Self { actor, critic, device }
    }
    
    pub fn forward(&self , state: &Tensor) -> (Tensor , Tensor){
        let action_logits = self.actor.forward(state);
        let critic_logits = self.critic.forward(state);
        
        (actor_logits , critic_logits)
    }
    
    pub fn get_action(&self, state: &Tensor, deterministic: bool) -> (Tensor, Tensor) {
        let (action_logits, value) = self.forward(state);
        
        if deterministic {
            // Greedy action selection
            (action_logits, value)
        } else {
            // Stochastic sampling with exploration
            let action = action_logits + Tensor::randn_like(&action_logits) * 0.1;
            (action, value)
        }
    }
}