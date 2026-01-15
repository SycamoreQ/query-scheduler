use std::collections::HashMap;
use tch::{nn, Device, IndexOp, Kind, Tensor};
use std::sync::{Arc, Mutex};
use std::f64;

pub struct WorldModel {
    representation_net: nn::Sequential,
    // Dynamics: predicts next latent state from current latent state + action
    dynamics_net: nn::Sequential,
    // Prediction: predicts reward and value from latent state
    reward_net: nn::Sequential,
    value_net: nn::Sequential,
    // Hidden dimension for latent states
    hidden_dim: i64,
}

impl WorldModel {
    pub fn new(vs: &nn::Path, state_dim: i64, action_dim: i64, hidden_dim: i64) -> Self {
        // Encoder: Raw state → Latent state
        let representation_net = nn::seq()
            .add(nn::linear(
                vs / "repr_1",
                state_dim,
                hidden_dim * 2,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "repr_2",
                hidden_dim * 2,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|x| x.tanh()); // Bounded latent space

        // Dynamics: (latent_state, action) → next_latent_state
        let dynamics_net = nn::seq()
            .add(nn::linear(
                vs / "dyn_1",
                hidden_dim + action_dim,
                hidden_dim * 2,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "dyn_2",
                hidden_dim * 2,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|x| x.tanh());

        // Reward prediction from latent state
        let reward_net = nn::seq()
            .add(nn::linear(
                vs / "rew_1",
                hidden_dim,
                128,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "rew_2", 128, 1, Default::default()));

        // Value prediction from latent state
        let value_net = nn::seq()
            .add(nn::linear(
                vs / "val_1",
                hidden_dim,
                128,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "val_2", 128, 1, Default::default()));

        Self {
            representation_net,
            dynamics_net,
            reward_net,
            value_net,
            hidden_dim,
        }
    }

    pub fn represent(&self, state: &SchedulerState) -> &Tensor {
        self.representation.forward(state)
    }

    pub fn predict_next(&self, latent_reward: &Tensor, action: &Tensor) -> Tensor {
        let input = Tensor::cat(&[latent_space, action], -1);
        self.dynamics_net.forward(&input)
    }

    /// Predict reward from latent state
    pub fn predict_reward(&self, latent_state: &Tensor) -> Tensor {
        self.reward_net.forward(latent_state)
    }

    /// Predict value from latent state
    pub fn predict_value(&self, latent_state: &Tensor) -> Tensor {
        self.value_net.forward(latent_state)
    }

    /// Full step: predict (next_state, reward, value)
    pub fn step(&self, latent_state: &Tensor, action: &Tensor) -> WorldModelOutput {
        let next_latent = self.predict_next(latent_state, action);
        let reward = self.predict_reward(&next_latent);
        let value = self.predict_value(&next_latent);

        WorldModelOutput {
            next_latent_state: next_latent,
            reward,
            value,
        }
    }

    pub fn unroll(&self, initial_latent: &Tensor, actions: &[Tensor]) -> Vec<WorldModelOutput> {
        let mut outputs = Vec::new();
        let mut current_latent = initial_latent.shallow_clone();

        for action in actions {
            let output = self.predict_next(&current_latent, action);
            current_latent = output.next_latent_state.shallow_clone();
            outputs.push(output);
        }

        outputs
    }

    pub fn train_step(
        &self,
        real_states: &[Tensor],
        actions: &[Tensor],
        real_rewards: &[f64],
    ) -> Tensor {
        // Encode initial state
        let mut latent = self.represent(&real_states[0]);

        let mut total_loss = Tensor::zeros(&[], (Kind::Float, Device::cuda_if_available()));

        for i in 0..actions.len().min(real_states.len() - 1) {
            // Predict next state
            let output = self.step(&latent, &actions[i]);

            // Real next state
            let real_next_latent = self.represent(&real_states[i + 1]);

            // Losses
            let state_loss = (&output.next_latent_state - &real_next_latent)
                .pow_tensor_scalar(2)
                .mean(Kind::Float);

            let reward_loss = (&output.reward - real_rewards[i])
                .pow_tensor_scalar(2)
                .mean(Kind::Float);

            total_loss = total_loss + state_loss + reward_loss;

            // Update latent for next iteration
            latent = output.next_latent_state;
        }

        total_loss / actions.len() as f64
    }
}

#[derive(Clone)]
pub struct WorldModelOutput {
    pub next_latent_state: Tensor,
    pub reward: Tensor,
    pub value: Tensor,
}




