use actor::TapFingerActor;
use critic::TapFingerCritic;
use std::collections::HashMap;
use structures::_graph::GraphTensors;
use tch::{Device, Kind, Tensor, nn};

pub struct MAPPOTrainer {
    actors: Vec<TapFingerActor>,
    critics: Vec<TapFingerCritic>,
    optimizer: nn::Optimizer,
    clip_epsilon: f64,
    value_loss_coef: f64,
    entropy_coef: f64,
}

impl MAPPOTrainer {
    pub fn new(
        vs: &nn::VarStore,
        num_agents: usize,
        state_dim: i64,
        hidden_dim: i64,
        num_gpus: usize,
        learning_rate: f64,
    ) -> Self {
        let mut actors = Vec::new();
        let mut critics = Vec::new();

        for i in 0..num_agents {
            let actor_vs = vs.root() / format!("actor_{}", i);
            let critic_vs = vs.root() / format!("critic_{}", i);

            actors.push(TapFingerActor::new(
                &actor_vs, state_dim, hidden_dim, num_gpus,
            ));
            critics.push(TapFingerCritic::new(&critic_vs, hidden_dim));
        }

        let optimizer = nn::Adam::default().build(vs, learning_rate).unwrap();

        Self {
            actors,
            critics,
            optimizer,
            clip_epsilon: 0.2,
            value_loss_coef: 0.5,
            entropy_coef: 0.01,
        }
    }

    pub fn compute_loss(
        &self,
        states: &[GraphTensors],
        actions: &[Tensor],
        old_log_probs: &[Tensor],
        advantages: &[Tensor],
        mask: &[Tensor],
    ) -> Tensor {
        let mut total_loss = Tensor::zeros(&[], (Kind::Float, Device::cpu));

        for (i, actor) in self.actors.iter().enumerate() {
            let (task_probs, resource_logits) = self.actor.forward(&states[i], &masks[i]);
            let log_probs = task_probs.log();
            let action_log_probs = log_probs.gather(1, &actions[i], false);

            let ratio = (action_log_probs - &old_log_probs[i]).exp();
            let surr1 = &ratio * &advantages[i];
            let surr2 =
                ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * &advantages[i];
            let policy_loss = -surr1.min_other(&surr2).mean(Kind::Float);

            // Value loss
            let value = self.critics[i].forward(&states[i].node_features);
            let value_loss = (value - &returns[i]).pow_tensor_scalar(2).mean(Kind::Float);

            // Entropy bonus
            let entropy = -(task_probs * task_probs.log())
                .sum_dim_intlist(&[1i64][..], false, Kind::Float)
                .mean(Kind::Float);

            total_loss = total_loss + policy_loss + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy;
        }

        total_loss
    }

    pub fn train_step(&mut self, batch: TrainingBatch) -> f64 {
        self.optimizer.zero_grad();

        let loss = self.compute_loss(
            &batch.states,
            &batch.actions,
            &batch.old_log_probs,
            &batch.advantages,
            &batch.returns,
            &batch.masks,
        );

        loss.backward();
        self.optimizer.step();

        f64::from(loss)
    }
}

pub struct TrainingBatch {
    pub states: Vec<GraphTensors>,
    pub actions: Vec<Tensor>,
    pub old_log_probs: Vec<Tensor>,
    pub advantages: Vec<Tensor>,
    pub returns: Vec<Tensor>,
    pub masks: Vec<Tensor>,
}
