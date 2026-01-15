use std::collections::HashMap;
use std::f64;
use std::sync::{Arc, Mutex};
use tch::{nn, Device, IndexOp, Kind, Tensor};

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

pub struct MCTSNode {
    pub latent_state: Tensor,
    // Parent node
    pub parent: Option<Arc<Mutex<MCTSNode>>>,
    // Children: (action, child_node)
    pub children: HashMap<ActionKey, Arc<Mutex<MCTSNode>>>,
    // Statistics
    pub visit_count: usize,
    pub total_value: f64,
    pub prior_prob: f64,
    // Which action led to this node
    pub action: Option<ActionKey>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ActionKey {
    pub task_idx: i64,
    pub cpu_bin: i64,
    pub gpu_idx: i64,
    pub mem_bin: i64,
}

impl ActionKey {
    pub fn from_tensor(task_action: &Tensor, resource_action: &Option<ResourceAction>) -> Self {
        let task_idx = i64::try_from(task_action).unwrap_or(0);

        if let Some(res) = resource_action {
            Self {
                task_idx,
                cpu_bin: i64::try_from(&res.cpu).unwrap_or(0),
                gpu_idx: i64::try_from(&res.gpu).unwrap_or(0),
                mem_bin: i64::try_from(&res.memory).unwrap_or(0),
            }
        } else {
            Self {
                task_idx,
                cpu_bin: 0,
                gpu_idx: 0,
                mem_bin: 0,
            }
        }
    }

    pub fn to_tensor(&self, device: Device) -> Tensor {
        Tensor::of_slice(&[
            self.task_idx as f32,
            self.cpu_bin as f32,
            self.gpu_idx as f32,
            self.mem_bin as f32,
        ])
        .to_device(device)
    }

    pub fn is_valid(&self, task_info: &TaskInfo, cluster: &ClusterResources) -> bool {
        // 1. Check if task selection is "No-Op" (always valid)
        if self.task_idx == 0 {
            return true;
        }

        // 2. Validate CPU: Bin must be >= task minimum AND <= cluster available
        let cpu_req = self.cpu_bin as usize;
        if cpu_req < task_info.min_cpu || cpu_req > cluster.cpu_available {
            return false;
        }

        // 3. Validate GPU: Check if the specific GPU has enough cores/memory
        if let Some(gpu) = cluster.gpus.get(self.gpu_idx as usize) {
            if (self.gpu_cores_bin as usize) < task_info.min_gpu_core
                || (self.gpu_cores_bin as usize) > gpu.core_available
            {
                return false;
            }
        } else {
            return false; // GPU index out of bounds
        }

        true
    }
}

impl MCTSNode {
    pub fn new(latent_state: Tensor) -> Self {
        Self {
            latent_state,
            parent: None,
            children: HashMap::new(),
            visit_count: 0,
            total_value: 0.0,
            prior_prob: 1.0,
            action: None,
        }
    }

    pub fn q_value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f64
        }
    }

    pub fn ucb_score(&self, parent_visits: usize, c_puct: f64) -> f64 {
        let exploitation = self.q_value();

        let exploration = c_puct * self.prior_prob * (parent_visits as f64).sqrt()
            / (1.0 + self.visit_count as f64);

        exploitation + exploration
    }

    pub fn select_child(&self, c_puct: f64) -> Option<(ActionKey, Arc<Mutex<MCTSNode>>)> {
        if self.children.is_empty() {
            return None;
        }

        let parent_visits = self.visit_count;

        self.children
            .iter()
            .max_by(|(_, a), (_, b)| {
                let score_a = a.lock().unwrap().ucb_score(parent_visits, c_puct);
                let score_b = b.lock().unwrap().ucb_score(parent_visits, c_puct);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|(action, node)| (action.clone(), Arc::clone(node)))
    }

    pub fn add_child(
        &mut self,
        action: ActionKey,
        child: MCTSNode,
        prior: f64,
    ) -> Arc<Mutex<MCTSNode>> {
        let mut child = child;
        child.action = Some(action.clone());
        child.prior_prob = prior;

        let child_ref = Arc::new(Mutex::new(child));
        self.children.insert(action, Arc::clone(&child_ref));
        child_ref
    }

    /// Backpropagate value up the tree
    pub fn backpropagate(&mut self, value: f64) {
        self.visit_count += 1;
        self.total_value += value;
    }

    /// Is this a leaf node?
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

pub struct MCTS {
    world_model: Arc<WorldModel>,
    actor: Arc<TapFingerActor>,
    c_puct: f64,
    num_simulations: usize,
    max_depth: usize,
    gamma: f64,
}

impl MCTS {
    pub fn new(
        world_model: Arc<WorldModel>,
        actor: Arc<TapFingerActor>,
        num_simulations: usize,
    ) -> Self {
        Self {
            world_model,
            actor,
            c_puct: 1.5,
            num_simulations,
            max_depth: 10,
            gamma: 0.99,
        }
    }

    pub fn search(&self, root_state: &GraphTensor) -> (ActionKey, f64) {
        let root_latent = self.world_model.represent(&root_state.node_features);
        let root = Arc::new(Mutex::new(MCTSNode::new(root_latent)));

        for _ in 0..self.num_simulations {
            self.simulate(Arc::clone(&root), root_state);
        }

        let root_lock = root.lock().unwrap();

        let (best_action, best_child) = root_lock
            .children
            .iter()
            .max_by_key(|(_, child)| child.lock().unwrap().visit_count)
            .map(|(action, child)| (action.clone(), Arc::clone(child)))
            .expect("No children found in root");

        let value = best_child.lock().unwrap().q_value();

        (best_action, value)
    }

    fn simulate(&self, node: Arc<Mutex<MCTSNode>>, ctx: &SchedulingContext) {
        let mut path = Vec::new();
        let mut current = Arc::clone();
        let mut depth = 0;

        loop {
            let node_lock = node.lock().unwrap();

            if node_lock.is_leaf() || depth >= self.max_depth {
                break;
            }

            if let Some((action, child)) = node_lock.select_child(self.c_puct) {
                path.push((Arc::clone(&current), action.clone()));
                drop(node_lock);
                current = child;
                depth += 1;
            } else {
                break;
            }
        }

        let should_expand = {
            let node_lock = current.lock().unwrap();
            node_lock.visit_count >= 1 && node_lock.is_leaf() && depth < self.max_depth
        };

        if should_expand {
            let latent_state = current.lock().unwrap().latent_state.shallow_clone();
            let device = latent_state.device();

            let mask = ActionMask::from_context(context, device);
            let valid_candidates = mask.get_valid_candidates(context);
            // Get Policy (Actor) probabilities for these specific candidates
            let action_probs = self.get_action_probs_from_candidates(&valid_candidates, &ctx.graph);

            let top_actions = self.select_top_k_actions(&action_probs, 8);

            for (action, prob) in top_actions {
                // Predict next state using world model
                let action_tensor = action.to_tensor(device);
                let output = self.world_model.step(&latent_state, &action_tensor);

                // Create child node
                let child = MCTSNode::new(output.next_latent_state);

                // Store the predicted reward from the World Model into the edge
                current
                    .lock()
                    .unwrap()
                    .add_child(action, child, prob, output.reward);
            }

            // Select one child to continue
            if let Some((action, child)) = current.lock().unwrap().select_child(self.c_puct) {
                path.push((Arc::clone(&current), action));
                current = child;
            }
        }

        // Evaluation: rollout or use value network
        let value = self.evaluate(&current);

        // Backpropagation: update all nodes in path
        for (node, _) in path.iter().rev() {
            node.lock().unwrap().backpropagate(value);
        }
        current.lock().unwrap().backpropagate(value);
    }

    /// Evaluate a node using value network or rollout
    fn evaluate(&self, node: &Arc<Mutex<MCTSNode>>) -> f64 {
        let latent_state = node.lock().unwrap().latent_state.shallow_clone();

        // Use value network for evaluation
        let value = self.world_model.predict_value(&latent_state);
        f64::try_from(&value).unwrap_or(0.0)
    }

    /// Get action probabilities from policy network
    fn get_action_probs(
        &self,
        state: &GraphTensors,
        ctx: &SchedulingContext,
    ) -> Vec<(ActionKey, f64)> {
        let mut candidates = Vec::new();

        let mask = ActionMask::from_environment(ctx);
        let (task_probs, resource_logits) = self.actor.forward(state, &mask);

        let candidates =
            mask.get_valid_candidates(&context.cluster_resources, &context.task_lookup);

        candidates
            .into_iter()
            .map(|key| {
                let prob = f64::try_from(task_probs.get(key.task_idx)).unwrap_or(0.0);
                (key, prob)
            })
            .collect()
    }

    fn get_action_probs_for_candidates(
        &self,
        candidates: &[ActionKey],
        state: &GraphTensors,
    ) -> Vec<(ActionKey, f64)> {
        let (task_logits, resource_logits) = self.actor.forward(state);

        candidates
            .iter()
            .map(|key| {
                // Joint Probability = P(Task) * P(Resources | Task)
                let t_prob = f64::try_from(task_logits.get(key.task_idx)).unwrap_or(0.0);

                // TODO: POSSIBLE : refine this by looking up specific resource bin logits
                (*key, t_prob)
            })
            .collect()
    }

    /// Select top-k actions by probability
    fn select_top_k_actions(
        &self,
        action_probs: &[(ActionKey, f64)],
        k: usize,
    ) -> Vec<(ActionKey, f64)> {
        let mut sorted = action_probs.to_vec();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sorted.into_iter().take(k).collect()
    }
}
