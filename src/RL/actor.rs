use tch::{nn, nn::Module, Device, Kind, Tensor};

pub struct TapFingerActor {
    pub task_selection: nn::Sequential,
    pub pointer_query: nn::Linear,
    pub task_encoder: nn::Sequential,
    pub pointer_key: nn::Linear,
    pub cpu_allocator: nn::Sequential,
    pub gpu_allocator: nn::Sequential,
    pub memory_allocator: nn::Sequential,
    pub hidden_dim: i64,
    pub resource_bins: i64,
}

impl TapFingerActor {
    pub fn new(vs: &nn::Path, hidden_dim: i64, num_resource_bins: i64) -> Self {
        let task_encoder = nn::seq()
            .add(nn::linear(
                vs / "task_enc_1",
                hidden_dim,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "task_enc_2",
                hidden_dim,
                hidden_dim,
                Default::default(),
            ));

        let pointer_query = nn::linear(vs / "ptr_q", hidden_dim, hidden_dim, Default::default());
        let pointer_key = nn::linear(vs / "ptr_k", hidden_dim, hidden_dim, Default::default());

        let cpu_allocator = nn::seq()
            .add(nn::linear(
                vs / "cpu_1",
                hidden_dim,
                hidden_dim / 2,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "cpu_2",
                hidden_dim / 2,
                num_resource_bins,
                Default::default(),
            ));

        let gpu_allocator = nn::seq()
            .add(nn::linear(
                vs / "gpu_1",
                hidden_dim,
                hidden_dim / 2,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "gpu_2",
                hidden_dim / 2,
                num_resource_bins,
                Default::default(),
            ));

        let memory_allocator = nn::seq()
            .add(nn::linear(
                vs / "mem_1",
                hidden_dim,
                hidden_dim / 2,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "mem_2",
                hidden_dim / 2,
                num_resource_bins,
                Default::default(),
            ));

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

    pub fn forward(
        &self,
        graph_tensors: &GraphTensors,
        action_mask: &ActionMask,
    ) -> (Tensor, Tensor) {
        // Get embeddings from HAN
        let full_embedding = self.han.forward(graph_tensors);

        // Extract cluster and pending task embeddings
        let cluster_embedding = self.extract_cluster_embedding(&full_embedding, graph_tensors);
        let pending_embeddings = self.extract_pending_embeddings(&full_embedding, graph_tensors);

        // Forward through detailed actor
        let output = self.forward_detailed(&cluster_embedding, &pending_embeddings, action_mask);

        // Convert to (task_probs, resource_logits) format for compatibility
        let resource_logits = if let Some(ref res) = output.resource_allocation {
            // Stack CPU, GPU, memory logits
            Tensor::cat(
                &[
                    res.cpu_logits.unsqueeze(0),
                    res.gpu_logits.unsqueeze(0),
                    res.memory_logits.unsqueeze(0),
                ],
                0,
            )
        } else {
            // No resource allocation (no-action selected)
            Tensor::zeros(&[3, 1], (Kind::Float, cluster_embedding.device()))
        };

        (output.task_probs, resource_logits)
    }

    pub fn forward_detailed(
        &self,
        cluster_embedding: &Tensor,
        pending_embedding: &Tensor,
        action_mask: &ActionMask,
    ) -> ActorOutput {
        let num_pending = pending_embeddings.size()[0];

        let no_action_emb = Tensor::zeros(&[1, self.hidden_dim], pending_embedding.kind());
        let task_embedding = Tensor::cat(&[&no_action_emb, pending_embedding], 0);

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
            let context = Torch::cat(&[cluster_embedding, &selected_task_emb.unsqueeze(0)], 1);

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
        } else {
            None
        };

        ActorOutput {
            task_action,
            task_probs,
            resource_allocation,
        }
    }

    fn extract_cluster_embedding(&self, full_embedding: &Tensor, graph: &GraphTensors) -> Tensor {
        if graph.cluster_indices.is_empty() {
            return Tensor::zeros(
                &[1, self.hidden_dim],
                (Kind::Float, full_embedding.device()),
            );
        }

        let cluster_idx =
            Tensor::of_slice(&graph.cluster_indices[0..1]).to_device(full_embedding.device());
        full_embedding.index_select(0, &cluster_idx)
    }

    fn extract_pending_embeddings(&self, full_embedding: &Tensor, graph: &GraphTensors) -> Tensor {
        if graph.pending_indices.is_empty() {
            return Tensor::zeros(
                &[0, self.hidden_dim],
                (Kind::Float, full_embedding.device()),
            );
        }

        let pending_idx =
            Tensor::of_slice(&graph.pending_indices).to_device(full_embedding.device());
        full_embedding.index_select(0, &pending_idx)
    }
}

pub struct ActionMask {
    pub task_mask: Tensor, // [num_pending+1] - 0 for valid, -inf for invalid
    pub cpu_mask: Tensor,  // [num_resource_bins]
    pub gpu_mask: Tensor,
    pub memory_mask: Tensor,
}

impl ActionMask {
    pub fn new(
        num_pending: i64,
        num_cpu_bins: i64,
        num_gpus: i64,
        num_memory_bins: i64,
        device: Device,
    ) -> Self {
        Self {
            task_mask: Tensor::zeros(&[num_pending + 1], (Kind::Float, device)),
            cpu_mask: Tensor::zeros(&[num_cpu_bins], (Kind::Float, device)),
            gpu_mask: Tensor::zeros(&[num_gpus + 1], (Kind::Float, device)),
            memory_mask: Tensor::zeros(&[num_memory_bins], (Kind::Float, device)),
        }
    }

    pub fn mask_task(&mut self, task_idx: String) {
        let _ = self.task_mask.get(task_idx).fill_(f64::NEG_INFINITY);
    }

    pub fn mask_cpu(&mut self, cpu_idx: String) {
        let _ = self.task_mask.get(cpu_idx).fill_(f64::NEG_INFINITY);
    }

    pub fn from_environment(env: &EdgeMLEnv, cluster_id: usize, device: Device) -> Self {
        let cluster = &env.clusters[cluster_id];
        let num_pending = cluster.pending_tasks.len() as i64;

        let mut mask = Self::new(num_pending, 17, 8, 20, device);

        // Mask tasks that don't fit in available resources
        for (i, task) in cluster.pending_tasks.iter().enumerate() {
            if task.min_cpu > cluster.cpu_cores.available
                || task.min_gpu_core > cluster.gpus.iter().map(|g| g.CoreAvailable).sum()
            {
                mask.mask_task(i as i64 + 1);
            }
        }

        // Mask CPU allocations beyond available
        let available_cpu = cluster.cpu_cores.available as i64;
        for cpu in (available_cpu + 1)..17 {
            mask.mask_cpu(cpu);
        }
        mask
    }

    pub fn get_valid_candidates(
        &self,
        cluster_res: &ClusterResources,
        task_lookup: &TaskLookup,
    ) -> Vec<ActionKey> {
        let mut candidates = Vec::new();

        candidates.push(ActionKey {
            task_idx: 0,
            cpu_bin: 0,
            gpu_idx: 0,
            mem_bin: 0,
        });

        let mask_vec: Vec<f32> = self
            .task_mask
            .shallow_clone()
            .try_into()
            .unwrap_or_default();
        for (idx, &mask_val) in mask_vec.iter().enumerate().skip(1) {
            if mask_val < 0.0 {
                continue;
            }

            let task_idx = idx as f64;

            let info = match task_lookup.get(&task_idx) {
                Some(i) => i,
                None => continue,
            };

            for &cpu_usage in &[info.min_cpu as i64, cluster_res.cpu_available as i64] {
                for (gpu_id, &gpu_cores) in cluster_res.gpu_available_cores.iter().enumerate() {
                    if gpu_cores > info.min_gpu_cores {
                        candidates.push(ActionKey {
                            task_idx,
                            cpu_bin: cpu_usage,
                            gpu_idx: gpu_id,
                            mem_bin: info.min_memory as i64,
                        });
                    }
                }
            }
        }

        candidates
    }

    pub fn to_tensor(&self, device: Device) -> Tensor {
        let data = vec![
            self.task_idx as f32,
            self.cpu_bin as f32 / 16.0,
            self.gpu_idx as f32 / 8.0,
            self.mem_bin as f32 / 20.0,
        ];

        Tensor::from_slice(&data)
            .to_device(device)
            .to_kind(Kind::Float)
    }

    pub fn no_op() -> Self {
        Self {
            task_idx: 0,
            cpu_bin: 0,
            gpu_idx: 0,
            mem_bin: 0,
        }
    }
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
