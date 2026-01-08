
pub struct TapFingerCritic {
    value_net: nn::Sequential,
}

impl TapFingerCritic {
    pub fn new(vs: &nn::Path, hidden_dim: i64) -> Self {
        let value_net = nn::seq()
            .add(nn::linear(vs / "v1", hidden_dim, hidden_dim, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "v2", hidden_dim, hidden_dim / 2, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "v3", hidden_dim / 2, 1, Default::default()));
        
        Self { value_net }
    }
    
    pub fn forward(&self, cluster_embedding: &Tensor) -> Tensor {
        self.value_net.forward(cluster_embedding)
    }
}