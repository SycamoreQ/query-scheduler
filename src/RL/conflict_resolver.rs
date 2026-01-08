pub struct ConflictResolver {
    temperature: f64,
}

impl ConflictResolver {
    pub fn resolve(
        &self,
        agent_actions: Vec<(usize, Option<String>, f64)>,
    ) -> Vec<(usize, Option<String>)> {
        let mut resolved = Vec::new();
        let mut assigned_tasks = std::collections::HashSet::new();

        let mut task_to_agents: HashMap<String, Vec<(usize, f64)>> = HashMap::new();

        for (agent_id, task_id, prob) in agent_actions {
            if let Some(tid) = task_id {
                task_to_agents
                    .entry(tid.clone())
                    .or_insert_with(Vec::new)
                    .push((agent_id, prob));
            } else {
                resolved.push((agent_id, None)); // No conflict for no-action
            }
        }

        for (task_id, agents) in task_to_agents {
            if agents.len() == 1 {
                // No conflict
                resolved.push((agents[0].0, Some(task_id)));
                assigned_tasks.insert(task_id);
            } else {
                // Multiple agents want this task - use Bayesian selection
                let total: f64 = agents.iter().map(|(_, p)| p).sum();
                let normalized_probs: Vec<_> =
                    agents.iter().map(|(aid, p)| (*aid, p / total)).collect();

                // Sample winning agent based on probabilities
                let winner = self.sample_categorical(&normalized_probs);
                resolved.push((winner, Some(task_id.clone())));
                assigned_tasks.insert(task_id);

                // Losers get no-action
                for (aid, _) in agents {
                    if aid != winner {
                        resolved.push((aid, None));
                    }
                }
            }
        }

        resolved
    }
}
