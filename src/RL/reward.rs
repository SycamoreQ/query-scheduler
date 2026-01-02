use serde::{Deserialize , Serialize};


#[derive(Debug, Clone)]
pub struct RewardMetrics {
    pub execution_time_sec: f32,
    pub estimated_time_sec: f32,
    pub gpu_efficiency: f32,     
    pub queue_wait_time_sec: f32,
    pub resource_match_score: f32, 
    pub priority_violation: bool,
    pub gpu_overload: bool,
}

pub fn calculate_reward(metrics: &RewardMetrics, query_priority: QueryPriority) -> f32 {
    let mut reward = 0.0;
    
    let time_ratio = metrics.estimated_time_sec / metrics.execution_time_sec.max(0.1);
    let time_reward = if time_ratio > 1.2 {
        5.0 * (time_ratio - 1.0).min(2.0)
    } else if time_ratio > 0.8 {
        2.0
    } else {
        -5.0 * (0.8 - time_ratio)
    };
    reward += time_reward;
    
    let efficiency_reward = metrics.gpu_efficiency * 10.0;
    reward += efficiency_reward;
    
    let resource_reward = metrics.resource_match_score * 5.0;
    reward += resource_reward;
    
    let wait_penalty = match query_priority {
        QueryPriority::Critical => -metrics.queue_wait_time_sec * 0.5,
        QueryPriority::High => -metrics.queue_wait_time_sec * 0.3,
        QueryPriority::Medium => -metrics.queue_wait_time_sec * 0.1,
        QueryPriority::Low => -metrics.queue_wait_time_sec * 0.05,
    };
    reward += wait_penalty;
    
    if metrics.priority_violation {
        reward -= 20.0;
    }
    
    if metrics.gpu_overload {
        reward -= 15.0;
    }
    
    let priority_multiplier = match query_priority {
        QueryPriority::Critical => 1.5,
        QueryPriority::High => 1.2,
        QueryPriority::Medium => 1.0,
        QueryPriority::Low => 0.8,
    };
    
    reward * priority_multiplier
}

pub fn calculate_resource_match_score(
    allocated: &ResourceAllocation,
    actual_used_core: f32,
    actual_used_memory: u64,
) -> f32 {
    let core_match = 1.0 - (allocated.core_percent - actual_used_core).abs() / 100.0;
    let memory_match = 1.0 - ((allocated.memory_mb as f32 - actual_used_memory as f32).abs() 
                              / allocated.memory_mb as f32);
    
    (core_match + memory_match) / 2.0
}
