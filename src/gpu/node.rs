use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use rater::Rater;
pub const GPUId: usize; 


pub struct NodeAllocator{
    pub rater : Rater, 
    pub gpus: GPU, 
    pub podsMap: 
}