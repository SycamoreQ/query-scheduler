use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUKVCache{ 
    pub gpu_id: String ,
    pub paper_cache: LRUCache<String , CachedPaper>,
    pub author_cache: LRUCache<String , CachedAuthor> ,
    pub comm_cache: LRUCache<String , CachedComm>,  
    pub cache_stats: CacheStatistics,
    pub memory_used_mb: f64,
    pub memory_limit_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedPaper {
    pub paper_id: String,
    pub title: String,
    pub year: u32,
    pub citation_count: usize,
    pub field_of_study: Vec<String>,
    pub venue: Vec<String>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: usize,
    pub size_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedAuthor {
    pub author_id: String,
    pub name: String,
    pub wrote_count: usize,
    pub h_index: u32,
    pub last_accessed: DateTime<Utc>,
    pub access_count: usize,
    pub size_bytes: usize,
}

#[derive(Debug , Clone , Serialize , Deserialize)]
pub struct CachedComm{
    pub comm_id: String ,
    pub nature: Option<CachedPaper , CachedAuthor> ,
    pub node_count: usize ,
    pub edge_count: usize,
    pub last_accessed : DateTime<Utc>, 
    pub access_count: usize,
    pub size_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub total_hits: u64,
    pub total_misses: u64,
    pub hit_rate: f32,
    pub eviction_count: u64,
    pub avg_access_time_ms: f32,
}

impl GPUKVCache {
    pub fn new(gpu_id: usize, memory_limit_mb: f64) -> Self {
        Self {
            gpu_id,
            paper_cache: LRUCache::new(100000),
            author_cache: LRUCache::new(10000),
            comm_cache: LRUCache::new(5000),
            cache_stats: CacheStatistics {
                total_hits: 0,
                total_misses: 0,
                hit_rate: 0.0,
                eviction_count: 0,
                avg_access_time_ms: 0.0,
            },
            memory_used_mb: 0.0,
            memory_limit_mb,
        }
    }
    
    
}

