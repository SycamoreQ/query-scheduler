use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use lru::LruCache;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUKVCache{
    pub gpu_id: String ,
    pub paper_cache: LRUCache<String , CachedPaper>,
    pub author_cache: LRUCache<String , CachedAuthor> ,
    pub venue_index: HashMap<String, HashSet<String>>,
    pub domain_index: HashMap<ResearchDomain, HashSet<String>>,
    pub domain_heatmap: HashMap<ResearchDomain, usize>,
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
            venue_index: HashMap::new(),
            field_index: HashMap::new(),
            domain_heatmap: HashMap::new(),
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
    
    pub fn insert_paper(&mut self , paper:CachedPaper) {
        let paper_id = self.paper.paper_id.clone();
        let size_mb = (paper.size_bytes as f64) / 1024.0 / 1024.0;
        
        if self.paper_cache.len() == self.paper_cache.cap().get(){
            if let Some((evicted_id , evicted_paper)) = self.paper_cache.pop_lru(){
                self.remove_paper_from_indices(&evicted_id, &evicted_paper);
                self.memory_used_mb -= (evicted_paper.size_bytes as f64) / 1024.0 / 1024.0;
                self.cache_stats.eviction_count += 1;
            }
        }
        
        for domain in &paper.field_of_study {
            self.domain_index.entry(*domain).or_default().insert(paper_id.clone());
            *self.domain_heatmap.entry(*domain).or_insert(0) += 1;
        }
    
        for venue in &paper.venue {
            self.venue_index.entry(venue.clone()).or_default().insert(paper_id.clone());
        }
    
        self.memory_used_mb += size_mb;
        self.paper_cache.put(paper_id, paper);
    }
    
    fn remove_paper_from_indices(&mut self, id: &str, paper: &CachedPaper) {
            for domain in &paper.field_of_study {
                if let Some(set) = self.domain_index.get_mut(domain) {
                    set.remove(id);
                }
                if let Some(count) = self.domain_heatmap.get_mut(domain) {
                    *count = count.saturating_sub(1);
                }
            }
            for venue in &paper.venue {
                if let Some(set) = self.venue_index.get_mut(venue) {
                    set.remove(id);
                }
            }
        }
        
    pub fn estimate_cache_coverage(&self , query_intent: &QueryIntent ) -> f32{
        
    }
    
    
}
