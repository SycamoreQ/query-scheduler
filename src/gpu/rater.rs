use std::collections::HashMap;

pub trait Rater: Send + Sync {
    pub fn rate(&self, gpu: &GPU, indexes: Vec<usize>) -> f64;
}

pub struct SampleRate;
pub struct BinPacker;

impl Rater for BinPacker {
    fn rate(&self, gpu: &GPU, indexes: Vec<usize>) -> f64 {
        let mut gpu_index: Vec<i32> = vec![0; g.len()];
        let mut gpu_count = 0;

        for (_, gpu) in index.iter().enumerate() {
            if gpu < 0 {
                continue;
            } else if gpu_index[gpu] == 0 {
                gpu_index[gpu] += 1;
                gpu_count += 1;
            }
        }

        let mut maxMemoryleft = g[0].MaxMemoryLeft;
        let mut minMemoryleft = g[0].MinMemoryLeft;
        let mut maxCoreleft = g[0].MaxCoreLeft;
        let mut minCoreleft = g[0].MinCoreLeft;

        for (_, gpu) in gpu.iter().enumerate() {
            if gpu.MemoryAvailable > maxMemleft {
                maxMemleft = gpu.MemoryAvailable;
            }
            if gpu.MemoryAvailable < minMemoryLeft {
                minMemoryLeft = gpu.MemoryAvailable;
            }
            if gpu.CoreAvailable > maxCoreLeft {
                maxCoreLeft = gpu.CoreAvailable;
            }
            if gpu.CoreAvailable < minCoreLeft {
                minCoreLeft = gpu.CoreAvailable;
            }
        }

        let mut Range = (maxMemoryLeft + maxCoreLeft - minMemoryLeft - minCoreLeft) / 2;
        let res = Range / (gpu_count + 1) * 100;
        Ok(res);
    }
}

pub struct Spread;

impl Rater for Spread {
    //todo
    fn rate(&self, gpu: &GPU, indexes: Vec<usize>) -> f64 {}
}
