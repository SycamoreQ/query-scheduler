use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

pub trait String {
    pub fn string(&self) -> String;
}

#[derive(Default, Clone, Debug)]
pub struct GPUUnit {
    pub Core: usize,
    pub Memory: usize,
    pub GPUCount: usize,
}

impl String for GPUUnit {
    fn string(&self) -> String {
        let g: GPUUnit;
        format!(
            "core:{} , memory:{} , count:{}",
            g.Core, g.Memory, g.GPUCount
        );
    }
}

pub trait GPUFuncs {
    pub fn Add(&mut self, resource: &GPUUnit) -> ();
    pub fn Sub(&mut self, resource: &GPUUnit) -> ();
    pub fn CanAllocate(&self, resource: &GPUUnit) -> bool;
}

#[derive(Debug, Deserialize, Serialize, Clone, Eq)]
pub struct GPU {
    pub CoreAvailable: usize,
    pub MemoryAvailable: usize,
    pub CoreTotal: usize,
    pub MemoryTotal: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone, Eq)]
pub struct GPUs;

pub trait GPUsFunction {
    pub fn String(&self) -> String;
    pub fn Trade(
        &self,
        rater: &Rater,
        request: &GPURequest,
    ) -> Result<option: GPUOption, std::error::Error>;
    pub fn Transact(&self, option: &GPUOption) -> std::error::Error;
    pub fn Cancel(&self, option: &GPUOption) -> std::error::Error;
    pub fn get_free_gpus(&self) -> Vec<usize>;
}

impl GPUFuncs for GPU {
    fn Add(&mut self, resource: &GPUUnit) {
        match resource.GPUCount {
            Some(count) if count > 0 => {
                self.core_available = 0;
                self.memory_available = 0;
            }
            _ => {
                self.core_available = self.core_available.saturating_sub(resource.core);
                self.memory_available = self.memory_available.saturating_sub(resource.memory);
            }
        }
    }

    fn Sub(&mut self, resource: &GPUUnit) {
        match resource.GPUCount {
            Some(count) if count > 0 => {
                self.core_available = self.CoreTotal;
                self.memory_available = self.MemoryTotal;
            }
            _ => {
                self.core_available = self.core_available.saturating_add(resource.core);
                self.memory_available = self.memory_available.saturating_add(resource.memory);
            }
        }
    }

    fn CanAllocate(&mut self, resource: &GPUUnit) -> bool {
        match resource.GPUCount {
            Some(count) if count > 0 => {
                self.CoreTotal == self.CoreAvailable && self.MemoryTotal == self.MemoryAvailable;
            }
            _ => {
                self.CoreAvailable >= self.CoreTotal && self.MemoryAvailable >= self.MemoryTotal;
            }
        }
    }
}

impl GPUsFunction for GPUs {
    fn String(&self) -> String {
        let r = serde_json::to_str(self).unwrap_or_else(|_| "{}".to_string());
        r;
    }

    fn trade(
        &mut self,
        rater: &dyn Rater,
        request: &GPURequest,
    ) -> Result<GPUOption, Box<dyn std::error::Error>> {
        let mut indexes: Vec<Vec<usize>> = vec![vec![]; request.len()];
        let mut found = false;
        let mut option = GPUOption::new(request);

        fn dfs(
            container_idx: usize,
            g: &mut GPUs,
            request: &GPURequest,
            indexes: &mut Vec<Vec<usize>>,
            found: &mut bool,
            option: &mut GPUOption,
            rater: &dyn Rater,
        ) {
            if container_idx == request.len() {
                *found = true;
                let mut rate_indexes = vec![0; indexes.len()];
                for (i, idx_list) in indexes.iter().enumerate() {
                    rate_indexes[i] = if idx_list.len() == 1 {
                        idx_list[0] as i32
                    } else {
                        -1
                    };
                }

                let curr_score = rater.rate(g, &rate_indexes);
                if option.score > curr_score {
                    return;
                }

                for (i, gpu_indices) in indexes.iter().enumerate() {
                    option.allocated[i] = gpu_indices.clone();
                }
                option.score = curr_score;
                return;
            }

            let req = &request[container_idx];
            info!("Start to allocate request on {} container", container_idx);

            if req.gpu_count > 0 {
                let free_gpus = g.get_free_gpus();
                if free_gpus.len() < req.gpu_count {
                    return;
                }

                let selected = free_gpus[..req.gpu_count].to_vec();
                for &gpu_idx in &selected {
                    g[gpu_idx].add(req);
                }
                indexes[container_idx] = selected.clone();

                dfs(container_idx + 1, g, request, indexes, found, option, rater);

                for &gpu_idx in &selected {
                    g[gpu_idx].sub(req);
                }
            } else {
                for i in 0..g.len() {
                    if g[i].can_allocate(req) {
                        g[i].add(req);
                        indexes[container_idx] = vec![i];

                        dfs(container_idx + 1, g, request, indexes, found, option, rater);

                        g[i].sub(req);
                    }
                }
            }
        }

        dfs(
            0,
            self,
            request,
            &mut indexes,
            &mut found,
            &mut option,
            rater,
        );

        if !found {
            return Err("no enough resource to allocate".into());
        }
        Ok(option)
    }

    fn Transact(&mut self, option: &GPUOption) -> Result<(), String> {
        debug!("GPU {:?} transacts {:?}", self, option);

        for (i, allocation) in option.allocated.iter().enumerate() {
            let request = &option.request[i];

            if request.gpu_count > 0 {
                for &gpu_index in allocation {
                    let gpu = &mut self[gpu_index];
                    if !gpu.can_allocate(request) {
                        let err_msg = format!(
                            "Fail to trade option on GPU {} because resources are insufficient",
                            gpu_index
                        );
                        error!("{}", err_msg);
                        return Err(err_msg);
                    }
                    gpu.add(request);
                }
            } else {
                if let Some(&gpu_index) = allocation.get(0) {
                    let gpu = &mut self[gpu_index];
                    if !gpu.can_allocate(request) {
                        let err_msg = format!(
                            "Fail to trade option on GPU {} because resources are insufficient",
                            gpu_index
                        );
                        error!("{}", err_msg);
                        return Err(err_msg);
                    }
                    gpu.add(request);
                }
            }
        }
        Ok(())
    }

    fn Cancel(&mut self, option: &GPUOption) -> Result<(), String> {
        debug!("Cancel option {:?} on GPU {:?}", option, self);

        for (i, request) in option.request.iter().enumerate() {
            let allocation = &option.allocated[i];

            if request.gpu_count > 0 {
                for &gpu_index in allocation {
                    self[gpu_index].sub(request);
                }
            } else {
                if let Some(&gpu_index) = allocation.get(0) {
                    self[gpu_index].sub(request);
                }
            }
        }
        Ok(())
    }

    /// GetFreeGPUs: Returns indices of GPUs with 100% available resources.
    fn get_free_gpus(&self) -> Vec<usize> {
        self.iter()
            .enumerate()
            .filter(|(_, gpu)| {
                gpu.core_available == gpu.core_total && gpu.memory_available == gpu.memory_total
            })
            .map(|(i, _)| i)
            .collect()
    }
}
