use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use k8s_openapi::api::core::v1::{Container, Pod, ResourceRequirements};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use bytes::{Bytes , buff , BytesMut}; 
use std::io::Write;
use std::hash::{DefaultHasher, Hash, Hasher};
use pod::*; 
use gpu::*;

pub const NOT_NEED_GPU:usize = -1;
pub const NOT_NEED_RATE: usize = -2 ;

#[derive(Debug , Hash , Deserialize , Serialize)]
pub struct GPURequest{
    pub units: Vec<GPUUnit> 
}

impl GPURequest{ 
    fn String(&self) -> String{
        let mut buffer = Vec::new();
        for (_ , r) in self.units{
            write!(buffer , r.to_string());
        }
        buffer.String()
    }
}

pub fn NewGPUReqeust(&mut self , pod: &Pod , core: &str , mem: &str) -> GPURequest{
    let mut request: Vec<GPUUnit> = vec![GPUUnit::default() , pod.spec.as_ref().unwrap().containers.len()];
    if let Some(spec) = &pod.spec {
        for (i , c) in pod.spec.as_ref().unwrap().containers(){
            let mut core = get_gpu_core_from_container(&c , core);
            let mut mem = get_gpu_mem_from_container(&c , mem);
            info!("container {} core:{}, memory: {}", c.Name, core, mem); 
            if core == 0 && mem == 0{
                request[i].Core = NOT_NEED_GPU;
                request[i].Memory = NOT_NEED_RATE;
            }
            if core >= utils.GPU_CORE_EACH_CARD {
                    request[i].gpu_count = core / utils.GPU_CORE_EACH_CARD;
                    continue;
            }
            
            request[i] = GPUUnit {
                core ,
                mem,
            }
        }
        
        info!("pod {} gpu request: {}", pod.Name, request);
        request 
    }
}

pub struct GPUOption{
    pub request: GPURequest, 
    pub Allocated: Vec<Vec<usize>>,
    pub Score: usize
}

pub fn NewGPUOption(&mut self , request: &GPURequest) -> GPUOption{
    let mut opt = GPUOption{ 
        request,
        vec![vec![0] ; &request.len()],
        0
    };
    opt
}

pub fn new_gpu_option_from_pod(pod: &Pod, core: &str, mem: &str) -> GPUOption {
    let request = new_gpu_request(pod, core, mem);
    let mut option = GPUOption::new(request);
    if let Some(spec) = &pod.spec {
        for (i, c) in spec.containers.iter().enumerate() {
            let key = format!("elasticgpu.io/container-{}", c.name);

            if let Some(annotations) = &pod.metadata.annotations {
                if let Some(v) = annotations.get(&key) {
                    log::debug!("container {} gpu key: {}", c.name, v);
                    
                    let ids_int: Vec<i32> = v
                        .split(',')
                        .filter_map(|s| s.trim().parse::<i32>().ok()) 
                        .collect();

                    if i < option.allocated.len() {
                        option.allocated[i] = ids_int;
                    }
                }
            }
        }
    }

    // Logging the final state
    log::debug!(
        "pod {}/{} allocated gpu: {:?}",
        pod.metadata.namespace.as_deref().unwrap_or("default"),
        pod.metadata.name.as_deref().unwrap_or("unknown"),
        option.allocated
    );

    option
}