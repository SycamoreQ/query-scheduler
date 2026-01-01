use k8s_openapi::api::core::v1::{Container, Pod, ResourceRequirements};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use std::collections::BTreeMap;

pub const RESOURCE_GPU_CORE: &str = "elasticgpu.io/gpu-core";
pub const RESOURCE_GPU_MEM: &str = "elasticgpu.io/gpu-memory";
pub const RESOURCE_QGPU_CORE: &str = "elasticgpu.io/qgpu-core";
pub const RESOURCE_QGPU_MEM: &str = "elasticgpu.io/qgpu-memory";
pub const RESOURCE_PGPU: &str = "elasticgpu.io/pgpu";

pub const ANNOTATION_EGPU_CONTAINER: &str = "elasticgpu.io/container-%s";
pub const EGPU_ASSUMED: &str = "elasticgpu.io/assumed";
pub const NOT_NEED_GPU: i32 = -1;
pub const GPU_CORE_EACH_CARD: i32 = 100;

pub fn is_completed_pod(pod: &Pod) -> bool {
    if pod.metadata.deletion_timestamp.is_some() {
        return true;
    }
    if let Some(status) = &pod.status {
        if let Some(phase) = &status.phase {
            return phase == "Succeeded" || phase == "Failed";
        }
    }
    false
}

pub fn is_gpu_pod(pod: &Pod) -> bool {
    let resources = [
        RESOURCE_GPU_CORE,
        RESOURCE_GPU_MEM,
        RESOURCE_QGPU_CORE,
        RESOURCE_QGPU_MEM,
        RESOURCE_PGPU,
    ];
    for res in resources {
        if is_resource_exists(pod, res) {
            return true;
        }
    }
    false
}

pub fn is_resource_exists(pod: &Pod, resource_name: &str) -> bool {
    if let Some(spec) = &pod.spec {
        for container in &spec.containers {
            if let Some(resources) = &container.resources {
                if let Some(limits) = &resources.limits {
                    if limits.contains_key(resource_name) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

pub fn get_resource_requests(pod: &Pod, resource_name: &str) -> u64 {
    let mut total = 0;
    if let Some(spec) = &pod.spec {
        for container in &spec.containers {
            if let Some(resources) = &container.resources {
                if let Some(limits) = &resources.limits {
                    if let Some(val) = limits.get(resource_name) {
                        // Quantity.0 is the String value in k8s-openapi
                        total += val.0.parse::<u64>().unwrap_or(0);
                    }
                }
            }
        }
    }
    total
}

pub fn get_updated_pod_annotation_spec(old_pod: &Pod, ids: Vec<Vec<i32>>) -> Pod {
    let mut new_pod = old_pod.clone();

    // Ensure Labels and Annotations exist
    if new_pod.metadata.labels.is_none() {
        new_pod.metadata.labels = Some(BTreeMap::new());
    }
    if new_pod.metadata.annotations.is_none() {
        new_pod.metadata.annotations = Some(BTreeMap::new());
    }

    let labels = new_pod.metadata.labels.as_mut().unwrap();
    let annotations = new_pod.metadata.annotations.as_mut().unwrap();

    if let Some(spec) = &new_pod.spec {
        for (i, container) in spec.containers.iter().enumerate() {
            if ids[i][0] == NOT_NEED_GPU {
                continue;
            }

            let ids_str: Vec<String> = ids[i].iter().map(|id| id.to_string()).collect();
            let key = ANNOTATION_EGPU_CONTAINER.replace("%s", &container.name);
            annotations.insert(key, ids_str.join(","));
        }
    }

    annotations.insert(EGPU_ASSUMED.to_string(), "true".to_string());
    labels.insert(EGPU_ASSUMED.to_string(), "true".to_string());

    new_pod
}

pub fn is_assumed(pod: &Pod) -> bool {
    pod.metadata
        .annotations
        .as_ref()
        .and_then(|ann| ann.get(EGPU_ASSUMED))
        .map(|val| val == "true")
        .unwrap_or(false)
}

pub fn get_container_assign_index(pod: &Pod, container_name: &str) -> anyhow::Result<Vec<String>> {
    let key = ANNOTATION_EGPU_CONTAINER.replace("%s", container_name);
    let annotations = pod
        .metadata
        .annotations
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("pod annotations are empty"))?;

    let index_array = annotations.get(&key).ok_or_else(|| {
        anyhow::anyhow!(
            "pod's annotation doesn't contain container {}",
            container_name
        )
    })?;

    Ok(index_array.split(',').map(|s| s.to_string()).collect())
}

pub fn get_gpu_core_from_container(container: &Container, resource: &str) -> i32 {
    container
        .resources
        .as_ref()
        .and_then(|r| r.requests.as_ref())
        .and_then(|req| req.get(resource))
        .and_then(|val| val.0.parse::<i32>().ok())
        .unwrap_or(0)
}

pub fn get_container_gpu_resource(pod: &Pod) -> BTreeMap<String, GPUUnit> {
    let mut maps = BTreeMap::new();
    if let Some(spec) = &pod.spec {
        for container in &spec.containers {
            if container.name.is_empty() {
                continue;
            }

            let mut unit = GPUUnit::default();
            let reqs = container
                .resources
                .as_ref()
                .and_then(|r| r.requests.as_ref());

            let core1 = reqs
                .and_then(|r| r.get(RESOURCE_GPU_CORE))
                .and_then(|v| v.0.parse::<i32>().ok())
                .unwrap_or(0);
            let core2 = reqs
                .and_then(|r| r.get(RESOURCE_QGPU_CORE))
                .and_then(|v| v.0.parse::<i32>().ok())
                .unwrap_or(0);

            if core1 >= GPU_CORE_EACH_CARD || core2 >= GPU_CORE_EACH_CARD {
                unit.gpu_count += (core1 / GPU_CORE_EACH_CARD) + (core2 / GPU_CORE_EACH_CARD);
                maps.insert(container.name.clone(), unit);
                continue;
            }

            unit.core = core1 + core2;
            let mem1 = reqs
                .and_then(|r| r.get(RESOURCE_GPU_MEM))
                .and_then(|v| v.0.parse::<i32>().ok())
                .unwrap_or(0);
            let mem2 = reqs
                .and_then(|r| r.get(RESOURCE_QGPU_MEM))
                .and_then(|v| v.0.parse::<i32>().ok())
                .unwrap_or(0);
            unit.memory = mem1 + mem2;

            maps.insert(container.name.clone(), unit);
        }
    }
    maps
}
