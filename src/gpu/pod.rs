use k8s_openapi::api::core::v1::Pod;
use kube::{Api, Client};
use serde_json::json;

pub fn is_completed_pod(pod: &Pod) -> bool {
    if pod.metadata.deletion_timestamp.is_some() {
        return true;
    }
    
    if let Some(status) = &pod.status {
        if let Some(phase) = &status.phase {
            if phase == "Succeeded" || phase == "Failed" {
                return true;
            }
        }
    }

    false
}

pub fn is_gpu_pod(&self , )