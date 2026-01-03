use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as Metadata;
use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CoalescePolicy {
    Earliest,
    Latest,
    All,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Trigger {
    Cron {
        expression: String,
        timezone: String,
    },
    Interval {
        seconds: u64,
    },
    Date {
        run_at: DateTime<Utc>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Query<'a> {
    pub id: Cow<'a, str>,
    pub content: Cow<'a, str>,
    pub domain: ResearchDomain,
    pub arrival_time: Option<Duration>,
    pub num_tokens: i64,
}

#[derive(Debug, Serialize, Deserialize, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Task<'a> {
    pub id: Cow<'a, str>,
    pub job_exec: Cow<'a, str>,
    pub misfire_grace_time: Option<Duration>,
    pub metadata: Metadata,
    pub running_jobs: i32,
    pub max_running_jobs: i32,
}

impl<'a> Task<'a> {
    pub fn new(id: Cow<'a, str>, job_exec: Cow<'a, str>) -> Self {
        Self {
            id,
            job_exec,
            misfire_grace_time: None,
            metadata: Metadata::Object(Default::default()),
            running_jobs: 0,
            max_running_jobs: 0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct MLTask {
    pub id: String,
    pub task_type: MLTaskType,
    pub model_type: ModelType,
    pub min_resources: ResourceRequirement,
    pub arrival_time: DateTime<Utc>,
    pub completion_time: Option<Duration>,
    pub progress: f64,
}

#[derive(Debug, Serialize, Deserialize, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ResourceRequirement {
    pub cpu_cores: usize,
    pub gpu_count: usize,
    pub gpu_memory_mb: usize,
    pub memory_mb: usize,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Schedule<'a> {
    pub id: Cow<'a, str>,
    pub task_id: Cow<'a, str>,
    pub trigger: Trigger,
    pub args: Vec<Metadata>,
    pub kwargs: Metadata,
    pub paused: bool,
    pub coalesce: CoalescePolicy,
    pub misfire_grace_time: Option<Duration>,
    pub max_jitter: Option<Duration>,
    pub job_executor: Cow<'a, str>,
    pub job_result_expiration_time: Duration,
    pub metadata: Metadata,
    pub next_fire_time: Option<DateTime<Utc>>,
    pub last_fire_time: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acquired_by: Option<Cow<'a, str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acquired_until: Option<DateTime<Utc>>,
}

impl<'a> PartialOrd for Schedule<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self.next_fire_time, other.next_fire_time) {
            (Some(s), Some(o)) => s.partial_cmp(&o),
            (None, Some(_)) => Some(std::cmp::Ordering::Greater), // None is "larger" (later)
            (Some(_), None) => Some(std::cmp::Ordering::Less),
            (None, None) => self.id.partial_cmp(&other.id),
        }
    }
}

impl<'a> Schedule<'a> {
    pub fn new(
        id: Cow<'a, str>,
        task_id: Cow<'a, str>,
        trigger: Trigger,
        coalesce: CoalescePolicy,
        job_executor: Cow<'a, str>,
    ) -> Self {
        Self {
            id,
            task_id,
            trigger,
            args: Vec::new(),
            kwargs: Metadata::Object(Default::default()),
            paused: true,
            coalesce,
            misfire_grace_time: None,
            max_jitter: None,
            job_executor,
            job_result_expiration_time: Duration::zero(),
            metadata: Metadata::Object(Default::default()),
            next_fire_time: None,
            last_fire_time: None,
            acquired_by: None,
            acquired_until: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Job<'a> {
    pub id: Cow<'a, str>,
    pub task_id: Cow<'a, str>,
    pub args: Vec<Metadata>,
    pub kwargs: Metadata,
    pub schedule_id: Option<Cow<'a, str>>,
    pub job_executor: Cow<'a, str>,
    pub jitter: Option<Duration>,
    pub start_deadline: Option<Duration>,
    pub result_expiration_time: Option<Duration>,
    pub schedule_fire_time: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acquired_by: Option<Cow<'a, str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acquired_until: Option<DateTime<Utc>>,
    pub metadata: Metadata,
    pub created_at: DateTime<Utc>,
}

pub enum JobOutcome {
    Success,
    Failure,
    Missed,
    Cancelled,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JobResult {
    pub job_id: String, // Or uuid::Uuid
    pub outcome: JobOutcome,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub return_value: Option<Metadata>, // Dynamic return value
    pub exception: Option<String>,
}
