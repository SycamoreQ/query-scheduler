use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use uuid::Uuid;
use _enum::{JobOutcome};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EventHeader {
    pub timestamp: DateTime<Utc>,
}

impl Default for EventHeader {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TaskAdded<'a> {
    #[serde(flatten)]
    pub header: EventHeader,
    pub task_id: Cow<'a, str>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TaskUpdated<'a> {
    #[serde(flatten)]
    pub header: EventHeader,
    pub task_id: Cow<'a, str>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TaskRemoved<'a> {
    #[serde(flatten)]
    pub header: EventHeader,
    pub task_id: Cow<'a, str>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ScheduleAdded<'a> {
    #[serde(flatten)]
    pub header: EventHeader,
    pub schedule_id: Cow<'a, str>,
    pub task_id: Cow<'a, str>,
    pub next_fire_time: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ScheduleUpdated<'a> {
    #[serde(flatten)]
    pub header: EventHeader,
    pub schedule_id: Cow<'a, str>,
    pub task_id: Cow<'a, str>,
    pub next_fire_time: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ScheduleRemoved<'a> {
    #[serde(flatten)]
    pub header: EventHeader,
    pub schedule_id: Cow<'a, str>,
    pub task_id: Cow<'a, str>,
    pub finished: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JobAdded<'a> {
    #[serde(flatten)]
    pub header: EventHeader,
    pub job_id: Uuid,
    pub task_id: Cow<'a, str>,
    pub schedule_id: Option<Cow<'a, str>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JobRemoved<'a> {
    #[serde(flatten)]
    pub header: EventHeader,
    pub job_id: Uuid,
    pub task_id: Cow<'a, str>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ScheduleDeserializationFailed<'a> {
    #[serde(flatten)]
    pub header: EventHeader,
    pub schedule_id: Cow<'a, str>,
    pub exception: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JobDeserializationFailed<'a> {
    #[serde(flatten)]
    pub header: EventHeader,
    pub job_id: Uuid,
    pub exception: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SchedulerStarted {
    #[serde(flatten)]
    pub header: EventHeader,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SchedulerStopped {
    #[serde(flatten)]
    pub header: EventHeader,
    pub exception: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JobAcquired<'a> {
    #[serde(flatten)]
    pub header: EventHeader,
    pub job_id: Uuid,
    pub scheduler_id: Cow<'a, str>,
    pub task_id: Cow<'a, str>,
    pub schedule_id: Option<Cow<'a, str>>,
    pub scheduled_start: Option<DateTime<Utc>>,
}

impl<'a> JobAcquired<'a> {
    pub fn from_job(job: &crate::_struct::Job, scheduler_id: Cow<'a, str>) -> Self {
        Self {
            header: EventHeader::default(),
            job_id: job.id,
            scheduler_id,
            task_id: Cow::Owned(job.task_id.to_string()),
            schedule_id: job
                .schedule_id
                .as_ref()
                .map(|id| Cow::Owned(id.to_string())),
            scheduled_start: job.next_fire_time,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JobReleased<'a> {
    #[serde(flatten)]
    pub header: EventHeader,
    pub job_id: Uuid,
    pub scheduler_id: Cow<'a, str>,
    pub task_id: Cow<'a, str>,
    pub schedule_id: Option<Cow<'a, str>>,
    pub scheduled_start: Option<DateTime<Utc>>,
    pub started_at: Option<DateTime<Utc>>,
    pub outcome: JobOutcome,
    pub exception_type: Option<String>,
    pub exception_message: Option<String>,
    pub exception_traceback: Option<Vec<String>>,
}

impl<'a> JobReleased<'a> {
    pub fn from_result(
        result: &crate::_struct::JobResult,
        scheduler_id: Cow<'a, str>,
        task_id: Cow<'a, str>,
        schedule_id: Option<Cow<'a, str>>,
        scheduled_fire_time: Option<DateTime<Utc>>,
    ) -> Self {
        let (ex_type, ex_msg) = if let Some(ref ex) = result.exception {
            (Some("BaseException".to_string()), Some(ex.clone()))
        } else {
            (None, None)
        };

        Self {
            header: EventHeader {
                timestamp: result.finished_at,
            },
            job_id: result.job_id,
            scheduler_id,
            task_id,
            schedule_id,
            scheduled_start: scheduled_fire_time,
            started_at: result.started_at,
            outcome: result.outcome,
            exception_type: ex_type,
            exception_message: ex_msg,
            exception_traceback: None,
        }
    }
}
