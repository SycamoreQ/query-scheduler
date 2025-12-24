use futures::future::BoxFuture;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub type EventCallback = Arc<dyn Fn(Box<dyn AnyEvent>) -> BoxFuture<'static, ()> + Send + Sync>;

pub trait AnyEvent: Send + Sync + 'static {
    fn event_type(&self) -> &'static str;
    fn as_any(&self) -> &dyn std::any::Any;
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum Event {
    TaskAdded(TaskAdded),
    TaskUpdated(TaskUpdated),
    TaskRemoved(TaskRemoved),
    ScheduleAdded(ScheduleAdded),
    ScheduleRemoved(ScheduleRemoved),
    ScheduleUpdated(ScheduleUpdated),
    JobAdded(JobAdded),
    JobRemoved(JobRemoved),
    ScheduleDeserializationFailed(SchedulerDeserializationFailed),
    JobDeserializationFailed(JobDeserializationFailed),
    JobAcquired(JobAcquired),
    ScheduleStarted(ScheduleStarted),
    ScheduleStopped(ScheduleStopped),
    JobReleased(JobReleased),
}

impl AnyEvent for Event {
    fn event_type(&self) -> &'static str {
        match self {
            Event::TaskAdded(_) => "TaskAdded",
            Event::TaskUpdated(_) => "TaskUpdated",
            Event::TaskRemoved(_) => "TaskRemoved",
            Event::ScheduleAdded(_) => "ScheduleAdded",
            Event::ScheduleRemoved(_) => "ScheduleRemoved",
            Event::ScheduleUpdated(_) => "ScheduleUpdated",
            Event::JobAdded(_) => "JobAdded",
            Event::JobRemoved(_) => "JobRemoved",
            Event::ScheduleDeserializationFailed(_) => "ScheduleDeserializationFailed",
            Event::JobDeserializationFailed(_) => "JobDeserializationFailed",
            Event::JobAcquired(_) => "JobAcquired",
            Event::ScheduleStarted(_) => "ScheduleStarted",
            Event::ScheduleStopped(_) => "ScheduleStopped",
            Event::JobReleased(_) => "JobReleased",
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct Subscription {
    pub id: u64,
    pub event_types: Option<Vec<String>>,
    pub callback: EventCallback,
    pub one_shot: bool,
}

pub struct LocalEventBroker {
    subscriptions: Arc<RwLock<HashMap<u64, Subscription>>>,
    next_token: std::sync::atomic::AtomicU64,
}

impl LocalEventBroker {
    pub fn new() -> Self {
        Self {
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            next_token: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub async fn subscribe(
        &self,
        callback: EventCallback,
        types: Option<Vec<String>>,
        one_shot: bool,
    ) -> u64 {
        let token = self
            .next_token
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let sub = Subscription {
            id: token,
            event_types: types,
            callback,
            one_shot,
        };
        self.subscriptions.write().await.insert(token, sub);
        token
    }

    pub async fn publish_local(&self, event: Arc<dyn AnyEvent>) {
        let mut subs = self.subscriptions.write().await;
        let event_type = event.event_type();
        let mut to_remove = Vec::new();

        for (token, sub) in subs.iter() {
            let matches = sub
                .event_types
                .as_ref()
                .map_or(true, |types| types.iter().any(|t| t == event_type));

            if matches {
                let cb = sub.callback.clone();
                let event_clone = event.clone();

                tokio::spawn(async move {
                    cb(event_clone).await;
                });

                if sub.one_shot {
                    to_remove.push(*token);
                }
            }
        }

        for token in to_remove {
            subs.remove(&token);
        }
    }
}

pub struct ExternalEventBroker {}

impl ExternalEventBroker {
    pub fn generate_notification<E: AnyEvent + Serialize>(&self, event: &E) -> &[u8] {}
}
