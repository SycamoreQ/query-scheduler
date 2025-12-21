use std::borrow::Cow;
use _struct::{Job, JobResult, Schedule, ScheduleResult, Task};
use std::iter::{Iterator};
use chrono::{DateTime, Utc, Duration};
use std::any::{Any};
use log::{info , warn};
use std::pin::Pin;

pub type AsyncCallback<E> = Box<dyn FnMut(E) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send>;

pub trait Trigger: Iterator<Item = DateTime<Utc>> {
    fn next_fire(&mut self) -> Option<DateTime<Utc>>;
    fn get_state(&self) -> serde_json::Value;
    fn set_state(&mut self, state: serde_json::Value);

    fn has_next(&mut self) -> bool {
        self.next_fire().is_some()
    }
}

pub trait Subscription{
    fn enter(&self) -> Result<Subscription>{
        Ok(Subscription);
    }
    
    fn exit(&self , exec_type: Any , exc_val: Any , exc_tb: Any) -> Result<None>{
        Ok(self.unsubscribe());
    }
    
    fn unsubscribe(&self) -> Result<None>;
}

pub trait EventBroker{
    #[tokio::main()]
    async fn start(&self) -> Result<None>;
    
    #[tokio::main()]
    async fn publish(&self , event: Event) -> Result<None>;
    
    #[tokio::main()]
    async fn publish_local(&self) -> Result<None>;
    
    #[tokio::main()]
    async fn subscribe<E , F>(&mut self , mut callable: F , mut event_types: Option<Vec<T_Event>> , one_shot:bool) -> Result<Subscription>
    where 
        F:'static + Send ,
        F: FnMut(E) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + 'static ; 
}


pub trait DataStore{
    #[tokio::main()]
    async fn start(&self , event_broker: EventBroker) -> Result<()>;
    
    #[tokio::main()]
    async fn add_task<'a , String>(&self , task_id: Cow<'a , String>) -> Result<()>;
    
    #[tokio::main()]
    async fn remove_task<'a>(&self , task_id: Cow<'a , String>) -> Result<()>;
    
    #[tokio::main()]
    async fn get_task<'a>(&self , task_id: Cow<'a , String>) -> Result<Task , Error>;
    
    #[tokio::main()]
    async fn get_task(&self) -> Result<Vec<Task> , Error>;
    
    #[tokio::main()]
    async fn get_schedules<'a>(&self , ids: Vec<Cow<'a , String>>) -> Result<Vec<Schedule> , Error>;
    
    #[tokio::main()]
    async fn add_schedule(&self , schedule: Schedule , conflictPolicy: ConflictPolicy) -> Result<()>;
    
    #[tokio::main()]
    async fn remove_schedule<'a , I>(&self , ids: I) -> Result<()>
    where I: Iterator<Cow<'a , String>>;
    
    #[tokio::main()]
    async fn acquire_schedule<'a>(&self , schedule_id: Cow<'a , String> , lease_duration: Option<Duration>, limit:Integer) -> Result<Vec<Schedule> , Error>; 
    
    #[tokio::main()]
    async fn release_schedule<'a>(&self , schedule_id: Cow<'a , String> , result: Vec<ScheduleResult>) -> Result<()>;
    
    #[tokio::main()]
    async fn get_schedule_next_runtime(&self) -> Result<DateTime<Utc> , ()>;
    
    #[tokio::main()]
    async fn add_job(&self , job: Job) -> Result<()>;
    
    #[tokio::main()]
    async fn get_jobs<I>(&self , ids: I) -> Result<Vec<Job> , Error>;
    
    #[tokio::main()]
    async fn acquire_job<'a>(&self , job_id: Cow<'a , String> , lease_duration: Option<Duration>, limit:Integer) -> Result<Vec<Job> , Error>; 
    
    #[tokio::main()]
    async fn release_job<'a>(&self , job_id: Cow<'a , String> , result: Vec<JobResult>) -> Result<()>;
    
    #[tokio::main()]
    async fn get_job_result<'a>(&self , job_id: Cow<'a , String>) -> Result<JobResult , ()>;
    
    #[tokio::main()]
    async fn extend_acquired_schedule_lease<'a>(&self , schedule_id: Cow<'a , String> , schedule_ids: Vec<Cow<'a , String>> , duration: Option<Duration>) -> Result<()>;
    
    #[tokio::main()]
    async fn extend_acquired_job_lease<'a>(&self , job_id: Cow<'a , String> , job_ids: Vec<Cow<'a , String>> , duration: Option<Duration>) -> Result<()>;
    
    #[tokio::main()]
    async fn reap_abandoned_jobs<'a>(&self , schedule_id: Cow<'a , String>) -> Result<()>;
    
    #[tokio::main()]
    async fn cleanup(&self) -> Result<() , Error>; 
}

pub trait JobExecutor{ 
    #[tokio::main()]
    async fn start(&self) -> Result<() , Error>;
    
    #[tokio::main()]
    async fn run_job<F>(&self , func : F , job: Job) -> Result<Any , Error>
    where F: 'static + Send ;
} 