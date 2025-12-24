use serde::{Deserialize , Serialize};
use async_nats::connect;
use tokio::time::{Duration, sleep};
use async_nats::ConnectOptions;
use futures::StreamExt;
use async_nats::jetstream::stream::{Config, RetentionPolicy, StorageType};

pub async fn setup_jetstream(js: async_nats::jetstream::Context)-> Result<() , async_nats::Error>{
    js.get_or_create_stream(Config{
        name: "JOBS_STREAM".to_string(),
                subjects: vec!["jobs.>".to_string()],
                retention: RetentionPolicy::WorkQueue, 
                storage: StorageType::File,           
                ..Default::default()
            }).await?;

    Ok(())
}

pub struct NatsEventBroker{
    client: async_nats::Client,
    jetstream: async_nats::jetstream::Context,
    subject: String,
    local_broker: LocalEventBroker,
}

impl NatsEventBroker{
    pub fn new(url : &str , context: String) -> Result<Self , async_nats::Error>{
        let client = async_nats::connect(url).await?;
        let jetstream = async_nats::jetstream::new(client.clone())?;

        Ok(Self{cleint , jetstream , subject});
    }

    pub async fn publish<E: AnyEvent + Serialize>(&self , event: &E) -> Result<() , async_nats::Error>{
        let subject = format!("jobs{}" , self.subject.to_string())?;
        let payload  = serde::json::to_vec(event)?;
        
        self.js.publish(subject , payload.into()).await?;
        
        self.local_broker.publish_local(Arc::new(event.clone())).await;
        
        Ok(())
        
    }
    
    pub async fn start_listening(&self) -> Return<() , Box<dyn std::error::Error>>{
        let stream = self.js.get_stream("JOB_STREAM").await?;
        let consumer = stream.get_or_create_consumer("schedule_listener" , async_nats::jetstream::consumer::pull::Config::default()).await?;
        
        let local = self.local_broker.clone();
                tokio::spawn(async move {
                    let mut messages = consumer.messages().await.unwrap();
                    while let Some(Ok(msg)) = messages.next().await {
                        if let Ok(event) = serde_json::from_slice<Event>(&msg.payload){
                            let shared_event = Arc::new(event);
                            self.local_broker.publish_local(shared_event).await?;
                        }
                        else{
                            self.logger.error("Failed to deserialize event from NATS");
                        }
                        msg.double_ack().await.ok(); 
                    }
                });
                
        Ok(())
            
    }
    
    
}
