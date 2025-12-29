use async_nats::jetstream::consumer::PullConsumer;
use futures::StreamExt;
use std::time::Duration;

pub struct GpuWorker {
    consumer: PullConsumer,
    batch_size: usize,
    max_wait: Duration,
}


impl GPUWorker{ 
    pub async fn process_batch(&self) -> Result<() , Box<dyn std::error::Error>> {
        let mut batch = self.consumer
            .fetch()
            .max_messages(batch_size)
            .expires(max_wait)
            .messages()
            .await?;
        
        let mut current_batch_data = Vec::new();
        let mut messages_to_ack = Vec::new();
        
        while let Some(Ok(msg)) = batch.next().await {
            if let Ok(event) = serde::json::from_slice<Event>(&msg.payload){
                current_batch_data.push(event);
                messages_to_ack.push(msg);
            }
        }
        
        if !current_batch_data.is_empty(){
            self.run_gpu_inference(current_batch_data).await?;
            
            for msg in messages_to_ack{
                msg.double_ack().await.ok();
            }
        }
    }
}