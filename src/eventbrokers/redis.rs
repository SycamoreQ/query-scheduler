use redis::AsyncCommands;
use redis::aio::PubSub;
use serde::{Deserialize, Serialize};

pub struct RedisEventBroker {
    client: redis::Client,
    channel: String,
    local_broker: LocalEventBroker,
    serializer: Box<dyn Serializer + Send + Sync>,
}

impl RedisEventBroker {
    pub fn new(url: &str, channel: String) -> Self {
        let client = redis::Client::open(url).unwrap();
        Self {
            client,
            channel,
            local_broker: LocalEventBroker::new(),
            serializer: Box::new(JSONSerializer::new()),
        }
    }

    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut conn = self.client.get_async_connection().await?;
        let mut pubsub = conn.into_pubsub();
        pubsub.subscribe(&self.channel).await?;

        let local = self.local_broker.clone();

        tokio::spawn(async move {
            let mut stream = pubsub.on_message();
            while let Some(msg) = stream.next().await {
                let payload: Vec<u8> = msg.get_payload().unwrap();
                if let Some(event) = self.reconstitute_event(payload) {
                    local.publish_local(Arc::new(event)).await;
                }
            }
        });

        Ok(())
    }

    pub async fn publish(&self, event: &dyn AnyEvent) -> Result<(), Box<dyn std::error::Error>> {
        let mut conn = self.client.get_async_connection().await?;

        let payload = self.generate_notification(event);

        conn.publish(&self.channel, payload).await?;
        Ok(())
    }

    fn generate_notification(&self, event: &dyn AnyEvent) -> Vec<u8> {
        let type_name = event.event_type();
        let serialized = self.serializer.serialize(event);

        let mut notification = type_name.as_bytes().to_vec();
        notification.push(b' ');
        notification.extend(serialized);
        notification
    }
}
