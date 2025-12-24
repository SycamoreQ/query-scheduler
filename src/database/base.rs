use crate::_enums::*;
use crate::_events::*;
use crate::_struct::*;
use chrono::{DateTime, Duration, Utc};
use std::borrow::Cow;

pub trait BaseDataStore: DataStore {
    pub type _event_broker: EventBroker;

    #[tokio::main()]
    async fn start(&self, event_broker: EventBroker) -> Result<()> {
        self._event_broker = event_broker;
        Ok(());
    }
}

pub trait BaseExternalDataStore: BaseDataStore {
    pub type start_from_scratch: bool = False;
}
