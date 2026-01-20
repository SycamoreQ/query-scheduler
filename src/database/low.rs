use anyhow::Result;
use fred::prelude::*;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct JobRecord {
    pub id: i64,
    pub job_id: String,
    pub task_type: String,
    pub cluster_id: i32,
    pub arrival_time: chrono::DateTime<chrono::Utc>,
    pub start_time: Option<chrono::DateTime<chrono::Utc>>,
    pub completion_time: Option<chrono::DateTime<chrono::Utc>>,
    pub status: String,
    pub cpu_allocated: i32,
    pub gpu_id: Option<i32>,
    pub gpu_cores_allocated: i32,
    pub memory_mb_allocated: i64,
    pub priority: f32,
    pub queue_wait_time_sec: Option<f32>,
    pub execution_time_sec: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ClusterMetrics {
    pub id: i64,
    pub cluster_id: i32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub cpu_utilization: f32,
    pub gpu_utilization: f32,
    pub memory_utilization: f32,
    pub queue_length: i32,
    pub active_jobs: i32,
    pub throughput: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCTSNodeData {
    pub node_id: String,
    pub parent_id: Option<String>,
    pub visit_count: usize,
    pub total_value: f64,
    pub prior_prob: f64,
    pub reward: f64,
    pub action: Option<ActionKeyData>,
    pub children_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionKeyData {
    pub task_idx: i64,
    pub cpu_bin: i64,
    pub gpu_idx: i64,
    pub mem_bin: i64,
}

#[async_trait::async_trait]
pub trait StorageLayer: Send + Sync {
    async fn store_job(&self, job: &JobRecord) -> Result<i64>;
    async fn update_job_status(&self, job_id: &str, status: &str) -> Result<()>;
    async fn get_job(&self, job_id: &str) -> Result<Option<JobRecord>>;
    async fn store_metrics(&self, metrics: &ClusterMetrics) -> Result<()>;
    async fn get_recent_metrics(&self, cluster_id: i32, limit: i64) -> Result<Vec<ClusterMetrics>>;
}

pub struct PostgresStorage {
    pool: PgPool,
}

impl PostgresStorage {
    pub async fn new(database_url: String) -> Result<Self> {
        let pool = PgPool::connect(database_url).await?;
        Ok(Self { pool })
    }

    pub async fn initialize(self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS jobs (
                id BIGSERIAL PRIMARY KEY
                task_type VARCHAR(225) UNIQUE NOT NULL
                cluster_id INTEGER NOT NULL
                arrival_time TIMESTAMPTZ NOT NULL,
                start_time TIMESTAMPTZ,
                completion_time TIMESTAMPTZ,
                status VARCHAR(50) NOT NULL,
                cpu_allocated INTEGER,
                gpu_id INTEGER,
                gpu_cores_allocated INTEGER,
                memory_mb_allocated BIGINT,
                priority REAL,
                queue_wait_time_sec REAL,
                execution_time_sec REAL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS cluster_metrics (
                id BIGSERIAL PRIMARY KEY,
                cluster_id INTEGER NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                cpu_utilization REAL NOT NULL,
                gpu_utilization REAL NOT NULL,
                memory_utilization REAL NOT NULL,
                queue_length INTEGER NOT NULL,
                active_jobs INTEGER NOT NULL,
                throughput REAL NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Create indices
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_jobs_job_id ON jobs(job_id)")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_jobs_cluster_id ON jobs(cluster_id)")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_metrics_cluster_timestamp ON cluster_metrics(cluster_id, timestamp DESC)")
            .execute(&self.pool)
            .await?;

        Ok(())
    }
}

#[async_trait::async_trait]
impl StorageLayer for PostgresStorage {
    pub async fn store_job(&self, job: &JobRecord) -> Result<i64> {
        let rec = sqlx::query!(
            r#"
            INSERT INTO jobs (
                job_id, task_type, cluster_id, arrival_time, start_time,
                completion_time, status, cpu_allocated, gpu_id,
                gpu_cores_allocated, memory_mb_allocated, priority,
                queue_wait_time_sec, execution_time_sec
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            RETURNING id
            "#,
            job.job_id,
            job.task_type,
            job.cluster_id,
            job.arrival_time,
            job.start_time,
            job.completion_time,
            job.status,
            job.cpu_allocated,
            job.gpu_id,
            job.gpu_cores_allocated,
            job.memory_mb_allocated,
            job.priority,
            job.queue_wait_time_sec,
            job.execution_time_sec,
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(rec.id)
    }

    pub async fn update_job_status(&self, job_id: &str, status: &str) -> Result<()> {
        let rec = sqlx::query!(
            r#"
            UPDATE jobs SET status = $1 WHERE job_id = $2
            "#,
            status,
            job.id,
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn get_job(&self, job_id: &str) -> Result<Option<JobRecord>> {
        let job = sqlx::query_as!(
            JobRecord,
            r#"
            SELECT * FROM jobs WHERE job_id = $1
            "#,
            job.id
        )
        .fetch_optional(&self.pool)
        .await?;
        Ok(job)
    }
}
