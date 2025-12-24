use crate::_struct::{Job, JobResult, Schedule, Task};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

#[derive(Copy, Debug, Serialize, Deserialize)]
pub struct MemoryDataStore {
    pub tasks: HashMap<String, Task>,
    pub schedules: Vec<Schedule>,
    pub schedules_by_id: HashMap<String, Schedule>,
    pub schedules_by_task_id: HashMap<String, HashSet<Schedule>>,
    pub jobs_by_id: HashMap<Uuid, Job>,
    pub jobs_by_task_id: HashMap<String, HashSet<Job>>,
    pub jobs_by_schedule_id: HashMap<String, HashSet<Job>>,
    pub job_results: HashMap<Uuid, JobResult>,
}

impl MemoryDataStore {
    pub fn find_schedule_index(&self, schedule: &Schedule) -> Option<usize> {
        match self.schedules.binary_search(schedule) {
            Ok(index) => Some(index),
            Err(_) => None,
        }
    }

    pub async fn get_schedules<'a>(&self, ids: Option<Vec<Cow<'a, str>>>) -> Vec<Schedule> {
        match ids {
            None => self.schedules.clone(),

            Some(id_list) => self
                .schedules
                .iter()
                .filter(|s| id_list.contains(&Cow::Borrowed(s.id.as_str())))
                .cloned()
                .collect(),
        }
    }

    pub async fn add_task(&mut self, mut task: Task) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(existing_task) = self.tasks.get(&task.id) {
            task.running_jobs = existing_task.running_jobs;

            let task_id = task.id.clone();
            self.tasks.insert(task_id.clone(), task);

            self._event_broker
                .publish(TaskUpdated {
                    header: EventHeader::default(),
                    task_id: Cow::Owned(task_id),
                })
                .await?;
        } else {
            let task_id = task.id.clone();
            self.tasks.insert(task_id.clone(), task);

            self._event_broker
                .publish(TaskAdded {
                    header: EventHeader::default(),
                    task_id: Cow::Owned(task_id),
                })
                .await?;
        }

        Ok(())
    }

    pub async fn remove_task<'a>(
        &mut self,
        task_id: Cow<'a, String>,
    ) -> Result<(), Box<dyn TaskLookupError>> {
        match _ {
            Some(task_id) => self.tasks.remove(&Cow::Owned(task_id)),
            None => TaskLookupError(task_id),
        }

        Ok(())
    }

    pub async fn get_task<'a>(
        &self,
        task_id: Cow<'a, String>,
    ) -> Result<Task, Box<dyn TaskLookupError>> {
        match _ {
            Some(task_id) => self.tasks.get(&task_id),
            None => TaskLookupError(task_id),
        }

        Ok(())
    }

    pub async fn get_tasks(&self) -> Result<Vec<Task>, ()> {
        let tasks = self._tasks.values().collect()?;

        Ok(tasks);
    }

    pub async fn add_schedule(
        &mut self,
        mut schedule: Schedule,
        conflict_policy: ConflictPolicy,
    ) -> Result<(), ConflictingError> {
        let mut old_schedule = self.schedules_by_id(&Cow::Borrowed(schedule.id))?;

        match old_schedule {
            Some(old_schedule) => {
                if conflict_policy == ConflictPolicy.do_nothing {
                    OK(());
                } else if conflict_policy == ConflictPolicy.exception {
                    ConflictPolicyError(schedule.id)
                } else {
                    let mut index = self._find_schedule_index(&old_schedule)?;
                    self._schedules.remove(&Cow::Borrowed(index))?;
                    if let Some(existing_schedule) = self.schedule_by_task_id.get(&schedule.task_id)
                    {
                        existing_schedule.remove(&old_schedule)?;
                    }
                }
            }

            None => {
                if let Some(schedule) = self.schedule_by_id(&schedule.id) {
                    self._schedule_by_task_id
                        .insert(Cow::Borrowed(schedule.id), schedule)?;
                    bisection::insort_right(self._schedules, schedule);

                    let mut event: Option<ScheduleAdded, ScheduleUpdated>;
                    match old_schedule {
                        Some(old_schedule) => {
                            event = ScheduleUpdated(
                                schedule_id = schedule.id,
                                task_id = schedule.task.id,
                                next_fire_time = schedule.next_fire_time,
                            )?
                        }

                        None(old) => {
                            event = ScheduleAdded(
                                schedule_id = schedule.id,
                                task_id = schedule.task.id,
                                next_fire_time = schedule.next_fire_time,
                            )
                        }
                    }

                    self.event_broker.publish(&event).await?;
                }
            }
        }

        Ok(());
    }

    pub async fn remove_schedules<I>(
        &mut self,
        ids: I,
        finished: bool,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        I: IntoIterator<Item = String>,
    {
        for id in ids {
            if let Some(schedule) = self.schedules_by_id.remove(&id) {
                if let Ok(index) = self.schedules.binary_search(&schedule) {
                    self.schedules.remove(index);
                }

                let event = ScheduleRemoved {
                    header: EventHeader::default(),
                    schedule_id: Cow::Owned(schedule.id.clone()),
                    task_id: Cow::Owned(schedule.task_id.clone()),
                    finished,
                };

                self._event_broker.publish(event).await?;
            }
        }

        Ok(())
    }

    pub async fn acquire_schedules<'a>(
        &mut self,
        scheduler_id: Cow<'a, str>,
        lease_duration: Duration,
        limit: usize,
    ) -> Result<Vec<Schedule>, String> {
        let now = Utc::now();
        let acquired_until = now + lease_duration;
        let mut acquired_schedules: Vec<Schedule> = Vec::new();

        for schedule in &mut self.schedules {
            match schedule.next_fire_time {
                None => break,
                Some(t) if t > now => break,
                _ => {}
            }

            if schedule.paused {
                continue;
            }
            if let Some(until) = schedule.acquired_until {
                if schedule.acquired_by.as_deref() != Some(scheduler_id.as_ref()) && now <= until {
                    continue;
                }
            }

            schedule.acquired_by = Some(scheduler_id.to_string());
            schedule.acquired_until = Some(acquired_until);

            acquired_schedules.push(schedule.clone());

            if acquired_schedules.len() == limit {
                break;
            }
        }

        Ok(acquired_schedules)
    }

    pub async fn release_schedules<'a>(
        &self,
        schedule_id: Cow<'a, String>,
        results: Vec<ScheduleResult>,
    ) -> Result<()> {
        for result in &mut results {
            let mut schedule = self._schedule_by_id.get(result.schedule_id)?;
            let mut index = self._find_schedule_index(&schedule)?;
            if Some(index) {
                self.schedules.remove(index)?;
            }

            &schedule.last_fire_time = &result.last_fire_time;
            &schedule.next_fire_time = &result.next_fire_time;
            &schedule.acquired_by = None;
            &schedule.acquired_unitl = None;

            bisect::insort_right(self.schedules, schedule);
            let event: ScheduleUpdated = {
                schedule_id = Cow::Owned(schedule_id.cloned());
                task_id = Cow::Owned(schedule.task_id.cloned());
                next_fire_time = schedule.next_fire_time;
            };

            self._event_broker.publish(event).await?;
        }

        Ok(())
    }

    pub async fn get_next_schedule_run_time(&self) -> Result<DateTime<Utc>, ()> {
        let mut target = self.schedule[0].as_ref()?;
        match schedules {
            Some(schedules) => target.next_fire_time?,
            _ => None,
        }?;

        Ok(());
    }

    pub async fn add_jobs(&mut self, job: Job) -> Result<()> {
        if let Some(job) = self.jobs_by_id.get(&job.id) {
            self.jobs_by_task_id.insert(&Cow::Owned(job.task_id), job)?;

            if Some(job.schedule_id) {
                self.jobs_schedule_id
                    .insert(&Cow::Owned(job.schedule_id), job);
            }
        }

        let event: JobAdded = {
            job_id = Cow::Owned(job.id.cloned());
            task_id = Cow::Owned(task.id.cloned());
            schedule_id = Cow::Owned(schedule.id.cloned());
        };

        self._event_broker.publish(event).await?;
    }

    pub async fn get_jobs<I>(&self, ids: I) -> Result<Vec<Job>, ()>
    where
        I: IntoIterator<Item = uuid::Uuid>,
    {
        let ids = HashSet::new();

        match ids {
            Some(ids) => ids = HashSet(ids)?,
            None => self.jobs_by_id.values().collect(),
            _ => {
                for job in self.jobs_by_id.values() {
                    if ids.contains(&jobs.id) || ids == None {
                        Ok(job)
                    }
                }
            }
        }

        Ok(())
    }

    pub async fn acquire_jobs(
        &mut self,
        scheduler_id: String,
        lease_duration: Duration,
        limit: Option<usize>,
    ) -> Result<Vec<Job>, Box<dyn std::error::Error>> {
        let now = Utc::now();
        let acquired_until = now + lease_duration;
        let mut acquired_jobs = Vec::new();
        let mut job_results_to_release = Vec::new();

        for job in self.jobs_by_id.values_mut() {
            let task = self.tasks.get_mut(&job.task_id).unwrap();

            // 1. Skip or Reclaim expired leases
            if let Some(until) = job.acquired_until {
                if until >= now {
                    continue;
                } else {
                    task.running_jobs = task.running_jobs.saturating_sub(1);
                }
            }

            // 2. Discard if deadline passed
            if let Some(deadline) = job.start_deadline {
                if deadline < now {
                    let result = JobResult {
                        job_id: job.id,
                        outcome: JobOutcome::MissedStartDeadline,
                        finished_at: now,
                        expires_at: now + job.result_expiration_time,
                    };
                    job_results_to_release.push((job.clone(), result));
                    continue;
                }
            }

            // 3. Check Task limits
            if let Some(max) = task.max_running_jobs {
                if task.running_jobs >= max {
                    continue;
                }
            }

            // 4. Acquire the job
            job.acquired_by = Some(scheduler_id.clone());
            job.acquired_until = Some(acquired_until);
            task.running_jobs += 1;
            acquired_jobs.push(job.clone());

            if let Some(l) = limit {
                if acquired_jobs.len() >= l {
                    break;
                }
            }
        }

        for job in &acquired_jobs {
            self._event_broker
                .publish(JobAcquired::from_job(job, &scheduler_id))
                .await?;
        }

        for (job, result) in job_results_to_release {
            self.release_job(&scheduler_id, job, result).await?;
        }

        Ok(acquired_jobs)
    }

    pub async fn release_job(
        &mut self,
        _scheduler_id: &str,
        job: Job,
        result: JobResult,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if result.expires_at > result.finished_at {
            self.job_results.insert(result.job_id, result.clone());
        }

        if job.acquired_by.is_some() {
            if let Some(task) = self.tasks.get_mut(&job.task_id) {
                task.running_jobs = task.running_jobs.saturating_sub(1);
            }
        }

        self.jobs_by_id.remove(&result.job_id);

        if let Some(task_jobs) = self.jobs_by_task_id.get_mut(&job.task_id) {
            task_jobs.retain(|j| j.id != job.id);
            if task_jobs.is_empty() {
                self.jobs_by_task_id.remove(&job.task_id);
            }
        }

        if let Some(sched_id) = &job.schedule_id {
            if let Some(sched_jobs) = self.jobs_by_schedule_id.get_mut(sched_id) {
                sched_jobs.retain(|j| j.id != job.id);
                if sched_jobs.is_empty() {
                    self.jobs_by_schedule_id.remove(sched_id);
                }
            }
        }

        self._event_broker
            .publish(JobReleased::from_result(result, job))
            .await?;

        Ok(())
    }

    pub async fn get_job_result(&mut self, job_id: Uuid) -> Option<JobResult> {
        self.job_results.remove(&job_id)
    }

    pub async fn cleanup(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let now = Utc::now();

        self.job_results.retain(|_, result| result.expires_at > now);

        let expired_jobs: Vec<Job> = self
            .jobs_by_id
            .values()
            .filter(|j| j.acquired_until.map_or(false, |until| until < now))
            .cloned()
            .collect();

        for job in expired_jobs {
            let scheduler_id = job.acquired_by.clone().unwrap_or_default();
            let result = JobResult {
                job_id: job.id,
                outcome: JobOutcome::Abandoned,
                finished_at: now,
                expires_at: now + Duration::hours(1),
            };
            self.release_job(&scheduler_id, job, result).await?;
        }

        let finished_schedule_ids: Vec<String> = self
            .schedules_by_id
            .iter()
            .filter(|(id, s)| {
                s.next_fire_time.is_none() && !self.jobs_by_schedule_id.contains_key(*id)
            })
            .map(|(id, _)| id.clone())
            .collect();

        self.remove_schedules(finished_schedule_ids, true).await?;

        Ok(())
    }
}
