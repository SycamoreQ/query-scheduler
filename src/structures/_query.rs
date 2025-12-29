use chrono::{DateTime, Duration, Utc};
use dsrs::Signature;
use serde::{Deserialize, Serialize};
use serde_json::Value as Metadata;
use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResearchDomain {
    #[serde(rename = "cs")]
    ComputerScience,
    Physics,
    Biology,
    Math,
    Chemistry,
    Medicine,
    Engineering,
    General,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QueryPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum target_entity {
    paper_title(String),
    author(String),
    venue(String),
    collaboration(String),
    communities(String),
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum operations {
    find,
    citations,
    references,
    collaborations,
    authors,
    papers,
    related,
    count,
    traverse,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Constraint {
    field(String),
    operator = vec![
        "equals",
        "contains",
        "greater_than",
        "less_than",
        "between",
        "in_list",
    ],
    value(Any),
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct QueryIntent {
    pub author: Optional<String>,
    pub paper_title: Optional<String>,
    pub venue: Optiona<String>,
    pub field_of_study: Optional<ResearchDomain>,
    pub min_citation_count: Optional<usize>,
    pub max_citation_count: Optional<usize>,
    pub min_author_count: Optional<usize>,
    pub sort_by: Optional<String>,
    pub limit: Optional<String>,
    pub operation: Optiona<operations>,
    pub priority: Optional<QueryPriority>,
    pub target_entity: Optional<target_entity>,
    pub constraint: Optional<Constraint>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[Signature]
pub struct Query {
    #[input]
    pub query: String,

    #[output]
    pub target_entity: String,

    #[output]
    pub operations: String,

    #[output]
    pub semantic: String,

    #[output]
    pub contraint_json: String,

    #[output]
    pub depth: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ParsedQuery {
    pub raw: String,
    pub estimated_input: u64,
    pub selectivity: f32,
    pub complexity_score: u32,
}

#[derive(Debug, Clone)]
pub struct ComplexityMetrics {
    pub data_volume_score: f64,
    pub computation_score: f64,
    pub graph_traversal_score: f64,
    pub string_operation_score: f64,
    pub join_complexity_score: f64,
}

impl ComplexityMetrics {
    pub fn total_complexity(&self, weights: &ComplexityWeights) -> f64 {
        weights.data_volume * self.data_volume_score
            + weights.computation * self.computation_score
            + weights.graph_traversal * self.graph_traversal_score
            + weights.string_operations * self.string_operation_score
            + weights.join_operations * self.join_complexity_score
    }
}

#[derive(Debug, Clone)]
pub struct ComplexityWeights {
    pub data_volume: f64,
    pub computation: f64,
    pub graph_traversal: f64,
    pub string_operations: f64,
    pub join_operations: f64,
}

impl Default for ComplexityWeights {
    fn default() -> Self {
        Self {
            data_volume: 1.0,
            computation: 1.5,
            graph_traversal: 2.0,
            string_operations: 1.2,
            join_operations: 1.8,
        }
    }
}

pub struct ComplexityCalculator {
    pub weights: ComplexityWeights,
    pub paper_title_count: HashMap<String, usize>,
    pub author_name_count: HashMap<String, usize>,
    pub venue_count: HashMap<String, usize>,
    pub field_count: HashMap<String, usize>, 
    pub avg_collabs: f64,
    pub knowledge_graph: Option<Graph<String, (), petgraph::Directed>>,
}

pub trait Parser {
    fn forward(&mut self, query: String) -> Result<QueryIntent, ()>;
    fn estimate_out_vol(&self) -> u64;
    async fn calculate_cost(&self) -> Result<float32, ()>;
    async fn calculate_total_job_cost(stages: Vec<ParsedQuery>) -> u32;
}

#[allow(async_fn_in_trait)]
impl Parser for ParsedQuery {
    fn forward(&mut self, query: String) -> Result<QueryIntent, ()> {
        let predictor = Predict::new(QueryIntent::new());
        result = predictor.forward(&query).await?;

        target_entity = result.target_entity.to_lowercase().trim();
        operations = result.operations.to_lowercase().trim();
        semantic = result.semantic.to_lowercase().trim();

        let mut constraints_str = result.constraint_json.trim()?;
        match constraints_str.startswith {
            Some("```") => &constraints_str.split("```").nth(1).unwrap_or()?,
            Some("json") => &constraints_str[4..],
        };

        let constraints_str = &constraints_str.trim()?;
        let constraints_data: Vec<Constraint> = serde_json::from_str(constraints_str)?;

        let valid_targets = vec!["papers", "authors", "collaborators", "communities"];
        if None(valid_targets.contains(&target_entity)) {
            target_entity = "papers".to_string();
        }

        let valid_ops = vec![
            "find",
            "citations",
            "references",
            "authors",
            "papers",
            "collaborators",
            "related",
            "count",
            "traverse",
        ];
        if None(valid_ops.contains(&operations)) {
            operations = "find".to_string();
        }

        Ok(QueryIntent(
            target_entity = target_entity,
            operation = operation,
            semantic = semantic,
            constraints = constraints,
        ))?;

        for constraint in constraint_data {
            match constraint.field {
                "venue" => QueryIntent.venue == "venue",
                "author" => QueryIntent.author == "author",
                "paper_title" => QueryIntent.paper_title == "paper_title",
                "field_of_study" => QueryIntent.field_of_study == "field_of_study",
                "year" => QueryIntent.year == "year",
                "citation_count" => QueryIntent.citation_count == "citation_count",
            };
        }
    }

    fn estimate_out_vol(&self) -> u64 {
        (self.estimated_input as f32 * self.selectivity) as u64
    }
}



impl ComplexityCalculator{
    pub fn new(weights: ComplexityWeights)-> Self{
        Self{
            weights,
            venue_paper_counts: HashMap::new(),
            field_paper_counts: HashMap::new(),
            avg_collaborations: 3.5,
            knowledge_graph: None,
        }
    }
    
    pub fn calculate_complexity(&self, parsed_query: &ParsedQuery) -> ComplexityMetrics {
        let data_volume = self.estimate_data_volume(parsed_query);
        let graph_traversal = self.estimate_graph_traversal(parsed_query);
        let string_ops = self.estimate_string_operations(parsed_query);
        let join_ops = self.estimate_join_complexity(parsed_query);

        ComplexityMetrics {
            data_volume_score: data_volume,
            computation_score: computation,
            graph_traversal_score: graph_traversal,
            string_operation_score: string_ops,
            join_complexity_score: join_ops,
        }
    }
    
    pub fn estimate_data_volume(&self , query_intent:&QueryIntent) -> f64{
        let mut base_cardinality= 1000.0;
        let mut selectivity = 1.0;
        
        if let Some(venue) = &query_intent.venue{
            let venue_cardinality = self.get_venue_cardinality(venue).await?;
            base_cardinality = venue_cardinality as f64;
            selectivity *= 0.1;
        }
        
        if let Some(author) = &query_intent.author{
            let author_cardinality = self.get_author_cardinality(author).await?;
            base_cardinality *= author_card as f64;
            selectivity *= 0.05; 
        }
        
        if let Some(field) = &query_intent.field_of_study {
            let field_card = self.field_count
                .get(&format!("{:?}", field))
                .copied()
                .unwrap_or(5000);
            base_cardinality *= field_card as f64 * 0.3;
        }
        
        let citation_selectivity = self.estimate_citation_selectivity(query_intent);
        selectivity *= citation_selectivity;
        
        if let Some(limit) = query_intent.limit {
            base_cardinality = base_cardinality.min(limit as f64);
        }
        
        let final_volume = base_cardinality * selectivity;
        
        Ok((final_volume.ln() + 1.0).max(1.0))
    }
    
    async fn get_venue_cardinality(&self , venue: &str) -> Result<usize , Box<dyn std::error::Error>>{
        if let Some(graph) = &self.knowledge_graph{
            let query = query(
                "MATCH (p:Paper)-[:PUBLISHED_IN]->(v:Venue {name: $venue})
                 RETURN count(p) as count"
            ).param("venue" , venue);
            
            let mut result = graph.execute(query).await?;
            if let Some(row) = result.next().await? {
                let count: i64 = row.get("count")?;
                return Ok(count as usize);
            }
        }
        
        Ok(self.venue_count.get(venue).copied().unwrap_or(100));
    }
    
    
    async fn get_author_cardinality(&self , author: &str) -> Result<usize , Box<dyn std::error::Error>>{
        if let Some(graph) = &self.knowledge_graph {
            let query = query(
                "MATCH (a:Author {name: $author})-[:AUTHORED]->(p:Paper)
                 RETURN count(p) as count"
            ).param("author", author);
            
            let mut result = graph.execute(query).await?;
            if let Some(row) = result.next().await? {
                let count: i64 = row.get("count")?;
                return Ok(count as usize);
            }
        }
        Ok(self.author_name_count.get(author).copied().unwrap_or(50))
    }
    
    fn estimate_citation_selectivity(&self , query_intent: &QueryIntent) -> f64{
        match(query_intent.min_citation_count , query_intent.max_citation_count){
            (Some(min) , Some(max)) => {
                let range_width = (max - min) as f64;
                let base_selectivity = 0.2;
                base_selectivity * (1.0 / (range_width / 10.0 + 1.0))
            }
            
            (Some(min), None) => {
                // High citation threshold is very selective
                if min > 100 {
                    0.05
                } else if min > 50 {
                    0.15
                } else {
                    0.3
                }
            }
            
            (None, Some(max)) => {
                // Low citation filter is less selective
                if max < 10 {
                    0.4
                } else {
                    0.7
                }
            }
            (None, None) => 1.0,  
        }
    }
    
    fn estimate_computation(&self, query_intent: &QueryIntent) -> f64 {
        let base_ops = match &query_intent.operation {
            Some(operations::find) => 1.0,
            Some(operations::count) => 1.2,
            Some(operations::authors) => 2.0,
            Some(operations::papers) => 2.0,
            Some(operations::citations) => 3.5,
            Some(operations::references) => 3.5,
            Some(operations::related) => 4.5,
            Some(operations::collaborations) => 5.0,
            Some(operations::traverse) => 6.0,
            None => 1.5,
        };
        
        let sort_multiplier = if query_intent.sort_by.is_some() {
            1.5
        } else {
            1.0
        };
        
        let aggregation_multiplier = match &query_intent.target_entity {
            Some(target_entity::communities) => 2.0,// Community detection is expensive
            _ => 1.0,
        };
        
        base_ops * sort_multiplier * aggregation_multiplier
    }
    
    async fn estimate_graph_traversal(&self , query_intent: &QueryIntent) -> Result<f64 , Box<dyn std::error::Error>>{
        let depth = self.estimate_graph_depth(query_intent).await?;
        
        let base_traversal_cost = match &query_intent.operations{
            Some(operations::citations) | Some(operations::references) => {
                let avg_citations  = 1.7;
                (avg_citations.powf(depth as f64)/2.0).ln();
            }
            
            Some(operations::collaborations) => {
                (self.avg_collabs.powf(depth as f64)).ln()
            }
            Some(operations::traverse) => {
                let avg_degree = 8.0;
                (avg_degree.powf(depth as f64 / 1.5)).ln()
            }
            Some(operations::related) => {
                depth as f64 * 2.5
            }
            _ => 1.0, 
        };
        
        Ok(base_traversal_cost.min(10.0).max(1.0))
    }
    
    async fn estimate_query_depth(
        &self,
        query_intent: &QueryIntent
    ) -> Result<u32, Box<dyn std::error::Error>> {
        
        // For citation/collaboration queries, query the graph
        if let Some(graph) = &self.knowledge_graph {
            match &query_intent.operation {
                Some(operations::citations) | Some(operations::references) => {
                    // Sample depth from a representative paper
                    let query = query(
                        "MATCH path = (p:Paper)-[:CITES*1..5]->(cited:Paper)
                         RETURN avg(length(path)) as avg_depth
                         LIMIT 100"
                    );
                    
                    let mut result = graph.execute(query).await?;
                    if let Some(row) = result.next().await? {
                        let depth: f64 = row.get("avg_depth")?;
                        return Ok(depth.ceil() as u32);
                    }
                }
                Some(operations::collaborations) => {
                    // Sample collaboration chain depth
                    let query = query(
                        "MATCH path = (a1:Author)-[:COLLABORATED_WITH*1..4]->(a2:Author)
                         RETURN avg(length(path)) as avg_depth
                         LIMIT 100"
                    );
                    
                    let mut result = graph.execute(query).await?;
                    if let Some(row) = result.next().await? {
                        let depth: f64 = row.get("avg_depth")?;
                        return Ok(depth.ceil() as u32);
                    }
                }
                _ => {}
            }
        }
        
        // Default heuristic depths
        Ok(match &query_intent.operation {
            Some(operations::citations) | Some(operations::references) => 3,
            Some(operations::collaborations) => 2,
            Some(operations::traverse) => 4,
            Some(operations::related) => 2,
            _ => 1,
        })
    }
    
    fn estimate_join_complexity(&self, query_intent: &QueryIntent) -> f64 {
        let mut join_count = 0;
        let mut join_types = Vec::new();
        
        // Count number of entities being joined
        if query_intent.author.is_some() {
            join_count += 1;
            join_types.push("author_paper");
        }
        
        if query_intent.venue.is_some() {
            join_count += 1;
            join_types.push("paper_venue");
        }
        
        if query_intent.field_of_study.is_some() {
            join_count += 1;
            join_types.push("paper_field");
        }
        
        // Operation-specific joins
        match &query_intent.operation {
            Some(operations::collaborations) => {
                join_count += 2; // Author-Paper-Author joins
                join_types.push("collaboration");
            }
            Some(operations::citations) | Some(operations::references) => {
                join_count += 1; // Paper-Citation-Paper
                join_types.push("citation");
            }
            Some(operations::related) => {
                join_count += 2; // Similarity requires multiple joins
                join_types.push("similarity");
            }
            _ => {}
        }
        
        if join_count == 0 {
            return 1.0;
        }
        
        // Join cost models [web:16]
        // WCO Join: cost = card(V1..Vk-1) * |edges| / |vertices|
        // Binary Join: 2*min(card(V1), card(V2)) + max(card(V1), card(V2))
        
        let base_join_cost = match join_count {
            1 => 2.0,
            2 => 4.5,
            3 => 8.0,
            _ => 12.0 + (join_count as f64 - 3.0) * 3.0,
        };
        
        // Expensive join types [web:44]
        let expensive_multiplier = if join_types.contains(&"collaboration") 
            || join_types.contains(&"similarity") {
            1.8
        } else {
            1.0
        };
        
        base_join_cost * expensive_multiplier
    }
    
    // FINAL COMPLEXITY SCORE
    pub async fn get_final_complexity_score(
        &self,
        query_intent: &QueryIntent
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let metrics = self.calculate_complexity(query_intent).await?;
        Ok(metrics.total_complexity(&self.weights))
    }
}

// Priority-based weight adjustment
impl ComplexityCalculator {
    pub fn adjust_weights_for_priority(&mut self, priority: QueryPriority) {
        match priority {
            QueryPriority::Critical => {
                // Critical queries should execute fastest
                // Lower weights = lower perceived complexity = higher priority
                self.weights.data_volume *= 0.7;
                self.weights.computation *= 0.7;
            }
            QueryPriority::High => {
                self.weights.data_volume *= 0.85;
                self.weights.computation *= 0.85;
            }
            QueryPriority::Low => {
                // Low priority queries can wait
                self.weights.data_volume *= 1.3;
                self.weights.computation *= 1.3;
            }
            _ => {}
        }
    }
    
    
    
}
