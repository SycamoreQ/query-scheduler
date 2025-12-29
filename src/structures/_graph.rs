use chrono::{DateTime, Duration, Utc};
use dsrs::Signature;
use serde::{Deserialize, Serialize};
use serde_json::Value as Metadata;
use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};


#[derive(Debug , Clone)]
pub struct Retrieval{
    pub estimated_records: usize,
    pub estimated_bytes: usize,
    pub data_density: DataDensity,
    pub interconnectedness: f64, 
    pub parsing_overhead: ParsingOverhead,
    pub join_fan_out: f64,
}

#[derive(Debug, Clone)]
pub enum DataDensity {
    Sparse { 
        sparsity_factor: f64
    },
    Dense { 
        density_factor: f64 
    },
    HyperDense { 
        avg_connections: f64
    }
}

#[derive(Debug , Clone)]
pub struct ParsingOverhead{ 
    pub paper_title_count: HashMap<String , usize>,
    pub author_name_count: HashMap<String , usize>,
    pub field_count: HashMap<String , usize>,
    pub venue_count: HashMap<String , usize>,
    pub nested_depth: HashMap<String , usize>,
    pub min_citation_count: HashMap<String , usize>,
    pub max_citation_count: HashMap<String , usize>,
    pub operation_count: HashMap<String , usize>
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum QueryPattern {
    ExactPaperLookup,           // Single paper by ID - very sparse
    AuthorByName,                // Few records
    VenueInYear,                 // Moderate records
    FieldOfStudyBroad,           // Many papers in CS, ML, etc.
    CollaborationNetwork,        // Dense interconnections
    CitationCascade,             // Dense citation chains
    FullTextSearch,              // Expensive parsing
    CrossVenueAnalysis,          // Multiple joins
    TemporalAggregation,         // Time-series processing
}

pub struct QueryPatternAnalyzer {
    pattern_stats: HashMap<QueryPattern, PatternStatistics>,
    total_papers: usize,
    total_authors: usize,
    avg_citations_per_paper: f64,
    avg_authors_per_paper: f64,
    venue_distributions: HashMap<String, VenueStats>,
    field_distributions: HashMap<String, FieldStats>,
}

#[derive(Debug, Clone)]
pub struct PatternStatistics {
    pub avg_records_retrieved: f64,
    pub avg_bytes_retrieved: f64,
    pub avg_interconnection_degree: f64,
    pub avg_parsing_time_ms: f64,
    pub memory_footprint_mb: f64,
}

#[derive(Debug, Clone)]
pub struct VenueStats {
    pub paper_count: usize,
    pub avg_citation_density: f64,
    pub avg_collaboration_size: f64,
}

#[derive(Debug, Clone)]
pub struct FieldStats {
    pub paper_count: usize,
    pub growth_rate: f64, // Papers per year
    pub cross_field_links: f64, // Interdisciplinary connections
}


impl QueryPatternAnalyzer{
    pub fn analyze_query(&self, parsed_query: &ParsedQuery) -> RetrievalCharacteristics {
        let pattern = self.identify_pattern(parsed_query);
        
        let base_stats = self.pattern_stats.get(&pattern)
            .cloned()
            .unwrap_or_else(|| self.estimate_unknown_pattern(parsed_query));
        
        let estimated_records = self.estimate_record_count(parsed_query, &pattern);
        let estimated_bytes = self.estimate_byte_size(parsed_query, estimated_records);
        let density = self.estimate_density(parsed_query, &pattern);
        let interconnectedness = self.estimate_interconnectedness(parsed_query, &pattern);
        let parsing = self.estimate_parsing_overhead(parsed_query);
        let join_fan_out = self.estimate_join_fan_out(parsed_query);
        
        RetrievalCharacteristics {
            estimated_records,
            estimated_bytes,
            data_density: density,
            interconnectedness,
            parsing_overhead: parsing,
            join_fan_out,
        }
    }
    
    fn identify_pattern(&self , parsed_query: &ParsedQuery)-> Result<QueryPattern>{
        if query.paper_id.is_some() && query.filters.is_empty(){
            Ok(QueryPattern::ExactPaperLookup);
        }
        
        if query.author_id.is_some() && query.filters.is_empty(){
            Ok(QueryPattern::AuthorByName);
        }
        if query.field_of_study.is_some() && query.time_range.is_none() {
            Ok(QueryPattern::FieldOfStudyBroad);
        }
        
        if query.requires_collaboration_analysis || query.collaboration_hops.is_some() {
            Ok(QueryPattern::CollaborationNetwork);
        }
        
        if query.citation_depth.is_some() && query.citation_depth.unwrap() > 2 {
            Ok(QueryPattern::CitationCascade);
        }
        
        // Processing-heavy patterns
        if query.full_text_search.is_some() {
            Ok(QueryPattern::FullTextSearch);
        }
        
        if query.venues.as_ref().map_or(false, |v| v.len() > 3) {
            Ok(QueryPattern::CrossVenueAnalysis);
        }
        
        if query.time_range.is_some() && query.requires_aggregation {
            Ok(QueryPattern::TemporalAggregation);
        }
    
        QueryPattern::VenueInYear
    }
    
    fn estimate_record_count(&self , query: &ParsedQuery , pattern: &QueryPattern) -> usize{
        if Some(query.target_entity) == "papers" && Some(query.operation)== "find" {
            match pattern{
                Query::ExactPaperLookup => 1,
                Query::SemanticSearch => query.
            }
        }
    }
}