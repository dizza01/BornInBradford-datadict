# Born in Bradford Data Model

This document describes the logical data model for the Born in Bradford cohort study, showing how tables and entities relate to each other.

## Overview

The BiB data model is organized around **people** (mothers, fathers, children), **pregnancies**, and **geographic locations**, with various data collected at different timepoints and through different studies.

## Core Entity-Relationship Model

```mermaid
erDiagram
    PERSON ||--o{ PREGNANCY : "mother_of"
    PERSON ||--o{ PREGNANCY : "father_of"
    PREGNANCY ||--o{ PERSON : "results_in"
    PERSON ||--o{ PERSON : "related_to"
    PERSON ||--o{ PERSON_ADDRESS : "lives_at"
    PERSON_ADDRESS }o--|| PROPERTY : "located_at"
    PROPERTY }o--|| LSOA : "within"
    
    PERSON {
        string person_id PK
        string person_type "mother|father|child"
        string ethnicity
        date date_of_birth
        string sex
    }
    
    PREGNANCY {
        string pregnancy_id PK
        string mother_id FK
        string father_id FK
        date estimated_delivery_date
        date recruitment_date
        string cohort "BiB|BiBBS"
    }
    
    PROPERTY {
        string uprn PK "Unique Property Reference Number"
        string postcode
        float latitude
        float longitude
    }
    
    LSOA {
        string lsoa_code PK
        string lsoa_name
        int imd_score "Index of Multiple Deprivation"
    }
    
    PERSON_ADDRESS {
        string person_id FK
        string uprn FK
        date move_in_date
        date move_out_date
    }
```

## Data Domain Structure

### 1. Baseline & Survey Data

```mermaid
erDiagram
    PERSON ||--o{ BASELINE_SURVEY : "completes"
    PERSON ||--o{ BIB1000_6M : "participates_in"
    PERSON ||--o{ BIB1000_12M : "participates_in"
    PERSON ||--o{ BIB1000_18M : "participates_in"
    PERSON ||--o{ BIB1000_24M : "participates_in"
    PERSON ||--o{ BIB1000_36M : "participates_in"
    PERSON ||--o{ GROWING_UP_SURVEY : "participates_in"
    PERSON ||--o{ STARTING_SCHOOL : "participates_in"
    PERSON ||--o{ PRIMARY_SCHOOL_YEARS : "participates_in"
    
    BASELINE_SURVEY {
        string person_id FK
        string survey_type "mother|father"
        int phase "1|2|3"
        date survey_date
    }
    
    BIB1000_12M {
        string person_id FK
        date assessment_date
        string child_id FK
    }
    
    GROWING_UP_SURVEY {
        string person_id FK
        string survey_type "adult|child_adult_comp|child_self_comp"
        date survey_date
    }
```

### 2. Health & Clinical Data

```mermaid
erDiagram
    PREGNANCY ||--o{ MATERNITY_RECORDS : "has"
    PREGNANCY ||--o{ ULTRASOUND_SCAN : "has"
    PREGNANCY ||--o{ PREGNANCY_BLOODS : "has"
    PERSON ||--o{ BIOSAMPLES : "provides"
    PERSON ||--o{ CHILD_GROWTH : "measured"
    PERSON ||--o{ BLOOD_PRESSURE : "measured"
    PERSON ||--o{ DXA_SCAN : "undergoes"
    
    MATERNITY_RECORDS {
        string pregnancy_id FK
        string mother_id FK
        string record_type "preg|baby|admiss"
    }
    
    BIOSAMPLES {
        string person_id FK
        string sample_type "blood|cord|urine"
        date collection_date
        string study_context
    }
    
    CHILD_GROWTH {
        string person_id FK
        date measurement_date
        string data_source "NCMP|research|primary_care"
        float height_cm
        float weight_kg
    }
```

### 3. Geographic & Environmental Data

```mermaid
erDiagram
    PROPERTY ||--|| AIR_QUALITY : "has"
    PROPERTY ||--|| BUILT_ENVIRONMENT : "has"
    PROPERTY ||--|| FOOD_ENVIRONMENT : "has"
    PROPERTY ||--|| GREEN_SPACE : "has"
    PROPERTY ||--o{ NDVI_LONGITUDINAL : "measured"
    LSOA ||--|| LSOA_CHARACTERISTICS : "has"
    
    AIR_QUALITY {
        string uprn FK
        float pm25
        float pm10
        float no2
        int year
    }
    
    FOOD_ENVIRONMENT {
        string uprn FK
        int fast_food_count_400m
        int supermarket_count_1000m
    }
    
    NDVI_LONGITUDINAL {
        string uprn FK
        date measurement_date
        float ndvi_100m
        float ndvi_300m
    }
```

### 4. Omics & Biomarker Data

```mermaid
erDiagram
    PERSON ||--o{ GENOTYPING : "has"
    PERSON ||--o{ EXOME_SEQUENCING : "has"
    PERSON ||--o{ DNA_METHYLATION : "has"
    BIOSAMPLES ||--o{ METABOLOMICS : "analyzed"
    BIOSAMPLES ||--o{ PROTEOMICS : "analyzed"
    BIOSAMPLES ||--o{ GLYCOMICS : "analyzed"
    
    GENOTYPING {
        string person_id FK
        string array_type
        string genotype_id
        date analysis_date
    }
    
    METABOLOMICS {
        string sample_id FK
        string platform "MS|NMR"
        string person_id FK
        string timepoint
    }
    
    DNA_METHYLATION {
        string person_id FK
        string array_type "450K|EPIC"
        string sample_id
    }
```

### 5. Education & Development Data

```mermaid
erDiagram
    PERSON ||--o{ EDUCATION_RECORDS : "has"
    PERSON ||--o{ SDQ_ASSESSMENTS : "completes"
    PERSON ||--o{ COGNITIVE_ASSESSMENTS : "completes"
    
    EDUCATION_RECORDS {
        string person_id FK
        string assessment_type "EYFSP|KS1|KS2|Phonics"
        int academic_year
        string school_id
    }
    
    SDQ_ASSESSMENTS {
        string person_id FK
        string study_context "BiB1000|Growing_Up|Primary_School|MeDALL"
        date assessment_date
        string respondent "parent|teacher|self"
    }
    
    COGNITIVE_ASSESSMENTS {
        string person_id FK
        string assessment_type "BPVS|CKAT|Executive_Function"
        date assessment_date
    }
```

### 6. Sub-Study Participation

```mermaid
erDiagram
    PERSON ||--o{ STUDY_PARTICIPATION : "enrolled_in"
    STUDY_PARTICIPATION }o--|| STUDY : "participates"
    
    STUDY_PARTICIPATION {
        string person_id FK
        string study_code FK
        date enrollment_date
        string consent_status
    }
    
    STUDY {
        string study_code PK
        string study_name
        string study_type
    }
```

## Key Linkage Identifiers

### Primary Identifiers

| Identifier Type | Scope | Used In | Description |
|----------------|-------|---------|-------------|
| `person_id` | Person | All person-level tables | Universal identifier for individuals |
| `pregnancy_id` | Pregnancy | Pregnancy-related tables | Links pregnancy data to mother and infant |
| `uprn` | Property | Geographic tables | Unique Property Reference Number |
| `lsoa_code` | Geographic Area | LSOA-level data | Lower Layer Super Output Area code |

### Study-Specific Identifiers

| Study | Identifier | Links To |
|-------|-----------|----------|
| BiB 1000 | `bib1000_id` | Longitudinal questionnaires at 6m, 12m, 18m, 24m, 36m |
| MeDALL | `medall_id` | Allergy studies, skin prick tests |
| Age of Wonder | `aow_id` | School visits and surveys |
| Growing Up | `growingup_id` | Adult and child surveys, DXA scans |
| PLATOON | `platoon_id` | Dental study data |

### Omics Identifiers

| Data Type | Identifier | Links To |
|-----------|-----------|----------|
| Genotyping | `genotype_id` | Person via linkage table |
| Exome Sequencing | `exome_id` | Person via linkage table |
| DNA Methylation | `dnam_id` | Person via linkage table |
| Metabolomics | `sample_id` | Biosample → Person |

## Data Granularity Levels

The BiB data exists at multiple levels of granularity:

```
Individual (Person)
    ├── Single timepoint measurements
    ├── Longitudinal measurements
    │   ├── Baseline
    │   ├── 6 months
    │   ├── 12 months
    │   ├── 18 months
    │   ├── 24 months
    │   ├── 36 months
    │   └── School age (4-11 years)
    └── Life course linkage
        ├── Pregnancy data (mother)
        ├── Birth data (child)
        ├── Childhood data
        └── Adolescence data

Property (Address)
    ├── Static characteristics
    ├── Temporal measures (NDVI over time)
    └── Proximity measures (food environment, etc.)

Area (LSOA)
    ├── Deprivation indices
    ├── Census data
    └── Aggregated characteristics
```

## Temporal Data Structure

### Longitudinal Study Design

```mermaid
gantt
    title BiB Cohort Timeline
    dateFormat YYYY-MM
    section Pregnancy
    Recruitment           :2007-01, 2011-01
    Baseline Survey       :2007-01, 2011-01
    Maternity Records     :2007-01, 2011-12
    
    section Early Childhood
    Birth                 :2007-09, 2011-12
    BiB 1000 - 6m        :2008-03, 2011-06
    BiB 1000 - 12m       :2008-09, 2012-01
    BiB 1000 - 18m       :2009-03, 2012-06
    BiB 1000 - 24m       :2009-09, 2013-01
    BiB 1000 - 36m       :2010-09, 2014-01
    
    section School Age
    Starting School       :2011-09, 2015-09
    Primary School Years  :2013-09, 2019-09
    
    section Later Studies
    Growing Up            :2016-01, 2020-12
    Age of Wonder         :2023-01, 2025-12
```

## Data Linkage Examples

### Example 1: Linking Mother's Pregnancy Data to Child Outcomes

```sql
-- Conceptual query structure
SELECT 
    p.person_id as child_id,
    pr.pregnancy_id,
    pr.mother_id,
    mb.biomarker_value as maternal_biomarker,
    cg.height_cm as child_height,
    ed.ks1_score
FROM person p
JOIN pregnancy pr ON p.pregnancy_id = pr.pregnancy_id
JOIN pregnancy_bloods mb ON pr.pregnancy_id = mb.pregnancy_id
JOIN child_growth cg ON p.person_id = cg.person_id
JOIN education_records ed ON p.person_id = ed.person_id
WHERE p.person_type = 'child'
```

### Example 2: Linking Person to Geographic Environment

```sql
-- Conceptual query structure
SELECT 
    p.person_id,
    pa.uprn,
    aq.pm25,
    ge.green_space_300m,
    fe.fast_food_count_400m,
    l.imd_score
FROM person p
JOIN person_address pa ON p.person_id = pa.person_id
JOIN property prop ON pa.uprn = prop.uprn
JOIN air_quality aq ON prop.uprn = aq.uprn
JOIN green_environment ge ON prop.uprn = ge.uprn
JOIN food_environment fe ON prop.uprn = fe.uprn
JOIN lsoa l ON prop.lsoa_code = l.lsoa_code
WHERE pa.move_out_date IS NULL  -- Current address
```

### Example 3: Longitudinal Within-Person Analysis

```sql
-- Conceptual query structure
SELECT 
    person_id,
    '6m' as timepoint, assessment_date, bmi_zscore 
FROM bib1000_6m
UNION ALL
SELECT 
    person_id,
    '12m', assessment_date, bmi_zscore 
FROM bib1000_12m
UNION ALL
SELECT 
    person_id,
    '18m', assessment_date, bmi_zscore 
FROM bib1000_18m
ORDER BY person_id, assessment_date
```

## Data Model Notes

### Design Principles

1. **Person-Centric**: Most data links back to individual persons via `person_id`
2. **Temporal Tracking**: Many tables include date fields for longitudinal analysis
3. **Hierarchical Geography**: Person → Address → Property → LSOA structure
4. **Study Flexibility**: Sub-study IDs allow participants to be in multiple studies
5. **Family Linkage**: Related pairs table captures family relationships

### Common Join Patterns

| To Link | Via |
|---------|-----|
| Mother's pregnancy data → Child's outcomes | `pregnancy.pregnancy_id` |
| Siblings | `related_pairs` table with `relationship_type = 'sibling'` |
| Mother-Child pairs | `pregnancy` table links mother_id to child via pregnancy_id |
| Person → Geographic environment | `person_id → person_address → uprn → property` |
| Longitudinal person data | Same `person_id` across different timepoint tables |
| Omics data → Person | Linkage tables (e.g., `genotyping_info`) |

### Data Cardinality

- **One Person** → Many addresses over time (residential moves)
- **One Pregnancy** → One mother, optionally one father, one or more children (twins)
- **One Person** → Many biosamples (different types, timepoints)
- **One Person** → Many SDQ assessments (different studies, timepoints)
- **One Property** → Many people over time
- **One LSOA** → Many properties

## Accessing the Data Model in the Dictionary

To explore these relationships in detail:

1. Open [docs/bib_cohortinfo_id_linkage.html](docs/bib_cohortinfo_id_linkage.html) for ID definitions
2. Check [docs/bib_cohortinfo_related_pairs.html](docs/bib_cohortinfo_related_pairs.html) for family relationships
3. Navigate to specific data domains to see available variables
4. Review linkage sections for omics data integration

## Version Information

- **Model Version**: 1.0
- **Last Updated**: February 2026
- **Based on**: BiB Data Dictionary Repository Structure
