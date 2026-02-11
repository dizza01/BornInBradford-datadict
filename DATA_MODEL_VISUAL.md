# Born in Bradford Data Model - Visual Guide

If the Mermaid diagrams don't render properly, this document provides alternative visualizations using ASCII art and tables.

## Core Entity Relationships (Visual)

### Primary Entities and Connections

```
┌─────────────┐
│   PERSON    │ (mother, father, child)
│  person_id  │
└─────┬───────┘
      │
      ├─── mother_of ──→ ┌─────────────┐
      │                   │  PREGNANCY  │
      ├─── father_of ──→  │pregnancy_id │
      │                   └─────┬───────┘
      │                         │
      ├─── results_in ←─────────┘
      │
      ├─── related_to (siblings, family)
      │
      └─── lives_at ──→ ┌──────────────────┐
                        │  PERSON_ADDRESS  │
                        │  person_id, uprn │
                        └────────┬─────────┘
                                 │
                                 └─→ ┌──────────┐      ┌──────────┐
                                     │ PROPERTY │ ───→  │   LSOA   │
                                     │   uprn   │       │lsoa_code │
                                     └──────────┘      └──────────┘
```

### Data Linkage Flow

```
                    PERSON (person_id)
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   BASELINE          SURVEYS         BIO SAMPLES
     DATA              DATA              DATA
        │                │                │
        │                ├─→ BiB 1000    │
        │                ├─→ Growing Up  │
        │                └─→ School Data │
        │                                │
        └────────────────┬────────────────┘
                         │
                         ▼
                   LINKED OMICS
                   (genotyping,
                    methylation,
                    metabolomics)
```

## Hierarchical Structure

### Geographic Hierarchy

```
Individual Person
    │
    ├─→ Person Address (temporal - can move)
    │        │
    │        └─→ Property (UPRN)
    │                 │
    │                 ├─→ Air Quality
    │                 ├─→ Built Environment  
    │                 ├─→ Food Environment
    │                 ├─→ Green Space
    │                 └─→ LSOA
    │                      │
    │                      └─→ Area Characteristics
    │                           ├─→ Deprivation Index
    │                           └─→ Census Data
```

### Temporal Hierarchy (Longitudinal Data)

```
PERSON
  │
  ├─→ Pregnancy (mother)
  │     ├─→ Baseline Survey
  │     ├─→ Maternity Records
  │     └─→ Pregnancy Bloods
  │
  ├─→ Birth
  │
  ├─→ Early Childhood
  │     ├─→ BiB 1000 - 6 months
  │     ├─→ BiB 1000 - 12 months  
  │     ├─→ BiB 1000 - 18 months
  │     ├─→ BiB 1000 - 24 months
  │     └─→ BiB 1000 - 36 months
  │
  ├─→ School Age
  │     ├─→ Starting School (Reception)
  │     ├─→ Primary School Years
  │     └─→ Age of Wonder
  │
  └─→ Adolescence
        └─→ Growing Up Study
```

## Table Relationship Matrix

Shows which identifiers link which tables:

| From Table | To Table | Link Via | Cardinality |
|------------|----------|----------|-------------|
| PERSON (mother) | PREGNANCY | person_id → mother_id | 1:Many |
| PERSON (father) | PREGNANCY | person_id → father_id | 1:Many |
| PREGNANCY | PERSON (child) | pregnancy_id | 1:Many |
| PERSON | PERSON | related_pairs table | Many:Many |
| PERSON | PERSON_ADDRESS | person_id | 1:Many |
| PERSON_ADDRESS | PROPERTY | uprn | Many:1 |
| PROPERTY | LSOA | lsoa_code | Many:1 |
| PERSON | BASELINE_SURVEY | person_id | 1:Many |
| PERSON | BIB1000_* | person_id | 1:Many |
| PERSON | BIOSAMPLES | person_id | 1:Many |
| BIOSAMPLES | METABOLOMICS | sample_id | 1:Many |
| PERSON | GENOTYPING | genotyping_info table | 1:1 |
| PERSON | DNA_METHYLATION | dnam_info table | 1:Many |
| PREGNANCY | MATERNITY_RECORDS | pregnancy_id | 1:Many |
| PREGNANCY | ULTRASOUND_SCAN | pregnancy_id | 1:Many |
| PERSON | CHILD_GROWTH | person_id | 1:Many |
| PERSON | EDUCATION_RECORDS | person_id | 1:Many |
| PERSON | SDQ_ASSESSMENTS | person_id | 1:Many |
| PROPERTY | AIR_QUALITY | uprn | 1:1 |
| PROPERTY | FOOD_ENVIRONMENT | uprn | 1:1 |
| PROPERTY | GREEN_SPACE | uprn | 1:1 |
| PROPERTY | NDVI_LONGITUDINAL | uprn | 1:Many |

## Key Identifier Reference

### Primary Keys

| Entity | Primary Key | Description | Example |
|--------|-------------|-------------|---------|
| PERSON | person_id | Unique individual identifier | P123456 |
| PREGNANCY | pregnancy_id | Unique pregnancy identifier | PRG78901 |
| PROPERTY | uprn | Unique Property Reference Number | 100012345678 |
| LSOA | lsoa_code | Lower Layer Super Output Area | E01010123 |

### Foreign Key Relationships

```
PREGNANCY table contains:
  - pregnancy_id (PK)
  - mother_id (FK → PERSON.person_id)
  - father_id (FK → PERSON.person_id)

PERSON_ADDRESS table contains:
  - person_id (FK → PERSON.person_id)
  - uprn (FK → PROPERTY.uprn)
  - move_in_date
  - move_out_date

BIOSAMPLES table contains:
  - sample_id (PK)
  - person_id (FK → PERSON.person_id)
  - sample_type
  - collection_date
```

## Common Data Integration Patterns

### Pattern 1: Mother → Pregnancy → Child

```
PERSON (mother)
  person_id = M123
       ↓
PREGNANCY
  pregnancy_id = PRG456
  mother_id = M123
  father_id = F789
       ↓
PERSON (child)
  person_id = C101
  pregnancy_id = PRG456
```

**SQL Pattern:**
```sql
SELECT 
  m.person_id as mother_id,
  p.pregnancy_id,
  c.person_id as child_id
FROM person m
JOIN pregnancy p ON m.person_id = p.mother_id
JOIN person c ON p.pregnancy_id = c.pregnancy_id
WHERE m.person_type = 'mother'
  AND c.person_type = 'child'
```

### Pattern 2: Sibling Identification

```
PERSON (child1) ←──┐
  person_id = C101  │
                    ├── same pregnancy_id = PRG456
PERSON (child2) ←──┘
  person_id = C102
```

**Or via RELATED_PAIRS:**
```
RELATED_PAIRS
  person_id_1 = C101
  person_id_2 = C102
  relationship_type = 'sibling'
```

### Pattern 3: Longitudinal Person Data

```
PERSON (child)
  person_id = C101
       │
       ├─→ BIB1000_6M  (age: 6m)
       ├─→ BIB1000_12M (age: 12m)
       ├─→ BIB1000_18M (age: 18m)
       ├─→ BIB1000_24M (age: 24m)
       └─→ BIB1000_36M (age: 36m)
```

**SQL Pattern:**
```sql
SELECT person_id, 6 as months, measurement_date, bmi
FROM bib1000_6m WHERE person_id = 'C101'
UNION ALL
SELECT person_id, 12, measurement_date, bmi
FROM bib1000_12m WHERE person_id = 'C101'
UNION ALL
SELECT person_id, 18, measurement_date, bmi
FROM bib1000_18m WHERE person_id = 'C101'
```

### Pattern 4: Person → Environment Exposure

```
PERSON
  person_id = C101
       ↓
PERSON_ADDRESS
  uprn = 100012345678
  move_in_date = 2010-01-15
  move_out_date = NULL (current)
       ↓
PROPERTY
  uprn = 100012345678
  postcode = BD3 8QH
  lsoa_code = E01010123
       ↓
    ┌──┴──┬─────────┬────────────┐
    ▼     ▼         ▼            ▼
AIR_QUALITY  FOOD_ENV  GREEN_SPACE  LSOA_CHARS
```

## Study-Specific Data Structures

### BiB 1000 Study Tables

```
BIB1000_6M_MAIN
BIB1000_12M_MAIN  
BIB1000_18M_MAIN  ◄── All link via person_id
BIB1000_24M_MAIN
BIB1000_36M_MAIN

Each timepoint has:
  - _MAIN table (primary data)
  - Multiple child tables for repeated measures
    (e.g., _B12F10, _B12F11 for food items)
```

### Growing Up Study Tables

```
PERSON
  │
  ├─→ GROWING_UP_PARTICIPANT_PATHWAY
  │
  ├─→ GROWING_UP_ADULT_SURVEY
  │     ├─→ BiB1000 sample subset
  │     └─→ MeDALL sample subset
  │
  ├─→ GROWING_UP_CHILD_SURVEY
  │     ├─→ Adult completed
  │     └─→ Child completed
  │
  ├─→ GROWING_UP_DXA_SCAN
  │     ├─→ Mother scan
  │     ├─→ Child scan  
  │     └─→ Father scan
  │
  └─→ GROWING_UP_BLOOD_PRESSURE
```

### Omics Data Linkage

```
PERSON
  person_id
       │
       ├─→ GENOTYPING_INFO (linkage table)
       │      └─→ GENOTYPING_DATA (array_id)
       │
       ├─→ EXOMESEQ_INFO (linkage table)
       │      └─→ EXOME_DATA (exome_id)
       │
       ├─→ DNAM_INFO (linkage table)
       │      └─→ METHYLATION_DATA (array_id)
       │
       └─→ BIOSAMPLES
              sample_id
                 │
                 ├─→ METABOLOMICS (sample_id)
                 ├─→ PROTEOMICS (sample_id)
                 └─→ GLYCOMICS (sample_id)
```

## Data Integration Examples

### Example 1: Full Family Profile

```
Query: Get mother's pregnancy data + child's growth + environmental exposure

PERSON (Mother M123)
    │
    ├─→ PREGNANCY (PRG456)
    │     ├─→ MATERNITY_RECORDS
    │     └─→ PREGNANCY_BLOODS
    │
    └─→ PERSON (Child C101)
          │
          ├─→ CHILD_GROWTH (multiple timepoints)
          │
          └─→ PERSON_ADDRESS → PROPERTY
                                  │
                                  ├─→ AIR_QUALITY
                                  └─→ GREEN_SPACE
```

### Example 2: Longitudinal BMI Trajectory

```
Track one child's BMI from birth through school age:

PERSON (C101)
    │
    ├─→ PREGNANCY → birth_weight
    ├─→ CHILD_GROWTH (0-36m) → research measurements
    ├─→ BIB1000_* (6m-36m) → reported weight/height
    ├─→ STARTING_SCHOOL → reception measurements
    ├─→ CHILD_GROWTH_NCMP → school measurements
    └─→ GROWING_UP → age 7-11 measurements
```

### Example 3: Multi-Generational Analysis

```
Analyze family patterns across generations:

PERSON (Grandmother)
    │
    └─→ PREGNANCY_1
          │
          └─→ PERSON (Mother) 
                │
                ├─→ BASELINE_SURVEY (as mother)
                │
                └─→ PREGNANCY_2
                      │
                      └─→ PERSON (Child/Grandchild)
                            │
                            └─→ BIB1000_*
                            └─→ EDUCATION_RECORDS
```

## Table Naming Convention Patterns

Tables follow these naming patterns which indicate relationships:

| Pattern | Example | Meaning |
|---------|---------|---------|
| `bib_cohortinfo_*` | `bib_cohortinfo_person_info` | Core cohort demographics |
| `bib_baseline_*` | `bib_baseline_base_m_survey` | Baseline data collection |
| `bib_1000_bib1000_Xm_*` | `bib_1000_bib1000_12m_main` | BiB 1000 at X months |
| `bib_geographic_bib_geog_*` | `bib_geographic_bib_geog_person` | Geographic at person level |
| `bib_biosamples_*` | `bib_biosamples_pregnancy_bloods` | Biological samples |
| `bib_metabolomics_met*_*` | `bib_metabolomics_metms_2k_s` | Metabolomics data |
| `bibbs_*` | `bibbs_cohortinfo_pregnancy` | BiBBS cohort (newer) |
| `bib4all_*` | `bib4all_geographic_*` | BiB4All extended cohort |

## Quick Reference: How to Link...

### To link siblings:
```
Use: related_pairs table
Match: person_id_1 & person_id_2 where relationship_type = 'sibling'
OR: Match pregnancy_id in PERSON table
```

### To link mother and child:
```
Use: pregnancy table
PERSON (mother).person_id = PREGNANCY.mother_id
PREGNANCY.pregnancy_id = PERSON (child).pregnancy_id
```

### To link person to environment:
```
Use: person_id → person_address → uprn → property → geographic tables
Join chain: PERSON → PERSON_ADDRESS → PROPERTY → [AIR_QUALITY|GREEN_SPACE|etc.]
```

### To link across timepoints:
```
Use: Same person_id in different timepoint tables
E.g., bib1000_6m.person_id = bib1000_12m.person_id
```

### To link omics to person:
```
Use: Linkage tables (genotyping_info, dnam_info, exomeseq_info)
PERSON.person_id → *_INFO.person_id → *_INFO.array_id → OMICS_DATA.array_id
```

---

**Note:** This visual guide complements [DATA_MODEL.md](DATA_MODEL.md). If Mermaid diagrams render properly in your viewer, use that document for interactive diagrams.
