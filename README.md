# Born in Bradford Data Dictionary

A comprehensive interactive data dictionary documenting all datasets, variables, and data structures from the **Born in Bradford (BiB)** longitudinal birth cohort study.

## 📖 About

Born in Bradford is a large-scale research program following families in Bradford, UK, to understand how genetic, nutritional, environmental, and social factors affect health and development from pregnancy through childhood and beyond. This repository contains the data dictionary documentation for the BiB cohort and related sub-studies.

## 🗂️ Repository Contents

This repository contains **321 HTML documentation pages** (~170 MB total) organized as an interactive bookdown website, providing detailed information about:

- **Variable definitions** and metadata
- **Data collection methods** and timepoints
- **Data linkage** information
- **Coding schemes** and value labels
- **Quality control** information

### 📊 Main Data Categories

The data dictionary covers the following major research areas and sub-studies:

#### **Core Cohort Data**
- **BiB Cohort Information** - Participant demographics, ethnicity, family relationships, ID linkage
- **BiB Baseline** - Maternal and paternal baseline surveys, food frequency questionnaires, exercise data
- **BiBBS (Born in Bradford's Better Start)** - Newer cohort recruitment and baseline data
- **BiB4All** - Extended cohort geographic and dental data

#### **Health & Clinical Data**
- **Pregnancy** - Maternity records, ultrasound scans, fetal renal data, blood pressure
- **Biosamples** - Blood tests, cord blood, GTT, biochemistry, haematology
- **Biobank** - Current biobank sample inventory
- **Congenital Anomalies** - Yorkshire & Humber CAR and GP linkage data
- **Dental** - Dental surveys, extractions, PLATOON study data

#### **Child Development & Growth**
- **BiB 1000** - Longitudinal questionnaires at 6m, 12m, 18m, 24m, 36m
- **Child Growth** - Anthropometry, bioimpedance, NCMP data, primary care records
- **Maternal Measurements** - Research measurement data
- **Starting School** - BPVS, letter identification, CKAT assessments
- **Primary School Years** - SDQ data, child quizzes, executive function tasks
- **Growing Up** - Adult and child surveys, DXA scans, blood pressure, renal studies
- **Age of Wonder** - Recent school visits, surveys, anthropometry (2023-2024 releases)

#### **Environmental Data**
- **Geographic Information** - Air quality, built environment, green space, food environment, NDVI, walkability
- **BREATHES** - Air quality survey data (Phase 1 & 2)

#### **Specialized Studies**
- **ALL IN** - Questionnaires on childcare, immunizations, household data (12m, 24m)
- **MeDALL** - Allergy questionnaires, skin prick tests, green space data
- **Education Records** - EYFSP, Key Stage assessments, phonics, contextual data

#### **Omics & Biomarkers**
- **Metabolomics** - MS and NMR metabolomics (mother, baby, child samples)
- **Proteomics** - Next generation sequencing, QPCR projects
- **Glycomics** - Total plasma glycome analyses (HILIC-UHPLC-FLR)
- **Genotyping** - Linkage to genotyping, exome sequencing, DNA methylation data

#### **Mental Health & Wellbeing**
- **SDQs** - Strength and Difficulties Questionnaires from multiple timepoints and studies
- **COVID-19 Surveys** - Phase 1, 2, and 3 survey data

## 🚀 How to Use

### Viewing the Data Dictionary

#### Quick Start - View in Browser

From the repository root directory, run:

```bash
# macOS
open docs/index.html

# Linux
xdg-open docs/index.html

# Windows
start docs/index.html

# Or use the full path
open /path/to/BornInBradford-datadict/docs/index.html
```

#### Clone and Browse

If you haven't cloned the repository yet:

```bash
git clone https://github.com/dizza01/BornInBradford-datadict.git
cd BornInBradford-datadict
open docs/index.html  # macOS
# or
xdg-open docs/index.html  # Linux
# or
start docs/index.html  # Windows
```

The data dictionary will open in your default web browser with full navigation and search functionality.

### Navigation

- Use the **left sidebar** to navigate between data modules
- Use the **search function** to find specific variables or tables
- Each table includes:
  - Variable names
  - Variable labels/descriptions
  - Value labels (for categorical variables)
  - Data types
  - Collection timepoints

### Finding Specific Information

**By Research Area**: Navigate using the main category pages (e.g., `bib_baseline.html`, `bib_pregnancy.html`)

**By Sub-study**: Look for specific study names (e.g., `bib_1000`, `bib_medall`, `bib_ageofwonder`)

**By Data Type**: Search for specific data types:
- Survey/questionnaire data (main tables with `_main.html` suffix)
- Geographic data (`bib_geographic` section)
- Clinical measurements (`bib_biosamples`, `bib_childgrowth`)
- Omics data (`bib_metabolomics`, `bib_proteomics`, `bib_glycomics`)

## 📋 Data Structure

### Naming Conventions

Data tables follow a hierarchical naming structure:

```
{cohort}_{study}_{subcategory}_{table}
```

**Examples:**
- `bib_baseline_base_m_survey` - BiB baseline maternal survey
- `bib_1000_bib1000_12m_main` - BiB 1000 study 12-month main questionnaire
- `bibbs_cohortinfo_pregnancy` - BiBBS cohort pregnancy information

### Data Relationships & Linkage

> **📊 [VIEW COMPLETE DATA MODEL](DATA_MODEL.md)** - Comprehensive Entity-Relationship diagrams showing all table relationships, identifiers, and linkage patterns.

#### Key Identifiers

The cohort uses various linking identifiers to connect tables:
- **Person-level IDs** - Unique identifiers for individuals (mothers, fathers, children)
- **Pregnancy IDs** - Link pregnancy-related data
- **Family relationship IDs** - Connect family members
- **Study-specific IDs** - Link to sub-study data (BiB 1000, MeDALL, etc.)
- **Property & LSOA codes** - Geographic linkage

#### Relationship Documentation

The data dictionary documents table relationships through:

1. **ID Linkage Section** ([bib_cohortinfo_id_linkage.html](docs/bib_cohortinfo_id_linkage.html))
   - Comprehensive guide to identifier types and how they link tables
   - Shows which IDs to use for joining datasets

2. **Related Pairs** ([bib_cohortinfo_related_pairs.html](docs/bib_cohortinfo_related_pairs.html))
   - Parent-child relationships
   - Sibling relationships  
   - Family linkage information

3. **Data Linkage Pages** - Specific sections for:
   - Genotyping data linkage
   - Exome sequencing data linkage
   - DNA methylation data linkage
   - GP record linkage to other datasets

4. **Hierarchical Structure** - Data is organized by:
   - **Person-level** → Individual participant data
   - **Pregnancy-level** → Linked to mothers via pregnancy ID
   - **Property-level** → Geographic data linked to residential addresses
   - **LSOA-level** → Area-level geographic and demographic data

5. **Temporal Relationships** - Longitudinal data linkage:
   - Same participants across timepoints (6m, 12m, 18m, 24m, 36m)
   - Baseline → Follow-up study linkage
   - Cross-sectional sub-studies linked to main cohort

## 🔍 Use Cases

This data dictionary is essential for:

1. **Data Analysts** - Understanding variable definitions before analysis
2. **Researchers** - Planning research proposals and identifying available data
3. **Collaborators** - Exploring what data exists within the BiB cohort
4. **Data Managers** - Reference for data structure and coding schemes
5. **Students** - Learning about longitudinal cohort study data organization

## 📦 Project Structure

```
BornInBradford-datadict/
├── docs/                      # 321 HTML documentation files
│   ├── index.html            # Main entry point
│   ├── bib_*.html            # BiB cohort documentation
│   ├── bibbs_*.html          # BiBBS cohort documentation
│   ├── bib4all_*.html        # BiB4All documentation
│   ├── covid19_*.html        # COVID-19 survey documentation
│   └── libs/                 # Supporting JavaScript/CSS libraries
├── datadict.Rproj            # R project file
├── README.md                 # This file
└── .nojekyll                 # Prevents Jekyll processing
```

## 🛠️ Technical Details

- **Generated using**: R bookdown package (v0.42)
- **Framework**: GitBook (v2.6.7)
- **Interactive tables**: reactable package
- **Total size**: ~170 MB (50 MB Git history + 120 MB working files)
- **HTML files**: 321 pages

## 📝 Contributing

This data dictionary is maintained by the Born in Bradford research team. For questions about:
- **Data access**: Contact the BiB Data Access Committee
- **Variable definitions**: Refer to study protocols or contact study leads
- **Technical issues**: Open an issue in this repository

## 📚 Related Resources

- [Born in Bradford Website](https://borninbradford.nhs.uk/)
- Data access requests and collaboration information
- Study protocols and publications

## ⚖️ License & Data Access

This data dictionary is publicly available for reference. However, **access to the actual data** requires:
- Approval from the BiB Data Access Committee
- Appropriate ethical approvals
- Signed data sharing agreements

The data dictionary helps researchers understand what data exists before applying for access.

## 📧 Contact

For more information about Born in Bradford or data access, please visit the official Born in Bradford website.

---

**Last Updated**: February 2026  
**Repository maintained by**: BiB Data Team
