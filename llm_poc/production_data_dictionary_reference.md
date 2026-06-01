# BiB Data Dictionary Quick Reference

This reference summarizes the structure of the Born in Bradford data dictionary for production assistant answers. Use it as orientation only. Exact variables, tables, labels, row counts, and completeness counts must come from the retrieved context or CSV-backed registry.

## Core Structure

- The data dictionary is organised into projects, tables, and variables.
- A full variable identifier has the form `Project.table.variable`, for example `BiB_CohortInfo.person_info.BiBPersonID`.
- A table identifier has the form `Project.table`, for example `BiB_AgeOfWonder_2024.survey_mod232_derived`.
- The variable registry is built from `docs/csv/all_variables_meta.csv` and enriched with section/question text parsed from `docs/*.html`.
- The table registry is built from `docs/csv/all_tables.csv`.
- When answering variable questions, prefer full `variable_id`, `variable`, `table`, `label`, `type`, and `n_complete` / non-missing count where available.
- Never expose or recommend row-level use of `BiBPersonID`; it is the privacy-sensitive linkage identifier.

## Main Study Contexts

- `Core Cohort`: person, pregnancy, mother, father, child, ethnicity, related-pair, and linkage metadata.
- `Baseline`: pregnancy/birth-era survey and questionnaire data from the original BiB recruitment period.
- `BiB 1000`: early-life follow-up waves, commonly labelled 6m, 12m, 18m, 24m, and 36m.
- `Starting School`: early school-age assessments, including school-readiness and assessment data.
- `Primary School Years`: primary-school-age assessment data, including CKAT and other child assessment tables.
- `Age of Wonder`: adolescent data collection, including 2023, 2024, and 2025 survey modules plus measurement tables.
- `Growing Up`: later childhood/adolescent follow-up surveys and measures.
- `BiBBS`: Better Start cohort data, including baseline, cohort info, dental, and geographic data.
- `COVID-19 Surveys`: pandemic-era adult and child survey phases.
- `Geographic Linkage`: environmental and spatial linkage data such as green space, air quality, food environment, LSOA, walkability, property, and transport.
- `Biosamples & Biobank`: bloods, cord bloods, pregnancy bloods, renal samples, and current sample metadata.
- `Metabolomics`, `Proteomics`, and `Glycomics`: omics assay outputs, annotations, QC, and result tables.
- `Pregnancy & Birth`: maternity records, ultrasound, fetal renal, congenital anomalies, and pregnancy-related clinical data.

## Registry Versus RAG Behaviour

- Use the deterministic registry for questions asking what variables, fields, measures, datasets, tables, or cohorts exist.
- Use paper/questionnaire retrieval for questions asking what a measure means, what a questionnaire asked, what a published study found, or how an instrument is described.
- If the user asks for “all variables” related to a topic, do not rely on the LLM to remember or invent the list. Use registry results and include the exportable set.
- If the user asks for cohorts/studies/waves where variables occur, summarize registry matches by `study_context`, with counts and examples.
- If the user asks for a specific variable, explain the variable from its registry row and any retrieved questionnaire or paper context.
- If the user asks for questionnaire item wording, prefer questionnaire/PDF evidence over derived score variables.

## Common Naming Notes

- `rcad`, `rcads`, `RCADS-25`, and `RCADS25` may refer to the Revised Child Anxiety and Depression Scale 25-item instrument. Derived score variables such as `rcad_ga`, `rcad_md`, and `rcad_total` are not the same thing as the item wording.
- `CKAT` appears in BiB as the Clinical Kinematic Assessment Tool and is associated with sensorimotor/cognitive assessment data in Starting School and Primary School Years contexts.
- `SDQ` refers to Strengths and Difficulties Questionnaire data.
- `BPVS` commonly refers to British Picture Vocabulary Scale data.
- `EYFSP`, `KS1`, `KS2`, and related terms usually refer to education-record outcomes.

## Answering Rules

- Ground concise factual answers in retrieved context or registry data.
- Do not turn every answer into a variable table. Include variable tables only when the user asks about variables, fields, measures, data availability, or export.
- For paper questions, list title, year, and the reason the paper is relevant. Do not infer findings beyond retrieved paper context.
- For questionnaire questions, cite the questionnaire title or PDF/source name when available.
- For ambiguous requests, briefly state the interpretation used, then answer from the most authoritative available source.
