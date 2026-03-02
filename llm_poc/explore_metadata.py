"""
Utility script to explore BiB CSV metadata
Helps understand what's in the metadata files before using in LLM POC
"""

import pandas as pd
from pathlib import Path
import sys

def explore_metadata(csv_path: str):
    """Explore the BiB metadata CSV files"""
    
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"❌ Path not found: {csv_path}")
        return
    
    print("="*70)
    print("BiB METADATA EXPLORER")
    print("="*70)
    
    # Load master files
    try:
        tables_df = pd.read_csv(csv_path / "all_tables.csv")
        variables_df = pd.read_csv(csv_path / "all_variables_meta.csv")
        
        print(f"\n✅ Successfully loaded metadata")
        print(f"   📊 Tables: {len(tables_df)}")
        print(f"   📝 Variables: {len(variables_df)}")
        
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return
    
    # Show table overview
    print("\n" + "="*70)
    print("TABLE OVERVIEW")
    print("="*70)
    
    print("\nProjects:")
    print(tables_df['project_name'].value_counts().head(10))
    
    print("\nTables with most variables:")
    top_tables = tables_df.nlargest(10, 'n_variables')[['table_name', 'n_variables', 'n_rows']]
    print(top_tables.to_string(index=False))
    
    # Show variable types
    print("\n" + "="*70)
    print("VARIABLE OVERVIEW")
    print("="*70)
    
    print("\nVariable types:")
    print(variables_df['value_type'].value_counts())
    
    print("\nVariables with topics:")
    topic_counts = variables_df['topic'].value_counts()
    print(topic_counts.head(15))
    
    # Show Age of Wonder example
    print("\n" + "="*70)
    print("AGE OF WONDER EXAMPLE")
    print("="*70)
    
    aow_tables = tables_df[tables_df['project_name'] == 'BiB_AgeOfWonder']
    print(f"\n{len(aow_tables)} Age of Wonder tables:")
    print(aow_tables[['table_name', 'n_variables', 'n_rows']].to_string(index=False))
    
    # Show RCADS variables
    print("\n\nRCADS Variables (from Age of Wonder survey):")
    rcads_vars = variables_df[
        (variables_df['table_id'] == 'BiB_AgeOfWonder.survey_mod02_dr23') &
        (variables_df['variable'].str.contains('rcads', case=False, na=False))
    ]
    
    if not rcads_vars.empty:
        print(rcads_vars[['variable', 'label', 'value_type']].head(10).to_string(index=False))
    
    # Interactive search
    print("\n" + "="*70)
    print("INTERACTIVE SEARCH")
    print("="*70)
    
    while True:
        keyword = input("\nSearch for variable (or 'quit' to exit): ").strip()
        
        if keyword.lower() in ['quit', 'exit', 'q', '']:
            break
        
        results = variables_df[
            variables_df['variable'].str.contains(keyword, case=False, na=False) |
            variables_df['label'].str.contains(keyword, case=False, na=False)
        ]
        
        if results.empty:
            print(f"   No variables found matching '{keyword}'")
        else:
            print(f"\n   Found {len(results)} variables:")
            print(results[['table_id', 'variable', 'label']].head(20).to_string(index=False))
            
            if len(results) > 20:
                print(f"\n   ... and {len(results) - 20} more")
    
    print("\n✅ Metadata exploration complete!\n")


if __name__ == "__main__":
    # Try to find metadata path
    possible_paths = [
        "/Users/dawud.izza/Desktop/BiB/BornInBradford-datadict/docs/csv",
        "BornInBradford-datadict/docs/csv",
        "../BornInBradford-datadict/docs/csv"
    ]
    
    metadata_path = None
    for path in possible_paths:
        if Path(path).exists():
            metadata_path = path
            break
    
    if metadata_path:
        explore_metadata(metadata_path)
    else:
        print("❌ Could not find BiB metadata CSV files")
        print("\nProvide path as argument:")
        print("  python explore_metadata.py /path/to/docs/csv")
        
        if len(sys.argv) > 1:
            explore_metadata(sys.argv[1])
