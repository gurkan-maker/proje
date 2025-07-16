# valve_database.py
import os
import pandas as pd
from valve import Valve

VALVE_DATA_FILE = "valve_database.xlsx"

def get_valve_sheet_name(valve):
    """Generate standardized sheet name for a valve"""
    return f"Valve_{valve.size}_{valve.rating_class}_{valve.valve_type}"

def load_valves_from_excel():
    """Load valve data from Excel file using new sheet-per-valve structure"""
    valves = []
    if not os.path.exists(VALVE_DATA_FILE):
        # Create initial Excel file with new structure if it doesn't exist
        initial_valves = [
            Valve(1, 600, 
                {0:0.0, 10:3.28, 20:7.39, 30:12.0, 40:14.2, 50:14.9, 60:15.3, 70:15.7, 80:16.0, 90:16.4, 100:16.8},
                {0:0.68, 10:0.68, 20:0.68, 30:0.68, 40:0.68, 50:0.68, 60:0.68, 70:0.68, 80:0.68, 90:0.68, 100:0.68},
                {10:0.581, 20:0.605, 30:0.617, 40:0.644, 50:0.764, 60:0.790, 70:0.809, 80:0.813, 90:0.795, 100:0.768},
                1, 1.0, 3),
            # ... [other initial valves] ...
        ]
        save_valves_to_excel(initial_valves)
    
    try:
        # Read main valve list sheet
        main_df = pd.read_excel(VALVE_DATA_FILE, sheet_name='Valve List')
        
        for _, row in main_df.iterrows():
            sheet_name = row['sheet_name']
            # Read valve-specific sheet
            valve_df = pd.read_excel(VALVE_DATA_FILE, sheet_name=sheet_name)
            
            # Convert to dictionaries
            cv_table = {}
            fl_table = {}
            xt_table = {}
            
            for _, r in valve_df.iterrows():
                opening = r['Opening (%)']
                if not pd.isnull(r['Cv']):
                    cv_table[opening] = r['Cv']
                if not pd.isnull(r['Fl']):
                    fl_table[opening] = r['Fl']
                if not pd.isnull(r['Xt']):
                    xt_table[opening] = r['Xt']
            
            valve = Valve(
                size_inch=row['size'],
                rating_class=row['rating_class'],
                cv_table=cv_table,
                fl_table=fl_table,
                xt_table=xt_table,
                fd=row['fd'],
                d_inch=row['diameter'],
                valve_type=row['valve_type']
            )
            valves.append(valve)
        return valves
    except Exception as e:
        print(f"Error loading valves: {e}")
        return [
            Valve(2, 600, 
                {0:0, 10:5, 20:15, 30:30, 40:45, 50:60, 60:75, 70:85, 80:92, 90:98, 100:100},
                {0:0.5, 10:0.55, 20:0.6, 30:0.65, 40:0.7, 50:0.75, 60:0.8, 70:0.82, 80:0.83, 90:0.84, 100:0.85},
                {0:0.2, 10:0.3, 20:0.4, 30:0.5, 40:0.6, 50:0.65, 60:0.7, 70:0.75, 80:0.78, 90:0.8, 100:0.81},
                0.7, 2.0, 3)
        ]

def save_valves_to_excel(valves):
    """Save list of Valve objects to Excel with new structure"""
    # Create main valve list
    main_data = []
    valve_sheets = {}
    
    for valve in valves:
        sheet_name = get_valve_sheet_name(valve)
        main_data.append({
            'size': valve.size,
            'rating_class': valve.rating_class,
            'valve_type': valve.valve_type,
            'fd': valve.fd,
            'diameter': valve.diameter,
            'sheet_name': sheet_name
        })
        
        # Create valve data table
        all_openings = sorted(set(list(valve.cv_table.keys()) + 
                              list(valve.fl_table.keys()) + 
                              list(valve.xt_table.keys())))
        
        valve_data = []
        for opening in all_openings:
            valve_data.append({
                'Opening (%)': opening,
                'Cv': valve.cv_table.get(opening, None),
                'Fl': valve.fl_table.get(opening, None),
                'Xt': valve.xt_table.get(opening, None)
            })
        
        valve_sheets[sheet_name] = pd.DataFrame(valve_data)
    
    # Create Excel writer
    with pd.ExcelWriter(VALVE_DATA_FILE, engine='openpyxl') as writer:
        # Save main list
        pd.DataFrame(main_data).to_excel(
            writer, sheet_name='Valve List', index=False
        )
        
        # Save each valve sheet
        for sheet_name, df in valve_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def add_valve_to_database(valve):
    """Add a new valve to the database"""
    valves = load_valves_from_excel()
    valves.append(valve)
    save_valves_to_excel(valves)

def delete_valve_from_database(size, rating_class, valve_type):
    """Delete a valve from the database"""
    valves = load_valves_from_excel()
    valves = [v for v in valves if not (
        v.size == size and 
        v.rating_class == rating_class and 
        v.valve_type == valve_type
    )]
    save_valves_to_excel(valves)