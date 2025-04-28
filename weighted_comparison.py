import pandas as pd
import numpy as np
import plotly.graph_objects as go
import webbrowser
import os
import datetime


def load_node_histories():
    """Load the node_history results from the .npy files"""
    #ACLED
    acled_result_file = 'acled_node_history_2.npy'
    #SATELLITE
    merged_result_file = 'merged_node_history_2.npy'
    
    acled_node_history = []
    merged_node_history = []
    
    if os.path.exists(acled_result_file) and os.path.exists(merged_result_file):
        print("Loading saved node history files...")
        
        acled_data = np.load(acled_result_file, allow_pickle=True)
        merged_data = np.load(merged_result_file, allow_pickle=True)
        
        print(f"ACLED data type: {type(acled_data)}")
        print(f"Merged data type: {type(merged_data)}")
        
        
        acled_node_history = process_node_history(acled_data)
        merged_node_history = process_node_history(merged_data)
        
        return acled_node_history, merged_node_history
    else:
        print(f"Could not find files: {acled_result_file} or {merged_result_file}")
        return [], []


def process_node_history(data):
    """Process the node history data into a list of monthly dataframes"""
    if len(data) == 0:
        return []
    
    
    if isinstance(data, list) or isinstance(data, np.ndarray):
        first_item = data[0]
        
        
        if isinstance(first_item, list) or isinstance(first_item, np.ndarray):
            print(f"Processing data with shape: {np.array(data).shape if hasattr(np, 'shape') else 'list'}")
            
            if len(first_item) > 0 and isinstance(first_item[0], (list, np.ndarray)) and len(first_item[0]) >= 3:
                
                processed_data = []
                
                for month_points in data:
                    
                    if len(month_points) > 0:
                        month_date = month_points[0][2] 
                        
                        
                        df = pd.DataFrame({
                            'latitude': [point[0] for point in month_points],
                            'longitude': [point[1] for point in month_points],
                            'timestamp': [point[2] for point in month_points]
                        })
                        df['month'] = month_date
                        processed_data.append(df)
                
                return processed_data
            
            
            elif len(first_item) >= 3:
                
                month_date = first_item[2]  
                
                df = pd.DataFrame({
                    'latitude': [point[0] for point in data],
                    'longitude': [point[1] for point in data],
                    'timestamp': [point[2] for point in data]
                })
                df['month'] = month_date
                return [df]
    
    
    print(f"Unknown data structure. First element: {data[0] if len(data) > 0 else 'empty'}")
    return []


def calculate_monthly_change(node_history):
    """Calculate the average movement of the frontline between consecutive months"""
    monthly_changes = []
    
    if len(node_history) <= 1:
        print("Need at least two months of data to calculate changes")
        return []
    
    for i in range(1, len(node_history)):
        prev_month = node_history[i-1]
        curr_month = node_history[i]
        
        
        prev_month_date = prev_month['month'].iloc[0]
        curr_month_date = curr_month['month'].iloc[0]
        
        
        if isinstance(prev_month_date, pd.Timestamp) or isinstance(prev_month_date, datetime.datetime):
            prev_month_name = prev_month_date.strftime('%b %Y')
            curr_month_name = curr_month_date.strftime('%b %Y')
            
            year = curr_month_date.strftime('%Y')
        else:
            prev_month_name = str(prev_month_date)
            curr_month_name = str(curr_month_date)
            year = curr_month_name.split()[-1]  
        
        period = f"{prev_month_name} to {curr_month_name}"
        
        prev_coords = np.column_stack((prev_month['latitude'], prev_month['longitude']))
        curr_coords = np.column_stack((curr_month['latitude'], curr_month['longitude']))
        
        min_length = min(len(prev_coords), len(curr_coords))
        prev_coords = prev_coords[:min_length]
        curr_coords = curr_coords[:min_length]
        distances = np.sqrt(np.sum((prev_coords - curr_coords)**2, axis=1))
        distances_km = distances * 111
        
        monthly_changes.append({
            'period': period,
            'year': year,  
            'avg_change_km': np.mean(distances_km),
            'max_change_km': np.max(distances_km),
            'median_change_km': np.median(distances_km),
            'total_change_km': np.sum(distances_km)
        })
    
    return monthly_changes


def compare_frontline_variations(acled_node_history, merged_node_history):
    """Compare the monthly variations between ACLED and merged datasets"""
    print("\nCalculating changes for ACLED dataset...")
    acled_changes = calculate_monthly_change(acled_node_history)
    
    print("Calculating changes for merged events dataset...")
    merged_changes = calculate_monthly_change(merged_node_history)
    
    if not acled_changes or not merged_changes:
        print("Error: Could not calculate changes for one or both datasets.")
        return
    
    acled_df = pd.DataFrame(acled_changes)
    merged_df = pd.DataFrame(merged_changes)
    
    print(f"ACLED changes: {len(acled_changes)} periods")
    print(f"Merged changes: {len(merged_changes)} periods")
    
    fig = go.Figure()

    acled_by_year = acled_df.copy()
    merged_by_year = merged_df.copy()
    
    fig.add_trace(go.Bar(
        x=acled_df['period'],
        y=acled_df['avg_change_km'],
        name='Combined Dataset',
        marker_color='blue',
        showlegend=False 
    ))
    
    
    fig_line = go.Figure()
    
    fig_line.add_trace(go.Scatter(
        x=acled_df['period'],
        y=acled_df['avg_change_km'],
        mode='lines+markers',
        name='Combined Dataset',
        line=dict(color='blue')
    ))
    
    fig_line.add_trace(go.Scatter(
        x=merged_df['period'],
        y=merged_df['avg_change_km'],
        mode='lines+markers',
        name='Satellite Dataset',
        line=dict(color='red')
    ))
    
    fig_line.update_layout(
        xaxis=dict(
            type='category',
            tickvals=[],
            ticktext=[],
            title=dict(
                text='By Month (Feb 2022-Mar 2025)',
                font=dict(size=30)
            )
        ),
        yaxis=dict(
            title=dict(
                text='Average Movement (km)',
                font=dict(size=30)
            )
        ),
        height=600
    )
    
    fig_line.write_html("frontline_comparison_line.html")
    

    webbrowser.open("frontline_comparison_line.html")
    
    comparison_table = []
    
    acled_periods = set(item['period'] for item in acled_changes)
    merged_periods = set(item['period'] for item in merged_changes)
    
    print("\nACLED Periods:", sorted(acled_periods))
    print("Merged Periods:", sorted(merged_periods))
    
    common_periods = acled_periods.intersection(merged_periods)
    print(f"Common periods: {len(common_periods)}")
    
    for period in sorted(common_periods):
        acled_value = next(item['avg_change_km'] for item in acled_changes if item['period'] == period)
        merged_value = next(item['avg_change_km'] for item in merged_changes if item['period'] == period)
        
        difference = merged_value - acled_value
        pct_diff = (difference / acled_value) * 100 if acled_value > 0 else 0
        
        comparison_table.append({
            'Period': period,
            'Combined (km)': round(acled_value, 2),
            'Satellite (km)': round(merged_value, 2),
            'Difference (km)': round(difference, 2),
            'Difference (%)': round(pct_diff, 1)
        })
    
    
    comparison_df = pd.DataFrame(comparison_table)
    print("\nComparison of Monthly Frontline Movement:")
    print(comparison_df)
    
    
    html_table = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Frontline Comparison Table</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
            th { background-color: #f2f2f2; text-align: center; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            h2 { color: #333; }
        </style>
    </head>
    <body>
        <h2>Comparison of Monthly Frontline Movement</h2>
        <table>
            <tr>
                <th>Period</th>
                <th>Combined (km)</th>
                <th>Satellite (km)</th>
                <th>Difference (km)</th>
                <th>Difference (%)</th>
            </tr>
    """
    
    for _, row in comparison_df.iterrows():
        html_table += f"""
            <tr>
                <td>{row['Period']}</td>
                <td>{row['Combined (km)']}</td>
                <td>{row['Satellite (km)']}</td>
                <td>{row['Difference (km)']}</td>
                <td>{row['Difference (%)']}</td>
            </tr>
        """
    
    html_table += """
        </table>
        <p>This table compares the average movement of the frontline between consecutive months for the Combined and Satellite datasets.</p>
    </body>
    </html>
    """
    
    with open("frontline_comparison_table.html", "w") as f:
        f.write(html_table)
    

if __name__ == "__main__":
    print("Loading frontline data from previous runs...")
    acled_node_history, merged_node_history = load_node_histories()
    
    if len(acled_node_history) > 0 and len(merged_node_history) > 0:
        print(f"Successfully loaded data: ACLED ({len(acled_node_history)} months), SATELLITE ({len(merged_node_history)} months)")
        compare_frontline_variations(acled_node_history, merged_node_history)
    else:
        print("Error")