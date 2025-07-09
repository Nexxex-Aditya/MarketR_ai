from src.main import Orchestrator
from pathlib import Path

if __name__ == '__main__':
    # Path to your configuration file
    config_path = Path('config/config.yaml')
    user_query = 'Analyze trends in premium spirits sales and their correlation with market clusters'
    user_inputs = """What if a new competitor enters the market, reducing our sales by 10%?"""  # Optionally, pass custom scenario inputs
    business_rules = '''
- Always mention % growth in sales and % impact on sales
- Highlight any region with >5% impact
- Summarize cross-category effects if present
- Provide actionable recommendations
'''

    print("Initializing Orchestrator...")
    orchestrator = Orchestrator(config_path=str(config_path))

    print("Running analysis pipeline...")
    results = orchestrator.run_analysis(user_query, user_inputs, business_rules)

    print('\n--- ORCHESTRATOR ANALYSIS RESULTS ---')
    # The results structure has changed to a single dictionary containing all outputs
    # Access report_output for structured summary and reports
    report_output = results.get('report_output', {})
    summary_data = report_output.get('summary_data', {})

    print('\n--- Executive Summary ---')
    print(summary_data.get('executive_summary', 'No executive summary.'))

    print('\n--- Key Headlines ---')
    if summary_data.get('headlines'):
        for headline in summary_data['headlines']:
            print(f"- {headline}")
    else:
        print('No key headlines.')

    print('\n--- Generated Reports ---')
    if report_output.get('generated_reports'):
        for report_type, path in report_output['generated_reports'].items():
            print(f"- {report_type.capitalize()} Report Path: {path}")
    else:
        print('No reports generated.')

    # For detailed agent outputs, you might need to inspect the full results dictionary
    # For example, to see raw forecast or news data:
    # print('\n--- Raw Forecast Results ---')
    # print(results.get('forecast_results', {}))
    # print('\n--- Raw News Results ---')
    # print(results.get('news_results', {}))
    # print('\n--- Raw Simulation Results ---')
    # print(results.get('simulation_results', {})) 