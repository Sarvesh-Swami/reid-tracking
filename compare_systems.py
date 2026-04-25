"""
Compare OSNet vs Gemini Person Tracking Systems
================================================
Runs both systems and generates a comparison report.

Usage:
    python compare_systems.py --source test_6.mp4
"""

import argparse
import subprocess
import json
import time
from pathlib import Path


def run_osnet_system(video_path, output_path):
    """Run the current OSNet-based system."""
    print("\n" + "="*60)
    print("RUNNING OSNET SYSTEM (Current)")
    print("="*60)
    
    start_time = time.time()
    
    cmd = [
        'python', 'track_attendance.py',
        '--source', video_path,
        '--output', output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        # Parse output for person count
        output = result.stdout
        person_count = None
        for line in output.split('\n'):
            if 'Total unique persons:' in line:
                person_count = int(line.split(':')[1].strip())
                break
        
        return {
            'success': result.returncode == 0,
            'person_count': person_count,
            'time': elapsed,
            'output': output_path,
            'error': result.stderr if result.returncode != 0 else None
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'time': time.time() - start_time
        }


def run_gemini_system(video_path, output_path, api_key=None):
    """Run the Gemini-based system."""
    print("\n" + "="*60)
    print("RUNNING GEMINI SYSTEM (New)")
    print("="*60)
    
    start_time = time.time()
    
    cmd = [
        'python', 'track_gemini.py',
        '--source', video_path,
        '--output', output_path
    ]
    
    if api_key:
        cmd.extend(['--api-key', api_key])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        # Parse output for person count
        output = result.stdout
        person_count = None
        for line in output.split('\n'):
            if 'Total unique persons:' in line:
                person_count = int(line.split(':')[1].strip())
                break
        
        return {
            'success': result.returncode == 0,
            'person_count': person_count,
            'time': elapsed,
            'output': output_path,
            'error': result.stderr if result.returncode != 0 else None
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'time': time.time() - start_time
        }


def print_comparison(osnet_result, gemini_result, actual_persons=4):
    """Print comparison report."""
    print("\n" + "="*60)
    print("COMPARISON REPORT")
    print("="*60)
    
    print(f"\nActual persons in video: {actual_persons}")
    print()
    
    # OSNet Results
    print("OSNet System (Current):")
    if osnet_result['success']:
        print(f"  ✓ Success")
        print(f"  Persons detected: {osnet_result['person_count']}")
        print(f"  Processing time: {osnet_result['time']:.1f}s")
        print(f"  Output: {osnet_result['output']}")
        if osnet_result['person_count']:
            accuracy = (actual_persons / osnet_result['person_count']) * 100
            print(f"  Accuracy: {accuracy:.1f}% ({actual_persons}/{osnet_result['person_count']})")
    else:
        print(f"  ✗ Failed: {osnet_result.get('error', 'Unknown error')}")
    
    print()
    
    # Gemini Results
    print("Gemini System (New):")
    if gemini_result['success']:
        print(f"  ✓ Success")
        print(f"  Persons detected: {gemini_result['person_count']}")
        print(f"  Processing time: {gemini_result['time']:.1f}s")
        print(f"  Output: {gemini_result['output']}")
        if gemini_result['person_count']:
            accuracy = (actual_persons / gemini_result['person_count']) * 100
            print(f"  Accuracy: {accuracy:.1f}% ({actual_persons}/{gemini_result['person_count']})")
    else:
        print(f"  ✗ Failed: {gemini_result.get('error', 'Unknown error')}")
    
    print()
    
    # Comparison
    if osnet_result['success'] and gemini_result['success']:
        print("Improvement:")
        
        if osnet_result['person_count'] and gemini_result['person_count']:
            improvement = ((osnet_result['person_count'] - gemini_result['person_count']) / 
                          osnet_result['person_count']) * 100
            print(f"  Duplicate reduction: {improvement:.1f}%")
            print(f"  ({osnet_result['person_count']} → {gemini_result['person_count']} persons)")
        
        time_diff = gemini_result['time'] - osnet_result['time']
        time_ratio = gemini_result['time'] / osnet_result['time']
        print(f"  Time difference: +{time_diff:.1f}s ({time_ratio:.1f}x slower)")
        
        print()
        print("Recommendation:")
        if gemini_result['person_count'] and gemini_result['person_count'] <= actual_persons + 2:
            print("  ✅ Gemini system is significantly better!")
            print("  Use Gemini for production.")
        elif gemini_result['person_count'] and gemini_result['person_count'] < osnet_result['person_count']:
            print("  ✅ Gemini system is better, but not perfect.")
            print("  Consider using Gemini with post-processing.")
        else:
            print("  ⚠️ Gemini system needs tuning.")
            print("  Try different batch size or model.")
    
    print("\n" + "="*60)
    
    # Save report
    report = {
        'actual_persons': actual_persons,
        'osnet': osnet_result,
        'gemini': gemini_result,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    report_file = 'comparison_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare OSNet vs Gemini Systems')
    parser.add_argument('--source', type=str, required=True, help='Input video path')
    parser.add_argument('--actual-persons', type=int, default=4, 
                        help='Actual number of persons in video')
    parser.add_argument('--api-key', type=str, help='Gemini API key')
    parser.add_argument('--skip-osnet', action='store_true', 
                        help='Skip OSNet system (only run Gemini)')
    parser.add_argument('--skip-gemini', action='store_true', 
                        help='Skip Gemini system (only run OSNet)')
    args = parser.parse_args()
    
    video_path = args.source
    
    # Check video exists
    if not Path(video_path).exists():
        print(f"ERROR: Video not found: {video_path}")
        return
    
    print("\n" + "="*60)
    print("SYSTEM COMPARISON")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Actual persons: {args.actual_persons}")
    print("="*60)
    
    # Run OSNet system
    osnet_result = None
    if not args.skip_osnet:
        osnet_result = run_osnet_system(video_path, 'output_osnet_compare.mp4')
    else:
        print("\nSkipping OSNet system...")
        osnet_result = {'success': False, 'error': 'Skipped'}
    
    # Run Gemini system
    gemini_result = None
    if not args.skip_gemini:
        gemini_result = run_gemini_system(video_path, 'output_gemini_compare.mp4', args.api_key)
    else:
        print("\nSkipping Gemini system...")
        gemini_result = {'success': False, 'error': 'Skipped'}
    
    # Print comparison
    print_comparison(osnet_result, gemini_result, args.actual_persons)


if __name__ == '__main__':
    main()
