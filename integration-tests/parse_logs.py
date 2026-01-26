import re
import csv
import sys
from datetime import datetime, timedelta

def parse_log_file(filename):
    tests = []
    current_test = None
    
    # Regex patterns
    # Remove ^ to be safe, matches pytest test Header
    test_start_pattern = re.compile(r'test/test_parametrized_deploydata\.py::(test_parametrized_deploy_data_(\d+)gb_total_(\d+)mb_per_chunk_(\d+)mb_propose)')
    
    running_state_pattern = re.compile(r'RunningStateEntered')
    received_deploy_pattern = re.compile(r'Received DeployData')
    chunk_pattern = re.compile(r'@(\d+)!')
    
    # Success pattern with duration extraction
    success_pattern = re.compile(r'Deploy and propose took (\d+\.\d+) seconds')
    
    crash_patterns = [
        re.compile(r'OutOfMemoryError', re.IGNORECASE),
        re.compile(r'socket closed', re.IGNORECASE),
        re.compile(r'java\.util\.concurrent\.TimeoutException', re.IGNORECASE),
        re.compile(r'Connection reset', re.IGNORECASE)
    ]
    
    test_end_pattern = re.compile(r'(FAILED|PASSED)\s+\[')
    
    # Summary section start
    summary_start_pattern = re.compile(r'^={10,}')

    with open(filename, 'r', errors='replace') as f:
        lines = f.readlines()

    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Stop if we hit the short test summary info section (end of file)
        # We ignore "test durations" because it can appear early for PYLINT
        if summary_start_pattern.match(line):
            if "short test summary info" in line:
                break
        
        # 1. Detect Test Start
        m_start = test_start_pattern.search(line)
        if m_start:
            if current_test:
                finalize_test(current_test)
                tests.append(current_test)
            
            test_name = m_start.group(1)
            chunk_mb = m_start.group(3)
            propose_mb = m_start.group(4)
            
            current_test = {
                'name': test_name,
                'mb_chunk': chunk_mb,
                'mb_propose': propose_mb,
                'started_at': None,
                'finished_at': None,
                'duration': None,
                'status': 'Running',
                'last_chunk': '0',
                'raw_start': None,
                'raw_end': None,
                'crash_reason': None,
                'last_activity_time': None,
                'raw_last_activity': None,
                'explicit_duration': None
            }
            i += 1
            continue

        if current_test:
            ts = extract_timestamp(line)
            if ts:
                # If test is explicitly finished, don't update activity time from subsequent logs
                # to avoid picking up unrelated timestamps before the next test starts
                if current_test['status'] != 'Finished':
                    current_test['last_activity_time'] = ts
                    current_test['raw_last_activity'] = parse_time(ts)

            if current_test['status'] == 'Running':
                if not current_test['started_at'] and running_state_pattern.search(line):
                    if i + 1 < len(lines):
                        next_line = lines[i+1]
                        if received_deploy_pattern.search(next_line):
                            ts_start = extract_timestamp(next_line)
                            if ts_start:
                                current_test['started_at'] = ts_start
                                current_test['raw_start'] = parse_time(ts_start)
                
                m_chunk = chunk_pattern.search(line)
                if m_chunk:
                    current_test['last_chunk'] = m_chunk.group(1)

                m_success = success_pattern.search(line)
                if m_success:
                    current_test['explicit_duration'] = float(m_success.group(1))
                    ts_end = extract_timestamp(line)
                    if ts_end:
                        current_test['finished_at'] = ts_end
                        current_test['raw_end'] = parse_time(ts_end)
                    current_test['status'] = 'Finished'
                
                for cp in crash_patterns:
                    if cp.search(line):
                        current_test['status'] = 'Crashed'
                        current_test['crash_reason'] = line.strip()[:50]
                        if ts:
                            current_test['finished_at'] = ts
                            current_test['raw_end'] = parse_time(ts)
                        break
                
                if test_end_pattern.search(line):
                    finalize_test(current_test)
                    tests.append(current_test)
                    current_test = None
        
        i += 1
        
    if current_test:
        finalize_test(current_test)
        tests.append(current_test)

    # Output CSV
    headers = ['Test Name', 'MB/Chunk', 'MB/Propose', 'Started At', 'Finished At', 'Duration', 'Crashed/Finished At Chunk']
    writer = csv.writer(sys.stdout)
    writer.writerow(headers)
    
    valid_tests = [t for t in tests if t['started_at'] is not None]
    
    for t in valid_tests:
        chunk_info = t['last_chunk']
        if t['status'] == 'Crashed':
            chunk_info = f"Crashed at chunk #{t['last_chunk']}"
        elif t['status'] == 'Finished':
            chunk_info = f"Finished (Last chunk #{t['last_chunk']})"
        elif t['status'] == 'Running':
            # Incomplete log? Assume crashed/stopped at last activity
            chunk_info = f"Crashed/Stopped at chunk #{t['last_chunk']}"
            
        writer.writerow([
            t['name'],
            t['mb_chunk'],
            t['mb_propose'],
            t['started_at'],
            t['finished_at'],
            t['duration'],
            chunk_info
        ])

def finalize_test(t):
    if not t['finished_at']:
        t['finished_at'] = t['last_activity_time']
        t['raw_end'] = t['raw_last_activity']
    
    if t['explicit_duration']:
        # Format explicit duration as H:M:S
        d = t['explicit_duration']
        td = timedelta(seconds=d)
        t['duration'] = str(td)
        
        # If we have start time and duration, we can backfill 'Finished At' if it was missing/stale
        if t['raw_start'] and not t['finished_at']:
            t['raw_end'] = t['raw_start'] + td
            t['finished_at'] = t['raw_end'].strftime("%H:%M:%S.%f")[:-3]

    elif t['raw_start'] and t['raw_end']:
        diff = t['raw_end'] - t['raw_start']
        if diff.total_seconds() < 0:
            diff = diff + timedelta(days=1)
        t['duration'] = str(diff)
    else:
        t['duration'] = "N/A"

def extract_timestamp(line):
    m = re.search(r'(\d{2}:\d{2}:\d{2}\.\d{3})', line)
    if m:
        return m.group(1)
    return None

def parse_time(ts_str):
    try:
        return datetime.strptime(ts_str, "%H:%M:%S.%f")
    except:
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_logs.py <logfile>")
        sys.exit(1)
    parse_log_file(sys.argv[1])
