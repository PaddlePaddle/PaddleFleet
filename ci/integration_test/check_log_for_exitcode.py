"""
ci.integration_test.check_log_for_exitcode
"""
import os
import sys
def check_ci_logs_for_error(log_file_path):
   """
   check_ci_logs_for_error - Docstring
   """
   if not os.path.isfile(log_file_path):
       raise FileNotFoundError(f"Log file not found: {log_file_path}")
   error_patterns = [
                        r"ERROR:",
                        r"FAILED",
                        r"Traceback",
                        r"Exception:",
                        r"segmentation fault",
                        r"core dumped"
                    ]
   with open(log_file_path, 'r') as log_file:
       for line in log_file:
           for pattern in error_patterns:
               if pattern in line:
                   print("Found an error pattern in the log file.")
                   print("Error pattern:", pattern)
                   print("Line:")
                   print(line)
                   return True
   return False

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "ci_log.txt"
    try:
        error_found = check_ci_logs_for_error(log_file)
        if error_found:
            print("Errors were found in the CI log.")
            sys.exit(1)
        else:
            print("No errors found in the CI log.")
            sys.exit(0)
    except FileNotFoundError as e:
        print(e)
        sys.exit(2)
   